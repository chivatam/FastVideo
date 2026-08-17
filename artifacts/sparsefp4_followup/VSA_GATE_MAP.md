# VSA gate/selector map (Phase F2.2)

Read before interpreting any F2 result. Every claim below is a code reading of the
**installed** kernel wheel, with file and line references, not an inference from
VSA's paper.

Sources:

- `fastvideo/attention/backends/video_sparse_attn.py` (FastVideo backend)
- `fastvideo/models/dits/wanvideo.py` (Wan DiT call site)
- `site-packages/fastvideo_kernel/ops.py` (kernel entry points)
- `site-packages/fastvideo_kernel/triton_kernels/fused_compress_topk.py` (pooling + top-k)

Installed wheel: `fastvideo_kernel` at `/mnt/nvme/scratch/fv-venv/lib/python3.12/site-packages/fastvideo_kernel`.

---

## Headline: `gate_compress` is **not** the selector

This is the single most important finding of F2.2 and it corrects a natural
reading of the phase brief, which asked us to test precision "at the VSA gate".

`gate_compress` is produced by a learned linear layer in the DiT block:

```458:462:fastvideo/models/dits/wanvideo.py
        self.to_gate_compress = ReplicatedLinear(dim,
```

and is threaded into the backend as `gate_compress`
(`wanvideo.py:542`, `:552`, `:560`), then handed to the kernel as the keyword
argument **`compress_attn_weight`**:

```318:326:fastvideo/attention/backends/video_sparse_attn.py
        if block_elements == 256 and video_sparse_attn_bshd is not None:
            return video_sparse_attn_bshd(query,
                                          key,
                                          value,
                                          attn_metadata.variable_block_sizes,
                                          attn_metadata.variable_block_sizes,
                                          cur_topk,
                                          block_size=VSA_TILE_SIZE,
                                          compress_attn_weight=gate_compress)
```

Inside the kernel it is used **only** as a multiplicative weight on the
*compression branch output*, after selection has already happened:

```141:143:site-packages/fastvideo_kernel/ops.py
    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s
```

So `gate_compress`:

- does **not** enter the score computation,
- does **not** enter the top-k selection,
- cannot change which blocks are retained,
- only rescales the dense-ish coarse branch that is *added* to the sparse branch.

**Consequence for the study:** quantizing `gate_compress` is not a routing
intervention at all. Intervention C as literally worded in the brief
("preserve the reference VSA `gate_compress`, quantize it before selection")
is not applicable to this implementation, because `gate_compress` is not
consumed by selection. Doing it anyway and calling it a routing result would be
wrong. We therefore run it as an explicitly-labelled **non-routing** control
(it perturbs the compression branch amplitude) and put the real routing
interventions where selection actually happens.

---

## The real selector

`video_sparse_attn` computes its own mask from Q and K:

```122:134:site-packages/fastvideo_kernel/ops.py
    # Compression branch (fused Triton: bf16 read → fp32 accumulate → div → bf16 write)
    q_c = fused_block_mean(q, q_variable_block_sizes, block_elements)
    k_c = fused_block_mean(k, variable_block_sizes, block_elements)
    v_c = fused_block_mean(v, variable_block_sizes, block_elements)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements, 1).view(batch, heads, q_seq_len, dim)

    # Sparse branch (fused Triton topk mask)
    mask = fused_topk_mask(scores, topk)
```

Answering F2.2's checklist directly:

| Question | Answer |
|---|---|
| 1. Where is the selector signal produced? | Inside the kernel entry point, `ops.py:123-127`. Not in FastVideo Python, and not in the DiT. |
| 2. What tensors feed it? | Q and K only (`v_c` feeds the compression output, never the mask). |
| 3. What kind of selector is it? | A **mean-pooled Q·K block score**. Not a learned gate, not a materialized coarse attention map. `scores` is reused for both `softmax`→`out_c` and `top-k`→`mask`, but selection reads the pre-softmax scores. |
| 4. Dtype before selection | Q/K arrive **bf16**. `fused_block_mean` reads bf16, accumulates **fp32**, writes back **bf16** (`fused_compress_topk.py:55-59`). The score matmul is `torch.matmul` on **bf16** inputs → tensor cores with **fp32 accumulation** → bf16 result. `fused_topk_mask` then upcasts to fp32 internally (`scores.to(tl.float32)`, line 233). |
| 5. Where is top-k applied? | `fused_topk_mask`, `ops.py:134`. |
| 6. Python, Triton, or fused kernel? | **Triton** (`_fused_topk_mask_kernel`), with a `torch.topk` fallback only when `next_power_of_2(kv_blocks) > 4096`. At Wan's 32760 tokens / 64-element tiles, `kv_blocks = 512`, so the **Triton path is used** and the fallback is not reached. |
| 7. Exact tie-breaking rule | 32-iteration fp32 bisection for the k-th value, then `scores > threshold` unconditionally selected, and ties *at* the threshold selected by **lowest key-block index first** via `cumsum(at_threshold) <= n_needed` (`fused_compress_topk.py:259-267`). This is a deterministic index-order tie-break. Measured against `torch.topk` on a constructed 20-way tie, both selected indices `[0,1,2,3,4]`, so the two rules **coincide in the cases tested** — we do not claim they differ, and F2 measures the effect rather than assuming it. |
| 8. Tile ordering and padding | `VSA_TILE_SIZE = (4, 4, 4)` → 64 tokens/tile, cube-ordered via `get_tile_partition_indices`. Ragged edge tiles carry `variable_block_sizes`, and the pooled mean divides by the **valid** token count, not by 64 (`fused_compress_topk.py:47`, `:57`). Padding slots are excluded from the mean rather than contributing zeros. |

`topk` itself:

```161:163:fastvideo/attention/backends/video_sparse_attn.py
def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks]."""
    return max(1, min(math.ceil((1 - sparsity) * num_blocks), num_blocks))
```

which is byte-for-byte the rule study 1 and F1 use.

---

## Is the gate "derived from Q/K in a way comparable to the proxy"?

**Yes — far more closely than study 1 was able to claim.** F2.2 was written to
allow for the answer being "no"; the code says otherwise, and that is a
substantive external-validity result rather than a formality.

VSA's selector and study 1's research scorer agree on all four structural
choices:

| Aspect | Study 1 research scorer | VSA (installed kernel) | Match |
|---|---|---|---|
| Pooling | mean over tokens in block, valid-count denominator | mean over tokens in block, valid-count denominator | **identical** |
| Score | pooled-Q · pooled-K, scaled by `1/sqrt(d)` | pooled-Q · pooled-K, scaled by `1/sqrt(d)` | **identical** |
| Selection | top-k per query block, fixed k | top-k per query block, same `compute_topk` | **identical** |
| Geometry | `64x64-cube` variant matched (4,4,4) tiles | (4,4,4) tiles | **identical in the cube arm** |

They differ in exactly two respects, both of which are *precision*, which is
precisely what this study is about:

1. **Pooling arithmetic.** Study 1 pooled in fp64. VSA pools bf16→fp32→bf16.
2. **Score arithmetic.** Study 1 scored in fp64. VSA scores in bf16 with fp32
   tensor-core accumulation, i.e. **exactly F1's `R4` arm**
   (`repr=bf16, pool=bf16/acc_fp32, score=bf16/acc_fp32`).

A third, smaller difference is the **tie-break**: VSA's index-order rule vs
`torch.topk`'s. F2 measures its effect rather than assuming it away.

This means F1's R4 result is directly load-bearing for VSA, and F1 already
found R4 produces **bit-identical masks to fp32 (R2) in 4320/4320 cells**. F2's
job is to confirm that on the real kernel rather than in the side-channel.

---

## Where precision interventions are therefore applied

Given the map, the brief's Interventions A/B/C translate as follows.

### Intervention A — gate-input representation (**applicable, primary**)
Quantize the Q/K that feed `fused_block_mean`. This is the true routing input.
Arms: bf16 (reference), FP8-E4M3, NVFP4. `V` is left untouched so the
intervention is confined to routing.

### Intervention B — selector arithmetic (**applicable, primary**)
Recompute pooling and the score matmul at fp64 / fp32 / bf16-fp32acc, then
select. Reference is the kernel's own bf16/fp32-acc arithmetic. This is the
axis F1 opened, now measured at VSA's real interface and geometry.

### Intervention C — `gate_compress` quantization (**not a routing intervention**)
Run and report, but labelled as a **compression-branch amplitude** control, with
an explicit statement that it cannot change the retained mask. Its value is as a
falsification check: if quantizing `gate_compress` changed the mask, our map
would be wrong.

### Additional intervention D — tie-break rule (**VSA-specific, not in the brief**)
Compare VSA's index-order threshold tie-break against `torch.topk`. Study 1's
mechanism claim rests on swaps landing at near-degenerate boundaries; a
deterministic tie-break at exactly those boundaries is a plausible confound that
only appears once the real kernel is in scope. On a constructed 20-way tie the two
rules selected the same blocks, so this is measured as a possible confound, not
asserted as a difference.

---

## Empirical verification of this map

Every structural claim above was checked against the installed kernel on
Wan-shaped tensors (`f2_selftest.py` re-runs these as permanent gates):

| Claim | Result |
|---|---|
| `fused_block_mean` returns bf16 | confirmed (`torch.bfloat16`) |
| score matmul result is bf16 | confirmed (`torch.bfloat16`) |
| `fused_block_mean` == `bf16(fp32 mean)` | confirmed **bit-identical** |
| `fused_topk_mask` retains exactly `k` per row | confirmed |
| tie-break selects lowest indices | confirmed (`[0,1,2,3,4]`) |
| `torch.topk` tie-break on same input | also `[0,1,2,3,4]` — rules coincide here |
| `VSA_TILE_SIZE` | `(4,4,4)` → 64 elements → BHSD path, not the 256 fastpath |
| `gate_compress` changes VSA output | confirmed (it scales the compression branch) |

The last two lines matter together: `gate_compress` demonstrably changes the
**output**, which is why it is easy to mistake for a routing signal, but it enters
only after `fused_topk_mask` has already produced the mask.

---

## Consequences for measurement design

1. **The mask must be read from the kernel's own code path.** F2 calls
   `fused_block_mean` + `torch.matmul` + `fused_topk_mask` — the same functions
   `video_sparse_attn` calls — rather than reimplementing them.
2. **VSA output is a sum of two branches** (`out_c * gate + out_s`). Attention
   error for a mask change must be measured on the **sparse branch** `out_s`, or
   on the total with the compression branch held fixed; otherwise the coarse
   branch's constant contribution dilutes the effect and understates it.
3. **VSA's reference is not fp64.** The deployed selector *is* bf16/fp32-acc, so
   F2 reports two references: VSA-as-deployed (the honest operating point) and
   fp64 (the scientific ideal), and states which each number is against.
4. **Wan's 64-element tiles use the BHSD `video_sparse_attn` path**, not the
   256-element BSHD fastpath, since `VSA_TILE_SIZE = (4,4,4)` → 64. The pooling in
   the 256 path is written in PyTorch with an explicit `.float()` accumulate
   (`ops.py:210-217`) and is *not* the code Wan exercises; F2 must not measure
   that path and report it as Wan's.
