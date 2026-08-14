# SparseFP4 Video Attention — Experimental Specification

Pre-registered protocol for Phases 1–2 of `SKILL.md`. This document is the
authority on *what* is measured and *how*; `SKILL.md` remains the authority on
research goals, integrity rules, and go/no-go policy.

The intent is that two engineers following this spec independently produce
comparable numbers. Details that depend on the actual FastVideo implementation
were originally left as explicit `TODO(codebase-map):` markers rather than
guessed. **The attention-stack survey is now complete
(`artifacts/sparsefp4/CODEBASE_MAP.md`, with the environment and native-NVFP4
verdict in `artifacts/sparsefp4/PHASE0.md`), and those markers have been
resolved in place with `file.py:LINE` citations.** Any remaining marker is
labelled `TODO(open):` and states precisely what is still unknown and how to
determine it — never resolve one with an assumption, and record every resolution
in the report. Where a resolved fact tightened or invalidated a pre-registered
choice, the change is logged in §12 rather than rewritten silently.

Freeze this file before collecting numbers. If a definition has to change after
data collection starts, record the change (old value, new value, reason, date)
in the "Protocol amendments" section at the bottom and re-run affected cells;
never retroactively re-interpret already-collected records.

---

## 1. Definitions

### 1.1 Sparsity vs retained fraction

The two terms are complements and must never be used interchangeably in output
columns, filenames, or figures.

```text
retained_fraction = 1 - sparsity
```

| `sparsity` | `retained_fraction` |
|---|---|
| 0.50 | 0.50 |
| 0.70 | 0.30 |
| 0.80 | 0.20 |
| 0.90 | 0.10 |
| 0.95 | 0.05 |

**Canonical on-disk field:** `sparsity` (float in `[0, 1)`), because that is the
field name in the `SKILL.md` raw-record schema. Any use of retained fraction is
derived at analysis time. Never store a percentage integer (e.g. `90`) in the
`sparsity` column.

### 1.2 Top-k per query block

Selection is **per query block, per head, per layer, per timestep**. For a query
block with `n_key_blocks` candidate key blocks:

```text
k = ceil(retained_fraction * n_key_blocks)
k = max(k, K_MIN)
k = min(k, n_key_blocks)
```

- `K_MIN = 1`. A query block must always retain at least one key block, so that
  every query row has a defined softmax denominator. Without this floor, high
  sparsity on short sequences produces `k = 0` and undefined output.
- `k` is computed from `n_key_blocks` for **that query block's candidate set**,
  not from a global block count. If the candidate set differs per query block
  (e.g. under causal masking), `k` differs per query block; this is expected and
  must be recorded per record.
- `k` is computed **once per (layer, head, timestep, query block, sparsity)**
  from the geometry alone, and is therefore *identical across all precision
  arms*. Precision may change *which* blocks are selected, never *how many*.
  Any arm whose `k` differs from the BF16 reference `k` for the same cell is a
  bug; fail the run rather than reporting it.

### 1.3 Tie-breaking

Ties in block score are broken by **ascending key-block index** (lowest index
wins). Implement this by using a stable descending sort on score with the
original index as the secondary ascending key, e.g. `np.argsort(-score,
kind="stable")` over a contiguous index axis, or `torch.topk(..., sorted=True)`
only if the backend's tie behaviour has been verified to be index-stable.

Rationale: low-precision quantization *creates* exact ties that do not exist in
BF16 (many distinct BF16 scores collapse to the same NVFP4 value). An unstable
tie-break turns a deterministic quantization artifact into run-to-run noise and
inflates the measured instability. Deterministic index-order tie-breaking makes
the measurement reproducible and, if anything, *conservative*.

Record the count of exact ties at the selection boundary
(`score[k] == score[k+1]` after sorting) in the per-record field
`boundary_ties` when the analysis pipeline supports it; it is a direct
mechanistic explanation for any instability observed at NVFP4.

### 1.4 Force-retained blocks

Many block-sparse video-attention methods force-retain a diagonal/local block
(the query block's own co-located key block) regardless of score.

Policy: **whatever the force-retention rule is, it must be byte-identical across
all precision arms and all sparsities within a comparison**, and it must be
recorded in the run config as `force_retain_diagonal: true|false` plus, if true,
the exact rule. Force-retained blocks consume budget: they count toward `k`,
they are *not* added on top of `k`. Because the rule is precision-independent,
force-retained blocks are guaranteed to agree between arms and therefore
*inflate* Jaccard and recall toward 1.0. Report both:

- `jaccard` — over the full selected set (what the kernel actually executes), and
- `jaccard_scored` — over the score-selected subset only, excluding
  force-retained blocks (the quantity that actually measures router
  instability).

If only one can be produced, produce `jaccard` (matches the schema) and state in
the report that force-retention dilutes the measured effect.

Default for the diagnostic scorer in this spec: `force_retain_diagonal: false`,
so the diagnostic measures the scorer alone. When integrating a real sparse
backend, adopt that backend's rule and say so.

**Resolved (VSA / `VIDEO_SPARSE_ATTN`): there is no force-retained diagonal or
local block.** The mask is pure per-`(batch, head, query-tile)` top-k over *all*
key tiles: `fused_topk_mask(scores, topk)` writes exactly `topk` `True` entries
per row (`fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py:298-313`,
kernel at `:211-277`), with `topk = compute_topk(VSA_sparsity, num_tiles)`
(`fastvideo/attention/backends/video_sparse_attn.py:161-167`). Nothing is added
on top of the budget and no key tile is exempt, so
`force_retain_diagonal: false` is the *correct* setting for the VSA-integrated
arms as well as for the diagnostic scorer, and `jaccard_scored == jaccard` there
(see §12). Two related facts worth carrying:

- VSA's tie-break at the selection threshold is deterministic and
  **index-ascending** — after bisecting to the k-th score value, ties are taken
  by `cumsum` order over the key axis, i.e. lowest key-block index wins
  (`fused_compress_topk.py:267-275`). That is exactly §1.3's rule, so the
  diagnostic scorer and VSA agree on tie handling.
- What VSA *does* add is an always-dense pooled **compression branch**
  (`out_c * compress_attn_weight + out_s`,
  `fastvideo-kernel/python/fastvideo_kernel/ops.py:108-129`; weight is Wan's
  learned `to_gate_compress`, `fastvideo/models/dits/wanvideo.py:458-462`).
  That is a second information path, **not** a force-retained block. It is a
  confound for the sparsity ablation and must be held fixed across arms or
  disabled via `gate_compress=None` (`fastvideo/attention/layer.py:210-213`,
  `:230-233`).
- For contrast, the only in-tree force-retention rule is VSA-**H3**'s "exempt"
  handling, which forces all text/cond/audio prefix tiles `True`
  (`fastvideo/attention/backends/video_sparse_attn_h3.py:222-231`) — that
  backend is not reachable from Wan (it is absent from
  `_supported_attention_backends`, `fastvideo/configs/models/dits/base.py:22-30`),
  so it does not apply here.

---

## 2. Where to instrument

### 2.1 The capture point

Q and K must be captured **exactly as the attention backend receives them**:

- after Q/K normalization (RMSNorm / LayerNorm on Q and K, if the model uses it),
- after rotary/positional embedding application,
- after any layout permutation/transpose/reshape the call site performs,
- immediately before the attention backend call, in the backend's expected
  layout and dtype.

**Forbidden:** comparing an early BF16 tensor (e.g. post-projection, pre-RoPE)
against a later transformed low-precision tensor. RoPE and Q/K-norm are not
precision-neutral, and mixing capture points attributes their effect to
quantization. Both arms of every comparison must be captured at the *same* point
in the graph.

Operationally: capture once in BF16 at the pre-backend point, then derive every
precision arm from that single captured tensor by applying the quantizer at that
point. This makes the capture point identical by construction. It does mean the
simulated arms model "quantize the final Q/K" rather than "propagate low
precision through Q/K-norm and RoPE"; state that limitation in the report, and
prefer native tensors when they can be observed (§4.4).

**Resolved.** The only point in the codebase where post-Q/K-norm, post-RoPE,
post-SP-layout Q and K exist as named tensors is inside the attention *layer*,
immediately before the backend call — **not** in the Wan model file:

- **Call site:** `fastvideo/attention/layer.py:145-147` —
  `q, k, v = qkv.chunk(3, dim=0)` then
  `output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)`
  (`DistributedAttention.forward`, `layer.py:82-164`). The VSA variant is
  `DistributedAttention_VSA.forward`, `layer.py:230-234`
  (five positional args: `q, k, v, gate_compress, ctx_attn_metadata`);
  cross-attention uses `LocalAttention.forward`, `layer.py:312-317`.
  Equivalently, the first two arguments of any `AttentionImpl.forward`
  (`fastvideo/attention/backends/abstract.py:186-194`) are exactly these tensors.
- **Argument names:** `query`, `key`, `value` on the impl
  (`abstract.py:186-194`; e.g. `fastvideo/attention/backends/flash_attn.py:234-240`).
- **Layout: `[B, S, H, D]` (BSHD)**, contiguous, dtype **bf16** for Wan
  (documented `fastvideo/attention/layer.py:96-98`; compute dtype from
  `layer.py:61` + `fastvideo/configs/pipelines/wan.py:49`). Backends needing
  BHSD transpose themselves *inside* `forward`
  (`fastvideo/attention/backends/video_sparse_attn.py:331-334`) or in
  `preprocess_qkv` (`fastvideo/attention/backends/sage_attn3.py:58-70`), so the
  capture point is BSHD regardless of backend.
- **Q/K-norm is applied OUTSIDE that module**, in the Wan block:
  `query = self.norm_q(query)` / `key = self.norm_k(key)` at
  `fastvideo/models/dits/wanvideo.py:398-401` (VSA block: `:544-547`).
  For Wan `qk_norm == "rms_norm_across_heads"`
  (`fastvideo/configs/models/dits/wanvideo.py:75`), so the RMSNorm is over the
  full `dim` on the flat `[B, S, dim]` tensor **before** the head split at
  `wanvideo.py:403-405`.
- **RoPE is applied INSIDE the attention layer**, after the SP all-to-all and
  the SP-padding trim: `fastvideo/attention/layer.py:130-132` (VSA:
  `:224-226`), on `qkv[:batch_size*2]` only — i.e. Q and K, never V.
  Implementation `_apply_rotary_emb`,
  `fastvideo/layers/rotary_embedding.py:105-149`, GPT-J-style interleaved
  (`is_neox_style=False`, `:143-144`), fp32 math cast back to bf16 (`:145-146`).
  The block only passes the `(cos, sin)` tuple in
  (`wanvideo.py:407-413`, built at `wanvideo.py:679-687`).

**Consequence — anything captured in `wanvideo.py` is pre-RoPE and pre-SP-layout
and is therefore forbidden by the rule above.** Practical hook options, ranked in
`CODEBASE_MAP.md` ("Recommended Phase 1 instrumentation plan"): a new
`AttentionImpl` subclass is the sanctioned mechanism; there is no existing hook
that reaches inside `AttentionImpl.forward`
(`fastvideo/hooks/hooks.py:33-35` sees only the layer's pre-RoPE arguments,
and `fastvideo/hooks/activation_trace.py:137-153` sees outputs only).
One caveat under VSA: `preprocess_qkv` runs at `layer.py:228` *before* the
chunk, so Q/K seen at the VSA backend boundary are already tiled and zero-padded
(`video_sparse_attn.py:290-296`) — see §2.3.
Note also that `FASTVIDEO_DISABLE_ATTENTION_COMPILE` defaults to `True`
(`fastvideo/attention/layer.py:18-35`), so a Python-side probe at this point is
safe and cheap.

### 2.2 Scope: self-attention over video tokens only

- **In scope:** self-attention within the video/latent token sequence of the DiT
  blocks.
- **Out of scope:** cross-attention to text embeddings, and any attention inside
  the text encoder or VAE.

Reason cross-attention is excluded: its key length is the (short) text sequence,
typically on the order of a few hundred tokens or fewer, so block-sparse routing
over key blocks has little or nothing to skip, and top-k over a handful of key
blocks is dominated by the `K_MIN` floor and rounding of `k`. It is a different
regime with a different cost profile, and including it would dilute the
video-token result that the paper is about. If a model variant fuses text tokens
into the self-attention sequence, that changes the analysis — record it and
handle the text-token blocks explicitly rather than ignoring them.

**Resolved: Wan2.1 self-attention carries only video/latent tokens.** Text
conditioning enters exclusively through cross-attention, so no text-token block
handling is needed and `seq_len` is prompt-independent.

- The Wan block calls `self.attn1(query, key, value, original_seq_len,
  freqs_cis=freqs_cis)` (`fastvideo/models/dits/wanvideo.py:407-413`; VSA block
  `:554-561`) with Q/K/V all derived from `norm_hidden_states`, i.e. the video
  latent stream (`wanvideo.py:393-396`). The layer's optional
  `replicated_q/k/v` arguments — the mechanism that *would* concatenate
  replicated text tokens into the self-attention sequence
  (`fastvideo/attention/layer.py:88-90`, `:137-143`) — are **not passed** by Wan.
- Text tokens go to `self.attn2` (`WanT2VCrossAttention`,
  `wanvideo.py:342-347`, called at `:424`), whose K/V come from
  `encoder_hidden_states` with `text_len = 512`
  (`fastvideo/configs/models/dits/wanvideo.py:65`).
- Cross-attention is also **excluded by construction**, not by filtering:
  `attn2` hard-pins `supported_attention_backends = (FLASH_ATTN, TORCH_SDPA)`
  (`wanvideo.py:174-175`), so any new research backend is silently ignored there
  and falls back with a warning (`fastvideo/attention/selector.py:279-288`).
  Cross-attention also never receives `freqs_cis` (`wanvideo.py:217`, `:270-272`),
  another reason its Q/K are not comparable to the self-attention capture point.
- `WanSelfAttention.forward` is a `pass` stub (`wanvideo.py:177-185`) and exists
  only as a base class for the cross-attention variants — do not instrument it.

Because `seq_len` is fixed by the latent geometry rather than the prompt, the
prompt-length confound in §9.3(3) does not arise for this model; verify it
anyway by asserting `seq_len` is constant within a comparison.

### 2.3 Sequence layout and token ordering

Record, per run: `seq_len`, `num_heads`, `head_dim`, latent frame count, latent
height/width, and the patchification/token-ordering rule that maps
(frame, y, x) to a linear token index. Block membership is entirely determined
by this ordering, so it is a first-class experimental variable, not a detail
(see §9).

**Resolved for Wan2.1-T2V-1.3B at 480x832 x 81 frames.**

- **Patchify rule:** the VAE latent is `(T, H, W) = (21, 60, 104)`, patchified by
  `patch_size = (1, 2, 2)` (`fastvideo/configs/models/dits/wanvideo.py:64`) to a
  DiT token grid `(21, 30, 52)`, flattened in **raster / row-major
  `(frame, y, x)` order**, i.e. `token_index = ((t * 30) + y) * 52 + x`
  (grid computed at `fastvideo/attention/backends/video_sparse_attn.py:211-216`;
  token order is plain `flatten` order over that grid).
- **`seq_len = 21 * 30 * 52 = 32760`**, confirmed independently by
  `tests/local_tests/test_nvfp4_fa4.py:47`
  (`MODEL_SEQLEN = 32760  # 480x832 video, 81 frames`). With `sp_size = 1` there
  is no SP padding, so `original_seq_len == seq_len` and head indices are global
  (`fastvideo/attention/layer.py:124-128`) — run the study at `num_gpus=1`.
- **Yes, VSA re-orders tokens into 3D spatio-temporal tiles.** `VSA_TILE_SIZE =
  (4, 4, 4)` = **64 tokens per tile**
  (`fastvideo/attention/backends/video_sparse_attn.py:28`), and query block ==
  key block == the tile volume (`video_sparse_attn.py:313`, enforced at
  `fastvideo-kernel/python/fastvideo_kernel/ops.py:96-106`). Tokens are permuted
  from raster order into tile-contiguous order by
  `get_tile_partition_indices` (`video_sparse_attn.py:31-47`) inside
  `preprocess_qkv` (`:254-296`) and permuted back in `postprocess_output`
  (`:298-303`).
- **Tile padding changes the sequence length the backend sees:**
  `num_tiles = (ceil(21/4), ceil(30/4), ceil(52/4)) = (6, 8, 13) = 624` tiles →
  padded `seq_len = 624 * 64 = 39936`, with boundary tiles zero-padded and the
  true per-tile token count carried in `variable_block_sizes`
  (`construct_variable_block_sizes`, `video_sparse_attn.py:59-99`). So a capture
  at the VSA backend boundary sees **39936 tiled tokens over 624 blocks**, while
  a capture on the dense path sees **32760 raster tokens**. Record which one a
  record refers to; for cross-arm comparability either capture before
  `preprocess_qkv` or carry `variable_block_sizes` / `untile_combined_index`
  (`video_sparse_attn.py:150-153`) alongside the capture (see §12).
- **The diagnostic scorer must use the same ordering as any backend arm it is
  compared against** (§9.3(1)); a raster-ordered 128x64 block map and a
  tile-ordered 64-token block map are not the same partition of tokens.

---

## 3. Block scorer

### 3.1 The diagnostic scorer

A deliberately simple, method-independent scorer, so that Phase 1 measures
*precision sensitivity of routing* rather than the quirks of one sparse method.

```python
q_blocks = pool_blocks(q, block=block_q).float()   # [H, n_q_blocks, D]
k_blocks = pool_blocks(k, block=block_k).float()   # [H, n_k_blocks, D]
scores = (q_blocks @ k_blocks.transpose(-1, -2)) * softmax_scale  # [H, n_q, n_k]
```

Rules:

1. Pooling is an arithmetic mean over the token axis within the block.
2. **Pool in the arm's precision, then cast to fp32 before the matmul.** That is:
   dequantized (or native) low-precision values enter the mean; the mean and the
   score matmul are computed in fp32. This isolates "the router saw low-precision
   Q/K" from "the router itself was computed sloppily". Accumulating the pooled
   mean in fp32 is required — a bf16 accumulator over 128 tokens adds its own
   error that is not attributable to NVFP4.
3. `softmax_scale` (typically `1/sqrt(head_dim)`) is a positive constant and does
   **not** change top-k ordering. Apply it anyway, so that `decision_margin` is
   on the same scale as real pre-softmax logits and is comparable across
   `head_dim` values. Record the value used.
4. The **identical** scorer implementation runs for every precision arm. Only the
   input tensor precision varies. Assert this in code by calling one function
   with a `precision` argument; do not maintain per-precision scorer copies.
5. No centering, no normalization, no temperature, no softmax on block scores.

### 3.2 Block geometry

- Default: `block_q = 128`, `block_k = 64`.
- Ablations: `64x64` and `128x128`.
- `block_q` and `block_k` are always recorded per record as `block_q` / `block_k`
  and are never inferred at analysis time.

### 3.3 Ragged final block

`seq_len` is generally not divisible by the block size.

**Chosen rule: keep the ragged final block and pool over its valid tokens only
(masked mean).** Do not zero-pad, and do not drop.

Justification:

- *Zero-padding* shrinks the pooled vector of the final block toward the origin
  by a factor `valid/block`, which systematically lowers its score and biases
  selection against the final block — an artifact unrelated to precision, and one
  whose magnitude differs between arms since NVFP4's relative error depends on
  magnitude.
- *Dropping* the ragged block silently removes real tokens from the analysis,
  changes `n_key_blocks` and therefore `k`, and (for the query side) leaves the
  tail of the video with no measured routing at all.
- The masked mean is the unbiased estimator of the same quantity the full blocks
  use, and it keeps `n_blocks = ceil(seq_len / block)` consistent across arms.

Every record carries `n_q_blocks`, `n_k_blocks`, `seq_len`, and `ragged_tail`
(the valid token count of the final block, equal to the block size when the split
is exact) so the ragged block can be excluded post hoc if it turns out to be an
outlier. Report whether excluding it changes any headline number.

---

## 4. Precision arms

### 4.1 Two independent axes

```text
router_precision    in {bf16, fp8_e4m3, nvfp4}
attention_precision in {bf16, fp8, nvfp4}
```

These are **independent**: the router may run at higher precision than the
attention compute. The scientific point of the study is exactly that a cheap
high-precision router can be paired with expensive low-precision compute, so any
code path that couples them (one dtype flag for both) is unusable here.

Phase 1 varies `router_precision` only (there is no attention compute in the
mask-stability measurement). Phase 2 varies both.

`reference_precision` is `bf16` everywhere in this study, and every record
records it explicitly rather than assuming it.

### 4.2 NVFP4 definition used in this study

- Element format: **e2m1** — 1 sign bit, 2 exponent bits, 1 mantissa bit.
  Representable magnitudes: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`;
  `E2M1_MAX = 6.0`.
- Microscaling: **block size 16 along the last (head-dim) axis**, one scale per
  16 contiguous elements.
- Scale format: **fp8** (`e4m3` unless the framework uses `e8m0`), optionally
  combined with a per-tensor fp32 global scale, as in the standard NVFP4 recipe.

Any deviation from the above found in the real implementation overrides this
section and must be recorded.

**Resolved — the FA4 path deviates from §4.2 in one respect, and that deviation
overrides §4.2 (logged in §12):**

| Property | FastVideo / FA4 actual | Citation |
|---|---|---|
| Element format | **E2M1** (`torch.float4_e2m1fn_x2`, packed 2-per-byte) | `fastvideo/attention/backends/flash_attn.py:85-86` |
| Scale dtype | **E4M3 (fp8), not e8m0** | `fastvideo/attention/backends/attn_qat_infer.py:57-62`, receipt `qk_mode=nvfp4(per-16-e4m3-sf)` at `:123-140` |
| Microscale block size | **16** (`sf_vec_size = 16`) | `flash_attn.py:70`; Triton reference `MXFP_BLOCK_SIZE = 16` at `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/nvfp4_utils.py:9` |
| Microscale axis | last axis, taken over the **`nheads*headdim` flattened row** (the tensor is reshaped to `(batch*seqlen_padded, nheads*headdim)` before quantization) | `flash_attn.py:78-82` |
| Per-tensor second-level scale | **none — the global scale is hard-wired to `1.0`** (`torch.ones(1)`), so there is *no* per-tensor amax rescale on this path | `flash_attn.py:81-82` |
| What is quantized | **Q and K only; V and PV stay BF16** | `flash_attn.py:340-352`, `fastvideo/attention/utils/flash_attn_cute.py:368-378`; verdict in `artifacts/sparsefp4/PHASE0.md` §6.1 |

Two consequences the protocol must respect:

1. **Set `global_fp32_scale: false` and keep it false** for the simulated NVFP4
   arm, so simulated and native arms use the same (absent) second-level scale.
   §4.3's per-16-group recipe is otherwise the correct model of this encoding.
   Note the *other* NVFP4 code paths in the repo **do** use a global scale —
   `nvfp4_utils.py:44-49` (`6*448/global_max`, optional) and
   `fastvideo/layers/fp4linear.py:18-22` (per-tensor `448*6/maxabs`) — so do not
   borrow their scaling when modelling the FA4 attention path.
2. Because `headdim = 128` and `sf_vec_size = 16`, each head contributes exactly
   8 scale groups (`flash_attn.py:93`) and **no microscale group straddles a head
   boundary**, so "per-16 along the head-dim axis" (§4.2) and "per-16 along the
   flattened `nheads*headdim` row" are numerically identical here. That
   equivalence is head_dim-dependent — re-check it if `head_dim % 16 != 0`.

Since PV is BF16, "dense NVFP4" in this study means **NVFP4 Q/K with BF16 PV**,
not fully-FP4 attention; label it that way in every table
(`artifacts/sparsefp4/PHASE0.md` §6.1).

### 4.3 Simulated quantizer (routing diagnostics only)

Deterministic, no stochastic rounding. For each 16-element group `g` of the last
axis:

```text
amax_g       = max(|x_i|) for i in g
scale_raw_g  = amax_g / E2M1_MAX                       # E2M1_MAX = 6.0
scale_g      = fp8_e4m3_rne(clamp(scale_raw_g, FP8_MIN_POS, FP8_MAX))
                                                        # FP8_MAX = 448.0
x_scaled_i   = x_i / scale_g                            # fp32 division
q_i          = nearest_e2m1_rne(clamp(x_scaled_i, -6.0, 6.0))
x_hat_i      = q_i * scale_g                            # fp32 multiply
```

- If `amax_g == 0`, set `scale_g = FP8_MIN_POS` and `x_hat_i = 0` for the group.
  Never divide by zero, never produce NaN.
- Rounding mode: **round-to-nearest-even (RNE)** at both steps — encoding the
  scale into fp8, and encoding `x_scaled` into e2m1. Ties-to-even matters
  because e2m1 has so few codes that boundary values are common.
- Clamping: e2m1 saturates at `±6.0`; fp8 e4m3 saturates at `±448.0`. Saturate,
  do not wrap, and do not emit inf/NaN. Count and record saturation events per
  tensor (`sat_frac_q`, `sat_frac_k`) — heavy saturation is a plausible mechanism
  for routing instability and must not be hidden.
- Compute the quantizer in fp32 and return an fp32 dequantized tensor
  (`x_hat`). The dequantized tensor is what the scorer consumes.

FP8 e4m3 simulated quantizer, for the `fp8_e4m3` router arm:

```text
scale   = amax(x, axis=fp8_scale_axis) / FP8_MAX        # FP8_MAX = 448.0
x_hat   = fp8_e4m3_rne(clamp(x / scale, -448.0, 448.0)) * scale
```

Default `fp8_scale_axis`: per-head, per-tensor (one scale per `(head)` slice of
Q and of K separately). Record the choice; per-tensor vs per-head vs per-token
scaling changes the result and must not vary within a comparison.

Reference implementation preference order:
`torch.float8_e4m3fn` cast (native dtype, exact RNE from hardware) >
hand-rolled bit manipulation > table lookup. For e2m1 there is no torch dtype
with a public elementwise cast on all versions, so a table/nearest-value
implementation over the 8 magnitudes is acceptable — verify it against known
values in a unit check before use.

### 4.4 Native quantizer is preferred

If the framework's real quantizer (or its quantize/dequantize helper) can be
applied to Q/K at the capture point, **use it** instead of §4.3 and label the arm
`native`. Simulated quantization is a fallback for routing diagnostics only, per
`SKILL.md` §1.3, and:

- every record and every table row must carry
  `native_or_simulated ∈ {native, simulated}`,
- simulated arms must be labeled "fake/simulated FP8" or "fake/simulated NVFP4"
  in all prose,
- simulated arms are excluded from all latency tables (no exceptions),
- a mixed table (some rows native, some simulated) must show the column, not a
  footnote.

**Resolved — yes for quantize, no for a matching dequantize, so this arm splits
in two:**

- **Callable native quantizer, usable on Q/K outside the fused kernel:**
  `_nvfp4_quantize_for_fa4(tensor_4d) -> (fp4_tensor, sf_tensor)`
  (`fastvideo/attention/backends/flash_attn.py:138-141`, real body at `:58-104`),
  a `torch.library.custom_op` (`:114-120`) over flashinfer
  `nvfp4_quantize` / `fp4_quantize_sm100` (`:38-55`). It takes **any**
  `(batch, seqlen, nheads, headdim)` fp16/bf16 tensor — it is not tied to a
  linear layer, and `flash_attn.py:340-341` calls it directly on post-RoPE
  `query`/`key`. `attn_qat_infer.py:146-162` memoizes the same pair
  `(_nvfp4_quantize_for_fa4, flash_attn_fp4_func)`, i.e. the framework itself
  reuses it; **use it verbatim** and set
  `native_quantizer_entrypoint` in the config accordingly. Output contract:
  `fp4` shape `(batch, seqlen_padded_to_128, nheads, headdim//2)` dtype
  `torch.float4_e2m1fn_x2`, plus a `uint8` `sf` in the FA4 MMA layout
  `(32, 4, rest_m, 4, rest_k, nheads, batch)` with `stride[3] == 1`
  (`flash_attn.py:62-65`, `:88-103`; pinned by
  `tests/local_tests/test_nvfp4_fa4.py:61-74`). The caller must slice
  `[:, :orig_seqlen]` before handing it to FA4 (`flash_attn.py:346-347`).
- **Can post-quantization Q/K be *observed* on the FA4 path? Only in packed
  form.** `_nvfp4_quantize_for_fa4` returns the FP4-packed tensor and its scale
  factors *before* the kernel call, so they are observable and their statistics
  (saturation counts, code histograms) are measurable — but **there is no
  dequantizer for that packed/swizzled layout anywhere in `fastvideo/`**.
  `flash_attn_fp4_func` (`fastvideo/attention/utils/flash_attn_cute.py:368-378`)
  consumes the packed tensor directly, and the FA4 kernel is fused, so a
  bf16-valued "post-quant Q/K" cannot be read back out of the native path.
  `attn_qat_train.py` does not help: it fake-quantizes *inside* a fused Triton
  kernel (`fastvideo/attention/backends/attn_qat_train.py:60-109`), so its
  quantized Q/K are equally unreadable.
- **Therefore the scorer cannot consume native-quantized Q/K values directly.**
  Two defensible routes, both of which must be labelled per §4.4:
  1. **Deterministic simulated NVFP4 round-trip** using the repo's own Triton
     reference pair — `_compute_quant_and_scale`
     (`fastvideo-kernel/python/fastvideo_kernel/triton_kernels/nvfp4_utils.py:12-133`)
     and `_compute_dequant` (`:136-239`), with `MXFP_BLOCK_SIZE = 16` (`:9`) and
     per-16 `block_max / 6` → E4M3 scale (`:54-58`). These are `@triton.jit`
     *device* functions, not host-callable ops, so they must be wrapped in a
     small host kernel; run them with the optional global scale **disabled**
     (`:44-49`) to match the FA4 encoding (§4.2). Same block geometry as the
     native path, so this is a faithful reference — but it is
     `native_or_simulated = "simulated"`.
  2. **§4.3's own fp32 quantizer**, verified elementwise against the FP4 codes
     emitted by `_nvfp4_quantize_for_fa4` for the same input. Also `simulated`.
- `fastvideo/layers/fp4linear.py:53-79` is **linear-layer-only** (needs a weight
  and calls `mm_fp4`, with a per-tensor global scale at `:18-22`); do not contort
  it onto Q/K.
- For the FP8 router arm, a native `torch.float8_e4m3fn` cast works on this
  machine (`artifacts/sparsefp4/PHASE0.md` §3.4, `_scaled_mm` verified); repo
  precedent for a per-`(b,h,d)` scaled E4M3 cast is
  `fastvideo/attention/backends/sla.py:520-526`.

**Net effect on the arm labels (logged in §12):** the *attention compute* path is
native NVFP4 (`artifacts/sparsefp4/PHASE0.md` §6), but the *router* arm that
consumes dequantized Q/K values is necessarily `simulated`, because no
dequantizer exists for the native packed format. Both facts must appear in the
`native_or_simulated` column rather than in a footnote.

---

## 5. Metrics

All metrics are computed per `(prompt_id, seed, layer, head, timestep, block_q,
block_k, sparsity, routing_precision)` cell. Within a cell, the per-query-block
values are summed/aggregated as described below; aggregation *across* cells
happens only in analysis (`scripts/analyze_masks.py`).

Notation, for one query block: `R` = key-block set selected from the reference
(BF16) scores, `C` = key-block set selected from the candidate
(`routing_precision`) scores, `|R| = |C| = k` (§1.2).

### 5.1 Set-overlap metrics

```text
intersection = |R ∩ C|
union        = |R ∪ C| = 2k - intersection
recall       = intersection / |R| = intersection / k
jaccard      = intersection / union = intersection / (2k - intersection)
```

Per cell, sum `intersection` and `union` over all query blocks in the cell and
store the sums, plus the ratio metrics computed from those sums (a
block-count-weighted micro-average). Storing the raw counts means analysis can
re-derive either micro- or macro-averages later; storing only the ratio cannot.

**Mandatory caveat (from `SKILL.md` §1.4):** because `|R| = |C| = k`, precision
== recall, and `jaccard = recall / (2 - recall)` is a monotone function of
recall. They are *one* measurement. Report recall and Jaccard if convenient for
readers, but never present them as independent corroborating evidence, and never
compute an F1 from them.

### 5.2 Fraction of routing decisions changed

```text
frac_decisions_changed = 1 - recall = (k - intersection) / k
```

Per cell this is the block-weighted mean over query blocks. It is a restatement
of recall, listed separately only because it is the more intuitive framing for
the paper. Same independence caveat applies.

Additionally record `frac_query_blocks_changed` — the fraction of query blocks
with `intersection < k`, i.e. at least one swapped block. This *is* extra
information (it separates "a few blocks each lost many" from "many blocks each
lost one") and should be reported alongside.

### 5.3 Decision margin

Sort the candidate's own scores for a query block descending as
`s_(1) ≥ s_(2) ≥ ... ≥ s_(n)` (ties per §1.3):

```text
margin_raw  = s_(k) - s_(k+1)
margin_norm = margin_raw / max(s_(1) - s_(n), eps)      # eps = 1e-12
```

- `margin_raw` is in pre-softmax logit units (i.e. `softmax_scale` already
  applied, §3.1).
- `margin_norm` is dimensionless and comparable across heads, layers, and
  `head_dim`. **The schema columns `decision_margin_reference` and
  `decision_margin_candidate` hold `margin_norm`**, computed on the reference
  scores and the candidate scores respectively; `margin_raw` values go in the
  optional columns `decision_margin_raw_reference` /
  `decision_margin_raw_candidate`.
- If `k == n_key_blocks` there is no `s_(k+1)`; store `null` (not 0, not -1) and
  exclude those cells from margin statistics, reporting how many were excluded.
- Per cell, store the median over query blocks (margins are heavy-tailed; a mean
  is dominated by a few high-contrast blocks).

The margin is the mechanistic link the paper needs: instability should
concentrate where the reference margin is small. Report the relationship (e.g.
`frac_decisions_changed` binned by reference-margin decile) rather than asserting
it.

### 5.4 Optional rank correlation

Spearman `rho` between reference and candidate score vectors over all
`n_key_blocks` for a query block, computed as Pearson correlation of
**average-tie-corrected ranks** (do not use the `1 - 6Σd²/(n(n²-1))` shortcut —
it is invalid with ties, and low-precision scores have many ties):

```text
rho = pearson(rank_avg(score_ref), rank_avg(score_cand))
```

Per cell, store the median over query blocks in `spearman_rho`. Optional: it is
useful for showing that global ordering is preserved even when the top-k boundary
moves, which is a different (and more interesting) statement than recall.

### 5.5 What "affected region" means

Fixed, pre-registered definition, used by the go/no-go rules and by the H3 test:

> An `(layer, timestep)` cell is **affected** at a given sparsity if its median
> BF16↔NVFP4 `jaccard` over heads is `< 0.90`.

A `(layer, head, timestep)` cell is affected under the same threshold applied
without the median over heads. Cells with `n < 20` observations are not eligible
to be called affected (§10.3). This threshold is chosen before looking at the
data and must not be tuned afterwards; if it turns out to select everything or
nothing, report that fact and show the full distribution instead of re-tuning.

---

## 6. Raw record schema

### 6.1 Format and location

- **Format: JSONL**, one JSON object per line, UTF-8, `\n`-terminated, no
  trailing commas, no NaN/Infinity literals (use `null`).
- One file per worker/shard, all under:

```text
artifacts/sparsefp4/raw/<run_id>/*.jsonl
```

- `run_id` format: `<YYYYmmdd-HHMMSS>-<short_git_sha>-<phase_tag>`, e.g.
  `20260813-191500-a1b2c3d-p1-stage1`. Never reuse a `run_id`; never append to a
  completed run's file. Write the effective resolved config to
  `artifacts/sparsefp4/raw/<run_id>/config.resolved.yaml` and the environment
  snapshot reference to `.../env.json` (or a relative link to the shared one).
- Chosen over Parquet/CSV because it is append-only and crash-tolerant (a
  truncated final line loses one record, not the file), needs no schema
  migration when optional columns appear, and is readable by the stdlib.
  Converting to Parquet for analysis afterwards is fine; JSONL remains the
  archival form.

### 6.2 Required minimum columns

Verbatim from `SKILL.md` §1.5 — all of these are **required** in every record:

```text
prompt_id
seed
layer
head
timestep
block_q
block_k
sparsity
routing_precision
reference_precision
intersection
union
selected_reference
selected_candidate
recall
jaccard
decision_margin_reference
decision_margin_candidate
```

Plus these **required additions**:

```text
native_or_simulated        # "native" | "simulated"
run_id                     # matches the containing directory name
git_commit                 # full 40-char sha of the code that produced the record
```

Semantics of the two set columns: `selected_reference` and `selected_candidate`
are the *counts* of selected key blocks summed over the query blocks of the cell
(so `selected_reference == selected_candidate == sum_over_query_blocks(k)`, and
`union = selected_reference + selected_candidate - intersection` holds as an
invariant the analysis script can check). If a run instead stores explicit index
lists, it must still emit the counts under these names and put the lists in
`selected_reference_idx` / `selected_candidate_idx`, and it must respect the
storage cap in §11.

### 6.3 Recommended optional columns

Emit when available; the analysis script tolerates their absence.

```text
k_per_query_block          n_q_blocks             n_k_blocks
seq_len                    num_heads              head_dim
ragged_tail                softmax_scale          force_retain_diagonal
jaccard_scored             frac_query_blocks_changed
decision_margin_raw_reference   decision_margin_raw_candidate
spearman_rho               boundary_ties          sat_frac_q   sat_frac_k
model_id                   model_revision         scheduler    num_inference_steps
guidance_scale             resolution             num_frames
attention_backend          quantizer_impl         stage         phase
wall_clock_utc             notes
```

### 6.4 Invariants the writer must assert

Fail the run loudly (do not write the record) if any of these break:

1. `0 <= intersection <= min(selected_reference, selected_candidate)`
2. `union == selected_reference + selected_candidate - intersection`
3. `selected_reference == selected_candidate` (equal budget, §1.2)
4. `abs(recall - intersection / selected_reference) < 1e-9`
5. `abs(jaccard - intersection / union) < 1e-9`
6. `reference_precision == "bf16"`
7. `routing_precision == "bf16"` implies `recall == 1.0` and `jaccard == 1.0`
   (the reference compared to itself; a self-consistency arm that must be run at
   least once per stage as a null control)
8. `0 <= sparsity < 1`
9. `native_or_simulated in {"native", "simulated"}`

---

## 7. Phase 2 — error decomposition

### 7.1 Configurations A–F

`SKILL.md` §Phase 2 lists six configurations. Fully specified:

| ID | Sparse? | Attention compute precision | Mask source precision | Native or simulated | Isolates |
|---|---|---|---|---|---|
| **A** | no | BF16 | n/a (dense) | native | **Reference.** Sole numerical reference for every other row. |
| **B** | no | low precision (NVFP4; FP8 optional) | n/a (dense) | native if the FP4/FP8 attention path exists, else simulated | **Quantization error** with no sparsity involved. |
| **C** | yes | BF16 | BF16 | native | **Sparsification error** with no quantization involved. |
| **D** | yes | BF16 | NVFP4 | native compute; mask source native or simulated per §4.4 | **Wrong-mask error**, isolated at high-precision compute: `err(D) - err(C)`. |
| **E** | yes | low precision (NVFP4) | NVFP4 | simulated unless a native sparse-NVFP4 kernel exists (Phase 4) | The **naive combined** config: quantized compute *and* quantized routing. |
| **F** | yes | low precision (NVFP4) | FP8 or BF16 | same as E | **Router-recoverable error**: `err(E) - err(F)` is the H3 quantity. |

Hard rules:

1. **A is the only reference.** Every error metric in Phase 2 is computed against
   A's output for the identical (prompt, seed, layer, timestep, head) cell. Do
   not use C as a reference for D/E/F "to isolate the mask effect" — instead
   report the *difference of errors against A*, which keeps one reference and
   makes rows comparable.
2. **C, D, E, F use an identical retained fraction** (identical `sparsity`,
   identical `k` per query block per §1.2). A row whose `k` differs is invalid.
3. E and F are almost certainly simulated at first. They are then
   **numerical-only; no native latency claim** — mark them exactly that way in
   every table and exclude them from latency tables (§4.4).
4. Report B for NVFP4 at minimum; B for FP8 is optional but cheap and makes the
   quantization-error axis two-pointed.
5. The differences in the table (`err(D) - err(C)`, `err(E) - err(F)`) are
   **attributions, not an exact additive decomposition**. Quantization and
   sparsification errors do not compose linearly. State this in the report and
   present the raw per-configuration errors alongside any difference.

**Resolved — yes to both, so B and D can be native.**

- **Native NVFP4 dense attention exists and runs on this machine.** Kernel entry
  `flash_attn_fp4_func` (`fastvideo/attention/utils/flash_attn_cute.py:368-378`)
  over the custom op at `:320-348`; consumed by the `FLASH_ATTN` backend's
  `_forward_nvfp4` (`fastvideo/attention/backends/flash_attn.py:334-360`,
  dispatched at `:318-319`), gated by `FASTVIDEO_NVFP4_FA4=1` or the
  `nvfp4_fa4` impl kwarg (`:226`) with hard capability asserts on
  `(10,0)/(10,3)` (`:227-232`). A second consumer is `ATTN_QAT_INFER`
  (`fastvideo/attention/backends/attn_qat_infer.py:276-310`).
  **Measured on this host** (8x B200, sm_100): dense NVFP4 smoke PASS, receipt
  `arch=sm_100 kernel=flash-attention-fp4 qk_mode=nvfp4(per-16-e4m3-sf)
  pv_mode=bf16`, emitted PTX
  `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`, and at the
  real Wan shape `cos = 0.99050 / rel_l2 = 0.13783` vs BF16 with a warmed
  kernel-only median of 4.013 ms vs 5.135 ms (1.28x) —
  `artifacts/sparsefp4/PHASE0.md` §6.1–6.3. So **B = native NVFP4 (Q/K FP4, PV
  BF16)**, not simulated. Operational requirements: `FASTVIDEO_FA4=1` must stay
  set (no FA2 is installed) and `use_fsdp_inference=False` is mandatory on the
  FP4 path (`artifacts/sparsefp4/PHASE0.md` §8).
- **No sparse *backend* accepts a caller-supplied mask, but the kernel layer
  underneath does.** `VideoSparseAttentionImpl.forward`
  (`fastvideo/attention/backends/video_sparse_attn.py:305-342`) passes only
  `variable_block_sizes` and `cur_topk`; the mask is built *inside* the kernel
  wrapper (`fastvideo-kernel/python/fastvideo_kernel/ops.py:120`) and there is no
  mask field on its metadata (`video_sparse_attn.py:140-158`) or builder
  (`:200-235`). The mask-taking public entry points are:
  `fastvideo_kernel.block_sparse_attn(q, k, v, block_map, variable_block_sizes)`
  (bool `[B,H,Qb,KVb]`,
  `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py:384-393`),
  `block_sparse_attn_from_indices(...)` (`:347-381`), and
  `block_sparse_attn_256(...)` / `block_sparse_attn_256_bshd(...)`
  (`fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py:115-131`,
  `:134-161`). `VIDEO_SPARSE_ATTN_H3` already demonstrates the full
  Python-level score/compute split end to end
  (`fastvideo/attention/backends/video_sparse_attn_h3.py:292-312`) — it is the
  template, though it is not reachable from Wan
  (absent from `_supported_attention_backends`,
  `fastvideo/configs/models/dits/base.py:22-30`). `NABLA_ATTN`'s `sta_mask`
  metadata field (`fastvideo/attention/backends/nabla.py:84-85`) is OR'd in, not
  substituted (`:55`), so it is not an external-mask input in the sense D needs.
  **So D (BF16 compute + externally supplied NVFP4-derived mask) is achievable
  natively** by calling `block_sparse_attn_256_bshd` with our own mask.
- **Honesty boundary for E/F:** `block_sparse_attn*` computes in the **input
  dtype** (bf16) — there is **no native sparse-NVFP4 kernel in this repo**. E and
  F therefore remain **numerical-only, no native latency claim**, exactly as
  rule 3 above states, until Phase 4 lands a real kernel. The repo's own
  precedent for refusing a silent low-precision fallback is
  `fastvideo/platforms/cuda.py:143-154`.

### 7.2 Error metrics

Let `ref` be configuration A's attention output for a cell and `x` the candidate
configuration's output, both cast to fp32 before the metric:

```text
rel_l2   = ||x - ref||_2 / ||ref||_2                     # primary
cosine   = <x_flat, ref_flat> / (||x_flat||_2 * ||ref_flat||_2)
max_abs  = max(|x - ref|)
```

- `rel_l2` is the **primary** metric. Report per-cell values, not just an
  aggregate.
- Guard `||ref||_2 == 0` (possible for a fully masked head): emit `null` and
  count the exclusion.
- `max_abs` is reported only when numerically stable — if a single outlier
  element dominates, say so; do not use it as the headline.
- **Optional but valuable:** post-residual hidden-state error, i.e. the same
  three metrics on the block's output *after* the attention output has been
  projected and added to the residual stream. This measures how much of the
  attention error survives into the network, and it is usually much smaller than
  the raw attention error. If reported, label the measurement point explicitly;
  never mix pre- and post-residual numbers in one table.
- Aggregate over cells with **median and IQR**, never mean alone: these
  distributions are heavy-tailed across heads/layers.

### 7.3 Where to run it

Phase 2 runs on a deliberately chosen subset, not everywhere:

- at least 3 `(layer, timestep)` regions identified as **affected** by Phase 1
  (per §5.5), and
- at least 3 regions identified as **unaffected**, as a negative control, and
- for the sparsity values `{0.80, 0.90}` at minimum, plus `0.95` if Phase 1 shows
  the effect growing with sparsity.

Selecting only affected regions and then reporting a large effect is a
development-set-selection artifact. Reporting both affected and unaffected
regions is what makes H2 (localized sensitivity) a claim rather than a hope.

---

## 8. The H3 test

### 8.1 The comparison

At **equal sparse compute budget** (identical sparsity, identical `k`, identical
attention compute precision), compare three router precisions feeding the same
sparse low-precision attention:

```text
E   : NVFP4 router  -> sparse NVFP4 attention
F8  : FP8   router  -> sparse NVFP4 attention
F16 : BF16  router  -> sparse NVFP4 attention
```

Everything except `router_precision` is held fixed, including the mask *size*.
The comparison is **paired** at the `(prompt_id, seed, layer, head, timestep,
sparsity)` level, so the statistic of interest is the per-cell paired difference,
not the difference of two independently-aggregated means.

### 8.2 Pre-registered effect size

From `SKILL.md`'s go/no-go heuristic:

> **Support for H3** requires, in affected regions (§5.5), a
> **≥ 20% relative reduction in median `rel_l2` versus A**, going from the NVFP4
> router (E) to the higher-precision router (F8 or F16):
>
> ```text
> reduction = (median rel_l2(E) - median rel_l2(F)) / median rel_l2(E) >= 0.20
> ```

Plus all of the following, which are part of the pre-registration:

1. **Report the full distribution, not just the mean or median.** Required:
   per-cell paired-difference distribution (histogram or ECDF), the
   `p10/p25/p50/p75/p90` of `rel_l2` for each arm, and the **fraction of paired
   cells that improve** (`rel_l2(F) < rel_l2(E)`). A 20% median reduction where
   only 55% of cells improve is a different — and weaker — result than one where
   95% of cells improve; both must be visible.
2. **Report unaffected regions too.** If higher-precision routing "helps"
   everywhere including regions with Jaccard ≈ 1.0, that is evidence of a
   confound (e.g. the arms are not actually matched), not of H3.
3. **Report `n`** for every quoted number (§10.3).
4. **Report the BF16-router null control** (`routing_precision == "bf16"` vs
   itself) to show the harness produces zero difference when nothing changes.
5. This is a **heuristic threshold, not a hypothesis test.** Do not compute a
   p-value against it and do not describe the result as "significant". If the
   observed reduction is near the threshold, say "borderline" and use the
   BORDERLINE rubric entry.

### 8.3 Interpretation guards

- A reduction driven entirely by a handful of extreme cells must be reported as
  such (show the trimmed statistic alongside).
- If FP8 routing recovers as much as BF16 routing, that is a *stronger*
  systems result (cheaper router), not a weaker scientific one — report it
  prominently rather than emphasizing BF16.
- If higher-precision routing does *not* help, H3 is unsupported. Write that
  plainly and go to the PIVOT branch of the rubric. Do not search additional
  configurations for a positive result without labeling the search.

---

## 9. Controls and confounders

### 9.1 Held identical across paired arms

Non-negotiable; a mismatch invalidates the pair:

- prompt text and `prompt_id`
- seed and RNG stream ordering
- model id **and pinned revision**
- scheduler, `num_inference_steps`, guidance scale
- resolution (480x832), `num_frames` (81)
- batch size and all tensor shapes
- `block_q`, `block_k`, sparsity, `k` per query block
- force-retention rule (§1.4)
- `torch.compile` mode / CUDA-graph state / attention backend selection, except
  where the backend *is* the variable under test
- capture point in the graph (§2.1)
- CPU/GPU offload settings (these change execution order and can change RNG
  consumption)

Record all of them in the resolved run config; the analysis script should refuse
to pair rows whose recorded values differ.

### 9.2 Determinism

- Set seeds for `random`, `numpy`, and torch (CPU + CUDA) from the run's `seed`.
- Prefer deterministic algorithms where available; if determinism must be
  disabled for a kernel, record that and quantify run-to-run variation by
  repeating one cell 3x and reporting the spread. Non-determinism smaller than
  the effect is acceptable *if measured*; assumed-negligible is not.
- Simulated quantization must be bit-deterministic (no stochastic rounding,
  §4.3).
- Log `torch.backends` flags (TF32, cudnn benchmark/deterministic, SDPA/FA
  backend choice) into the env snapshot.

### 9.3 Confounders to watch and report

1. **Token ordering / patchification** — which video tokens land in which block
   is set by the patchify and any tile re-ordering. Blocks that are spatially
   coherent behave differently from blocks that straddle frame boundaries.
   Record the ordering; if the sparse backend re-orders tokens, the diagnostic
   scorer must use the *same* ordering or the two are not comparable.
2. **`seq_len` not divisible by block size** — see §3.3. Check whether the ragged
   block is an outlier before it influences a headline number.
3. **Prompt-length variation** — different prompts give different text-embedding
   lengths; if text tokens are ever concatenated into the self-attention
   sequence, `seq_len` and therefore `n_key_blocks` and `k` vary by prompt, which
   confounds cross-prompt aggregation. Record `seq_len` per record and verify it
   is constant within a comparison.
4. **Timestep sampling** — instability is expected to vary over the denoising
   trajectory. Use the *same* timestep set for every arm, record the actual
   scheduler timestep value (not only the step index), and never compare step
   index `i` of one scheduler/step-count against step index `i` of another.
5. **Head-dim scaling** — `softmax_scale` placement changes `decision_margin`
   magnitudes (not the ordering). Fix and record it (§3.1).
6. **Where the framework applies the attention scale relative to quantization** —
   scaling Q *before* quantization changes the amax and therefore the microscale,
   which changes the quantization error; scaling after does not. This can
   single-handedly explain a difference between the simulated and native arms.
   **Resolved: `softmax_scale` is applied strictly AFTER Q/K quantization, and
   there is no Q pre-scale on the NVFP4 path.** In `_forward_nvfp4`
   (`fastvideo/attention/backends/flash_attn.py:334-360`), `query` and `key` go
   into `_nvfp4_quantize_for_fa4` unmodified (`:340-341`) and `softmax_scale` is
   passed as a *kernel argument* to `flash_attn_fp4_func` (`:355`), which forwards
   it to the fused FA4 op (`fastvideo/attention/utils/flash_attn_cute.py:334-347`,
   `:368-378`) where it multiplies the accumulated QK product. `self.softmax_scale`
   is only stored at construction (`flash_attn.py:225`) and never applied to Q.
   Consequently the microscale amax per 16-element group is computed on the
   *unscaled* Q/K, so **`softmax_scale` cannot influence the quantization error at
   all** on this path — and since the FA4 global scale is hard-wired to `1.0`
   (`flash_attn.py:81-82`), there is no second-level scale that could absorb it
   either. The same ordering holds for the sparse router: VSA pools first and
   divides by `sqrt(dim)` on the *block scores*, after any Q/K transformation
   (`fastvideo-kernel/python/fastvideo_kernel/ops.py:109-113`).
   **Implication for this study:** the simulated arm must likewise quantize the
   raw Q/K and apply `softmax_scale` only to the block scores (§3.1 rule 3), which
   is what §4.3 already specifies — so this confounder is eliminated by
   construction rather than merely monitored. Record the value used and assert the
   ordering in the harness.
7. **Warm-up / caching effects** on anything timed (§ benchmark protocol in
   `SKILL.md` Phase 5); never compare a first-iteration compile against a warmed
   iteration.
8. **CFG / conditional-unconditional batching** — if classifier-free guidance
   runs conditional and unconditional passes, they are different attention calls
   with different statistics. Record which pass a record came from
   (`cfg_branch`) or restrict the study to one branch and say so.

---

## 10. Sample sizing and order of execution

### 10.1 Order (from `SKILL.md` "Default experiment matrix")

Do **not** run the full Cartesian product first.

| Stage | Scope | Sparsities | Purpose |
|---|---|---|---|
| 1 | 1 prompt x 1 seed x **all** layers/heads/timesteps | 0.80, 0.90 | Does the effect exist at all; find the structure (H1/H2). |
| 2 | 10 prompts x 1 seed, all layers/heads/timesteps | full sweep 0.50–0.95 | Is the structure stable across content and sparsity. |
| 3 | Error decomposition A–F on representative **affected and unaffected** regions | 0.80, 0.90 (+0.95 if trending) | H3, numerical attribution. |
| 4 | End-to-end video generation | selected | Does the numerical result show up in output. |
| 5 | Native kernel work | selected | H4, stretch. |

Advance a stage only after the previous stage's records are written *and*
analyzed. Stage 1 with a null control (§6.4 rule 7) is the smoke test for the
whole harness.

### 10.2 Cell counts

Stage 1 enumerates every `(layer, head, timestep)` for one prompt, so `n` per
`(sparsity, routing_precision)` cell is `n_layers * n_heads * n_timesteps` —
large, but from a single prompt, so it supports **within-model structure** claims
(H1/H2) and not **content-generality** claims. Say so explicitly in the report.

**Resolved for Wan2.1-T2V-1.3B.**

| Quantity | Value | Citation |
|---|---|---|
| `n_layers` (transformer blocks) | **30** | `fastvideo/tests/golden_gate/test_wan_t2v.py:32` |
| `n_heads` | **12** | `fastvideo/tests/golden_gate/test_wan_t2v.py:29`; `tests/local_tests/test_nvfp4_fa4.py:45` |
| `head_dim` | **128** | `fastvideo/tests/golden_gate/test_wan_t2v.py:30`; `tests/local_tests/test_nvfp4_fa4.py:46` |
| `hidden_size` | 1536 (= 12 x 128) | `fastvideo/tests/golden_gate/test_wan_t2v.py:20` |
| self-attention `seq_len` | **32760** (39936 tiled under VSA, §2.3) | `tests/local_tests/test_nvfp4_fa4.py:47` |
| `num_inference_steps` | **50** | `fastvideo/pipelines/basic/wan/presets.py:60` |
| `guidance_scale` | **3.0** | `fastvideo/pipelines/basic/wan/presets.py:59` |
| scheduler | **`FlowUniPCMultistepScheduler(shift=flow_shift)`**, `flow_shift = 3.0` | `fastvideo/pipelines/basic/wan/wan_pipeline.py:11,28`; `fastvideo/configs/pipelines/wan.py:41` |
| `seed` | 1024 (SamplingParam default; the preset does not override) | `fastvideo/api/sampling_param.py:82` |
| resolution / frames / fps | 480x832, 81, 16 | `fastvideo/pipelines/basic/wan/presets.py:55-58` |

⚠️ **Do not read the geometry off the arch-config defaults.**
`fastvideo/configs/models/dits/wanvideo.py:66-73` declares
`num_attention_heads=40`, `num_layers=40`, `ffn_dim=13824` — that is Wan2.1-T2V-**14B**.
The 1.3B values arrive from the checkpoint's `transformer/config.json` via
`update_model_arch` (`fastvideo/configs/models/base.py:60-69`). Likewise resolve
sampling defaults through `SamplingParam.from_pretrained(model_path)`
(`fastvideo/api/sampling_param.py:212`) or the preset — the *class-level*
`SamplingParam` defaults are the generic 720x1280 / 125-frame /
`guidance_scale=1.0` ones (`sampling_param.py:85-95`).

**Stage-1 cell counts:** `30 layers x 12 heads x 50 timesteps = 18,000` records
per `(sparsity, routing_precision)` per CFG branch, i.e. **36,000 per prompt**
when both the conditional and unconditional passes are recorded (CFG runs as two
separate forwards — `fastvideo/pipelines/stages/denoising.py:505-511`, `:550-555`;
tag with `cfg_branch` per §9.3(8)). That clears the `n >= 1000` bar in §10.3 by
more than an order of magnitude. The framework defaults above need no override,
so `SKILL.md`'s "preserve normal inference defaults" is satisfied by doing
nothing.

### 10.3 What counts as "enough" (pre-registered)

- **Every quoted aggregate must carry `n`.** A number without `n` is not
  reportable. `scripts/analyze_masks.py` emits `n` for every cell by
  construction; do not hand-copy numbers around it.
- A `(layer, timestep)` or `(layer, head, timestep)` cell needs **n >= 20**
  paired observations before its median may be quoted or before it may be labeled
  affected/unaffected. Cells with `n < 20` appear in tables with an
  `insufficient_n` flag and are excluded from headline claims.
- A `(sparsity, routing_precision)` aggregate needs **n >= 1000** paired
  observations to be called non-anecdotal (easily satisfied by stage-1
  enumeration).
- A **cross-content** claim (i.e. "this holds across prompts") needs **>= 10
  prompts**, and even then is a development-set statement, never a benchmark
  claim (`SKILL.md` Phase 5).
- A **seed-robustness** claim needs >= 3 seeds on at least a subset; with 1 seed,
  write "single seed" next to the number.
- Report every exclusion: count, reason, and the filter that produced it
  (`SKILL.md` §9 forbids silent discards).

---

## 11. Storage budget

Per `SKILL.md` §1.1 (Phase 1): compute metrics **online** and do not dump full
Q/K activations for all layers/steps.

Rules:

1. The metric pipeline consumes Q/K in-process, emits one JSONL record per cell,
   and frees the tensors. Full-tensor dumps are not part of the normal path.
2. **Cap on retained raw metric data: 5 GiB total** under
   `artifacts/sparsefp4/raw/`. At roughly 400 bytes per JSONL record that is
   >10M records — far beyond what any planned stage produces. If a stage
   projects past the cap, reduce enumerated cells (subsample heads or timesteps
   with a recorded, deterministic rule), never silently truncate output.
3. **Cap on tensor dumps: 2 GiB total**, and only for a **single** debug case:
   one `(prompt_id, seed, layer, timestep, head)` at one block geometry, written
   to `artifacts/sparsefp4/raw/<run_id>/debug/` as compressed `.npz`/`.pt`, with
   a `README.md` in that directory stating why it was kept. Permitted contents
   for that one case: pre-backend Q and K (BF16), the dequantized NVFP4 Q/K, the
   fp32 block-score matrices, and the reference/candidate mask index arrays.
4. `selected_*_idx` index lists (§6.2) are permitted only when the projected
   record size keeps the run under the cap; otherwise store counts only.
5. Figures must be accompanied by the CSV of their plotted values (the analysis
   script does this automatically) — that CSV, not the PNG, is the archival
   artifact.
6. Never overwrite a completed run directory (`SKILL.md` "Artifact layout").
   Deleting a run requires recording the deletion and its reason in
   `artifacts/sparsefp4/STATUS.md`.

---

## 12. Protocol amendments

Append-only log. Any change to §1–§11 after data collection has begun goes here.

| Date (UTC) | Section | Old | New | Reason | Affected runs re-run? |
|---|---|---|---|---|---|
| 2026-08-13 | §4.2 / config `quantization.nvfp4.global_fp32_scale` | "optionally combined with a per-tensor fp32 global scale, as in the standard NVFP4 recipe" | **No second-level scale.** The FA4 path hard-wires the global scale to `1.0` (`fastvideo/attention/backends/flash_attn.py:81-82`); `global_fp32_scale: false` is now mandatory, not a default. | Codebase-map resolution of the §4.2 marker. §4.2 already stated that the real implementation overrides it; recording the specific override. Other repo NVFP4 paths (`nvfp4_utils.py:44-49`, `fp4linear.py:18-22`) *do* use a global scale and must not be borrowed as the model of the attention path. | None — no measured runs existed at the time of this edit (Phase 1 in flight). |
| 2026-08-13 | §4.4 / config `quantization.prefer_native_quantizer` | "If the framework's real quantizer … can be applied to Q/K at the capture point, **use it** … and label the arm `native`." | **Split by arm.** Native *quantization* of Q/K is callable outside the kernel (`_nvfp4_quantize_for_fa4`, `flash_attn.py:138-141`), but **no dequantizer exists for its packed/swizzled layout**, so a router arm that needs dequantized Q/K *values* is necessarily `native_or_simulated = "simulated"`. Attention-compute arms remain `native`. | Resolution of the §4.4 marker. The preference order in §4.4 is unchanged; what changed is that it is unsatisfiable for the router input, which must be stated rather than implied. | None. |
| 2026-08-13 | §1.4 / config `topk.force_retain_diagonal` | Policy allowed for a backend force-retention rule to be adopted "when integrating a real sparse backend". | **VSA has no force-retention rule** (pure top-k, `fused_compress_topk.py:298-313`), so `force_retain_diagonal: false` holds for the VSA-integrated arms too and `jaccard_scored == jaccard` there. The dilution caveat in §1.4 does not apply to any currently planned arm. | Resolution of the §1.4 marker. Tightening only — no metric definition changed. | None. |
| 2026-08-13 | §3.2 (block geometry) — **advisory, no change to the pre-registered values** | Default `block_q = 128`, `block_k = 64`; ablations 64x64 and 128x128. | Values stand **for the diagnostic scorer**. They are **not expressible in VSA**, which forces `block_q == block_k == prod(VSA_TILE_SIZE) ∈ {64, 256}` (`fastvideo/attention/backends/video_sparse_attn.py:313`, `fastvideo-kernel/python/fastvideo_kernel/ops.py:96-106`). Any VSA-integrated arm must therefore report 64x64 (or 256x256) and state the geometry on every table; 128x64 is natively available only via SLA's `get_block_map` (`fastvideo/attention/backends/sla.py:78-110`, which additionally applies smooth-k at `:99` and omits `1/sqrt(d)` at `:102`). | Consequence of the §2.3 and §7.1 marker resolutions. Logged rather than silently rewriting §3.2, because the diagnostic-scorer geometry is unchanged. | None. |
| 2026-08-13 | §2.3 / §9.3(1) | `seq_len` recorded per run. | **Two distinct sequence lengths must be distinguished:** 32760 raster tokens on the dense path vs 39936 tile-padded tokens over 624 tiles at the VSA backend boundary (`video_sparse_attn.py:211-216`, `:28`, `:59-99`). Records must state which ordering/length they refer to. | Resolution of the §2.3 marker; block indices are not comparable across the two orderings. | None. |
| 2026-08-13 | §9.3(6) | Listed as a live confounder to watch and report. | **Eliminated by construction on this path:** `softmax_scale` is a kernel argument applied after Q/K quantization (`flash_attn.py:340-341`, `:355`), and no Q pre-scale exists, so it cannot affect the microscale amax. Still recorded and asserted, but it can no longer explain a native-vs-simulated gap. | Resolution of the §9.3(6) marker. | None. |
| 2026-08-14 | §4.3 / block scorer arithmetic | Scorer accumulation dtype unspecified; `fp32` assumed adequate. | **All block scores are computed in `fp64`.** Phase 1's follow-up analysis showed the pooled scores carry magnitude far above the discriminative spread between competing key blocks, so an `fp32` matmul quantizes the top-k margin onto a power-of-two grid and manufactures exact boundary ties. Measured on real Wan Q/K (Phase 2 `table9_score_resolution_trap8.csv`): `fp32` produces ~1.6k boundary ties per cell where `fp64` produces zero, and `fp32` flips ~0.5% of top-k decisions at `sparsity=0.80`. Critically the penalty is **not equal across router arms**, and H3 *is* the FP8-vs-NVFP4 router comparison, so `fp32` would partly measure float resolution instead of quantization precision — biased **against** H3. The `bf16` null control cannot detect this because both sides of that identity land on the same grid; a passing null control does not certify scorer resolution. | `artifacts/sparsefp4/STATUS.md` trap 8. | Phase 2 uses `fp64` from the first measured run; `score_dtype == "float64"` is a hard gate in `phase2_analyze.verify`. Phase 1's `fp32` tables are superseded by its `fp64` control run, not silently dropped. |
| 2026-08-14 | §7.1 (Phase 2 configurations) | Configurations A–F. | **Two configurations added, and F split by router.** `B_sim` = dense simulated-NVFP4 Q/K + BF16 PV, the simulation control that makes the sparse NVFP4 rows (which have no native kernel) interpretable by bounding the simulation's own error against native `B`. `C_rand` = sparse BF16 compute with a BF16-scored mask into which exactly as many blocks have been swapped, at random, as the NVFP4 mask swaps — the equal-magnitude random-perturbation contrast control for the decision-margin mechanism. `F` is reported as `F8` (FP8 router) and `F16` (BF16 router) so the H3 arms are separate rows rather than a collapsed cell. `D` likewise gains `D8` (FP8 router, BF16 compute) so wrong-mask error is measured for both routers at BF16 compute. | The mechanism claim ("quantization only perturbs harmless boundary blocks") is not falsifiable without an equal-magnitude random contrast; and a simulated sparse row without a dense simulated control cannot separate simulation error from the effect under test. | None — added before the first measured Phase 2 run. |
| 2026-08-14 | §7.2 (Phase 2 statistics) | Median/IQR per configuration. | **The H3 comparison is paired per cell**, at `(prompt, layer, head, timestep, cfg_branch, sparsity)`, not a difference of independently-pooled medians. Every arm shares one dense-BF16 denoising trajectory and the same `k`, so the per-cell paired difference is available and is the correct statistic; the report gives its full distribution plus the fraction of cells that improve. | Pooled medians would hide sign-flipping across cells and overstate precision. | None. |
| 2026-08-14 | §3.2 (block geometry) — Phase 2 execution | Diagnostic scorer at `block_q=128`, `block_k=64`. | Scoring geometry is unchanged, but the block-sparse kernel's query grid is 64 rows, so each 128-token query block is **expanded into its two constituent 64-row kernel blocks** before execution. The executed mask is therefore exactly the scored 128x64 mask. Verified against a masked dense reference in `phase2_selftest.py` (`expanded_128x64_mask_matches_masked_reference`), and the tiling is asserted at runtime (`assert_query_grid_alignment`) rather than assumed. | Phase 2 executes masks rather than only scoring them, which Phase 1 did not. | None. |
