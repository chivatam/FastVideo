# F2 — Precision intervention on the actual VSA selector

Run ID `20260816-215059-6e886a9c-f2-full`, commit `6e886a9c`, 1,166,400 records, all 17
validator checks PASS (`raw/f2_full_validation.json`).

The implementation map this phase depends on is in `VSA_GATE_MAP.md`; this document
covers the intervention, the results and the decision.

## Why this phase exists

F1 and study 1 both measure a *controlled proxy* scorer: mean-pooled Q·K with an explicit
top-k. That is a defensible model of dynamic sparse routing, but it is not what ships. If
the deployed selector had a materially different failure mode, the whole result would be
an artifact of the proxy. This phase measures VSA's own selector, on a genuine VSA
trajectory, using VSA's own kernels.

## Setup

`VSA_PRECISION_PROBE_ATTN` (`fastvideo/attention/backends/vsa_precision_probe_attn.py`)
**subclasses** `VideoSparseAttentionImpl` rather than reimplementing it. The model
therefore runs real `VIDEO_SPARSE_ATTN` — tiling, both branches, `gate_compress`
weighting and all — while the probe measures on the already-tiled Q/K/V the kernel is
about to consume. That is the only point where the routing input exists in the exact form
VSA routes on.

Every record asserts this: `model_trajectory_backend` is
`"VIDEO_SPARSE_ATTN (real, unmodified)"` and `routing_interface` is
`"actual_vsa_fused_block_mean_plus_fused_topk_mask"`, both validator-gated.

- Geometry: VSA 4x4x4 cube tiles, 64 tokens each, 624 blocks, padded length 39,936,
  ragged tail of 8 tokens.
- Execution sparsity fixed at 0.90 (one declared operating point for the trajectory),
  measured sparsities 0.80/0.90/0.95.
- Lattice: 10 prompts x 30 layers x 6 timesteps x 2 CFG branches x 12 heads x 3
  sparsities x 9 arms = 1,166,400 records, zero holes.

## The 9 arms

| Arm | Kind | Routing repr. | Pool | Score | Selection |
|---|---|---|---|---|---|
| V0 | deployed baseline | BF16 | kernel `fused_block_mean` | kernel bf16 matmul, fp32 acc | kernel `fused_topk_mask` |
| V0_FP64 | **rescue** | BF16 | fp64 | fp64 | exact stable sort |
| VA_FP8 | intervention A | FP8-E4M3 | kernel | kernel | kernel |
| VA_NVFP4 | intervention A | NVFP4 | kernel | kernel | kernel |
| VB_FP32 | intervention B | BF16 | fp32 | fp32 | exact stable sort |
| VB_BF16_LOW | intervention B | BF16 | bf16, bf16 acc | bf16, bf16 acc | kernel |
| VA_NVFP4_VB_FP64 | A+B | NVFP4 | fp64 | fp64 | exact stable sort |
| VC_GATE_NVFP4 | **invariant test** | BF16 | kernel | kernel | kernel |
| VD_TORCH_TOPK | tie-break contrast | BF16 | kernel | kernel | `torch.topk` |

Two design points matter:

**Higher-precision arms must not use the bf16 kernel.** Pushing fp64 scores through
`fused_topk_mask` (which only accepts bf16) would discard the precision the arm exists to
test, silently turning intervention B into a no-op. Those arms use an
`exact_index_order` rule — a stable descending sort, which reproduces VSA's *rule* (k
largest, ties toward the lower key-block index) at full precision. The self-test asserts
this rule is bit-identical to the kernel on bf16 scores, so the substitution is verified,
not assumed. A validator check fails if any fp32/fp64 arm is ever routed through the
kernel.

**`VC_GATE_NVFP4` is a falsification test, not a decoration.** `VSA_GATE_MAP.md` claims
`gate_compress` never reaches the selector. This arm quantizes `gate_compress` to NVFP4
for real and asserts the resulting mask is bit-identical to the deployed mask. If
quantizing the gate could move the mask, the map would be wrong and this phase's framing
would collapse.

## Results at sparsity 0.90

| Arm | Jaccard vs deployed | Damage share | Isolation [95% CI] |
|---|---|---|---|
| VA_FP8 | 0.98686 | 4.530e-04 | 310 [216, 495] |
| VA_NVFP4 | 0.96865 | 1.118e-03 | 225 [124, 707] |
| VB_BF16_LOW | 0.96217 | 8.752e-04 | 47.7 [41.4, 55.1] |
| VA_NVFP4_VB_FP64 | 0.96865 | 1.752e-03 | **9.97 [8.17, 12.04]** |
| V0_FP64 (rescue) | 0.98912 | +1.303e-03 (signed) | 3.0 |

**The gate invariant holds exactly.** `VC_GATE_NVFP4` produced a mask bit-identical to
the deployed mask in **129,600 / 129,600 records** at every sparsity, with the gate
genuinely quantized in every one (`gate_compress_quantized: true`). `VSA_GATE_MAP.md`'s
central claim is confirmed by the strongest available test.

**The result transfers.** NVFP4 routing moves the selected block set by 3.1% in Jaccard
terms — one swapped block in 74% of query blocks — for 0.11% of VSA's own sparsification
error, with matched-random 225x more damaging. Same qualitative structure as the proxy.

**The high-precision rescue does not rescue.** Routing at fp64 while executing the
identical sparse kernel is *worse* than the shipped bf16 selector: signed excess
+9.05e-04 / +1.303e-03 / +1.513e-03 at sparsities 0.80/0.90/0.95, and fp64 wins in only
9.9%–11.3% of cells. This is the phase's most consequential finding — it is direct
evidence that routing precision is not the binding constraint, and it removes the premise
of any "precision-decoupled routing recovers quality" claim.

## Decision

`INDETERMINATE_ISOLATION_THRESHOLD` at sparsities 0.80 and 0.90; `PARTIAL_SUPPORT` at
0.95.

The verdict turns entirely on one arm, `VA_NVFP4_VB_FP64`, whose isolation ratio is
9.04 / 9.97 / 10.49 against a hard 10x criterion. Its bootstrap interval spans 10x at all
three sparsities, so **the data does not determine which side it falls on**. Reporting
`SCOPE_REVISION_REQUIRED` from the point estimate at 0.80/0.90 while reporting
`PARTIAL_SUPPORT` at 0.95 would be reading sampling noise as a finding, so the aggregator
downgrades those to explicitly indeterminate. Resolving it needs more prompts, not more
cells — prompts are the unit of replication.

No arm reaches the 1% damage threshold at any sparsity, and the rescue is under 1% of
total sparse error, so no criterion independently forces a revision.

## Incidental finding: VSA's top-k kernel can exceed its own budget

`fused_topk_mask` returns `topk + 1` blocks when the k-th and (k+1)-th block scores tie
exactly. Its 32-iteration fp32 bisection converges toward the k-th value from *below* and
never lands on it, so both tied scores test as strictly above the threshold, the tie-fill
branch computes a negative requirement, and the row keeps one block too many.

- Rate: 5,376 selector rows across 1,166,400 records; worst single cell 6.68e-04 of rows.
- Scale-invariant: multiplying an offending score row by 2 or 0.5 keeps it failing.
- **Affects the shipped `V0` path** (456 records), not only the probe — so it is a
  property of VSA as deployed.
- Reproduced deterministically from a saved score row by `configs/f2_kernel_topk_bug.py`;
  mechanism traced iteration-by-iteration by `configs/f2_explain_violation.py`.

Because F2's measurand *is* VSA's selector, this is recorded per row
(`selector_budget_*` fields) and gated for negligibility rather than treated as an error —
aborting would have discarded a real finding about the kernel. Lesson:
`.agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md`.

## Artifacts

- Tables: `tables/f2_full/` (6 tables: headline, by region, timestep, CFG branch, layer,
  prompt)
- Validation: `raw/f2_full_validation.json` (17 checks); self-test: `raw/f2_selftest.json`
- Figure: `figures/figureB_vsa_selector.{png,pdf}`
- Implementation map: `VSA_GATE_MAP.md`
- Cache: `raw/cache/f2_full.npz` (13 MB from 3.5 GB of JSONL)
