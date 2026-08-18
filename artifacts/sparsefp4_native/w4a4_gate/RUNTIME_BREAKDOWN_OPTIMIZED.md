> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state: REPORT_V4.md (see REPORT_CANONICAL.md). Human-readable summaries: supplementary/w4a4_gate/.

# RUNTIME_BREAKDOWN_OPTIMIZED — D1 component decomposition (Phase 3)

Same methodology as `RUNTIME_BREAKDOWN_EAGER.md`; kernel-trace
categorization (module hooks are incompatible with compiled blocks, so the
kernel trace is the primary instrument here; categories receipts in
`full_dqvsa/flop_breakdown.py`). Traces:
`prof_dqvsa/trace{480,720}_opt.json`.

## Kernel GPU-time decomposition, D0 vs D1

### 480p (per 2 forwards: D0 0.89 s -> D1 0.65 s GPU)

| Category | D0 eager | D1 optimized |
|---|---|---|
| index/scatter/copies (VSA tiling, rope, layout) | 41.7% | 40.5% |
| norm/elementwise | 37.5% | 14.7% eager-residual + 7.9% inductor-fused = 22.6% |
| **linear GEMMs** | **11.2%** | **23.7%** |
| attention fine kernel (FP4 FA4) | 8.9% | 12.2% |
| other | 0.7% | 1.0% |

### 720p (per 2 forwards: D0 2.13 s -> D1 1.59 s GPU)

| Category | D0 eager | D1 optimized |
|---|---|---|
| index/scatter/copies | 39.8% | 37.6% |
| norm/elementwise | 35.2% | 13.5% + 7.3% fused = 20.8% |
| **linear GEMMs** | **10.8%** | **21.9%** |
| attention fine kernel (FP4 FA4) | 13.6% | 19.0% |
| other | 0.6% | 0.7% |

## Where the removed time went

Compilation removed ~27% of total GPU time (0.89->0.65 s at 480p), almost
entirely from the eager norm/modulation/elementwise chains (37.5% -> 22.6%
combined) — inductor fuses LayerNorm+scale/shift+residual+GELU chains and
replaces the eager GEMM launches with autotuned templates
(`triton_tem_fused_addmm_*`, plus the remaining `nvjet` cuBLAS calls).
What it could NOT remove: the `index_elementwise`/`direct_copy` mass
(~40%) — that is the VSA256 tiling scatter/gather, RoPE indexing, and BHSD
`.contiguous()` copies living inside/around the graph-breaking sparse
attention custom ops.

## The critical gate quantity

Linear GEMM GPU-time fraction:

| | eager D0 | optimized D1 |
|---|---|---|
| 480p | 11.2% | **23.7%** |
| 720p | 10.8% | **21.9%** |

This lands at the bottom edge of the Gate-B band (">=20-25% of D1 measured
GPU time"), so the W4A4 PTQ measurement (Phases 4-5) decides — a credible
path to >=10% E2E must be shown by real W4A4 kernels, not FLOPs. Upper
bound if W4A4 GEMMs were infinitely fast at D1: 1/(1-0.237) = 1.31x DiT
GPU-time; with a realistic ~2x GEMM speedup: ~1.13x. The remaining ~40%
index/copy mass is the larger lever and belongs to the sparse-attention
integration, not to linear arithmetic.
