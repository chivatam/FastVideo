> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state: REPORT_V4.md (see REPORT_CANONICAL.md). Human-readable summaries: supplementary/w4a4_gate/.

# SERVING_OPTIMIZATIONS — D1 definition (Phase 2)

Goal: reduce serving-stack overhead WITHOUT changing model math, weights,
sparsity, precision, scheduler, or steps. D1 must equal D0 within normal
backend numerical tolerance.

## What D1 is

D1 = D0 (B0 weights, native P4 operator) with **`torch.compile` applied to
every transformer block** (`mode="max-autotune-no-cudagraphs"`,
`dynamic=False`): the standard PyTorch inductor path — no new kernels, no
architecture changes. The sparse-attention custom ops (packed-FP4 kernel,
Triton selector ops) graph-break naturally and keep running their existing
kernels; inductor fuses the surrounding norms/modulation/elementwise chains
and takes over the GEMMs with autotuned templates.

Tried and adopted:
- torch.compile on blocks (above) — the entire win.

Tried and rejected/deferred:
- CUDA graphs (`max-autotune`): the dynamic sparse mask + data-dependent
  packing in the attention path is graph-unsafe in the current backend;
  would require the mask-capture refactor that is explicitly out of scope.
- Whole-model compile: graph breaks at the attention boundary make it
  equivalent to per-block compile; per-block keeps compile time bounded
  (~60 s once per shape).

Not touched (per the brief): sparsity, attention precision, selector,
scheduler/steps, model math, weights.

## Equivalence receipt (D0 vs D1, same weights, same inputs, 480p)

```text
rel-L2 3.35e-02   cosine 0.999440   max|Δ| 1.25e-01   all finite
```

(`d0_d1_equivalence.py`; magnitude consistent with BF16 reduction-order
changes accumulated over 30 blocks — the same class of difference as a
BF16 kernel-backend swap.)

## Speed effect (one transformer forward, B=1)

| | D0 eager | D1 compiled | speedup |
|---|---|---|---|
| 480p | 460.4 ms | 331.2 ms | **1.39x** |
| 720p | 1072.1 ms | 801.7 ms | **1.34x** |

(Receipts: `profile_components.py` JSONs; wall per forward, median of 3
after 3 warmups. E2E multiplies by 100 forwards + VAE/encode overhead.)
