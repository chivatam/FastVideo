> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state: REPORT_V4.md (see REPORT_CANONICAL.md). Human-readable summaries: supplementary/w4a4_gate/.

# W4A4_PTQ_PERFORMANCE + THROUGHPUT/MEMORY + NATIVE PROOF (Phases 4-5, condensed)

All runs: B0 weights, native P4 attention operator, 1x B200,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, median of 5 forwards
after 3 warmups (`w4a4_forward_bench.py`, raw JSONs in `raw/`).

## Native W4A4 implementation (proof summary)

- Weights: packed `float4_e2m1fn_x2` + per-16 E4M3 SFs via production
  `flashinfer.nvfp4_quantize` (128x4 SF layout, unshuffled — the layout the
  cudnn backend consumes; correctness matrix in `raw/backend_layout_matrix`:
  cudnn+noshuffle rel-L2 0.134 = the NVFP4 arithmetic floor, wrong layouts
  give rel-L2 1.37).
- Activations: production `nvfp4_quantize` per call (frozen per-module
  global scale calibrated on first call).
- GEMM: `flashinfer.mm_fp4` cudnn backend — native Blackwell block-scaled
  FP4 tensor-core GEMM; no dequant-to-BF16 GEMM anywhere (source:
  `W4A4Linear.forward`).
- Backend engineering receipts: trtllm backend (prebuilt
  `Gemm_Bfloat16_E2m1E2m1_..._sm100f` cubins) is numerically correct
  standalone but its PDL launches deadlock against the persistent FA4
  attention kernel in-model; cutlass backend JIT-compiles >10 min/shape.
  cudnn + `enable_pdl=False` is the working path.

## Kernel-level W4A4 vs BF16 (Wan shapes, 30-rep medians)

| GEMM (m,k,n) | BF16 ms | FP4 ms (pre-quant) | FP4+act-quant ms | speedup incl. quant |
|---|---|---|---|---|
| QKV 480p (32760,1536,4608) | 0.294 | 0.221 | 0.275 | 1.07x |
| o_proj 480p (32760,1536,1536) | 0.104 | 0.211 | 0.269 | **0.39x (slower)** |
| FFN-in 480p (32760,1536,8960) | 0.567 | 0.218 | 0.279 | 2.03x |
| FFN-out 480p (32760,8960,1536) | 0.588 | 0.216 | 0.339 | 1.74x |
| FFN-in 720p (75600,1536,8960) | 1.272 | 0.567 | 0.659 | 1.93x |
| FFN-out 720p (75600,8960,1536) | 1.460 | 0.452 | 0.962 | 1.52x |

## In-model forward latency (the decision numbers)

Eager (D0-stack):

| Config | 480p ms (Δ) | 720p ms (Δ) |
|---|---|---|
| W0 all-BF16 | 447.9 | 1076.1 |
| W1 +W4A4 FFN | 476.7 (**+6.4% slower**) | 1102.2 (+2.4%) |
| W2 +o_proj | 490.7 (+9.6%) | 1122.2 (+4.3%) |
| W3 +QKV | 533.0 (+19.0%) | 1167.1 (+8.4%) |

Compiled (D1-stack, `default` mode):

| Config | 480p ms (Δ) | 720p ms (Δ) |
|---|---|---|
| W0 all-BF16 | 308.8 | 754.7 |
| W1 +W4A4 FFN | 355.3 (**+15.1% slower**) | 788.4 (+4.5%) |
| W3 all target linears | 411.2 (+33.2%) | 848.1 (+12.4%) |

W4A4 is net-negative in EVERY configuration. Mechanism: each W4A4 linear
adds an activation-quantize kernel + graph break; the FFN GEMM saving
(~0.3-0.9 ms/layer) is smaller than the added quant/launch/copy overhead,
o_proj/QKV GEMMs are memory-bound at K=1536 and gain nothing (o_proj
actively loses 2.5x), and under torch.compile the dynamo-opaque module
additionally forfeits inductor's fused epilogues.

## Throughput / concurrency / memory (Gates C/D quantities)

- Batching does not change component shares (B=1/2/4 within 0.5pt,
  `RUNTIME_BREAKDOWN_EAGER.md`) — per-sample work already saturates the
  GPU, so throughput == 1/latency here; latency is worse ⇒ throughput is
  worse. No concurrency benefit exists to offset it.
- Memory: 4-bit weights would shrink the ~3.0 GB BF16 transformer weights
  by ~2.2 GB, but measured runtime peak INCREASED (480p: 3.35 -> 4.58 GB
  eager W3) from transient quant buffers, and capacity is nowhere near
  binding on B200-183GB (720p B=4 peaks at 28.3 GB; feasible concurrency is
  compute-limited, not memory-limited). No serving-economics change.

## Phase-6 quality triage: intentionally skipped

The gate fails on performance alone in every configuration; per-GEMM
quantization error is documented (rel-L2 0.134 vs BF16 per linear). Running
video-quality triage for a system that is strictly slower would not affect
the decision.
