# C8 — Performance

## Attention-kernel latency (CUDA events, median of 50, Wan shape B=1 S=39936 H=12 D=128, pre-quantized inputs)

| Arm | retained | median ms | p90 ms |
|---|---|---|---|
| A0_dense_bf16_kernel | dense | 7.414 | 7.446 |
| B0_dense_fp4_kernel | dense | 6.017 | 6.297 |
| D0_sparse_fp4 | 1.00 | 6.010 | 6.069 |
| C0_sparse_bf16 | 1.00 | 7.591 | 7.790 |
| D0_sparse_fp4 | 0.50 | 3.074 | 3.313 |
| C0_sparse_bf16 | 0.50 | 3.815 | 4.135 |
| D0_sparse_fp4 | 0.25 | 1.654 | 1.751 |
| C0_sparse_bf16 | 0.25 | 1.814 | 2.041 |
| D0_sparse_fp4 | 0.10 | 0.800 | 0.808 |
| C0_sparse_bf16 | 0.10 | 0.830 | 0.837 |

## Fine-branch wall-clock per call (24%-kept mask, incl. host overhead)

| Branch | wall ms | CUDA-event ms |
|---|---|---|
| fine_bf16 | 1.653 | 1.860 |
| fine_fp4 | 1.438 | 1.752 |
| fine_fp4_prequant | 1.431 | 1.527 |

Quantize overhead: 0.173 ms per call (Q+K), ~0.5 s per 50-step CFG video.

## End-to-end at 480x832x81 (median steady-state reps; 50 steps, seed 1234, prompt p00; first gen excluded as warmup/JIT)

| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB |
|---|---|---|---|---|
| P0 dense BF16 (FA4) | 46.92 | 1.000x | 44.38 | 8888 |
| P1 dense native NVFP4 | 44.44 | 1.056x | 42.05 | 8888 |
| P2 deployed VSA@0.9 (Triton fine) | 49.98 | 0.939x | 47.44 | 8893 |
| P2G VSA sel. + FA4 BF16 fine (24% kept) | 48.67 | 0.964x | 46.28 | 8893 |
| P3 VSA sel. + native NVFP4 fine (24% kept) | 53.09 | 0.884x | 50.71 | 8893 |
| P4G VSA256-FA4 BF16 fine (10% kept, exact) | 45.62 | 1.029x | 43.22 | 8893 |
| P4 VSA256-FA4 native NVFP4 fine (10% kept, exact) | 47.33 | 0.991x | 44.96 | 8893 |

## End-to-end at 720x1280x81 (median steady-state reps; 50 steps, seed 1234, prompt p00; first gen excluded as warmup/JIT)

| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB |
|---|---|---|---|---|
| P0 dense BF16 (FA4) | 148.76 | 1.000x | 144.45 | 19026 |
| P1 dense native NVFP4 | 135.78 | 1.096x | 131.47 | 19027 |
| P2 deployed VSA@0.9 (Triton fine) | 131.72 | 1.129x | 127.47 | 19033 |
| P4G VSA256-FA4 BF16 fine (10% kept, exact) | 106.09 | 1.402x | 101.76 | 19032 |
| P4 VSA256-FA4 native NVFP4 fine (10% kept, exact) | 250.89 | 0.593x | 241.99 | 19032 |

Notes: all arms share checkpoint/scheduler/steps/resolution/frames/
guidance/seed/negative prompt; no torch.compile/CUDA graphs; one
process per arm. Kernel table excludes quantization (pre-quantized);
E2E includes everything. Never inferred from FLOPs.
