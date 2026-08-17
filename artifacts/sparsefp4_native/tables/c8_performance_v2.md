# C8 — Performance (V2, unified allocator configuration)

Every E2E arm below ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in a fresh process (receipts: `raw/performance/perf_v2/`, logs: `logs/perf_v2/`). The old `tables/c8_performance.md` mixed pre-allocator-fix numbers (e.g. P4 720p 250.9 s) and is retained as historical/root-cause evidence only — see `P4_PERF_ROOT_CAUSE.md`.

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

## End-to-end at 480x832x81 (median of 3 steady-state reps; 50 steps, seed 1234, prompt p00; first gen excluded as warmup/JIT; expandable_segments allocator in every arm)

| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB | alloc_conf |
|---|---|---|---|---|---|
| P0 dense BF16 (FA4) | 46.62 | 1.000x | 44.18 | 8888 | expandable_segments:True |
| P1 dense native NVFP4 | 44.85 | 1.039x | 42.43 | 8888 | expandable_segments:True |
| P2 deployed VSA@0.9 (Triton fine) | 49.71 | 0.938x | 47.22 | 8893 | expandable_segments:True |
| P4G VSA256-FA4 BF16 fine (10% kept, exact) | 45.62 | 1.022x | 43.24 | 8893 | expandable_segments:True |
| P4 VSA256-FA4 native NVFP4 fine (10% kept, exact) | 47.51 | 0.981x | 45.01 | 8893 | expandable_segments:True |

## End-to-end at 720x1280x81 (median of 3 steady-state reps; 50 steps, seed 1234, prompt p00; first gen excluded as warmup/JIT; expandable_segments allocator in every arm)

| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB | alloc_conf |
|---|---|---|---|---|---|
| P0 dense BF16 (FA4) | 149.13 | 1.000x | 144.82 | 19022 | expandable_segments:True |
| P1 dense native NVFP4 | 133.99 | 1.113x | 129.71 | 19022 | expandable_segments:True |
| P2 deployed VSA@0.9 (Triton fine) | 131.49 | 1.134x | 127.23 | 19028 | expandable_segments:True |
| P4G VSA256-FA4 BF16 fine (10% kept, exact) | 106.20 | 1.404x | 101.93 | 19028 | expandable_segments:True |
| P4 VSA256-FA4 native NVFP4 fine (10% kept, exact) | 112.58 | 1.325x | 108.28 | 19028 | expandable_segments:True |

Notes: all arms share checkpoint/scheduler/steps/resolution/frames/
guidance/seed/negative prompt; no torch.compile/CUDA graphs; one
process per arm; identical `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Kernel table excludes quantization (pre-quantized); E2E includes
everything. Never inferred from FLOPs.
