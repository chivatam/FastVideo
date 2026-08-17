# DQ-VSA final performance — trained checkpoints through the native P4 path

Protocol identical to `c8_performance_v2.md`: fresh process per arm,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, median of 3
steady-state reps (first gen excluded as warmup/JIT), 50 steps, seed 1234,
prompt p00. Receipts: `raw/performance/dqvsa_final/`, logs
`logs/t_final/dqvsa-*.log`.

## 480x832x81

| System | E2E s | DiT s | Peak MB |
|---|---|---|---|
| P4 untrained (c8_performance_v2 reference) | 47.51 | 45.01 | 8893 |
| P4 + DQ-VSA **T3-c500** (= B0) | 47.58 | 45.16 | 8893 |
| P4 + DQ-VSA T2-c250 | 47.44 | 45.02 | 8893 |

## 720x1280x81

| System | E2E s | DiT s | Peak MB |
|---|---|---|---|
| P4 untrained (c8_performance_v2 reference) | 112.58 | 108.28 | 19028 |
| P4 + DQ-VSA **T3-c500** (= B0) | 111.32 | 107.07 | 19028 |
| P4 + DQ-VSA T2-c250 | 112.39 | 108.02 | 19028 |

Deltas vs untrained P4 are <=1.2% (within rep-to-rep dispersion) with
identical peak memory — the operational confirmation that DQ-VSA is a
weights-only change: same kernel, same exact 10% sparse work, same
selector, same Q/K packing/scales, no BF16 Q/K materialization
(`DQVSA_NATIVE_SERVING_PROOF.md`).
