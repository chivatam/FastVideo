# Experiment Protocol — Native Sparse NVFP4

## Primary question
Can sparse execution and native NVFP4 attention arithmetic compose cleanly on Blackwell?

## Controlled operator matrix

```text
A0 dense BF16
B0 dense native NVFP4 QK + BF16 PV
C0 sparse BF16 + frozen real VSA mask
D0 native sparse NVFP4 QK + BF16 PV + exact same mask as C0
```

The mask must be identical between C0/D0.

Primary metrics:
- MSE
- relative L2
- cosine similarity
- SNR

All vs A0. Report median, IQR, p10/p90, n.

## Production matrix

```text
P0 dense BF16
P1 dense native NVFP4
P2 deployed VSA BF16
P3 VSA + native sparse NVFP4 fine branch
```

P2/P3 must share selector, coarse branch, top-k, geometry, and generation config.

## Video quality

Development: 10 prompts x one fixed seed for debugging only.

Paper evaluation:
- official/repo-integrated VBench-compatible protocol,
- VBench metrics/aggregate available,
- PSNR,
- SSIM,
- LPIPS,
- paired by prompt and seed.

Do not silently drop failed dimensions.

## Performance

Kernel:
- warmup first,
- >=20 repetitions where practical,
- CUDA synchronization,
- median + dispersion.

DiT step:
- >=10 measured steady-state steps where practical.

End-to-end:
- >=5 repeated steady-state generations where practical.

Report:
- kernel latency,
- DiT-step latency,
- E2E latency,
- peak memory.

## Native validity

D0/P3 are invalid if:
- Q/K are dequantized before sparse QK,
- all K tiles are computed,
- only a BF16 sparse kernel runs,
- no runtime/source proof establishes the low-precision sparse path.

## Interaction interpretation

Primary evidence is the raw A0/B0/C0/D0 table.

Describe:
- quant-only error = B0 vs A0,
- sparse-only error = C0 vs A0,
- joint error = D0 vs A0.

Do not assume additivity.

If a derived interaction statistic is introduced, label it exploratory unless it directly follows a cited formulation.

## Failure diagnosis

If D0 is materially worse:
1. MSE/SNR by timestep,
2. error by sparsity,
3. error by resolution,
4. QK vs PV precision,
5. tile/scaling granularity,
6. only then custom routing diagnostics.

## Claims

Can claim native SparseFP4 only when:
- D0/P3 satisfy native proof,
- quality is measured,
- latency is measured.

Cannot claim it from simulation/dequantization or theoretical FLOPs.
