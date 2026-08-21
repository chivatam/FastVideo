# Phase 4 Profiling Method

- GPU: NVIDIA B200 (device 0), persistence on, SM 1845 MHz / mem 3996 MHz.
- CUDA 13.0 (nvcc 13.0.88 pip layout), torch 2.12.0+cu130, branch build
  `TORCH_CUDA_ARCH_LIST=10.0a`; for source correlation the extension was
  rebuilt with `-lineinfo` (codegen-neutral; benchmark latencies unchanged
  within noise: 5.53-5.57 ms).
- Profiler: Nsight Compute 2025.3.1 (extracted from the CUDA 13.0.2
  runfile), run under sudo for counter permissions.
- Workload: one kernel launch via `artifacts/vsa_local_reuse_phase3/profile_one.py
  --mode baseline` (real Phase-1 720p mask, 12 heads, K=144, bf16).
- Commands:

```bash
ncu --profile-from-start off --launch-count 1 --section LaunchStats --section Occupancy ...
ncu --profile-from-start off --launch-count 1 --set detailed --section SourceCounters \
    --import-source yes -o phase4_baseline ...
ncu --import phase4_baseline.ncu-rep --page source --csv > phase4_source.csv
# PC -> file:line mapping via extras/python/ncu_report (action.source_info(addr))
```

- Stall attribution: per-SASS-instruction `stall_long_sb` etc. from the
  source page, aggregated by correlated source line; percentages are from
  PC sampling and therefore approximate (±1-2 points run-to-run).
- Warp-role mapping: source lines are role-exclusive in this kernel
  (warp_id branches), so line-level attribution identifies the role
  directly.
- Latency numbers: CUDA events, median of 30 iters after 8 warmups
  (`run_sweeps.py`, `bench_block_sparse_local_reuse_sm100a.py`).
- ncu classifies mbarrier phase-check spin loops (SYNCS.PHASECHK + BRA
  back-edge, NANOSLEEP suspend) under long-scoreboard / sleep; we report
  them as synchronization waits with the barrier identity taken from the
  source line, which is the load-bearing interpretation step of this phase.
