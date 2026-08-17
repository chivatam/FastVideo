# SparseFP4 Native Composition — STATUS

_Last updated: 2026-08-17 00:02 ET (updates every ~10 min while work is active)_

## Live state

- **P4/P4G runs in flight:** perf reps (GPUs 1/3) + 20 quality generations
  (GPUs 4-7). Both smokes passed; keep fraction 0.1012 exact (no coarsening).
- **VBench complete (7 dims, means over 10 prompts):** within the sparse
  family P3 beats P2 on temporal_flickering (0.966 vs 0.951),
  motion_smoothness (0.980 vs 0.975), aesthetic_quality (0.410 vs 0.399);
  trails on subject_consistency (0.846 vs 0.899), background_consistency
  (0.918 vs 0.946), imaging_quality (0.610 vs 0.644). No catastrophic
  degradation on any dim. Raw: `raw/quality/pq_s090_vbench.jsonl`.
- **Paired quality complete:** P3 closer to dense reference than P2
  (PSNR 14.4 vs 11.5 dB, LPIPS 0.585 vs 0.650).
- Next: fold P4/P4G into quality+perf tables, write RESULTS_DECISION.md,
  REPORT.md, PAPER_UPDATE.md; commit study code.

## Four-arm operator result (median over 25 real-VSA cells, rel-L2 vs A0)

| Arm | rel-L2 | cosine | SNR (dB) |
|---|---|---|---|
| B0 dense NVFP4 (quant-only) | 0.098 | 0.9952 | 20.2 |
| C0 sparse BF16 (frozen mask, 24.2% kept) | 0.128 | 0.9929 | 17.9 |
| D0 native sparse NVFP4 (same mask) | 0.204 | 0.9801 | 13.8 |
| D0 vs C0 (quant cost *on top of* sparsity) | **0.096** | 0.9954 | 20.4 |
| C0_TRITON64 (deployed 64x64 mask, 10.1% kept) | 0.206 | 0.9849 | 13.7 |
| D0 vs dequantized oracle (kernel exactness) | 0.0017 | 0.999999 | 55.2 |

**Composition does not amplify error:** adding native NVFP4 QK to the sparse
path costs rel-L2 0.096 — the same as adding it to the dense path (0.098).
Mask coarsening (VSA 64x64 -> FA4 256x128 any-pool) raises retention
0.101 -> 0.242; the deployed finer 10% mask alone (C0_TRITON64, 0.206) sits
at D0's total joint error, so geometry choice dominates the error budget,
not precision. Timestep trend: all arms' error decreases with denoising
progress; B0/C0/D0 ordering stable.

## E2E performance (median of 5 steady-state reps, 50 steps, 480x832x81)

| Arm | E2E s | DiT s | Peak MB |
|---|---|---|---|
| P0 dense BF16 | 46.9 | 44.4 | 8888 |
| P1 dense NVFP4 | 44.4 | 42.1 | 8888 |
| P2 deployed VSA@0.9 | 50.0 | 47.4 | 8893 |
| P2G VSA sel. + FA4 BF16 fine | 48.7 | 46.3 | 8893 |
| P3 VSA sel. + native NVFP4 fine | 53.1 | 50.7 | 8893 |
| P4G / P4 (VSA-on-FA4 geometry) | pending | pending | |

## Phase ledger

| Phase | State | Evidence |
|---|---|---|
| C0 code-path audit | **DONE** | `CODE_PATH_AUDIT.md` |
| C2 native sparse NVFP4 | **DONE** | fork `e650c04e`; envelope limit documented (empty Q-row + multi-wave deadlock — unreachable under VSA topk>=1) |
| C3 native proof | **DONE — all 7 native conditions met** | `NATIVE_PROOF.md`: FP4 kernel symbol in profiler, work-scaling 6.01→0.80 ms (100%→10% retained), packed E2M1+E4M3 receipts |
| C4 correctness vs oracle | **DONE — PASS** | D0 vs dequant oracle: cos 0.999997, rel-L2 2.3e-3 — identical to dense kernels' own oracle deviation (32 cells incl. Wan seqlen 39936) |
| C5 capture | **DONE** | 25 genuine-VSA cells @ sparsity 0.90 |
| C5 2x2 matrix (A0/B0/C0/D0) | **RUNNING** | `configs/c5_operator_matrix.py` |
| P3 production backend | **DONE + smoke passed** | `SPARSEFP4_NATIVE_VSA_ATTN` (deployed selector/coarse branch + native NVFP4 fine branch); 5-step E2E smoke OK |
| C7 video quality (P0-P3) | **RUNNING** | `p_launch.sh pq-s090`, then `p_quality.py` (PSNR/SSIM/LPIPS) + VBench pass |
| C8 performance | kernel part DONE (in NATIVE_PROOF); DiT-step/E2E pending | perf reps after quality sweep |
| C10 decision + report | pending | — |

## Headline evidence so far

1. **D0 native path exists, is correct, and skips work.** Kernel-only at Wan
   shape: dense BF16 7.41 ms → native sparse NVFP4 @10% retained 0.80 ms
   (9.3x vs dense BF16). Profiler shows the `flash_fwd_sm100_fp4` symbol.
2. **No extra numerical cost from composing:** D0's oracle deviation equals
   the dense FP4 kernel's (both 2.3e-3 rel-L2, cos 0.999997).
3. P3 end-to-end works: deployed VSA selector + coarse branch + native
   NVFP4 fine branch generated a valid video (27 s for 5 steps incl. quant).

## Environment

fv-venv (torch 2.12.0+cu130), 8x B200, fa4-fork editable @ `e650c04e`,
repo @ `0a942986` + study backends (uncommitted). Receipts in `env/`.

## Risks / notes

- INVALID/INCOMPLETE rule armed; so far every native gate has passed.
- C5 matrix mask-coarsening (VSA 64x64 → FA4 256x128 any-pool) raises
  retained fraction; C0/D0 share the mask byte-for-byte, and a
  C0_TRITON64 control quantifies the coarsening separately.
