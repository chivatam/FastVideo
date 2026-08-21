# CLAIM_LEDGER — geometry-alignment spin-out paper

Claim-ledger gate (Claude Scholar `ml-paper-writing`): no claim enters
manuscript prose unless an evidence record supports it; allowed wording is
binding; forbidden stronger wording is listed explicitly. Parent-study
wording constraints from `PAPER_CLAIMS_FINAL.md` are inherited.

| ID | Claim (allowed wording) | Evidence record | Forbidden stronger wording |
|---|---|---|---|
| C1 | Deployed VSA at 0.90 sparsity yields 1.13x E2E at 720p and 0.94x at 480p vs dense BF16 FA4 | `tables/c8_performance_v2.md` (P2 vs P0 rows) | "sparse attention never helps" |
| C2 | The 64-token selector tile cannot be consumed by the FA4 SM100 block-sparse kernel; the deployed fine branch runs a Triton kernel on sm_100 | `CODE_PATH_AUDIT.md` §4 | "Triton kernels are inherently slow" |
| C3 | Any-pool coarsening of a 64-tile mask onto FA4's 256x128 granularity inflates retention ~2.4x; coarsened arms measured 48.7-53.1 s at 480p (worse than dense) | backend docstring `sparsefp4_vsa256_fa4.py`; `REPORT_V2.md` §7 (demoted arms) | inflation figures at other sparsities/geometries (unmeasured) |
| C4 | Selecting on (4,8,8)=256-token tiles maps 1:1 onto FA4 sparse granularity with exact retention (median keep 0.1006) | `sparsefp4_vsa256_fa4.py`; `tables/c5_matrix_vsa256_exact10.md` | "lossless selection" (selection quality differs; see C8) |
| C5 | Aligned sparse BF16 kernel time tracks dense-x-retention: 8.93x at 10% kept (0.830 vs 7.414 ms, medians of 50) | `tables/c8_performance_v2.md` kernel table | presenting 8.93x as an E2E number |
| C6 | P4G achieves 1.40x E2E at 720p vs dense BF16 (149.13 -> 106.20 s) and 1.24x vs deployed VSA; 1.02x at 480p; flat peak memory | `tables/c8_performance_v2.md` E2E tables | attributing 1.40x to FP4/quantization; extrapolating beyond 720p/81f |
| C7 | The speedup is precision-independent: the NVFP4 twin (P4) is slower E2E than BF16 P4G (112.58 vs 106.20 s), for attributed integration reasons | `tables/c8_performance_v2.md`; `P4_PERF_ROOT_CAUSE.md` | "FP4 attention is useless" (kernel-level FP4 wins are real) |
| C8 | P4G quality is comparable to deployed VSA on measured dimensions with small Holm-significant trade-offs: aesthetic -0.030, motion smoothness -0.018, background -0.008; dynamic degree +0.139; subject/imaging/temporal n.s. | `tables/p4g_vs_p2_quality_bootstrap.md` (326 prompts, Holm) | "statistically indistinguishable"; any non-inferiority claim; "quality-neutral" |
| C9 | P4G is slightly farther from the dense reference than P2 on pixel metrics (dPSNR -0.25, dSSIM -0.006, dLPIPS +0.023) | same table, pixel block | interpreting pixel proximity to P0 as quality |
| C10 | Sparsification error at the aligned geometry: rel-L2 0.192/0.218 (480p/720p) vs dense attention outputs at exact 10% | `tables/c5_matrix_vsa256_exact10.md` | treating operator error as end-video quality |
| C11 | VSA's own ablation prefers B=64 for pretraining loss (0.13162 vs 0.13375 at B=256) | `SOTA_RECOVERY_LIT_REVIEW.md` §1 (verified primary source) | "VSA chose the wrong tile size" |

## Hypotheses (must stay marked, never stated as findings)

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Gains grow beyond 720p/81 frames | untested — mechanism prediction only |
| H2 | Fine-tuning at B=256 would remove the small quality trade-offs | untested |
| H3 | The alignment principle transfers to non-Blackwell kernels | untested — principle plausible, magnitudes platform-specific |

## Citation verification status

All arXiv-cited works were fetched from primary sources on 2026-08-17
(`SOTA_RECOVERY_LIT_REVIEW.md`, verification appendix). The FA4 paper itself
has no located arXiv ID; FA4 kernel characterization is second-hand via
arXiv:2603.00040 and must be cited as such.
