# PAPER_ARTIFACT_MAP — each paper section -> exactly one canonical source

Paper frozen at V4. Purpose: prevent paper-writing agents from pulling
stale V2/V3/V5 claims. When two documents disagree, the file named here
wins.

| Paper section | Canonical source |
|---|---|
| Introduction / claims | `PAPER_CLAIMS_FINAL.md` |
| Title / framing / guardrails | `PAPER_UPDATE_V4.md` |
| Method: native kernel | `NATIVE_PROOF.md` (+ `CODE_PATH_AUDIT.md` for context) |
| Method: DQ-VSA recovery | `TRAINING_RECOVERY_PLAN.md` |
| Operator results (2x2) | `tables/c5_matrix_vsa256_exact10.md` |
| Performance (kernel/DiT/E2E) | `tables/c8_performance_v2.md` |
| Performance root cause | `P4_PERF_ROOT_CAUSE.md` |
| Quality: dense NVFP4 effect | `tables/p1_vs_p0_quality_bootstrap.md` |
| Quality: sparse NVFP4 effect | `tables/p4_vs_p4g_quality_bootstrap.md` |
| Quality: factorial interaction | `tables/nvfp4_sparsity_interaction.md` |
| Quality: geometry trade-off | `tables/p4g_vs_p2_quality_bootstrap.md` |
| DQ-VSA results + decision | `REPORT_V4.md` + `RESULTS_DECISION_V4.md` |
| DQ-VSA statistics | `tables/dqvsa_recovery_bootstrap.md` (+ `raw/statistics/dqvsa_recovery.json`) |
| DQ-VSA serving/perf invariance | `DQVSA_NATIVE_SERVING_PROOF.md` + `tables/dqvsa_final_performance.md` |
| DQ-VSA training-arm triage (context) | `tables/t_matrix_gates.md` (dev gate — filter only, never selection evidence) |
| Related work | `SOTA_RECOVERY_LIT_REVIEW.md` (primary sources only) |
| Limitations | `REPORT_V4.md` §5 |
| Supplement / appendix (W4A4) | `supplementary/w4a4_gate/` (`W4A4_EXPLORATORY_STUDY.md`, `W4A4_AMDAHL_ANALYSIS.md`, `W4A4_NEGATIVE_RESULT.md`) |

Explicitly NOT canonical sources: `REPORT.md`, `REPORT_V2.md`,
`REPORT_V3.md`, `REPORT_V5.md`, `PAPER_UPDATE.md`, `PAPER_UPDATE_V2.md`,
`PAPER_UPDATE_V3.md`, `PAPER_UPDATE_V5.md`, `tables/c8_performance.md`
(pre-allocator-fix), `tables/paper_scale_quality.md` (superseded by the
Holm-corrected V3 tables), anything under `w4a4_gate/` or `full_dqvsa/`
outside the supplement mapping above.
