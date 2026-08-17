# RESULTS_DECISION_V4 — DQ-VSA training recovery at paper scale

## Verdict: **STRONG RECOVERY (Outcome A, with honest residual-gap wording)**

A short (<=500-step) teacher-preserving quantization-aware velocity
distillation recovers most of native SparseFP4's quality degradation while
leaving sparsity (exact 10%), VSA256/FA4 geometry, BF16 selector, native
NVFP4 QK serving, and E2E latency unchanged. No candidate is claimed to be
at "full parity": the criteria-clean winner keeps a small, statistically
significant residual gap to the BF16 twin.

Receipts: `tables/dqvsa_recovery_bootstrap.md`,
`raw/statistics/dqvsa_recovery.json`, `DQVSA_NATIVE_SERVING_PROOF.md`,
`tables/dqvsa_final_performance.md`. Protocol identical to the V2/V3
contrasts (326 prompts, 7 VBench dims, 10k paired bootstrap, Holm).

## Arms (kept distinct everywhere)

- **T1** — standard flow-matching/task-loss QAT. Excluded from paper scale
  after the dev gate showed teacher-drift and motion collapse
  (dynamic_degree 0.30 < untrained 0.50; PSNR-to-teacher decreasing) —
  the failure mode QAD predicts for task-loss QAT.
- **T2** — velocity distillation from the frozen P4G-operator teacher,
  fake-quant NVFP4 forward, **naive/high-precision-style attention
  backward**.
- **T3** — same distillation, **Attn-QAT-consistent backward** (backward
  attention probabilities recomputed from the saved fake-quantized Q/K).

## Paper-scale results (candidate vs P4G teacher; Holm-corrected)

Pre-declared deficit (P4 - P4G): imaging -0.101, dynamic -0.250.

| | T2-c250 | T3-c250 | T3-c500 |
|---|---|---|---|
| imaging recovery | **85%** (residual -0.015, Holm n.s.) | 17% (residual -0.084, sig) | 56% (residual -0.045, sig) |
| dynamic recovery | **100%** (Δ 0.000) | 100% (Δ 0.000) | 56% (residual -0.111, sig) |
| subject vs P4G | -0.012 (sig) | -0.014 (sig) | **+0.026 (sig, better)** |
| background vs P4G | -0.005 (sig) | -0.011 (sig) | **+0.007 (sig, better)** |
| temporal vs P4G | -0.026 (sig) | -0.023 (sig) | **+0.007 (sig, better)** |
| motion vs P4G | -0.002 (n.s.) | +0.013 (sig) | **+0.020 (sig, better)** |
| aesthetic vs P4G | **+0.022 (sig, better)** | n.s. | n.s. |
| vs untrained P4 | improves imaging/dynamic/aesthetic (sig); regresses temporal/motion/background (sig, small) | improves dynamic; imaging n.s. | **improves 5 dims (sig), regresses none** |

## Success-criteria audit (A-H)

| Criterion | T2-c250 | T3-c500 |
|---|---|---|
| A imaging gap >=50% closed | YES (85%) | YES (56%) |
| B dynamic gap >=50% closed | YES (100%) | YES (56%) |
| C subject/background no material regression | marginal (-0.012/-0.005, sig but small) | YES (improves) |
| D temporal/motion no material regression | **NO** (temporal -0.026 sig) | YES (improves) |
| E exact 10% sparsity | YES (keep256=0.1012 receipts) | YES |
| F BF16 selector | YES | YES |
| G same native serving kernel | YES (`DQVSA_NATIVE_SERVING_PROOF.md`) | YES |
| H E2E latency unchanged | YES (`tables/dqvsa_final_performance.md`) | YES |

## Decision

- **B0 = T3-c500** — the only candidate satisfying ALL pre-declared
  criteria: "substantial recovery with a small residual gap" (imaging
  -0.045, dynamic -0.111 vs P4G) and significantly better than the teacher
  on four stability dimensions; strictly no regression vs untrained P4.
- **T2-c250 is the max-target-recovery alternative**: imaging residual
  statistically indistinguishable after Holm, dynamic at exact parity —
  at the cost of small but significant temporal/subject/background
  regressions (-0.026/-0.012/-0.005). A deployment prioritizing the two
  headline dimensions could reasonably prefer it; the paper reports both.

## Backward-semantics ablation (T2 vs T3) — honest verdict

At matched budget (250 steps), the **naive backward (T2) beats the
Attn-QAT-consistent backward (T3) on imaging recovery (85% vs 17%)** with
comparable flicker cost and identical dynamic recovery. The
theoretically-cleaner backward provides no measurable benefit in this
QK-only NVFP4 / BF16-PV setting — consistent with Attn-QAT's own analysis
that its O' requirement vanishes when PV stays high-precision (O = O'),
leaving only the backward-P re-quantization, which evidently does not bind
here. Caveats: one seed, one LR (1e-5 both), recovery-vs-step-count
entangled (T3 continues improving on imaging to c500 while T2-c500
dev-collapsed on dynamic; step-wise dynamics differ between the recipes).

## What T1 established

Task-loss flow-matching QAT drifts the model off the teacher (dev gate:
aesthetic inflates toward the fine-tuning data, PSNR-to-teacher falls
monotonically, dynamic_degree collapses below untrained). This is the
diffusion analogue of QAD's finding that QAT "silently rewrites the output
distribution" — and the direct empirical justification for the
teacher-preserving loss in DQ-VSA.

## Note on the dev gate

The 10-prompt triage mis-ranked candidates (it put T3-c500's imaging at
teacher level and T2-c250 behind; paper scale reverses this). Dev gates
remain go/no-go filters only — never selection evidence.
