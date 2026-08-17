# REPORT_V4 — SparseFP4 Native Composition + DQ-VSA Recovery

Supersedes REPORT_V3.md where they differ; V3 history preserved. All
numbers trace to receipts under `raw/`, `tables/`, `logs/`; environment in
`env/`. Model Wan2.1-T2V-1.3B, VSA sparsity 0.90, seed 1234, 50 steps,
8x B200 (sm_100), CUDA 13.0. Unified statistics throughout: 326-prompt
paired protocol, 10k prompt-level bootstrap, 95% CI, two-sided bootstrap p,
Holm across the 7 scorable VBench dimensions per contrast.

Tags: **[E]** experimentally established, **[S]** statistically
established, **[L]** literature-supported, **[H]** proposed/hypothesized.

## 1. Research question and V3 recap

Can native NVFP4 attention and block-sparse video attention be composed on
Blackwell for real speedup without unacceptable quality loss — and if the
quality loss is real, can training remove it?

V3 established (unchanged, receipts in REPORT_V3):

1. [E] Native block-sparse NVFP4 attention exists and is proven on B200
   (`NATIVE_PROOF.md`); priority wording guarded per the primary-source
   audit (`SOTA_RECOVERY_LIT_REVIEW.md` §10c).
2. [E] Operator-level composition is clean: conditional quant effect ==
   dense quant effect (rel-L2 0.092-0.096 vs 0.095-0.102), both
   resolutions (`tables/c5_matrix_vsa256_exact10.md`).
3. [E] Performance under one allocator config
   (`tables/c8_performance_v2.md`): P4G 1.40x E2E at 720p (BF16); P4 1.33x;
   FP4's kernel increment is geometry-dependent (1.26x -> 1.04x at 480p
   retention curve, still 1.26x at 720p plain lists). Bottleneck-composition
   mechanism wording is binding (PAPER_UPDATE_V3).
4. [S] NVFP4 QK carries a real quality cost: dense imaging -0.144 /
   dynamic -0.306; sparse imaging -0.101 / dynamic -0.250; the factorial
   interaction shows sparsity does NOT amplify it (imaging/aesthetic
   attenuated, subject slightly amplified, dynamic undetectable at this n).
5. [S] Geometry alignment (P4G vs P2): comparable with small significant
   trade-offs (aesthetic -0.030, motion -0.018, background -0.008,
   dynamic +0.139).

## 2. DQ-VSA: training-based recovery (new in V4)

### 2.1 Method [E for infrastructure, L for design]

Teacher-preserving quantization-aware velocity distillation
(`TRAINING_RECOVERY_PLAN.md`): frozen P4G-operator teacher (BF16 fine QK),
student with production fake-quant NVFP4 fine QK, identical VSA256 mask
policy, loss `||u_student - u_teacher||^2` on identical (x_t, t, c).
Backends/pipelines: `SPARSEFP4_QAT_VSA256_ATTN`,
`WanDQVSADistillationPipeline`. Data: 1,290 motion-diverse self-generated
clips over the full 946-prompt VBench corpus with dynamic/human-action
oversampling (seeds disjoint from eval; `configs/t_datagen.py`). Loss
choice and mechanics anchored in the QAD/SpargeAttention2/Attn-QAT
literature (lit review §10a).

Arms — kept distinct in every table:

- **T1**: standard flow-matching/task-loss QAT (LR 1e-6).
- **T2**: velocity distillation + fake-quant NVFP4 forward + **naive
  high-precision backward** (LR 1e-5).
- **T3**: velocity distillation + fake-quant NVFP4 forward + **Attn-QAT-
  consistent backward** (backward P from saved fake-quantized Q/K; LR 1e-5).

500 steps each, same data/batch/compute; gates at 100/250/500.

### 2.2 Dev triage [E]

`tables/t_matrix_gates.md` (10 prompts, go/no-go only): T1 exhibits
teacher-drift (PSNR-to-teacher falls monotonically, aesthetic inflates
toward fine-tuning data, dynamic_degree collapses to 0.30 < untrained
0.50) — excluded from paper scale. T2/T3 advance. The dev gate's
*ranking* of T2/T3 candidates was later contradicted at paper scale;
it is a filter, not selection evidence.

### 2.3 Paper-scale recovery [S]

`tables/dqvsa_recovery_bootstrap.md` — all candidates generated through
the NATIVE P4 serving path (receipts: `DQVSA_NATIVE_SERVING_PROOF.md`;
keep256=0.1012, `Selected backend: SPARSEFP4_VSA256_FA4_ATTN`, fine=nvfp4).

Recovery of the pre-declared targets (deficit: imaging -0.101,
dynamic -0.250):

| Candidate | imaging | dynamic | regressions vs P4G (Holm-sig) |
|---|---|---|---|
| T2-c250 | **85%** (residual -0.015, Holm n.s.) | **100%** (Δ 0.000) | temporal -0.026, subject -0.012, background -0.005; aesthetic +0.022 improves |
| T3-c250 | 17% | 100% | temporal -0.023, subject -0.014, background -0.011 |
| T3-c500 | 56% (residual -0.045, sig) | 56% (residual -0.111, sig) | **none** — subject +0.026, motion +0.020, temporal +0.007, background +0.007 all significantly BETTER than teacher |

vs untrained P4: T3-c500 improves five dimensions significantly and
regresses none; T2-c250 improves imaging/dynamic/aesthetic but regresses
temporal (-0.035)/motion (-0.019)/background (-0.008).

**B0 = T3-c500** by the pre-declared A-H criteria
(`RESULTS_DECISION_V4.md`); T2-c250 documented as the max-target-recovery
alternative. Wording: "substantial recovery with a small residual gap" —
NOT "full parity".

### 2.4 Backward-semantics ablation [S, honestly reported]

At matched 250 steps the naive backward (T2) beats the Attn-QAT-consistent
backward (T3) on imaging recovery (85% vs 17%), equal dynamic recovery,
comparable flicker cost. **No measurable benefit from the
precision-consistent backward in this QK-only NVFP4 / BF16-PV setting** —
consistent with Attn-QAT's own analysis that the O' requirement vanishes
when PV is high precision. Caveats: one seed, one LR, step-count dynamics
differ between recipes (T3 keeps improving imaging to c500; T2-c500
dev-collapsed on dynamic).

### 2.5 Serving and performance invariance [E]

`DQVSA_NATIVE_SERVING_PROOF.md` + `tables/dqvsa_final_performance.md`:
trained checkpoints are weights-only swaps into the untouched native P4
pipeline; E2E latency at 480p/720p matches the untrained P4 rows of
`c8_performance_v2.md` within noise under the same allocator config.

## 3. Updated verdict

Direction B is upgraded: the systems story (bottleneck composition,
geometry alignment as the dominant speed lever) stands, and the quality
objection to native SparseFP4 is now **actionable**: a <=500-step
teacher-preserving distillation recovers most of the NVFP4 QK degradation
(85-100% of the two headline gaps for the per-target-best candidate;
56%/56% with zero regressions for the criteria-clean candidate) with the
serving operator byte-identical.

## 4. Claims (V4, binding wording)

1. [E] Claims 1-4 of REPORT_V3 §9 stand unchanged.
2. [S] Training-free native SparseFP4 exhibits a measurable quality loss
   (imaging -0.101, dynamic -0.250 vs its BF16 twin), but a short
   teacher-preserving quantization-aware distillation stage recovers most
   of the loss while leaving sparsity, geometry, and native NVFP4
   inference unchanged (T3-c500: all pre-declared criteria met, small
   significant residual gap; T2-c250: imaging residual n.s./dynamic parity
   with small stability trade-offs).
3. [S] Task-loss QAT is the wrong recovery loss for this setting: it
   drifts the model off the teacher and collapses motion (dev-gate
   evidence, consistent with QAD).
4. [S] The Attn-QAT-consistent backward provides no measurable benefit
   over naive STE backward for QK-only NVFP4 with BF16 PV (matched-budget
   comparison; caveats in §2.4).
5. [H] Full-DQ-VSA (W4A4 NVFP4 linears + this recovery recipe) is the
   remaining compute opportunity — staged, gated, not yet run.

## 5. Limitations

Single model/sparsity/seed as in V3; recovery evaluated at one paper-scale
seed (paired design); T2/T3 ablation is single-seed single-LR; dev gates
n=10; dynamic_degree CIs wide (n=72, near-binary metric); training corpus
self-generated by the 1.3B model (QAD-style justification; a stronger
teacher corpus is untested); T2-c500/T3-c250 paper-scale rows exist only
where generated (T2-c500 excluded by dev gate).
