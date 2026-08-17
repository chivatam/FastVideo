# TRAINING_RECOVERY_PLAN — DQ-VSA (Distilled Quantization-Aware VSA)

Goal: recover the NVFP4 quality degradation of the P4 arm (VSA256/FA4 native
sparse NVFP4) while preserving, unchanged:

- exact VSA256/FA4-aligned sparsity geometry ((4,8,8)=256-token tiles, 1:1
  FA4 mask mapping),
- 10% retained attention (VSA_sparsity=0.90),
- native NVFP4 QK inference (packed E2M1 + per-16 E4M3 SFs, FA4 kernel),
- BF16 selector,
- BF16 PV (initially).

The quantity to recover, measured (tables/p4_vs_p4g_quality_bootstrap.md,
326 prompts, paired, Holm-corrected):

| Dim | P4 - P4G mean Δ | 95% CI |
|---|---|---|
| imaging_quality | -0.101 | [-0.112, -0.090] |
| dynamic_degree | -0.250 | [-0.347, -0.153] |
| temporal_flickering | +0.009 | [+0.006, +0.011] |
| motion_smoothness | +0.016 | [+0.014, +0.019] |

The dense contrast (P1 - P0) shows the same signature with larger
magnitudes (imaging -0.144, dynamic -0.306, aesthetic -0.055), and the
factorial interaction test (tables/nvfp4_sparsity_interaction.md) shows the
NVFP4 penalty is NOT amplified by sparsity — on imaging/aesthetic it is
significantly *smaller* under sparsity. Consequence for training design:
the target of recovery is NVFP4's own quantization error on retained tiles,
so a teacher that differs from the student ONLY in QK precision (Stage 2)
is the correct, unconfounded experiment.

Literature grounding: see `SOTA_RECOVERY_LIT_REVIEW.md` (primary sources
only, verified 2026-08-17). The load-bearing supports:

- **Loss choice.** For NVFP4 recovery specifically, distillation from the
  original-precision twin beats task-loss QAT (NVIDIA QAD, arXiv:2601.20088
  §3.1-3.2: QAT silently rewrites the output distribution, KL-to-teacher
  0.311 vs 0.004 for QAD, and can degrade below PTQ on heavily post-trained
  models). The diffusion analogue is velocity distillation
  `L_VD = ||u_student - u_teacher||^2` on identical (x_t, t, c), which
  SpargeAttention2 (arXiv:2602.13515 §4.2, Table 6) shows beats plain
  diffusion-loss fine-tuning on every quality metric and is robust to
  fine-tuning-data mismatch.
- **QAT mechanics.** Attn-QAT (arXiv:2603.00040 §2.2-2.3) is the only
  systematic FP4-attention QAT study: fake-quant forward + STE, with (R1)
  backward-recomputed attention probabilities re-fake-quantized to forward
  precision and (R2) a high-precision auxiliary O' for the softmax-Jacobian
  scalar. With NVFP4 restricted to QK and PV kept BF16 — our exact P4
  configuration, which they also find is the faster B200 configuration —
  O = O' and R2 vanishes; R1 reduces to recomputing S from the
  fake-quantized Q/K in backward, which our autograd kernel does by
  construction (it saves the fake-quantized inputs).
- **Degradation signature match.** Attn-QAT measures the same
  training-free NVFP4 signature on Wan1.3B we do (imaging quality and
  dynamic_degree collapse; their dense FP4 DD -0.276 vs our dense -0.306 /
  sparse -0.250), and plain QAT restored imaging fully but dynamic_degree
  only partially (0.3039 vs BF16 0.3923) — motivating distillation and
  motion-diverse data for exactly the dimension we most need to recover.
- **Curriculum.** No verified paper needs a sparsity curriculum when the
  checkpoint is already sparse-adapted and geometry is frozen (VSA's anneal
  is for introducing sparsity into dense checkpoints). Stage 1 below is
  therefore optional.
- **Hyperparameter anchors.** Attn-QAT: AdamW LR 1e-6, wd 0.01, 3-4k steps,
  global bs ~16, best checkpoint ~3k, Wan-14B-synthesized latents; QAD:
  LR window 1e-6..1e-5; SLA2/SpargeAttention2: 500-step adaptations suffice
  for moderate operator changes.

---

## Implemented infrastructure (smoke-tested, receipts below)

### Training operator: `SPARSEFP4_QAT_VSA256_ATTN`
`fastvideo/attention/backends/sparsefp4_qat_vsa256.py`

- VSA256 metadata/selector at the exact serving geometry (reuses
  `SparseFP4VSA256FA4MetadataBuilder`; same `compute_topk`, same
  `fused_block_mean`/`fused_topk_mask`, BF16 selector + coarse branch,
  differentiable).
- Fine branch: **production fake-quant NVFP4** Q/K — the same flashinfer
  packing + per-16 E4M3 scale pipeline used by the serving kernel, decoded
  by the validated oracle pair (`nvfp4_fake_quant_ste`, reused from the
  Track-D backend) — with a straight-through estimator.
- Fine kernel: autograd Triton route-A of
  `fastvideo_kernel.block_sparse_attn_256` (logical 256-blocks expanded to
  the 64-block Triton kernel). Backward recomputes attention probabilities
  from the *saved fake-quantized* Q/K and uses the saved O/logsumexp —
  the low-precision-consistent backward semantics Attn-QAT prescribes
  (up to FP32 softmax accumulation inside the kernel, which serving also
  uses).
- `fine_qat` per-impl switch: `False` = P4G operator (BF16 fine QK, same
  mask policy). `set_fine_qat(model, flag)` flips a whole model.

Known, documented deviation from serving: the training fine kernel is
Triton, not the FA4 CuTe forward kernel. Identical mask, identical
quantized values, exact softmax — differences are kernel-numerics-level
(the C4 correctness study bounded FA4-vs-oracle at the FP4 arithmetic
floor). The STE makes gradients invariant to this level of discrepancy.

### Stage-2 trainer: `WanDQVSADistillationPipeline`
`fastvideo/training/wan_dqvsa_distillation_pipeline.py`

- Teacher: second transformer instance from the same pretrained checkpoint,
  frozen, eval, `fine_qat=False` -> the P4G operator.
- Student: trainable, `fine_qat=True` -> the (fake-quant) P4 operator.
- Both see identical x_t, timestep, text conditioning and the same VSA256
  metadata inside one forward context.
- Loss: `L_QVD = || u_student_nvfp4(x_t,t,c) - u_teacher_bf16(x_t,t,c) ||^2`
  (velocity distillation; ground-truth flow-matching loss logged as a
  no-grad diagnostic).
- `FASTVIDEO_DQVSA_GRAD_CHECK=1` asserts finiteness of every gradient.

### Smoke test (RUN, PASSED) — `configs/dqvsa_smoke.sh`, `logs/dqvsa_smoke.log`

20 steps, 1x B200, 480x832x77 latents (crush-smol), sparsity 0.90:

- forward precision semantics: teacher logs `fine_qat=False`, student
  `fine_qat=True`, 30 impls each, keep256=0.1000 exactly;
- backward correctness/finiteness: all 20 steps pass the full-parameter
  finite-grad check; distill loss 0.0008-0.06, grad_norm 0.1-4.2;
- memory: ~2.5 s/step with full gradient checkpointing, teacher+student
  fit with headroom on one B200;
- checkpoint save/load: `checkpoints/dqvsa_smoke/checkpoint-20`
  (distributed + diffusers weight-only);
- native serving after training: trained transformer symlink-assembled and
  served through the real P4 arm (`SPARSEFP4_VSA256_FA4_ATTN`, fine=nvfp4,
  FA4 kernel) for a 10-step 480p generation — `DQVSA_SERVE_RC=0`.

---

## Stage 1 — OPTIONAL BF16 sparse adaptation

Teacher: dense BF16 Wan (P0 operator). Student: VSA256 BF16 (P4G operator).
`L_velocity = ||u_student_sparse - u_teacher_dense||^2`, with a VSA-style
sparsity curriculum (`VSA_decay_rate`/`VSA_decay_interval_steps` already in
the trainer: start near-dense, decay to exact 10%). Selector stays BF16.

Status: **deferred**. The geometry test (tables/p4g_vs_p2_quality_bootstrap.md)
shows P4G is broadly comparable to deployed VSA — small significant
trade-offs (aesthetic -0.030, motion_smoothness -0.018, background -0.008,
dynamic_degree +0.139) but no collapse. The sparsity-family cost vs dense
(aesthetic, subject consistency) is shared by P2/P4G/P4 alike and is not
the quantity this project must recover. Run Stage 1 only if Stage 2
converges and the remaining gap to P0-family quality matters for the paper.

## Stage 2 — LOAD-BEARING NVFP4 recovery (the clean experiment)

Teacher: frozen P4G-operator model (original weights). Student: identical
init, identical mask policy, fake-quant NVFP4 fine QK. Teacher and student
differ ONLY in QK precision -> trains away exactly P4 - P4G without
confounding sparsity adaptation. Implemented + smoke-tested above.

Attn-QAT-faithful semantics implemented:
- exact production NVFP4 fake quantization for Q/K (not a synthetic
  quantizer);
- STE through quantization;
- backward attention probabilities recomputed from the same fake-quantized
  values used in forward (Triton kernel saves q_fq/k_fq and recomputes);
- softmax backward uses the saved O and logsumexp (the auxiliary needed for
  consistent FlashAttention-style backward);
- softmax numerically protected (FP32 accumulation in-kernel);
- PV BF16.

## Stage 3 — OPTIONAL dense-teacher joint recovery

After Stage 2 converges: teacher = dense BF16 P0 operator, student = the
Stage-2 P4 student; short stage of
`L = ||u_student_sparse_nvfp4 - u_teacher_dense_bf16||^2` so the model can
compensate for both approximations while keeping the final inference
operator. Infrastructure delta: load teacher without the VSA256 backend
(dense FLASH_ATTN/FA4) — one loader-scope change in the pipeline. Do not
substitute for Stage 2.

## Optional internal distillation (only if velocity distillation stalls)

Attention-OUTPUT matching at a few representative layers:
`L_attn = mean_l NMSE(O_student^l, stopgrad(O_teacher^l))`, implemented as
forward hooks on the attention outputs of both models (no NxN maps stored).
QuantSparse-style multi-scale attention supervision is a further escalation;
do not introduce before velocity distillation is given a fair budget.

---

## Training data

The 47-video static-camera crush-smol set is smoke-test-only (its
motion-poverty was implicated in the Track-D dynamic_degree collapse).
Main-result corpus:

1. Generate a few thousand synthetic clips with a stronger Wan teacher
   (Wan2.1-14B or Wan2.2 family) over a broad VBench-like prompt
   distribution: varied camera motion, human/object motion, spatial detail,
   scene transitions, low/high motion strata. Precedent: VSA adapted
   Wan-1.3B on 80k Wan-14B-synthesized videos; Attn-QAT used 81k
   Wan-14B-synthesized 480p latents. Because Attn-QAT's dynamic_degree
   recovery stayed partial on such data, additionally bias toward motion:
   FPSAttention's optical-flow-magnitude filter (0.05-2.0) is the only
   verified recipe that *raised* dynamic_degree above baseline after QAT.
2. Preprocess to latents+text embeddings with the existing
   `fastvideo/pipelines/preprocess` wan flow (same as crush-smol was).
3. Velocity distillation reads the *teacher's* velocity at sampled
   (x_t, t), which reduces dependence on matching the original pretraining
   distribution — the dataset supplies x_0 coverage, not supervision.

Feasibility scale first: ~2-5k clips, 480p, 77-81 frames.

## First experiment matrix (same data, same compute budget)

| Arm | Recipe | Operator |
|---|---|---|
| T0 | current P4, no training | (baseline) |
| T1 | standard flow-matching QAT | `wan_training_pipeline` + `SPARSEFP4_QAT_VSA256_ATTN` |
| T2 | velocity distillation only | `WanDQVSADistillationPipeline` |
| T3 | Attn-QAT mechanics + velocity distillation | T2 (mechanics already Attn-QAT-faithful) with `FASTVIDEO_FA4_BWD_FALLBACK` audit + backward-consistency verification |
| T4 | T3 + timestep-aware precision schedule | needs the timestep-degradation profile below |

Evaluate at 100 / 250 / 500 steps (weight-only checkpoints at each) before
scaling. LR: start at 1e-6 (Attn-QAT anchor) for T1/T3; allow up to 1e-5
for T2 per the QAD window (the smoke test used 1e-5 without instability at
20 steps). Evaluation at each checkpoint: 10-prompt paired dev protocol
(p_run.py arms T_k vs P4G) for triage; full 326-prompt protocol +
PSNR/SSIM/LPIPS + performance receipt (native P4 serving unchanged) for the
final candidate only.

Success criteria (all required):
1. imaging_quality gap (-0.101) materially closes (>=50% of the gap, CI
   excluding the untrained value);
2. dynamic_degree gap (-0.250) materially closes;
3. temporal_flickering / motion_smoothness / subject & background
   consistency do not regress (paired CIs);
4. sparsity stays exactly 10% (keep256 receipt);
5. native inference path unchanged (same backend, same kernel, serving
   receipt);
6. speed unchanged except checkpoint effects (perf receipt vs
   c8_performance_v2 P4 row).

## Optional denoising-step strategy (fallback/ablation, not primary)

Measure P4-P4G divergence by denoising timestep first (cheap: env-gated
capture256 hooks already dump per-timestep cells; extend the C5 aggregate to
report rel-L2 by timestep, or diff intermediate latents per step from two
paired runs). If the error concentrates in a timestep band, test an
FPSAttention-style schedule: NVFP4 on tolerant steps, BF16 (or MXFP8 if the
fork gains it) on sensitive steps — and train T4 with exactly the same
schedule as inference.

## Explicitly out of scope for the first matrix

- FP8/NVFP4 PV (fork build blocks `MmaF8F6F4Op` under dsl 4.5.3);
- selector precision changes (stays BF16 by design);
- tile-geometry changes (the whole point is to freeze P4 geometry);
- multi-thousand-step runs before the 100/250/500 gates pass.
