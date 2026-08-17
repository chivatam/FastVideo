# REPORT_V3 — SparseFP4 Native Composition (paper-validation pass)

Supersedes REPORT_V2.md where they differ. All numbers trace to receipts
under `raw/`, `tables/`, `logs/`; environment in `env/`. Model
Wan2.1-T2V-1.3B, VSA sparsity 0.90, seed 1234, 50 steps, 8x B200 (sm_100),
CUDA 13.0. Statistics: unified prompt-level paired percentile bootstrap
(10k resamples, two-sided bootstrap p, Holm correction across the 7 VBench
dimensions per contrast family; `configs/paired_stats_v2.py`, receipts in
`raw/statistics/`).

This document separates: **[E]** experimentally established results,
**[S]** statistical conclusions, **[L]** literature-supported design
choices, **[H]** proposed/new hypotheses.

## 1. Research question

Can native NVFP4 attention compute and block-sparse video attention be
composed on Blackwell for real speedup without unacceptable quality loss?

## 2. Native implementation [E]

Unchanged from `NATIVE_PROOF.md`: packed `float4_e2m1fn_x2` Q/K + per-16
E4M3 SFs, `flash_fwd_sm100_fp4` block-scaled MMA over retained-tile-only
load/MMA/softmax loops, BF16 PV, no BF16 Q/K materialization; profiler +
work-scaling receipts. Priority wording (per `SOTA_RECOVERY_LIT_REVIEW.md`
§10c, primary-source audit of the required 2025-2026 corpus): no verified
paper demonstrates **native block-sparse NVFP4 attention**; the closest are
Attn-QAT (arXiv:2603.00040 — native NVFP4 attention, dense only; in-kernel
block-sparse capability, no sparse experiments; sparse+QAD stated as future
work), SLA2 (arXiv:2602.12675 — sparse + INT8/FP8, not FP4), and
FPSAttention (arXiv:2506.04648 — sparse + FP8). A first-ness claim is
defensible **only** with all qualifiers ("native block-sparse NVFP4
attention for video diffusion, with paired quality evaluation") and should
be framed as "to our knowledge, no published system…" given the
unverifiable FlashAttention-4 paper and Attn-QAT's announced follow-up.

## 3. Canonical exact-10% operator matrix [E]

`tables/c5_matrix_vsa256_exact10.md` (unchanged): rel-L2 vs A0, medians —

| Resolution | quant-only B0 | sparse-only C0_256 | joint D0_256 | conditional D0 vs C0 |
|---|---|---|---|---|
| 480x832x81 | 0.0951 | 0.1918 | 0.2885 | 0.0957 |
| 720x1280x81 | 0.1018 | 0.2175 | 0.3005 | 0.0918 |

Conditioned on the same mask, NVFP4 on retained tiles perturbs attention
outputs by the same magnitude as dense NVFP4, at both resolutions
(operator level; raw values, no additivity assumed).

## 4. Performance (V2 canonical, unified allocator) [E]

`tables/c8_performance_v2.md` — every E2E arm rerun in a fresh process
under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (no mixing of
pre/post-allocator-fix numbers; the old `tables/c8_performance.md` with the
pre-fix P4 720p 250.9 s is historical root-cause evidence only,
`P4_PERF_ROOT_CAUSE.md`):

| System | 480p E2E s (x) | 720p E2E s (x) |
|---|---|---|
| P0 dense BF16 | 46.6 (1.00) | 149.1 (1.00) |
| P1 dense NVFP4 | 44.9 (1.04) | 134.0 (1.11) |
| P2 deployed VSA | 49.7 (0.94) | 131.5 (1.13) |
| P4G VSA256-FA4 BF16 (10%) | 45.6 (1.02) | **106.2 (1.40)** |
| P4 VSA256-FA4 NVFP4 (10%) | 47.5 (0.98) | 112.6 (1.33) |

Kernel-level (pre-quantized, CUDA events): FP4-over-BF16 increment decays
dense 1.26x -> 50% 1.24x -> 25% 1.10x -> 10% 1.04x at 480p geometry, but
remains 1.26x at 720p geometry with plain sparse lists (3.16 vs 3.98 ms).
**The retention curve is geometry-dependent; do not universalize it.**
The 9.3x kernel number (sparse FP4 vs dense BF16) is a *sparsity* speedup.
1.40x E2E belongs to BF16 P4G, never to FP4.

Mechanism phrasing (binding): sparsification changes the kernel's
bottleneck composition. As QK MMA becomes cheaper, non-MMA costs — softmax,
sparse validity predicates, scale-factor handling, quantization,
launch/integration overhead, and allocator behavior — can dominate, so
arithmetic speedups do not automatically translate into multiplicative E2E
gains. The residual 720p P4-P4G gap (6.4 s) is attributed to the vbs
mask_mod predicate on the softmax-bound FP4 kernel (+42% kernel cost,
`p4_maskmod_ab.json`), per-call quantization (~1 s, `quant_overhead.json`),
and small integration overhead.

## 5. Quality at paper scale — the four contrasts [S]

326 official VBench prompts, 7 scorable dimensions, paired seed/config.

### 5.1 Dense NVFP4 effect (P1 - P0), `tables/p1_vs_p0_quality_bootstrap.md`

Holm-significant negatives: imaging_quality **-0.144** [-0.168, -0.119],
dynamic_degree **-0.306** [-0.417, -0.194], aesthetic_quality **-0.055**
[-0.072, -0.039], temporal_flickering -0.004; significant positive:
background_consistency +0.005. Subject consistency +0.008 (Holm p=0.052,
n.s.); motion smoothness n.s. P1's similarity to P0: PSNR 15.8 [15.6, 16.0].

### 5.2 Sparse NVFP4 effect (P4 - P4G), `tables/p4_vs_p4g_quality_bootstrap.md`

Same unified implementation as 5.1 (no parallel stats scripts).
Holm-significant negatives: imaging_quality **-0.101** [-0.112, -0.090],
dynamic_degree **-0.250** [-0.347, -0.153]; significant positives:
temporal_flickering +0.009, motion_smoothness +0.016. Subject, background,
aesthetic n.s. after Holm. Pixel similarity to P0: P4 is *closer* to the
dense reference than P4G on all three metrics (ΔPSNR +0.97 [+0.88, +1.06],
ΔSSIM +0.017, ΔLPIPS -0.045, all p=0.0002) — noted honestly; we do not
interpret pixel-proximity to P0 as quality.

### 5.3 Factorial interaction I = (P4-P4G) - (P1-P0), `tables/nvfp4_sparsity_interaction.md` [S]

Does sparsity amplify the NVFP4 quality effect? **Mixed, mostly the
opposite.** Holm-significant *positive* interactions (NVFP4 penalty smaller
under sparsity): imaging_quality **+0.043** [+0.016, +0.069],
aesthetic_quality **+0.049** [+0.031, +0.067], temporal_flickering +0.012,
motion_smoothness +0.014. Holm-significant *negative* interaction (penalty
larger under sparsity): subject_consistency **-0.014** [-0.024, -0.005].
dynamic_degree +0.056 [-0.083, +0.194] and background n.s. — for
dynamic_degree specifically: no detectable interaction at this sample size
(CI includes zero; note the CI is wide, so a moderate interaction is not
excluded either).

Wording constraints honored: we do not claim "no interaction" from p>0.05;
where CIs exclude zero we report the interaction. On no dimension does
sparsity significantly *amplify* the NVFP4 penalty except
subject_consistency (small, -0.014); on the two headline damaged
dimensions the dense penalty is at least as large (imaging/aesthetic:
significantly larger dense). This is a standard difference-in-differences;
no custom damage ratio is used. Relation to QuantSparse (arXiv:2509.23681
App. F: super-additive attention-map shift under joint quant+sparsity): not
a contradiction — different quantization target (INT linear layers vs
NVFP4 QK), different observable (attention-map MSE vs end-video VBench),
different sparsifier; our §3 operator-level result is conditional on the
same mask at our specific geometry.

### 5.4 Geometry alignment (P4G - P2), `tables/p4g_vs_p2_quality_bootstrap.md` [S]

The V2 claims "quality-neutral" / "statistically indistinguishable" are
**retired — not supported**. Holm-significant differences: aesthetic_quality
**-0.030** [-0.040, -0.020], motion_smoothness **-0.018** [-0.020, -0.015],
background_consistency -0.008, dynamic_degree **+0.139** [+0.069, +0.222].
Subject consistency, imaging quality, temporal flickering n.s. Pixel:
P4G slightly farther from P0 than P2 (ΔPSNR -0.25, ΔSSIM -0.006,
ΔLPIPS +0.023, p=0.0002). Licensed wording: P4G is **comparable on
measured dimensions** with small significant trade-offs — it gives up a
little aesthetic quality, motion smoothness, and background consistency
against deployed VSA while producing markedly more motion. No
non-inferiority margin was pre-registered, so no non-inferiority claim is
made.

## 6. Verdict (updated Direction B) [E+S]

Native SparseFP4 is correct and kernel-positive; at the production
operating point it trails its BF16 twin E2E (112.6 vs 106.2 s at 720p) for
fully-attributed integration reasons, and NVFP4 QK carries a real,
Holm-significant no-reference quality cost (imaging, dynamism) that is
present dense and sparse alike — with no evidence sparsity amplifies it on
those dimensions (it attenuates on imaging/aesthetic, small amplification
only on subject consistency). The positive systems result is
geometry-aligned sparse attention: 1.40x E2E at 720p in BF16, with a
quality profile comparable to deployed VSA up to the small trade-offs in
§5.4.

## 7. Training-based recovery (Part B) [L+H, smoke-tested infrastructure E]

`TRAINING_RECOVERY_PLAN.md` (design; literature-anchored) +
`SOTA_RECOVERY_LIT_REVIEW.md` (primary sources). Implemented and
smoke-tested (20 steps, receipts `logs/dqvsa_smoke.log`):

- `SPARSEFP4_QAT_VSA256_ATTN` training backend: exact VSA256 serving
  geometry, production fake-quant NVFP4 QK + STE, autograd fine kernel
  whose backward recomputes attention probabilities from the saved
  fake-quantized Q/K (Attn-QAT R1; R2's high-precision O' is automatic
  because PV is BF16, so O = O').
- `WanDQVSADistillationPipeline`: frozen P4G-operator teacher, fake-quant
  student, velocity distillation
  `L_QVD = ||u_student_nvfp4 - u_teacher_bf16||^2` — teacher and student
  differ only in fine QK precision (the clean Stage-2 causal experiment).
- Smoke gates all passed: precision semantics (30 teacher BF16 impls vs 30
  student fake-quant impls, keep256=0.1000), gradient finiteness (20/20
  steps), memory (~2.5 s/step, 1x B200 with checkpointing), checkpoint
  save/load, and native P4 serving of the trained weights
  (`DQVSA_SERVE_RC=0`).
- No large training run was started (per instruction). Next: T0-T4 matrix
  at 100/250/500 steps on a motion-diverse corpus (plan §matrix).

## 8. Limitations

Single model (Wan2.1-1.3B), single sparsity point (0.90), one seed at
paper scale (paired design mitigates); 7/16 VBench dimensions scorable
here; E2E medians from 3 steady-state reps of one prompt per arm
(dispersion across reps <0.5%); FP8/NVFP4 PV untested (fork build limit);
training fine kernel is Triton route-A, operator-equivalent but not
bit-identical to the FA4 serving kernel; dynamic_degree is a binary-ish
VBench dimension with wide CIs at n=72.

## 9. Exact defensible claims (V3, binding wording)

1. [E] Native block-sparse NVFP4 attention (packed E2M1 QK, per-16 E4M3
   SF, retained-tile-only execution, BF16 PV) exists and is proven on
   B200; to our knowledge no published system demonstrates this natively
   (guarded per §2).
2. [E] Conditioned on identical masks at the exact deployment geometry,
   NVFP4 on retained tiles perturbs attention outputs by the same
   magnitude as dense NVFP4 (rel-L2 0.092-0.096 vs 0.095-0.102), both
   resolutions.
3. [E] Sparsification changes the kernel's bottleneck composition: FP4's
   kernel increment decays with retention at 480p geometry (1.26x -> 1.04x)
   while remaining 1.26x at 720p geometry with plain lists; integration
   costs (validity predicate on the softmax-bound FP4 pipeline,
   quantization, allocator behavior) can invert the E2E ordering (112.6 vs
   106.2 s). Arithmetic speedups do not automatically compound with
   sparsity; neither do they universally collapse.
4. [E+S] Selector-geometry alignment is the dominant lever: 1.40x E2E vs
   dense at 720p (BF16), 1.24x vs deployed VSA, at comparable quality on
   measured dimensions (small significant trade-offs: aesthetic -0.030,
   motion smoothness -0.018, background -0.008; dynamic degree +0.139).
5. [S] NVFP4 QK causes Holm-significant no-reference quality losses in
   both dense (imaging -0.144, dynamic -0.306, aesthetic -0.055) and
   sparse (imaging -0.101, dynamic -0.250) settings, with small
   significant gains on temporal stability metrics in the sparse setting;
   the factorial interaction shows the penalty is not amplified by
   sparsity on those damaged dimensions (imaging/aesthetic attenuated;
   subject consistency slightly amplified, -0.014; dynamic degree: no
   detectable interaction at this sample size).
6. [L+H] Distillation-based QAT (DQ-VSA) is the literature-indicated
   recovery: velocity distillation from the BF16 sparse twin under
   Attn-QAT-faithful semantics, geometry frozen. Infrastructure exists and
   passed a 20-step smoke including native serving; recovery itself is
   unproven (hypothesis, matrix pending).
