# PAPER_CLAIMS_FINAL — source of truth for paper writing

Paper frozen at V4. Every claim below maps to receipts via
`PAPER_ARTIFACT_MAP.md`. Nothing outside section 1 may appear as a
contribution; nothing in section 3 may appear in title/abstract/intro
paragraph 1/contribution bullets/primary system diagram.

## 1. MAIN CLAIMS

1. **Native block-sparse NVFP4 attention** exists and is proven on B200
   (packed E2M1 QK, per-16 E4M3 SFs, retained-tile-only execution, BF16
   PV); first-ness worded as "to our knowledge…" with the full qualifier
   string, per the primary-source audit.
2. **Operator composition is numerically clean**: conditioned on identical
   masks at the exact deployment geometry, NVFP4 on retained tiles perturbs
   attention outputs by the same magnitude as dense NVFP4 (rel-L2
   0.092-0.096 vs 0.095-0.102, both resolutions).
3. **Bottleneck migration (sparse attention, narrowly scoped)**:
   sparsification changes the attention kernel's bottleneck composition —
   as QK MMA becomes cheaper, softmax, sparse validity predicates,
   quantization, allocator behavior, and integration overhead can
   dominate, so FP4's arithmetic advantage does not automatically compound
   with sparsity at E2E (kernel increment 1.26x -> ~1.04x at 480p
   retention curve, still 1.26x at 720p plain lists; P4 112.6 s vs P4G
   106.2 s).
4. **Geometry-aligned sparsity is the dominant speed lever**: VSA256/FA4
   alignment gives P4G ~1.40x E2E at 720p vs dense BF16 (1.24x vs deployed
   VSA), at comparable quality with small significant trade-offs.
5. **Quality characterization**: NVFP4 QK causes Holm-significant
   imaging/dynamic degradation (dense: -0.144/-0.306; sparse:
   -0.101/-0.250), and sparsity does not significantly amplify the two
   headline degradation dimensions (factorial interaction: attenuation on
   imaging/aesthetic, small amplification only on subject consistency,
   undetectable for dynamic degree at this n).
6. **DQ-VSA recovery (established result, NOT future work)**:
   "Training-free native SparseFP4 exhibits measurable quality
   degradation. A short teacher-preserving quantization-aware
   velocity-distillation stage substantially recovers that degradation
   while leaving sparsity, geometry, and native NVFP4 inference
   unchanged." Nuance is binding: T3-c500 is the criteria-clean candidate
   (~56% imaging / ~56% dynamic recovery, no significant regressions vs
   P4G); T2-c250 is the maximum target-recovery candidate (~85% imaging /
   100% dynamic, small temporal/background trade-offs). Never "full
   recovery" or "parity".

## 2. SUPPORTING / ABLATION RESULTS

- **Allocator root cause**: the 250.9 s P4 720p anomaly was CUDA
  caching-allocator thrash from transient FP4 buffers; fixed with
  `expandable_segments:True` (controls unchanged); all canonical
  performance numbers use the unified allocator config.
- **Backward-semantics ablation (T2 vs T3)**: at matched 250-step budget,
  naive/high-precision backward matches or beats the Attn-QAT-consistent
  backward in the QK-only NVFP4 / BF16-PV setting (one seed, one LR;
  scoped, not generalized).
- **T1 task-loss drift**: plain flow-matching QAT drifts off the teacher
  (dev gate: dynamic collapses below untrained, aesthetic inflates toward
  fine-tuning data) — the QAD-predicted failure, motivating the
  teacher-preserving loss.
- **P4G-vs-P2 trade-off**: geometry alignment is "comparable on measured
  dimensions" with enumerated significant trade-offs (aesthetic -0.030,
  motion smoothness -0.018, background -0.008, dynamic +0.139); never
  "statistically indistinguishable".
- **Retention/geometry observations**: FP4 kernel increment decays with
  retention at 480p geometry but not at 720p plain lists — the curve is
  geometry-dependent, not universal; vbs mask_mod predicate costs the
  softmax-bound FP4 kernel +42% while free on BF16.
- **Dev-gate caution**: the 10-prompt triage mis-ranked candidates vs the
  326-prompt protocol; dev gates are go/no-go filters only.

## 3. SUPPLEMENTARY / EXCLUDED RESULTS

- **W4A4 / Full-DQ-VSA gate** (`supplementary/w4a4_gate/`): linears are
  63-78% of nominal FLOPs but 11% of eager GPU time (22-24% after
  compilation); native W4A4 was E2E-negative in every configuration; the
  training extension was gated off. Maximum main-paper exposure: one
  Discussion/Limitations paragraph (wording in
  `W4A4_EXPLORATORY_STUDY.md`). NOT a contribution, NOT the thesis, NOT in
  the title.
- **FLOP-vs-time decomposition + Amdahl analysis**
  (`W4A4_AMDAHL_ANALYSIS.md`): appendix only; no universal claims.
- **W4A4 kernel/engineering receipts** (backend layout matrix, PDL
  deadlock, torch.compile interactions): appendix/reproducibility notes.
- **Serving-stack compilation observation** (torch.compile blocks:
  1.34-1.39x forward, outputs equivalent): supplementary observation from
  the gate study; the canonical performance tables remain the
  no-compile V2 protocol numbers.
- Legacy exploratory diagnostics (routing/Jaccard/scorer-precision):
  appendix mechanism context only (unchanged V3 ruling).
