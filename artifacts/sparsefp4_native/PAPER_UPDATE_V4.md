# PAPER_UPDATE_V4 — framing after the DQ-VSA recovery result

> **CANONICAL PAPER FRAMING (paper frozen at V4).** PAPER_UPDATE_V5 does
> NOT supersede this file; it is a supplementary appendix note for the
> gated-off W4A4 exploration. Title, abstract, thesis, and contribution
> bullets come from THIS file + `PAPER_CLAIMS_FINAL.md`.

Supersedes PAPER_UPDATE_V3.md; V3 guardrails remain binding except where
explicitly upgraded here.

## Title direction

"Geometry, Bottlenecks, and Recovery: Composing Block-Sparse and NVFP4
Attention for Video Diffusion on Blackwell"
(alt: keep V3's "Why Sparsity and FP4 Do Not Automatically Compound…" with
DQ-VSA as the constructive final act)

## Narrative arc (V4)

1. Build native block-sparse NVFP4 attention (guarded first-ness wording).
2. Numerics compose cleanly at the operator level (exact-geometry 2x2).
3. Sparsification changes the kernel's bottleneck composition; arithmetic
   speedups do not automatically compound (verbatim mechanism sentence
   from PAPER_UPDATE_V3).
4. Geometry alignment is the dominant speed lever (P4G 1.40x E2E, BF16).
5. NVFP4 QK has a real, Holm-significant quality cost — not amplified by
   sparsity (factorial interaction).
6. **DQ-VSA recovers it** (new main contribution): teacher-preserving
   quantization-aware velocity distillation, <=500 steps, serving operator
   unchanged. Claim wording (binding):
   "Training-free native SparseFP4 exhibits a measurable quality loss, but
   a short teacher-preserving quantization-aware distillation stage
   recovers most of the loss while leaving sparsity, geometry, and native
   NVFP4 inference unchanged."
7. Ablations that make it a paper, not a recipe: (a) task-loss QAT fails
   by teacher-drift/motion-collapse (QAD's prediction, shown for video
   DiTs); (b) Attn-QAT-consistent backward buys nothing over naive STE in
   the QK-only/BF16-PV regime; (c) dev-gate vs paper-scale ranking reversal
   as a methodological caution.

## Main figures/tables (V4 additions)

- T4: recovery table — candidates x {recovery fraction, residual-vs-P4G CI,
  regressions} (`tables/dqvsa_recovery_bootstrap.md`).
- F3: trajectory figure — imaging/dynamic recovery vs training step
  (100/250/500 gates) for T1/T2/T3, showing T1 collapse and T2/T3 paths.
- T5: serving/performance invariance (`tables/dqvsa_final_performance.md`).

## Wording guardrails (V4)

- All V3 guardrails stand (allocator-unified perf numbers only; 1.40x is
  BF16 P4G; guarded first-ness; no universal FP4-collapse; interaction
  wording; P4G-vs-P2 trade-off enumeration).
- Recovery: "substantial recovery with a small residual gap" — never
  "full parity" or "lossless". T3-c500 residuals: imaging -0.045
  [-0.058, -0.032], dynamic -0.111 [-0.194, -0.028]. T2-c250: imaging
  residual Holm-n.s. (report the CI, not "no difference"), dynamic Δ=0.000.
- Keep T1/T2/T3 definitions explicit in every table; never collapse T2/T3
  into "velocity distillation".
- The backward ablation is a negative result about OUR setting (QK-only,
  BF16 PV); do not generalize to PV-quantized or backward-native regimes.
- Pixel metrics are descriptive; winner selection used VBench dimensions
  and pre-declared criteria only.
- Dev-gate numbers never appear as evidence — only as the triage
  methodology caution.
- DQ-VSA claims cite the native serving receipts
  (`DQVSA_NATIVE_SERVING_PROOF.md`); training-time fake quantization is
  irrelevant to the claim.

## Ready-to-write paper skeleton

1. Intro: composition question + the three results (bottleneck-composition,
   geometry, recovery).
2. Native SparseFP4 implementation + proof (V2/V3 material).
3. Operator-level composition study (exact-geometry 2x2).
4. Systems study: kernel/E2E under unified allocator; root-cause waterfall.
5. Paper-scale quality: four contrasts + factorial interaction.
6. DQ-VSA: method, T-matrix, paper-scale recovery, ablations.
7. Related work (from `SOTA_RECOVERY_LIT_REVIEW.md`, primary sources).
8. Limitations (REPORT_V4 §5) + one optional Discussion paragraph on the
   supplementary W4A4 exploration (gated off; appendix material only —
   wording in `supplementary/w4a4_gate/W4A4_EXPLORATORY_STUDY.md`).
