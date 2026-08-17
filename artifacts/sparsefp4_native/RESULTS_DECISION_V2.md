# RESULTS_DECISION_V2 — after re-audit, exact-geometry rerun, and paper-scale evaluation

## Verdict: **Direction B — NEGATIVE/BOTTLENECK systems result for native SparseFP4; POSITIVE result for geometry-aligned sparse attention (BF16)**

Native sparse NVFP4 is real, numerically clean, and kernel-fast — but at the
target production shape it does **not** beat its BF16 sparse twin end-to-end
(111.7 vs 106.0 s at 720p), and at paper scale (326 prompts) NVFP4 causes a
**statistically significant quality penalty on top of the same sparse
baseline** on imaging_quality (-0.101, CI [-0.112, -0.090]) and
dynamic_degree (-0.250, CI [-0.361, -0.153]). The defensible paper is the
regime-change story, with the geometry-alignment result (P4G: 1.40x vs
dense, quality at deployed-VSA level) as the positive systems contribution.

## What each pillar now says (receipts inline)

1. **Nativeness — intact.** D0/P4 receipts unchanged (`NATIVE_PROOF.md`).
2. **Numerical composition — clean, replicated at exact geometry.**
   Canonical exact-10% VSA256 matrix (`tables/c5_matrix_vsa256_exact10.md`,
   both resolutions): conditional quant effect 0.096/0.092 rel-L2 ~= dense
   quant effect 0.095/0.102. No amplification. (Narrow claim only; no
   QuantSparse-contradiction wording.)
3. **Kernel speed — sparsity dominates; FP4's increment shrinks with sparsity.**
   Dense FP4/BF16 1.26x -> 50% 1.24x -> 25% 1.10x -> 10% 1.04x
   (with plain sparse lists FP4 still wins at 720p geometry: 3.16 vs
   3.98 ms). The 9.3x number is a *sparsity* speedup, not an FP4 speedup.
4. **E2E — FP4 loses to its BF16 twin.** After fixing the allocator
   pathology (250.9 -> 111.7 s; controls unchanged — `P4_PERF_ROOT_CAUSE.md`),
   P4 trails P4G by 5.7 s at 720p; deficit fully attributed to the vbs
   mask_mod predicate in the softmax-bound FP4 kernel (+42% kernel cost)
   plus per-call quantization.
5. **Quality at paper scale — NVFP4 is NOT free.**
   (`tables/paper_scale_quality.md`, 326 prompts, paired bootstrap)
   P4-P4G significant: imaging_quality -0.101, dynamic_degree -0.250,
   small positives on temporal_flickering (+0.009) and motion_smoothness
   (+0.016); subject/aesthetic n.s. Crucially the same signature appears in
   *dense* NVFP4 (P1 vs P0: imaging 0.506 vs 0.650, dynamic 0.458 vs 0.764)
   — the penalty is NVFP4's own, not a composition effect, consistent with
   pillar 2.
6. **Geometry alignment — the positive result.** VSA256/FA4-aligned
   selection keeps exact 10% retention, matches deployed-VSA quality
   (subject consistency 0.869 vs 0.869 at paper scale), and delivers 106.1 s
   vs dense 148.8 s (1.40x) and deployed VSA 131.7 s (1.24x) — as a
   **BF16 sparse** result.

## Defensible paper claims (exact wording constraints)

- "Native sparse NVFP4 attention on Blackwell is implementable and
  numerically well-behaved: conditioned on the same mask, NVFP4 on retained
  QK tiles perturbs outputs by the same magnitude as dense NVFP4."
- "Sparsity and FP4 acceleration are not multiplicative: FP4's kernel
  advantage falls from 1.26x (dense) to ~1.04x at 90% sparsity as QK MMA
  ceases to dominate; integration overheads (validity predicates in the
  softmax-bound FP4 pipeline, quantization, allocator pressure) can invert
  the ordering end-to-end."
- "Aligning the sparse selector's tile geometry to the kernel's sparse
  granularity is worth more than reducing arithmetic precision: it yields
  1.40x end-to-end over dense at 720p in BF16, at deployed-baseline quality."
- "At paper scale, NVFP4 QK carries a measurable no-reference quality cost
  (imaging quality, motion dynamism) that appears equally in dense and
  sparse settings; QAT-style recovery is the indicated remedy (supplementary
  feasibility evidence only)."

## Explicitly retired claims

1.40x as an FP4 result; 9.3x as an FP4 speedup; "no quality penalty" for
NVFP4; any QuantSparse contradiction; QAT-restores-quality as a main claim;
the coarsened-24% C0/D0 table as canonical (superseded by
`c5_matrix_vsa256_exact10.md`).
