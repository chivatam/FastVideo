# PAPER_UPDATE_V3 — framing after the paper-validation pass

> **HISTORICAL — superseded by `PAPER_UPDATE_V4.md` (canonical, paper
> frozen at V4).** Note: V3's "QAT/DQ-VSA recovery: future work /
> supplementary" guardrail is OBSOLETE — DQ-VSA is an established V4 main
> result.

Supersedes PAPER_UPDATE_V2.md. Incorporates the unified-allocator
performance rerun, the four-contrast Holm-corrected statistics, the
primary-source literature audit, and the DQ-VSA recovery design.

## Title direction

"Why Sparsity and FP4 Do Not Automatically Compound in Video Attention"
(alt: "Geometry Beats Precision: Composing Block-Sparse and NVFP4
Attention for Video Diffusion on Blackwell")

## Narrative arc

1. **Build the real thing first.** Native block-sparse NVFP4 attention on
   B200 (7 latent bugs repaired in the public FP4 FlashAttention fork).
   Priority wording guarded: "to our knowledge, no published system
   demonstrates native block-sparse NVFP4 attention" — supported by the
   primary-source audit (`SOTA_RECOVERY_LIT_REVIEW.md` §10c: Attn-QAT is
   dense NVFP4 with sparse+QAD as announced future work; SLA2 is
   sparse+INT8/FP8; FPSAttention is sparse+FP8).
2. **Numerics compose cleanly at the operator level.** Exact-geometry 2x2,
   both resolutions: conditional quant effect == dense quant effect
   (~0.09 rel-L2).
3. **The bottleneck composition changes.** As QK MMA gets cheaper under
   sparsity, non-MMA costs (softmax, sparse validity predicates,
   scale-factor handling, quantization, launch/integration overhead,
   allocator behavior) can dominate; FP4's kernel increment decays
   1.26x -> 1.04x with retention at 480p geometry but stays 1.26x at 720p
   geometry with plain lists — the retention curve is geometry-dependent,
   not universal. E2E under one allocator config: P4 112.6 vs P4G 106.2 s.
4. **Quality: NVFP4 has a real cost; sparsity does not amplify it.**
   326-prompt paired, Holm-corrected: dense NVFP4 imaging -0.144 /
   dynamic -0.306 / aesthetic -0.055; sparse NVFP4 imaging -0.101 /
   dynamic -0.250 with small temporal-stability gains. The factorial
   interaction (difference-in-differences) is *positive* on
   imaging/aesthetic (penalty smaller under sparsity), slightly negative
   on subject consistency (-0.014), undetectable for dynamic degree at
   this sample size.
5. **What actually wins: geometry.** VSA256/FA4 alignment gives 1.40x E2E
   vs dense at 720p in BF16 — at quality comparable to deployed VSA on
   measured dimensions, with small significant trade-offs (aesthetic
   -0.030, motion smoothness -0.018, background -0.008, dynamic +0.139).
   State the trade-offs; do not say "indistinguishable" or "neutral".
6. **Path forward.** DQ-VSA: velocity distillation from the BF16 sparse
   twin with Attn-QAT-faithful fake-quant semantics under frozen geometry
   (infrastructure smoke-tested; recovery pending — future-work section,
   not a claim).

## Main figures/tables

- T1: exact-10% 2x2 operator matrix (`tables/c5_matrix_vsa256_exact10.md`).
- F1: FP4-vs-BF16 sparse-kernel latency vs retained fraction at both
  geometries (the geometry-dependent decay).
- T2: E2E 5-arm x 2-resolution, unified allocator
  (**`tables/c8_performance_v2.md`** — never cite the stale
  `c8_performance.md` numbers; that file is root-cause history).
- T3: four-contrast quality statistics
  (`tables/p1_vs_p0_quality_bootstrap.md`,
  `tables/p4_vs_p4g_quality_bootstrap.md`,
  `tables/nvfp4_sparsity_interaction.md`,
  `tables/p4g_vs_p2_quality_bootstrap.md`).
- F2: P4 root-cause waterfall (allocator / predicate / quantize)
  (`P4_PERF_ROOT_CAUSE.md`).

## Wording guardrails (V3, binding)

- "First native sparse NVFP4": only with the full qualifier string and
  "to our knowledge"; cite the audit. If in doubt, lead with the measured
  composition result instead of first-ness.
- 1.40x belongs to BF16 P4G, never to FP4. 9.3x is a sparsity speedup.
- No universal "FP4 speed collapses with sparsity": the 480p 1.26x->1.04x
  retention curve does not hold at 720p geometry (1.26x retained).
- Mechanism sentence (use verbatim): "Sparsification changes the kernel's
  bottleneck composition. As QK MMA becomes cheaper, non-MMA costs such as
  softmax, sparse validity predicates, scale-factor handling,
  quantization, launch/integration overhead, and allocator behavior can
  dominate, so arithmetic speedups do not automatically translate into
  multiplicative E2E gains."
- No NVFP4 "quality parity"; report the paired CIs.
- No QuantSparse-contradiction language: different quantization target,
  observable, and sparsifier; our result is conditional on identical masks
  at one geometry.
- Interaction: never claim "no interaction" from p>0.05. Use "no
  detectable interaction at this sample size" only where the CI includes
  zero (dynamic_degree, background); report the significant
  positive/negative interactions as such.
- P4G-vs-P2: "comparable on measured dimensions" with the enumerated
  trade-offs; "statistically indistinguishable"/"quality neutral" are
  retired.
- QAT/DQ-VSA recovery: future work / supplementary; never a main claim
  until the T-matrix produces paired evidence.
- All performance numbers from the unified-allocator receipts
  (`raw/performance/perf_v2/`); pre-fix numbers only in the root-cause
  narrative, clearly labeled historical.
