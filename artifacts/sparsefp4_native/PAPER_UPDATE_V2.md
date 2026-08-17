# PAPER_UPDATE_V2 — framing after the re-audit

> **Superseded by `PAPER_UPDATE_V3.md`.** T2 must cite
> `tables/c8_performance_v2.md` (unified allocator config), not
> `tables/c8_performance.md` (contains the pre-allocator-fix P4 720p
> 250.9 s row; historical/root-cause evidence only).

Supersedes PAPER_UPDATE.md. Direction B per `RESULTS_DECISION_V2.md`.

## Title direction

"Why Sparsity and FP4 Do Not Automatically Compound in Video Attention"
(alt: "Geometry Beats Precision: Composing Block-Sparse and NVFP4 Attention
for Video Diffusion on Blackwell")

## Narrative arc

1. **Build the real thing first.** First working native block-sparse NVFP4
   attention on B200 (repairing a never-run fwd+bwd path in the public FP4
   FlashAttention fork — 7 latent bugs; systems-archaeology sidebar).
2. **Numerics are not the problem.** Exact-geometry 2x2 (both resolutions):
   conditional quant effect == dense quant effect (~0.09 rel-L2). The
   operator composes cleanly.
3. **The regime change is the problem.** FP4's kernel increment decays
   1.26x -> 1.04x as retention drops to 10%; per-element validity
   predicates cost the softmax-bound FP4 pipeline +42% while free on BF16;
   200 MB/call quantization transients interact catastrophically with the
   caching allocator (2.25x E2E swing) until mitigated. Net: FP4 loses to
   its BF16 twin E2E at the production point (111.7 vs 106.0 s).
4. **Quality is not free either.** Paper-scale paired evaluation (326
   VBench prompts): NVFP4 costs imaging_quality -0.101 and dynamic_degree
   -0.25 on top of the same sparse baseline — and the same signature in
   dense NVFP4, i.e. it is the format's cost, not the composition's.
5. **What actually wins: geometry.** Aligning the VSA selector tile to the
   FA4 sparse granularity (256-token tiles, 1:1 mask mapping, zero
   inflation) gives 1.40x E2E vs dense at 720p in BF16 at deployed-baseline
   quality — precision-independent, and the practical recommendation.
6. **Path forward** (future work, partially evidenced): eliminate the
   validity predicate via tile-aligned padding; preallocated quantize
   workspaces; block-scaled PV (needs newer fork branches); QAT recovery of
   the NVFP4 quality cost (feasibility shown; needs motion-diverse corpus).

## Main figures/tables

- T1: exact-10% 2x2 operator matrix, both resolutions
  (`tables/c5_matrix_vsa256_exact10.md`).
- F1: FP4-vs-BF16 sparse-kernel latency vs retained fraction (the decay
  curve), 480p+720p geometries.
- T2: E2E 5-arm x 2-resolution performance (`tables/c8_performance.md`).
- T3: paper-scale VBench + paired P4-P4G bootstrap CIs
  (`tables/paper_scale_quality.md`).
- F2: P4 root-cause waterfall (allocator / predicate / quantize)
  (`P4_PERF_ROOT_CAUSE.md`).

## Wording guardrails (from the re-audit, binding)

- 1.40x belongs to BF16 P4G, never to FP4.
- 9.3x is sparsity; FP4's increment at 10% is ~1.04x kernel-level.
- No "quality parity" claim for NVFP4; report the paired CIs.
- No QuantSparse-contradiction claim; state the narrow conditional result.
- QAT stays out of the abstract; supplementary feasibility only.
- Routing/Jaccard/scorer-precision material: appendix mechanism context
  only (used solely to justify the BF16 selector design choice).
