# VSA Phase 3 — Local KV-Stationary Execution in PR #1719's 2-Q CTA (B200)

**Question:** can local cross-query KV reuse (dual-consumer KV ring) beat the
merged PR #1719 sm_100a kernel on real VSA masks?

**Answer: NO-GO on B200 at VSA shapes — the reuse the masks expose is real
(Phase 1/2) and the kernel implementation of it is correct and does eliminate
30% of L2 KV reads, but the baseline was already serving 97.3% of KV requests
from L2, so there is no memory traffic left to recover; the pair machinery
costs 6–14% instead.** Full analysis: `DESIGN_PHASE3.md`.

## What was built (all milestones, kernel in production source tree)

- `PAIR_SHARED` instantiation of the blk64 sm_100a kernel: leading
  `pair_shared_tiles` K-tiles fetched once, consumed by both m-tiles,
  `empty_bar[slot]` committed after the second consumer; masking via a
  per-(row, position) threshold table reusing the existing block-mask path.
  Baseline instantiation byte-identical, all 29 baseline tests unregressed.
- `pair_metadata.build_pair_metadata` (union = B2, shared-private = A),
  ascending order preserved, no runtime sort.
- New op `block_sparse_sm100a_pair_fwd` + wrapper; 24 new correctness tests.
- `tests/bench_block_sparse_local_reuse_sm100a.py` (modes / resolutions /
  real masks / controlled overlap), `profile_one.py` for ncu.

## Results (B200, CUDA events, real Phase-1 720p masks, 12 heads)

| | baseline | B2 | A |
|---|---|---|---|
| 720p | 5.55 ms | 8.50 ms (0.65x) | 6.45 ms (0.86x) |
| 480p | 0.98 ms | 2.23 ms (0.44x) | 1.35 ms (0.72x) |

Overlap sweep (A, 720p): 0.83–0.94x at every overlap **including 1.0**.
Correctness vs baseline kernel: max-abs ~1e-3 bf16 (reassociation from
K-tile regrouping), cosine 0.999994, LSE ≤ 7e-6.

## The decisive profile (ncu, real masks)

Baseline: **L2 hit 97.3%**, 0.93 GB DRAM vs ~80 GB logical KV requests;
62% long-scoreboard stalls at 1 CTA/SM. A: L2 read sectors **−30%**
(mechanism works), DRAM unchanged-to-worse, runtime worse. The theoretical
1.56x KV reuse was already captured by the cache hierarchy; the kernel is
latency-bound, not bandwidth-bound. 3E tuning deliberately skipped — the
premise, not the implementation, is absent on this hardware.

Reproduce: `results/*.csv`, commands in file headers; baseline env in
`results/baseline_env.json`.
