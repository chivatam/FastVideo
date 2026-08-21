# Phase 3 Design & Findings — Dual-Consumer KV Ring in the sm_100a VSA Kernel

## Barrier lifecycle change (milestone 3A)

Baseline (per K-tile, per m-tile; ring slot = 32 KB, 4 stages):

    load warp:  wait empty_bar[s] -> TMA fetch (m-tile i's blocks) -> full_bar[s] (tx)
    MMA warp:   wait full_bar[s]  -> bmm1/bmm2 atoms for m-tile i
                -> tcgen05_commit(empty_bar[s])          # slot recycled after ONE consumer
    sequence:   V0(k) K0(k+1) V1(k) K1(k+1)              # each m-tile fetched separately

PAIR_SHARED (K-tile j < pair_shared_tiles):

    load warp:  wait empty_bar[s] -> ONE TMA fetch (union/shared blocks) -> full_bar[s]
    MMA warp:   wait full_bar[s]  -> bmm1/bmm2 atoms for m-tile 0 (S0/O0 TMEM)
                                  -> bmm1/bmm2 atoms for m-tile 1 (S1/O1 TMEM)
                -> tcgen05_commit(empty_bar[s])          # committed AFTER the 2nd consumer
    sequence:   Vsh(k) Ksh(k+1)                          # one fetch per K-tile

K-tiles j >= pair_shared_tiles keep the exact baseline lifecycle (this is the
whole private phase of Strategy A; with pair_shared_tiles = ceil(|union|/4)
the entire walk is dual-consumer = Strategy B2). Per-m-tile masking moved
from `variable_block_sizes[q2k_idx[...]]` to a per-(row, position)
`block_thresholds` table (0 = whole block -> -inf), so union non-membership,
shared-phase padding, and vbs raggedness use ONE mechanism. Online-softmax
state, correction, epilogue, LSE: untouched; all mbarrier arrival counts
unchanged; only the empty_bar commit position moved.

## Correctness (gates all pass; 53 tests)

vs the baseline kernel on identical inputs: max |out| diff ~1e-3 bf16,
rel-L2 3.4e-3, cosine 0.999994, LSE <= 7e-6 — NOT bit-exact, and provably
cannot be: regrouping blocks into different 256-token K-tiles changes the
online-softmax segmentation (per-tile row-max/rescale points), a pure fp
reassociation. Overlap 0/one-shared/partial/full, ragged vbs, seq lengths,
K=144 at 1440 blocks, real Phase-1 rows: all within 2.5e-2 gate.

## Performance (milestones 3B/3C) — B200, CUDA events, real 720p masks, 12 heads

| mode | 720p | 480p | vs baseline |
|---|---|---|---|
| PR#1719 baseline | 5.55 ms | 0.98 ms | 1.00x |
| B2 union-dense | 8.50 ms | 2.23 ms | 0.65x / 0.44x |
| A shared/private | 6.45 ms | 1.35 ms | 0.86x / 0.72x |

Controlled overlap sweep (720p): A = 0.83–0.94x at EVERY overlap level,
including 1.0 where dual-consumer halves all KV fetches. 40-head check:
same pattern (0.88–0.90x).

## Why it loses (milestone 3D, ncu on real 720p masks)

| metric | baseline | A shared/private | B2 |
|---|---|---|---|
| kernel time | 6.93 ms | 9.25 ms | 13.46 ms |
| L2 read sectors (tex) | 2.52 G | **1.77 G (-30%)** | — |
| **L2 hit rate** | **97.3%** | 95.6% | — |
| DRAM bytes read | 0.93 GB | 1.12 GB | 0.95 GB |
| SM throughput | 69% | 51% | 47% |
| warps active | 25% (1 CTA/SM, smem-limited) | 25% | 25% |
| stall: long scoreboard | 62% | 60% | 64% |
| stall: barrier / membar | ~0.5% / 0% | ~0.5% / 0% | ~0.2% / 0% |

Three findings:

1. **The dual-consumer ring works exactly as designed**: L2 read sectors
   drop 30% — the predicted union-level KV fetch reduction, delivered.
2. **But the redundant fetches were already nearly free**: baseline L2 hit
   rate is 97.3%; of ~80 GB of logical KV requests only 0.93 GB reaches
   DRAM. B200's L2 (plus TMA multicast-friendly access pattern across
   adjacent CTAs) already exploits the very support-coherence we measured
   in Phase 1. The theoretical "1.56x KV reuse" was real but was being
   captured one level down the hierarchy.
3. The kernel is **latency-bound, not bandwidth-bound**: 62% long-scoreboard
   stalls at 1 CTA/SM. Removing L2 traffic doesn't shorten the critical
   path; the pair path *adds* to it: per-position threshold-table reads
   (bigger, colder than the tiny vbs array), wider padded metadata rows,
   +2% (A) / +33% (B2) pipeline iterations, and shared-slot consumption
   serializing the two m-tiles' MMA issue. Isolation: pair kernel with
   shared_tiles=0 (baseline schedule, pair metadata) = 6.07 ms → ~0.45 ms
   is fixed pair-path overhead, ~0.19 ms is the dual-consumer phase itself.

## Milestone 3E (targeted tuning): intentionally NOT performed

Tuning rules allow it only where profiling implicates the new structure.
Profiling instead shows the *mechanism's premise* is absent on this
hardware: there is no DRAM traffic to recover and no L2-bandwidth wall.
No amount of ring/barrier tuning changes a 97%-hit baseline. Recorded as
a deliberate decision, not an omission.

## Decision: NO-GO for local cross-query KV reuse on B200 at VSA shapes

The clean ablation (untouched PR#1719 vs B2 vs A, same masks, same inputs)
attributes the loss entirely to the reuse transformation. The Phase-1/2
overlap statistics remain valid and the dual-consumer primitive is proven
correct — but on B200 the L2 already monetizes local support coherence.
The idea could still pay where that premise flips: KV working sets far
exceeding L2 (much longer sequences / more heads per device), L2-thrashing
multi-kernel co-residency, or hardware with weaker L2:DRAM ratios.
