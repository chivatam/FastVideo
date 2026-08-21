# Phase 2 Design Note — Mapping Local KV Reuse onto PR #1719's sm_100a Kernel

Source read: `fastvideo-kernel/csrc/attention/block_sparse_kernel_sm100a.cuh`
(blk64 config: `BLOCK=64`, `M_TILE=64`, `M_TILES_PER_CTA=2`, `K_TILE=256`
= 4 KV blocks per pipeline iteration, `NUM_KV_STAGES=4` ring stages of 32 KB,
warps: 8 softmax (4 per m-tile), 4 correction, 1 MMA, 1 epilogue, 1 load,
1 scheduler).

## How the current kernel executes a Q pair

- `decode_workitem()` gives the CTA `mtile0 = 2p`, `mtile1 = 2p+1`, each
  tile's own `q2k_num`, and one **shared trip count**
  `num_kv_blocks = max(cnt0, cnt1)` (floored at 1). Every warp role derives
  `num_k_tiles = ceil(num_kv_blocks / 4)` identically — the pipeline already
  runs BOTH tiles for the same number of iterations and masks the shorter
  tile's padding iterations (clamped `q2k_idx` reads in the load warp,
  threshold-0 block masking in the softmax warps).
- Load warp, per iteration: `load_v(0,k); load_k(0,k+1); load_v(1,k);
  load_k(1,k+1)` — **each m-tile's KV is fetched separately** into ring
  slots; a slot is filled by TMA for exactly one m-tile and consumed by
  exactly one `bmm1`/`bmm2` call, then recycled via `empty_bar[slot]`
  (single-arrival mbarrier, committed by the MMA warp's `tcgen05_commit`).
- MMA warp interleaves per iteration: `bmm2(0) bmm1(0) bmm2(1) bmm1(1)`.
  TMEM has fully independent per-m-tile S regions (`i*S_COLS`) and O
  accumulators (`2*S_COLS + i*O_COLS`).
- Softmax warp groups (one per m-tile) keep independent online-softmax state
  (`m_run`, `l_run`) and already mask whole 64-token KV blocks per m-tile via
  per-block thresholds read from `variable_block_sizes[q2k_idx[row, j]]`,
  with positions `>= cnt` forced to threshold 0 (block → -inf).
- Correction warps rescale each m-tile's O by `alpha` each iteration; the
  `alpha == 1.0` fast path skips the TMEM round trip. Epilogue is per m-tile.

Two structural facts matter for reuse:

1. **Pair-max padding already exists.** Unequal per-tile counts are the
   normal case today; nothing assumes `cnt0 == cnt1`.
2. **Ring-slot recycling is single-consumer.** Sharing one K/V slot between
   both m-tiles' MMAs requires committing `empty_bar[slot]` only after the
   *second* consumer's MMA — a one-line change in the MMA warp (issue both
   m-tiles' MMAs from the slot, single commit after the last), plus halving
   the load warp's fetch sequence.

## Strategy A — shared/private phased execution

Executable form that preserves barrier symmetry (this is the key design
finding): run

    phase 1 (shared):   ceil(|S0∩S1| / 4) iterations — ONE K/V fetch per
                        iteration, consumed by both m-tiles' bmm1/bmm2.
    phase 2 (private):  ceil(max(|P0|,|P1|) / 4) iterations — exactly
                        today's loop shape (per-m-tile fetch + compute),
                        with lists = the private lists; the shorter side
                        runs masked padding iterations as today.

Per-m-tile pipeline iterations: `ceil(shared/4) + ceil(max(p0,p1)/4)` —
measured on the real pairs: **median 37 vs today's 36 (+2.1% total)**. KV
tiles fetched: **median 47 vs 72 (-32.6%)**.

| Kernel element | Change |
|---|---|
| `WorkItem` | + `n_shared`, per-tile private counts; trip count = shared + max(private) phases |
| `q2k_idx` consumption | new per-pair metadata rows: shared list + two private lists (same padded int32 layout) |
| Load warp | phase-dependent: 1 fetch/iter in shared phase, 2/iter in private phase |
| KV ring | shared-phase slots have 2 consumers → `empty_bar` commit after 2nd MMA (or 2-arrival barrier) |
| bmm1/bmm2 | shared phase: both m-tiles read the same slot (different TMEM S/O); private phase: unchanged |
| Softmax warps | unchanged math; per-block thresholds now read from the pair metadata rows (shared row for phase 1, own private row for phase 2) |
| Correction warps | unchanged (same trip count both m-tiles) |
| Barriers | per-m-tile mbarrier *counts* unchanged; only shared-phase ring commit timing changes |
| SMEM | unchanged (ring size/stages unchanged) |
| TMEM | unchanged (independent S/O per m-tile already) |
| Online softmax | independent per m-tile, unchanged; block order changes (shared first) → fp reassociation only, measured ≤ 5e-5 bf16 max-abs vs baseline |
| Epilogue / LSE | unchanged |

Answers to the design questions: (1) yes — phase 2 *is* the existing loop;
(2) yes, with the private phases fused into ONE padded phase (pair-max
mechanism reused), no barrier asymmetry arises; (3) the pipeline requires a
common iteration count and this schedule keeps it; (4) no deadlock: every
mbarrier keeps one producer/consumer per iteration, only the shared-phase
`empty_bar` needs its commit moved after the second MMA; (5) yes, that is
the entire point of phase 1; (6) yes, TMEM S/O regions are already
independent; (7) yes; (8) yes — correction sees the same per-iteration
`alpha` protocol, phase lengths are equal for both m-tiles by construction.

Classification: **MODERATE** (one new load schedule, dual-consumer ring
commit, metadata format; everything else preserved).

## Strategy B1 — union + conditional MMA

Membership varies per 64-token block *inside* a 256-token K-tile, but
`bmm1` issues MMA atoms across the full K-tile from one descriptor stream.
Conditional per-block execution means predicating MMA atom groups and
reshaping the P/V consumption per m-tile per iteration — irregular tcgen05
issue, phase-dependent `full_bar_spo` arrival patterns, and scheduling
complexity in the softmax warps for skipped blocks. All to save FLOPs that
Strategy A already saves with a regular schedule. Classification: **HARD** —
dominated by A on both axes; not recommended.

## Strategy B2 — union + dense grouped MMA

Both m-tiles compute every union block; non-member blocks are masked to
-inf. The masking mechanism ALREADY EXISTS: the softmax warps mask whole
blocks via per-block thresholds — feeding per-pair, per-m-tile threshold
rows (`member ? vbs[blk] : 0`) makes non-membership indistinguishable from
padding. Bit-exact by construction (measured: 0 ULP vs baseline in both
fp32 and bf16 references, because each Q still consumes its member blocks in
the same ascending order and masked blocks contribute exactly zero).

| Kernel element | Change |
|---|---|
| `WorkItem` | both m-tiles share one count: `|union|` |
| `q2k_idx` | one union row per pair (both m-tiles read it) |
| Load warp | fetch the union list once per iteration (drop the second fetch) |
| KV ring | every slot has 2 consumers (same commit change as A's shared phase, applied uniformly) |
| bmm1/bmm2 | both m-tiles read every slot — perfectly symmetric |
| Softmax warps | threshold source becomes per-pair per-m-tile rows; math unchanged |
| Correction / barriers / TMEM / epilogue | unchanged |

Cost: per-m-tile iterations `ceil(|U|/4)` — **median 47 vs 36 (+32.7%)**:
+32% softmax-warp work, +32% correction work, +32% MMA issue, for the same
-33.7% KV fetch reduction that A achieves. The softmax warps are a known
throttle in this design (`SOFTMAX_THROTTLE` exists because of it), so the
inflation lands on the most contended resource. The `alpha==1` skip and
threshold-0 masking make masked iterations cheaper than real ones, but the
TMEM S load/P store per iteration is not skippable.

Classification: **EASY–MODERATE** (smallest structural diff of the three).

## Metadata

Both A and B2 need per-pair rows in the existing padded `q2k_idx` layout
plus (B2) per-m-tile threshold rows or (A) three lists + counts. Measured
construction cost at full 720p layer scale (8640 pairs, K=144, generic
PyTorch composites, B200): **0.95 ms (union+membership) / 1.17 ms
(shared/private)** — against a measured **5.52 ms sm_100a kernel runtime**
per 720p layer call (12 heads) and the **1.10 ms `map_to_index` pass that
already runs per call today**. As a standalone pass the overhead is ~17-21%
of kernel time; the production path fuses it into (or replaces) the
existing `map_to_index` scan — it already visits every mask row, and the
pair variants OR/AND two rows instead — so the expected marginal cost is
near zero. Sortedness makes this trivial: `q2k_idx` rows are ascending by
construction (`map_to_index` scans the bool map in increasing KV order), so
intersection/union is a linear merge — no sort, no hash table (verified on
all 270 captured shards).

Metadata memory at 720p, per (layer, head-batch) call: shared/private
15.0 MB, union+membership 12.5 MB (int32 padded; int16 halves it) — vs
q2k_idx today: 8640 rows x 1440 x int32 ≈ 50 MB allocated (`map_to_index`
allocates [B,H,Nq,Nk]). Negligible.

## Decision matrix

| Metric | A shared/private | B1 union conditional | B2 union dense |
|---|---|---|---|
| Exact sparse semantics | yes (reassoc ≤5e-5 bf16) | yes (bit-exact) | yes (bit-exact) |
| KV-load reduction (ring granularity, real pairs) | **-32.6%** | -33.7% | -33.7% |
| Extra MMA FLOPs | 0% | 0% | +31.7% |
| Extra pipeline iterations (softmax/correction) | **+2.1%** | irregular | +32.7% |
| Metadata bytes (720p call) | 15.0 MB | 12.5 MB | 12.5 MB |
| Metadata construction (prototype / production) | 1.17 ms / fuse into map_to_index | 0.95 ms / same | 0.95 ms / same |
| Barrier complexity | med (dual-consumer commit in shared phase only) | high | low-med (uniform dual-consumer commit) |
| PR1719 compatibility | good — phase 2 IS the current loop | poor | best — reuses pair-max masking as-is |
| Expected implementation | MODERATE | HARD | EASY-MODERATE |
| Likely B200 friendliness | best (flat softmax work, -33% HBM) | poor | good if HBM-bound, hurt if softmax-throttled |
| Recommendation | **GO-A** | reject | fallback / stepping stone |

## Recommendation: GO-A, with B2 as the explicitly cheaper-to-land fallback

A delivers the same ~33% KV-traffic cut as B2 while keeping per-Q pipeline
iterations flat (+2.1% vs +32.7%). B2's simplicity is real (it is nearly a
metadata-only change), and because both share the identical dual-consumer
ring commit, **B2 is a natural milestone on the way to A**: land the union
walk with threshold masking first (validates the shared-slot ring), then
split the walk into shared + padded-private phases to reclaim the +32%
iteration inflation. If profiling shows the kernel fully HBM-bound with
idle softmax warps, stopping at B2 is defensible; the phase-1/2 data says
don't bet on it.
