# Phase 4 — Why PR #1719's sm_100a Kernel Stalls: Attribution & Decision

Diagnostic only; production kernel untouched. Method: `PROFILE_METHOD.md`.

## Baseline confirmed (B200, 1845 MHz SM, CUDA 13.0, torch 2.12+cu130, `10.0a`, real 720p masks, 12 heads)

5.55 ms · L2 hit 97.3% · DRAM read 0.93 GB · long-scoreboard 62.0% · 1 CTA/SM.

## Part 1 — occupancy: THREE simultaneous hard limits

| resource | usage | CTA limit |
|---|---|---|
| dynamic smem | 200.3 KB (+1 KB driver) of 233.5 | 1 |
| registers | 128/thread × 512 = **entire 64K regfile** | 1 |
| **TMEM** | **512 of 512 columns** (2×S128 + 2×O128) | **1 (hard)** |

**Can smem reduction alone yield 2 CTA/SM? NO.** Registers and TMEM
independently cap at 1; TMEM is architectural (the kernel's per-pair
S/O layout consumes the full 512-column TMEM). 2 CTA/SM would require
halving TMEM *and* registers *and* smem — a different kernel.

## Parts 2–4 — where "long scoreboard" actually lives

62% "long scoreboard" is **not memory latency**. ~81% of LSB samples sit on
four mbarrier spin-wait back-edges (BRA/NANOSLEEP of `mbarrier_wait_parity`
phase-check loops, which ncu classifies under long-SB):

| rank | % of LSB | % of all | who waits, on what |
|---|---|---|---|
| 1 | 37.4 | 23.3 | **softmax warps waiting `full_bar_spo`** — for the MMA warp's QK scores (tcgen05→TMEM) |
| 2 | 17.3 | 11.3 | correction warps waiting `full_bar_alpha` (softmax's per-iter alpha) |
| 3 | 17.3 | 11.1 | correction warps waiting `full_bar_l`/`o_acc` (epilogue handoff) |
| 4 | 14.0 | 9.1 | suspend/NANOSLEEP waits — **load warp waiting for EMPTY ring slots**, epilogue, scheduler |
| 5 | 5.0 | 5.5 | CLC scheduler idle |
| 6 | 1.1 | 0.7 | MMA warp waiting `empty_bar_spo` (softmax) |
| — | **≤2** | **≤1.5** | **sparse metadata loads (`q2k_idx`, `variable_block_sizes`)** |

Critical path (ASCII):

    MMA warp: bmm1 QK (tcgen05, S_COLS=128/tile) ────────┐ paces everything
       softmax w0-7: [SPIN 23%] ld TMEM S -> mask -> max -> ex2 -> P store
          correction w8-11: [SPIN 22%] alpha -> O rescale (skipped if alpha==1)
             MMA: bmm2 PV (waits empty_spo only 0.7%)
    load warp: TMA K/V [SLEEPS on empty_bar -> pipeline has LOAD SLACK]

## Part 5–6 — metadata verdict

Source facts: load warp caches `q2k_idx` in a 32-wide register window
(shfl-broadcast, refilled every 32 blocks — effective prefetch distance
0 iterations at refill points); softmax warps *independently* re-read
`q2k_idx` and do the dependent `variable_block_sizes[q2k_idx[i]]` lookup
(also 32-wide cached). Both tables are L1/L2 resident. And it **does not
matter**: contiguous synthetic indices (Experiment C) give 5.38 ms vs 5.51
(real) / 5.57 (random) — **~2-3%**. The dependent-load chain is real but
off the critical path. Experiments A/B unnecessary given C's null result
(and Phase-3's `block_thresholds` variant already showed removing the
dependent lookup while widening metadata is net-negative).

## Parts 7–10 — pipeline classification

**CASE B: consumer-paced.** The load warp sleeps on `empty_bar` (slots not
recycled fast enough) while consumers almost never wait on `full_bar` KV
(<1%) — NUM_KV_STAGES=4 is sufficient; TMA has slack; deeper TMA pipelines
buy nothing. Inside the consumer side, the pacing chain is
QK(tcgen05) → softmax(TMEM round-trip + ex2 chain) → alpha→correction →
P→bmm2 — enforced by `full_bar_spo`/`empty_bar_spo`/`full_bar_alpha` each
iteration. SPLIT_P and DEFER_ROWSUM already soften it; the softmax warps
still spend 23% of all samples waiting for new scores while correction
spends 22% waiting for softmax: 12 of 16 warps idle in lockstep behind one
tcgen05 producer whose S tile must round-trip through TMEM every 256
tokens. CLC/scheduler: ~6% idle-by-design, not implicated.

## Part 11 — scaling

Latency is linear in K (720p: 2.89/5.53/13.6/27.4 ms at 5/10/25/50%) and
in S (480p 0.95 → 720p 5.53 → 1.6x-blocks synthetic 26.6 ms at fixed
density) — steady-state per-iteration cost, no fixed-overhead knee, no
bandwidth knee. Consistent with an iteration-serialized consumer pipeline.

## Parts 12–13 — ceilings & ranking (evidence-based)

See `results/counterfactual_ceiling.csv`. Ranking:

| direction | evidence | upside | complexity | novelty | confidence |
|---|---|---|---|---|---|
| 5. QK/softmax/PV pipeline redesign (deeper S double-buffering in TMEM, decouple softmax from per-tile handshake) | 23% softmax spin + 22% corr spin, MMA rarely waits | HIGH | HIGH | MEDIUM | MEDIUM |
| 6. correction/epilogue restructuring (remove per-iter alpha handshake; fold rescale) | 22% corr spin | MEDIUM | MEDIUM | LOW | MEDIUM |
| 9. 2-CTA MMA / cluster g=4 (amortize TMEM round-trip over more Q) | indirect | MEDIUM | HIGH | HIGH | LOW |
| 8. 2 CTA/SM via smem diet | **killed**: regs+TMEM also at 100% | — | — | — | HIGH (negative) |
| 3. deeper TMA pipeline | load warp has slack | ~0 | LOW | LOW | HIGH (negative) |
| 1/2/4. metadata prefetch / packed descriptors / decoupled metadata pipeline | Exp C: 2-3%; ≤2% of LSB | ~2% | LOW-MED | LOW | HIGH (negative) |
| 7. CLC redesign | 6% idle-by-design | ~0 | MED | LOW | HIGH (negative) |

## Decision

**Primary next direction: QK/SOFTMAX/PV PIPELINE REDESIGN** — specifically,
increasing the number of in-flight S tiles per m-tile (a second S buffer in
TMEM is impossible at 512 columns with the current 2×(S+O) layout, so this
means either S_COLS splitting, staggering the two m-tiles' phases so the
tcgen05 producer never idles behind one m-tile's softmax, or absorbing the
correction handshake into the softmax warps). The metadata/TMA/occupancy
directions are dead on evidence.
