# PAPER_PLAN — geometry-alignment spin-out paper

Per ARIS `paper-plan`: story, one-sentence contribution, claims-evidence
matrix, section outline, figure plan — locked before drafting. Archetype:
**kernel/inference systems paper** (hardware bottleneck -> profiling evidence
-> insight -> design -> microbenchmark -> E2E -> quality -> limitations).

## One-sentence contribution

Matching a dynamic sparse-attention selector's tile geometry to the attention
kernel's native sparse granularity — a one-constant change — converts a
deployed 90%-sparse video DiT from 1.13x to 1.40x end-to-end speedup on
Blackwell at comparable quality, and is worth more wall-clock time than the
choice of attention arithmetic precision.

## Story (Problem -> Gap -> Insight -> Method -> Evidence -> Implication)

1. **Problem:** video DiT attention is quadratic; block-sparse attention
   promises ~10x at 90% sparsity.
2. **Gap:** the deployed system realizes only 1.13x E2E at 720p and loses
   time at 480p (C1) — theoretical sparsity is not being converted into time.
3. **Insight:** the loss is at the selector/kernel boundary: sparsity is
   *decided* at 64-token tiles but the fastest kernel can only *skip* at
   256x128 granularity; every route across that boundary loses (kernel
   downgrade C2, or ~2.4x retention inflation C3).
4. **Method:** re-align the selector geometry itself — (4,4,4)->(4,8,8) —
   so decision granularity equals skip granularity (C4). No new kernel, no
   training, unchanged sparsity budget.
5. **Evidence:** kernel converts retention almost ideally (C5); E2E 1.40x at
   720p / repairs the 480p slowdown (C6); precision-independent (C7);
   quality comparable with enumerated trade-offs (C8-C10).
6. **Implication:** selector geometry is a first-class systems parameter;
   the tile triple deserves the same hardware-awareness as kernel tiling.

## Claims-evidence matrix

| Claim | Section | Figure/Table |
|---|---|---|
| C1 deployed VSA underdelivers | §1, §5.2 | Fig 3, §5.2 tables |
| C2 kernel downgrade | §1, §2, §6.1 | — (code audit receipt) |
| C3 coarsening inflation | §1, §3.2, §6.1 | Fig 1 |
| C4 exact 1:1 mapping | §3.2 | Fig 1 |
| C5 kernel scaling | §5.1 | Fig 2, §5.1 table |
| C6 1.40x E2E | abstract, §1, §5.2 | Fig 3, §5.2 tables |
| C7 precision independence | §6.2 | §5.2 tables (P1/P4 rows) |
| C8 quality trade-offs | §5.3 | Fig 4, §5.3 table |
| C9 pixel honesty note | §5.3 | §5.3 |
| C10 operator error | §5.4 | §5.4 |
| C11 VSA tile ablation tension | §2, §7 | — |

## Section outline (with per-section thesis)

- **Abstract** — Farquhar 5-part: what / why hard / how / evidence /
  strongest number (1.13x->1.40x).
- **§1 Introduction** — thesis: the sparsity budget is lost at the
  selector/kernel geometry boundary, and re-aligning the selector recovers
  it. Contributions bulleted, each mapped to a claim ID.
- **§2 Background & Related Work** — organized by *methodological category*
  (trainable dynamic selection; static/sliding patterns; kernel-side sparse
  granularity), each paragraph ending with positioning; not paper-by-paper.
- **§3 Method** — preserved algorithm family; exact 1:1 mapping; boundary
  handling; what is explicitly unchanged.
- **§4 Experimental setup** — arms, protocol, statistics.
- **§5 Results** — setup -> mechanistic (kernel) -> main (E2E) -> cost
  (quality) -> operator matrix; each experiment opens with the claim it
  tests ("what to observe" sentences).
- **§6 Analysis** — decomposition of the win; precision independence;
  resolution scaling (H1 marked as untested prediction).
- **§7 Limitations** — honest; includes B=64-training tension (C11/H2).
- **§8 Conclusion** — restated contribution (rephrased), general lesson.
- **References; Appendix A (reproducibility); Appendix B (claim boundaries).**

## Figure plan

Already executed (see `figures/FIGURE_CONTRACT.md`): Fig 1 hero mechanism,
Fig 2 kernel scaling, Fig 3 main E2E result, Fig 4 quality forest.

## Writing rules bound for this draft

- Claim-ledger gate: every quantitative sentence maps to a ledger row.
- One message per paragraph, topic sentence first; reverse-outline test
  after drafting.
- Experiments state the claim they test before the numbers.
- No AI watch-words (delve/pivotal/landscape/underscore/notably/...); no
  significance inflation; vary sentence openings.
- Hedging removed except where the ledger marks genuine uncertainty (H1-H3).
- Consistent terminology: "selector tile", "kernel sparse granularity",
  "retention", "deployed VSA (P2)", "aligned configuration (P4G)" — never
  renamed mid-paper (Banana Rule).
