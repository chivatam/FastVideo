# Phase 2 — Numerical error decomposition and the H3 test

**Status:** complete. **Recommendation: PIVOT.**

**H3 is falsified.** Higher-precision routing does not meaningfully reduce
sparse-attention output error. The pre-registered success criterion was a **≥20%**
relative reduction in attention-output error from replacing an NVFP4 router with a
higher-precision one. The measured reduction is **0.04%–0.10%** — between 200x and
500x short of the threshold — over `n = 20,400` exactly paired cells per sparsity
level, at every sparsity tested and in every region of the network.

The reason is now measured rather than conjectured: **NVFP4 quantization only ever
flips the top-k decision at the near-degenerate boundary, where the blocks carry
almost no attention mass.** There is therefore almost no error for a better router
to recover. This was the falsifiable prediction stated in the brief, and it is
confirmed on all four of its independent implications (§4).

The result is a clean, well-powered negative for the study's central hypothesis,
and it simultaneously establishes a *positive* engineering finding: because the
NVFP4 router costs essentially nothing in accuracy, **routing can be done in the
same low precision as the compute**, which is the cheap design, not the expensive
one. The pivot is to stop pursuing precision-decoupled routing and instead pursue
the sparsification error itself, which dominates by three to four orders of
magnitude (§3).

---

## 1. What ran

| Item | Value |
|---|---|
| Run ID | `20260814-025500-8208536-p2-main` |
| Raw records | 615,380 across 10 shards (one per prompt), gzipped under `artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main/` |
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 50 denoising steps, seed 1234, `sp_size=1` |
| Attention consumed by the model | **Dense BF16 (configuration A), byte for byte** — every arm shares one denoising trajectory |
| Prompts | 10 (`p01`–`p10`) |
| Layers | 17: `0 1 2 5 6 8 10 11 13 16 20 23 24 25 27 28 29` |
| Timesteps | 5: `0 1 10 25 40` |
| CFG branches | both (`positive`, `negative`) |
| Heads | all 12 |
| Sparsity | `0.80`, `0.90`, `0.95` |
| `seq_len` | 32,760 raster tokens (constant, asserted) |
| Block geometry | `block_q=128`, `block_k=64` scored; executed on the kernel's 64-row query grid by expansion |
| **Block-score dtype** | **`fp64` everywhere** (`STATUS.md` trap 8), gated in analysis |
| Paired cells per sparsity | 20,400 = 10 prompts x 17 layers x 12 heads x 5 timesteps x 2 branches |
| Verification | `PASS`, zero failures — `artifacts/sparsefp4/tables/phase2_main/verification.json` |

Nothing was dropped for being inconvenient. The only exclusion is recorded in §9.

**Terminology, held to throughout.** "Low precision" means **NVFP4 Q/K with BF16
PV** (the FA4 kernel's `qk_mode=nvfp4`, `pv_mode=bf16`) — never "fully FP4"
attention. Every table row carries `native_or_simulated` for the compute path and
`router_native_or_simulated` for the mask path independently, because they differ.
All geometry is the raster-order 128x64 diagnostic geometry, **not** VSA's 64-token
(4,4,4) spatio-temporal cubes; block indices are not comparable across the two.

---

## 2. Configurations

Implemented per `EXPERIMENT_SPEC.md` §7.1, at matched retained fraction, with
**dense BF16 (A) as the sole numerical reference**. Two configurations were added
to the pre-registered set and `D`/`F` were split by router; all four changes are
logged in the spec's append-only amendment table and were made **before** the
first measured run.

| Config | Attention compute | Mask source | Isolates | Native / simulated |
|---|---|---|---|---|
| **A** | dense BF16 (FA4) | — | reference | native |
| **B** | dense NVFP4 Q/K + BF16 PV | — | quantization only | **native** |
| **B_sim** | dense simulated NVFP4 Q/K + BF16 PV | — | quantization only | simulated — *simulation control* |
| **C** | sparse BF16 (Triton) | BF16 scores | sparsification only | native |
| **D** | sparse BF16 | NVFP4 scores | + NVFP4 wrong-mask | native compute, native router |
| **D8** | sparse BF16 | FP8-E4M3 scores | + FP8 wrong-mask | native compute, simulated router |
| **C_rand** | sparse BF16 | BF16 scores, *N* blocks swapped at random | + equal-magnitude **random** wrong-mask | native compute, synthetic control |
| **E** | sparse simulated NVFP4 Q/K + BF16 PV | NVFP4 scores | naive combined | simulated — numerical only |
| **F8** | sparse simulated NVFP4 Q/K + BF16 PV | FP8-E4M3 scores | **H3 arm** | simulated — numerical only |
| **F16** | sparse simulated NVFP4 Q/K + BF16 PV | BF16 scores | **H3 arm** | simulated — numerical only |

`B_sim` and `C_rand` are the two additions. `B_sim` exists because `E`/`F8`/`F16`
have **no native sparse-NVFP4 kernel in this repository**, so they are necessarily
simulated; without a *dense* simulated control there is no way to separate the
simulation's own error from the effect under test. `C_rand` exists because the
mechanism claim is not falsifiable without an equal-magnitude random contrast
(§4.3).

**Simulation fidelity, measured on real Q/K.** `B` (native) vs `B_sim` (simulated)
agree to a **median relative difference of 9.6e-8** in output error (p90 = 3.9e-7;
max absolute difference 1.7e-6; 34% of cells bit-identical; `n = 20,400`). The
simulated NVFP4 path is therefore ~5 orders of magnitude tighter than the effect
sizes discussed below, and the `E`/`F` rows are safe to interpret numerically.
They still carry **no latency claim** — `numerical_only = true` on every such row.

**Equal budget.** `k` is identical across all arms at a cell by construction
(`k = 103 / 52 / 26` at sparsity `0.80 / 0.90 / 0.95`), asserted in the analysis
gate: zero cells disagree.

---

## 3. Error decomposition against dense BF16

Median relative L2 `||x - A||_2 / ||A||_2` per head-cell, `n = 20,400` per row.
Full table with IQR, p10/p90, cosine and max-abs:
`artifacts/sparsefp4/tables/phase2_main/table1_af_decomposition.csv`.

| Config | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| **B** — quantization only (native) | 0.0520 | 0.0520 | 0.0520 |
| **C** — sparsification only | 0.1058 | 0.1762 | 0.2740 |
| **D** — C + NVFP4 wrong-mask | 0.1057 | 0.1765 | 0.2745 |
| **D8** — C + FP8 wrong-mask | 0.1057 | 0.1762 | 0.2741 |
| **C_rand** — C + random wrong-mask | 0.1145 | 0.1797 | 0.2761 |
| **E** — NVFP4 compute + NVFP4 router | 0.1553 | 0.2126 | 0.2961 |
| **F8** — NVFP4 compute + FP8 router | 0.1552 | 0.2126 | 0.2958 |
| **F16** — NVFP4 compute + BF16 router | 0.1552 | 0.2125 | 0.2959 |

Cosine similarity to dense BF16 stays high throughout (0.9894 for `E` at 0.80,
0.9628 at 0.95), so these are magnitude-scale errors, not directional collapse.

![A–F error decomposition](figures/phase2_main/fig1_af_decomposition.png)

### 3.1 Attribution

Differences of errors against the single reference A. These are **not an additive
decomposition** — quantization and sparsification errors do not compose linearly,
and indeed compose **sub-additively** here (`E - B - C` is negative, median
-0.034 to -0.043, i.e. the two errors partially cancel). Table:
`table4_error_attribution.csv`.

| Term | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| Quantization (`B`) | 0.0520 | 0.0520 | 0.0520 |
| Sparsification (`C`) | 0.1058 | 0.1762 | 0.2740 |
| **Wrong-mask, NVFP4 (`D - C`)** | **3.10e-05** | **3.33e-05** | **9.52e-05** |
| **Wrong-mask, FP8 (`D8 - C`)** | **7.9e-08** | **-4.0e-07** | **1.3e-06** |
| Random wrong-mask (`C_rand - C`) | 8.36e-04 | 7.20e-04 | 9.59e-04 |
| Router-recoverable (`E - F16`) | 6.6e-06 | 8.4e-06 | 4.8e-05 |
| Share of `E` from quantization | 33.5% | 24.5% | 17.6% |
| Share of `E` from sparsification | 68.1% | 82.9% | 92.5% |
| **Share of `E` from wrong-mask** | **0.020%** | **0.016%** | **0.032%** |

The wrong-mask term — the *entire* quantity H3 proposes to remove — is
**0.02%–0.03%** of the combined error. It is three to four orders of magnitude
below both quantization and sparsification, and at sparsity 0.90 the FP8 router's
wrong-mask term is *negative*, i.e. indistinguishable from zero.

---

## 4. The H3 test

**Pre-registered criterion:** ≥20% relative reduction in attention-output error
from higher-precision routing, holding compute precision fixed.

The comparison is **paired per cell** at `(prompt, layer, head, timestep,
cfg_branch, sparsity)`, not a difference of independently-pooled medians. Every
arm shares one dense-BF16 trajectory and the same `k`, so the per-cell difference
is available and is the correct statistic. Table: `table3_h3_paired.csv`;
per-cell reduction ECDF: `figures/phase2_main/fig2_h3_paired_reduction_ecdf.csv`.

| Comparison | s | median rel. reduction | 5%-trimmed mean | paired diff p10 / p50 / p90 | frac. cells improved | n | ≥20%? |
|---|---|---|---|---|---|---|---|
| `E → F8` (NVFP4→FP8 router) | 0.80 | **0.050%** | 0.018% | -3.0e-04 / +6.7e-06 / +3.6e-04 | 55.5% | 20,400 | **No** |
| `E → F8` | 0.90 | **0.037%** | 0.021% | -5.4e-04 / +1.0e-05 / +7.1e-04 | 52.9% | 20,400 | **No** |
| `E → F8` | 0.95 | **0.104%** | 0.038% | -9.7e-04 / +4.5e-05 / +1.3e-03 | 54.0% | 20,400 | **No** |
| `E → F16` (NVFP4→BF16 router) | 0.80 | **0.051%** | 0.018% | -3.0e-04 / +6.6e-06 / +3.6e-04 | 55.4% | 20,400 | **No** |
| `E → F16` | 0.90 | **0.055%** | 0.021% | -5.4e-04 / +8.4e-06 / +6.7e-04 | 52.2% | 20,400 | **No** |
| `E → F16` | 0.95 | **0.073%** | 0.040% | -9.7e-04 / +4.8e-05 / +1.3e-03 | 54.7% | 20,400 | **No** |

![H3 verdict: measured reduction vs threshold, and the per-cell distribution](figures/phase2_main/fig2_h3_verdict.png)

Three points make this a strong negative rather than a weak one:1. **The effect is 200–500x below threshold**, not marginally below it. No
   plausible amount of additional statistical power changes the verdict.
2. **The sign is barely better than a coin flip.** Only 52–56% of cells improve at
   all; the paired difference distribution straddles zero symmetrically (p10 is
   negative in every row). A *perfect* BF16 router makes the output worse in
   roughly 45% of cells.
3. **A BF16 router — the exact upper bound of H3 — is no better than an FP8
   router.** `F16` and `F8` are indistinguishable from each other. H3 does not
   merely fail at FP8; it fails at infinite router precision, which means the
   ceiling on precision-decoupled routing is itself ~0.05%.

Null control: `F16` vs itself is exactly 0 across all 61,200 rows. Per the trap-8
correction this is reported as an **arithmetic-identity check only** and does *not*
certify scorer resolution.

Per-region breakdown (`table3_h3_paired.csv`, `region` column) shows the same
verdict in `affected`, `unaffected` and `broad` layer sets; no region reaches even
1% reduction.

---

## 5. Why H3 fails: the decision-margin mechanism, measured

The brief's prediction was that quantization flips exactly those blocks sitting at
a near-degenerate top-k boundary — the ones contributing least to the output — so
there would be almost nothing for a better router to recover. This was tested
directly, not assumed, by computing the **exact dense attention mass** each key
block contributes to each sampled query block (softmax over the full key axis,
then summed within block; verified to sum to 1.0 in the self-test).

Measured at 34,560 query-block observations per sparsity across 6 layers x 2
timesteps. Table: `table5_margin_mechanism.csv`.

| Quantity (median, sparsity 0.90, all regions) | Value |
|---|---|
| Blocks swapped per query block (mean) | 1.00 of `k`=52 |
| Query blocks with at least one swap | 65.6% |
| **Attention mass of a swapped (dropped) block** | **0.00283** |
| **Attention mass of a block both masks agree to keep** | **0.01320** |
| Attention mass of an average *excluded* block | 0.00070 |
| Total mass retained by the BF16 top-k | 0.676 |
| Normalized score gap of the swapped pair | **0.0047** of the score spread |

Four independent implications of the mechanism, all confirmed:

![Mechanism: block mass and the random contrast control](figures/phase2_main/fig3_mechanism.png)

**5.1 Swapped blocks are near-worthless.** A swapped block carries **4.7x less**
attention mass than a block the two masks agree on (`0.00283` vs `0.01320`). It
sits much closer to the *excluded* population (`0.00070`) than to the retained
one. The total mass displaced by NVFP4's disagreement is `7.5e-04` — under **0.1%
of the attention distribution**, against `0.676` retained.

**5.2 The swaps happen at a vanishing margin.** The reference-score gap between
the dropped and added blocks is **0.47% of the score spread** within that query
block's key axis. The decision is a near-tie by any measure; NVFP4's quantization
error is simply larger than that margin and smaller than every other margin.

**5.3 Output error is concentrated where the margin is *not* the driver.** Binning
query blocks by the score gap of their swapped pair (`table6_error_vs_score_gap_decile.csv`)
shows the wrong-mask excess error is **0.0 in 8 of 10 deciles** and never exceeds
`1.0e-05` in any decile, while the *sparsification* error `C` varies by more than
10x across the same deciles (0.187 → 0.018). Where output error is large, it is
large for both `C` and `D` alike — the mask difference contributes nothing. And
the mass of dropped blocks *falls* monotonically as the gap widens (0.0038 →
0.0005), exactly as a boundary-effect account predicts.

**5.4 The contrast control isolates the mechanism.** This is the decisive test.
`C_rand` changes **exactly as many blocks** as NVFP4 does, from the same BF16
baseline mask, under the same budget — but chooses them at random rather than by
quantization error. Table: `table7_random_perturbation_contrast.csv`.

| Excess rel-L2 over `C` (median) | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| Quantization-chosen swaps (`D - C`) | 3.10e-05 | 3.33e-05 | 9.52e-05 |
| **Random swaps of equal count (`C_rand - C`)** | **8.36e-04** | **7.20e-04** | **9.59e-04** |
| **Ratio, random / quantization** | **27.0x** | **21.7x** | **10.1x** |
| Cells where random is worse | 63.2% | 61.6% | 61.6% |

Removing the same *number* of blocks at random costs **10x to 27x more** output
error than letting NVFP4's quantization choose them. The same ordering appears in
the mass measurement independently: random-dropped blocks carry `0.00525` of mass
versus `0.00283` for quantization-dropped, a **1.9x** ratio (`table5`).

Quantization is therefore not making a generic mask error of a certain magnitude —
it is making the **cheapest possible** error of that magnitude. It perturbs only
the harmless boundary. That is precisely why a higher-precision router has nothing
to recover, and it is a mechanistic explanation of the H3 failure rather than a
restatement of it.

**Interpretation caveat.** `C_rand` is a synthetic control, not a system anyone
would build; its role is to establish that the *magnitude* of NVFP4's mask
perturbation is not what makes it harmless — the *location* is. It is labeled
`router_native_or_simulated = "synthetic_control"` in the raw data so it can never
be mistaken for a measured precision arm.

---

## 6. Are the "sensitive" edge layers real, or just wider activations?

Phase 1 found routing instability concentrated at the network edges (layers 0–2,
27–29). The boring alternative is that those layers simply have wider activations,
so more elements clip at the e2m1 maximum. Both were measured per layer:
`table8_saturation_vs_layer_sensitivity.csv`.

**The saturation explanation is rejected, and so is the sensitivity framing.**

![Saturation control](figures/phase2_main/fig5_saturation_control.png)

- NVFP4 saturation fraction is **flat across all 17 layers** (0.0958–0.1098 for Q;
  0.0977–0.1099 for K). Layer 0, the most extreme Phase 1 outlier, has the
  *lowest* saturation of any layer measured.
- Spearman correlation between a layer's wrong-mask excess error and its Q
  saturation fraction is **-0.25** — the wrong sign for the saturation account.
- Correlation with Q absolute max is **+0.36** and with intra-group dynamic range
  **+0.22**: weakly positive, far from explanatory.

So activation width does not explain the layer ranking. But the more important
finding is that **the layer ranking does not matter**: the largest per-layer
wrong-mask excess across all 17 layers is `2.0e-04` (layer 2), and the largest
*relative* excess is `1.7e-03`. Even the single worst layer in the network is two
orders of magnitude below the H3 threshold. Phase 1's `affected` layers do show
more mask churn — they have more swaps per query block (1.32 vs 0.72 at 0.90) —
but their swapped blocks are correspondingly *even less* important
(`dropped_over_agreed_ratio` 0.13 in affected vs 0.32 in unaffected). More churn,
less consequence. Targeting the sensitive layers with a better router would not
help, because that is where the swaps are most harmless.

One genuine asymmetry: layer 0 has an intra-group dynamic range of **12.2**,
roughly 4x every other layer (3.0–4.4). That is a real property of layer 0's Q/K
distribution and would matter for a *weight* or *activation* quantization study.
It does not translate into routing error here.

---

## 7. Trap 8 confirmed on real data, and it mattered

The correction to use `fp64` block scores was verified rather than taken on faith
(`table9_score_resolution_trap8.csv`, `n = 1,700` cells per row):

| Router | s | boundary ties, fp32 | boundary ties, fp64 | frac. top-k decisions fp32 ≠ fp64 |
|---|---|---|---|---|
| bf16 | 0.80 | 1,429 | **0** | 0.41% |
| bf16 | 0.95 | 1,235 | **0** | 1.17% |
| fp8_e4m3 | 0.80 | 1,430 | **0** | 0.41% |
| nvfp4 | 0.80 | 1,430 | **0** | 0.41% |

`fp32` manufactures ~1,400 exact boundary ties per cell where `fp64` produces
**zero**, and flips 0.4%–1.2% of top-k decisions. For scale: the *entire* NVFP4-vs-BF16
mask disagreement is ~1 block per query block. An `fp32` scorer's own arithmetic
noise is comparable to the effect being measured. Phase 2 used `fp64` from the
first measured run, and `score_dtype == "float64"` is a hard gate in the analysis
verifier — no Phase 2 number exists that was computed in `fp32`, so nothing needed
quarantining on this account.

I note one honest deviation from the briefing's expectation: with `fp64` scores the
three routers show **near-identical** tie counts and flip rates (1,429 / 1,430 /
1,430), i.e. I could not reproduce the "fp32 penalizes FP8 1.6x harder" asymmetry
at this geometry and these sparsities. The correction was still the right call —
`fp64` removes ~1,400 spurious ties per cell regardless — but the specific
directional bias against H3 is not visible in my measurement, and I am not going to
claim support for it that I do not have. Either way, the H3 verdict is not close
enough to threshold for scorer resolution to be the deciding factor.

---

## 8. Threats to validity

| Threat | Status |
|---|---|
| **Sparse NVFP4 compute is simulated** | Real. No native sparse-NVFP4 kernel exists here. Bounded by the `B` vs `B_sim` control: median relative disagreement `9.6e-08`, ~5 orders below the effect sizes. Every affected row carries `numerical_only = true` and **no latency claim is made anywhere in this report.** |
| **FP8 router is simulated** | Real, and unavoidable: native NVFP4 quantization is callable but has no dequantizer for its packed layout, and there is no native FP8 attention quantizer to borrow. Mitigated by `F16` (BF16 router, native values) landing in the same place — the H3 ceiling is established by a non-simulated arm. |
| **Diagnostic geometry ≠ VSA geometry** | Real and stated on every table. 128x64 raster blocks, not VSA's 64-token (4,4,4) cubes. A VSA-integrated arm could in principle behave differently; this is the first item in §10. |
| **Single model, single resolution** | Real. Wan2.1-T2V-1.3B only. Effect is 200x+ from threshold, so scale would have to change the mechanism qualitatively, not just quantitatively. |
| **Error measured at the attention output, not the final video** | Deliberate — it is the quantity H3 is about, and it is the *most favorable* place to look for a router effect, since downstream residual mixing can only dilute it. A ≥20% attention-output reduction was the pre-registered gate. |
| **Trajectory is dense BF16, not the sparse trajectory** | Deliberate: it is what makes 20,400 exactly-paired comparisons possible. It means these are *single-step* errors and do not capture accumulation across 50 steps. See §10. |
| **Mechanism sampled on 6 layers x 2 timesteps** | 34,560 query-block observations per sparsity, but not the full layer set. Region split (affected/unaffected/broad) is consistent across all three, and the A–F decomposition itself covers all 17 layers. |

## 9. Data exclusions

One exclusion, recorded per the no-silent-drops rule:

- **`QUARANTINED-20260814-023500-p2-main-mixed-code`** (8 complete shards + 1
  partial). This first main-sweep attempt was launched before a mypy-driven
  refactor of the record-emission path landed; prompt `p10` then crashed on a
  stale local variable. Rather than mix code versions within one run, the whole
  run was quarantined under its original name on scratch and **all 10 prompts were
  re-run from scratch** under the final code as `20260814-025500-8208536-p2-main`.
  No quarantined number appears anywhere in this report. Its cause was a code-version
  hygiene decision, not an unfavorable result — the quarantined shards' medians
  agree with the final run's to ~3 significant figures.

All 10 shards of the reported run wrote **identical record counts (61,538 each)**,
which is itself a check that no layer/timestep/head cell silently fell off the
research backend.

---

## 10. Recommendation: **PIVOT**

**Do not pursue precision-decoupled routing.** H3 is falsified with a large,
well-powered margin and a measured mechanism. The ceiling on the entire idea is
~0.05% of attention-output error, established by a BF16 router, which is the
best any router can possibly be. Spending FP8 or BF16 bandwidth on the router buys
nothing.

**The positive finding to carry forward is the inverse of H3.** Because NVFP4
routing is essentially free in accuracy — its wrong-mask error is 0.02%–0.03% of
total error and 10–27x cheaper than a random perturbation of the same size — the
router can be run at the *same* low precision as the compute. This is a real, if
modest, engineering result: it removes a design question ("what precision must the
router be?") by answering "the cheapest one", and it is supported by the mechanism
rather than by a single benchmark.

**Where the error actually is.** Sparsification dominates: 68% of combined error at
sparsity 0.80, rising to **93% at 0.95**, versus 0.02% for wrong-mask. Any further
work on sparse + NVFP4 video attention should target the retained-block *budget and
selection criterion*, not the arithmetic precision of the selection. The
sub-additivity finding (`E - B - C` ≈ -0.04, consistently negative) is a concrete
lead: quantization and sparsification errors partially cancel, so the two should be
co-designed rather than independently minimized.

**Suggested next steps, in priority order.**

1. **Confirm at VSA's real geometry.** The one threat that could change the
   conclusion is geometry, since VSA forces `block_q == block_k ∈ {64, 256}` on
   spatio-temporal cubes rather than raster blocks. Re-run `C`/`D`/`C_rand` only
   (cheap — no new configurations) at 64x64 cubes. If the mechanism holds there,
   the negative result is robust for the deployed path.
2. **Measure end-to-end, not per-step.** Run the sparse trajectory for real
   (config `E` as the consumed attention) for a handful of prompts and compare
   final-video SSIM/LPIPS against dense BF16. Single-step attention error of 0.15–0.30
   is large enough that the interesting question is now whether *sparsification
   itself* is viable at these sparsities, independent of precision.
3. **Attack the budget, not the precision.** The `k` sweep shows error roughly
   doubling from 0.80 to 0.95. Non-uniform per-layer or per-head budgets are the
   obvious lever, and §6's data already indicates regions differ substantially in
   how much mass their top-k retains at sparsity 0.95: 0.70 in the edge
   (`affected`) layers versus 0.35–0.37 in the mid-stack ones. A uniform budget is
   leaving accuracy on the table in exactly the layers that need it most.

**For the workshop paper.** This is publishable as a negative result with a
mechanism, which is more useful than a marginal positive: *"quantization-induced
sparse-mask instability is real but benign, because low-precision routing errs
exactly at the near-degenerate top-k boundary; we show a same-magnitude random
perturbation is 10–27x more damaging, and conclude that sparse-attention routers
should be run at the compute precision."* The `C_rand` contrast and the
mass-vs-margin measurement are the paper's core evidence. No latency claim can be
made from this data.

---

## 11. Artifacts

| Kind | Path |
|---|---|
| Raw JSONL (gzipped, 10 shards) | `artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main/` |
| Per-run configs and summaries | same directory, `phase2_config_p*.json`, `run_summary_p*.json` |
| Verification gate | `.../verification.json` (also in tables dir) |
| Tables (CSV + Markdown) | `artifacts/sparsefp4/tables/phase2_main/table1`–`table9`, `summary.json` |
| Figure data (CSV) | `artifacts/sparsefp4/figures/phase2_main/fig1`–`fig5` |
| Rendered figures (PNG) | `.../fig1_af_decomposition.png`, `fig2_h3_verdict.png`, `fig3_mechanism.png`, `fig5_saturation_control.png` |
| Correctness self-test | `artifacts/sparsefp4/raw/phase2_selftest.json` (12 checks, `PASS`) |
| Backend | `fastvideo/attention/backends/precision_sparse_attn.py` |
| Shared numerics | `fastvideo/attention/backends/sparsefp4_numerics.py` |
| Driver / launcher / analysis | `artifacts/sparsefp4/configs/phase2_{run.py,launch.sh,analyze.py,selftest.py,figures.py}` |
| Protocol amendments | `.agents/skills/sparsefp4-video-attention/references/EXPERIMENT_SPEC.md` §12 |

Reproduce:

```bash
source artifacts/sparsefp4/configs/env.sh
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_selftest.py --out /tmp/selftest.json
bash artifacts/sparsefp4/configs/phase2_launch.sh <new_run_id>
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_analyze.py \
  --raw artifacts/sparsefp4/raw/<new_run_id> \
  --out-tables artifacts/sparsefp4/tables/<tag> \
  --out-figures artifacts/sparsefp4/figures/<tag>
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_figures.py \
  --tables artifacts/sparsefp4/tables/<tag> \
  --figures artifacts/sparsefp4/figures/<tag>
```
