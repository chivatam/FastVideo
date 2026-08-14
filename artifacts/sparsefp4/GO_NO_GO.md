# GO / NO-GO — SparseFP4 Video Attention

Written after Phases 1–2 as the SKILL requires. Decision owner: study lead.
Date: 2026-08-14. Repo `8208536` on `exp/sparsefp4-mask-stability`.

## Decision: **PIVOT** (to a negative + mechanistic result, plus systems composability)

The study's originally-proposed method — decoupling router precision from attention-compute
precision — is **falsified**, cleanly and with a measured explanation. The pre-registered
NO-GO/PIVOT condition is met on both of its clauses, so this is the outcome the protocol
committed to in advance, not a retreat.

Per the SKILL: *"A strong negative result is preferable to manufacturing a method."*

## Why: the pre-registered criteria, evaluated

The SKILL's PIVOT condition is: BF16 and NVFP4 masks nearly identical almost everywhere
(e.g. Jaccard > 0.95 at 80–90% sparsity) **and** higher-precision routing does not improve
output error. Both hold.

### Clause 1 — masks are nearly identical (Phase 1, n=72,000/cell)

| sparsity | fp8 router | nvfp4 router | cells > 0.95 Jaccard |
|---|---|---|---|
| 0.80 | 0.9924 | 0.9807 | 97.1% |
| 0.90 | 0.9891 | 0.9738 | 89.4% |
| 0.95 | 0.9827 | 0.9611 | — |

Spearman rho of block scores 0.9997. The null control (bf16 vs bf16) is exact at 1.0 across
187,200 records, and the record lattice was verified complete, so no layer silently fell off
the probe backend.

### Clause 2 — higher-precision routing does not help (Phase 2, n=20,400 exactly paired)

Measured relative error reduction from a higher-precision router, against the pre-registered
**>=20%** support threshold:

| arm | s=0.80 | s=0.90 | s=0.95 | verdict |
|---|---|---|---|---|
| NVFP4 -> FP8 router | 0.050% | 0.037% | 0.104% | No |
| NVFP4 -> BF16 router | 0.051% | 0.055% | 0.073% | No |

**H3 fails by 200–500x, not marginally.** Only 52–56% of cells improve at all — near coin-flip.
Decisively, the **BF16 router is the theoretical ceiling of the entire idea and is no better
than FP8**, so this is not an FP8 shortfall; the premise itself is wrong.

## The reason, measured rather than asserted

The wrong-mask error term — the *entire* quantity H3 proposes to remove — is
**0.016%–0.032%** of total error. It is dwarfed by sparsification (68% -> 93% as sparsity
rises) and quantization (34% -> 18%):

| term | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| quantization (`B`) | 0.0520 | 0.0520 | 0.0520 |
| sparsification (`C`) | 0.1058 | 0.1762 | 0.2740 |
| wrong-mask, NVFP4 (`D-C`) | 3.10e-05 | 3.33e-05 | 9.52e-05 |
| wrong-mask, FP8 (`D8-C`) | 7.9e-08 | **-4.0e-07** | 1.3e-06 |

At sparsity 0.90 the FP8 wrong-mask term is *negative* — indistinguishable from zero.

### The mechanism: quantization makes the cheapest possible error

Quantization-induced swaps land on near-degenerate top-k boundaries:

- a swapped block carries **4.7x less** attention mass than an agreed-on block
  (0.00283 vs 0.01320) and sits closer to the *excluded* population (0.00070);
- the swapped pair's score gap is **0.47% of the score spread** — a near-tie;
- total displaced mass is 7.5e-04, under **0.1%** of the attention distribution;
- wrong-mask excess error is **0.0 in 8 of 10** score-gap deciles.

### The control that makes this an explanation, not a shrug

`C_rand` changes **exactly as many blocks** from the same baseline mask under the same budget,
but picks them at random instead of by quantization error:

| excess rel-L2 over `C` | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| quantization-chosen swaps | 3.10e-05 | 3.33e-05 | 9.52e-05 |
| random swaps, equal count | 8.36e-04 | 7.20e-04 | 9.59e-04 |
| **ratio random / quantization** | **27.0x** | **21.7x** | **10.1x** |

So the null is *not* "the errors were too few to matter." Same count, 10–27x the damage.
Low-precision routing errs **only where erring is nearly free**. That is a positive,
falsifiable finding.

## Hypotheses

| | verdict |
|---|---|
| H1 — NVFP4 perturbs routing | **Supported in direction, effect small.** Monotone in sparsity and precision; Jaccard 0.9807 @0.80, 0.9611 @0.95. Real but not consequential. |
| H2 — instability is localized | **Unsupported.** Affected cells (<0.90): timesteps 0/50, heads 0/12, layers 0/30. 50-step spread 0.0095. No schedulable structure. Edge-layer residue explained away — saturation is flat across layers (0.096–0.110) and correlates with wrong-mask error at rho **-0.25**, the wrong sign. |
| H3 — precision-decoupled routing recovers quality | **Falsified.** 0.04–0.10% vs a 20% threshold, with the BF16-router ceiling no better than FP8. |
| H4 — native sparse-NVFP4 wall-clock benefit | **Untested, and now materially less attractive than it first looked.** The dense NVFP4 win is **1.28x on the attention kernel in isolation** but only **1.055x end-to-end** (44.436 s vs 46.876 s, warmed, n=5, identical 8518 MB peak) and 1.061x per DiT step. See the Amdahl bound below before investing. |

### Amdahl reality check on H4 (derived from two measured numbers, not directly measured)

Per-step DiT latency falls 890.5 -> 839.7 ms when attention alone gets 1.28x faster. Solving
`T_attn * (1 - 1/1.28) = 50.8 ms` puts attention at **~232 ms of the 890.5 ms step, i.e. ~26%**.

Consequences, which should temper Phase 4 expectations sharply:

- Making attention **infinitely fast** caps end-to-end speedup at about **1.35x** at this
  configuration (1.3B params, 480x832, 81 frames, seq 32760, sp_size=1).
- A sparse kernel achieving a generous 3.3x on attention would yield roughly **1.22x per step**.
- A native sparse-NVFP4 kernel must therefore beat **44.436 s** end-to-end, and the entire prize
  is a fraction of a small number.

This does not make H4 uninteresting — a measured sparse-NVFP4 kernel is the one thing that could
lift this study to STRONG GO — but it does mean the kernel work is a systems contribution with a
modest ceiling here, and the ceiling grows only with longer sequences (higher resolution / more
frames / larger models), where attention's share rises.

## The pivot

Adopt the SKILL's pivot options **(3) negative result** and **(1) systems composability**:

1. **Low-precision routing is safe.** Route at the *same* precision as compute — the cheaper
   design. No high-precision side-path is needed. This is directly actionable for anyone
   building sparse attention on Blackwell.
2. **The mechanism explains why**, and generalizes as a principle: top-k selection is robust to
   quantization precisely because quantization error is small relative to all margins *except*
   the ones that do not matter.
3. **Remaining error is sparsification, not routing** (93% at sparsity 0.95). Attack the budget
   and selection rule, not the router's number format.
4. **H4 is the open systems question** with a measured bar to beat, and the FP4 kernel source is
   editable CuTeDSL Python rather than a C++ rebuild.

### Smallest defensible claim

> In a Wan2.1-1.3B DiT at 480x832x81, deriving block-sparse attention routing from
> NVFP4-quantized Q/K instead of BF16 Q/K changes 1.9–3.9% of selected blocks (sparsity
> 0.80–0.95) but changes attention output by <0.1% relative L2 — **27–76x less than random block
> swaps of identical count**, across three block geometries including VSA's deployed 64-token
> `(4,4,4)` cubes. Higher-precision routing recovers 0.04–0.10% against a pre-registered 20%
> threshold, and **produces no detectable end-to-end video-quality change** (0 of 24 tests
> surviving Holm-Bonferroni, on tests where sparsity is significant 8/8). Low-precision routing is
> therefore safe, because quantization perturbs only near-degenerate selection boundaries.

## Required scoping and open threats

1. **Geometry — CLOSED, and the mechanism STRENGTHENS.** Phase 2B re-ran `C`/`D`/`C_rand` at
   three geometries, separating block size from token ordering. The `C_rand`/`D` isolation ratio
   *rises* at VSA's deployed 64-token `(4,4,4)` cube geometry, most emphatically where 128x64 was
   weakest:

   | `C_rand`/`D` ratio | s=0.80 | s=0.90 | s=0.95 |
   |---|---|---|---|
   | 128x64 raster | 27.0x | 21.7x | 10.1x |
   | 64x64 raster | 47.1x | 23.3x | 24.2x |
   | **64x64 cube (deployed)** | **75.6x** | **44.6x** | **35.7x** |

   The agreed/swapped mass gap widens from 3.8–5.7x to 6.9–11.1x, cells where random is worse rise
   61.6% -> 69.8%, and cube Jaccard IQR is 25% *narrower*. Mechanistically, coherent cube tiles
   *separate* block scores, pushing near-ties further into the mass tail. All 18 cells point the
   same way. **Token ordering matters more than block size** on the mass measures (+61–77% vs
   +6–11%).
   Conservative bias worth noting: the cube arm retains 5–6% *fewer* tokens at equal block
   sparsity (0.191 vs 0.201 at s=0.80), biasing that arm **against itself**; all mechanism ratios
   are paired within-geometry so this cannot explain them.
   Padding was handled with VSA's own utilities and gate-checked four ways — pad slots hold zeros
   (bit-identical round trip), pooling divides by true tile count (0.0 difference), `all_pad_blocks
   = 0` so top-k cannot select a pad-only block, and perturbing all 7,176 pad V slots by +100 left
   the output bit-identical.

   **Two limits this does NOT license.** H3 was *not* re-tested at cube geometry — the
   NVFP4-compute arms were deliberately not re-run, so H3's falsification remains a 128x64 result.
   And the cube arm ran on a research block-sparse kernel, **not** through VSA's own scorer, so
   nothing may be claimed about VSA end-to-end quality.
2. **"NVFP4" means NVFP4 Q/K with BF16 PV** — the FA4 kernel is `qk_mode=nvfp4, pv_mode=bf16`.
   Never state or imply fully-FP4 attention.
3. **No latency claim for any sparse arm.** Sparse-NVFP4 compute has no native kernel here and
   is simulated (dense native-vs-simulated control bounded at 9.6e-8). Only the dense
   attention-kernel microbenchmark is a measured performance number.
4. **fp64 block scores are mandatory** (trap #8). fp32 manufactures ~1,400 boundary ties/cell
   where fp64 gives zero, flipping 0.4–1.2% of top-k decisions.
5. **The Phase1-vs-Phase2 discrepancy is RESOLVED — both phases were right about different
   denominators.** Phase 1's `boundary_ties` reduced over `n_q_blocks` only, i.e. **per (cell,
   head)**: 115 of 256. Phase 2 summed the whole head x q-block grid, i.e. **per cell**: 1,430 of
   3,072. Rescaling, 115 x 12 heads = 1,380 vs 1,430 — **4% agreement, one measurement, not two**.
   Tie *rate* 0.449 vs 0.465. Confirmed by a fresh `tie_diagnostic` record emitting both
   denominators at all three geometries (per-head 110–111, per-cell 1,429–1,430). fp64 gives
   **exactly 0 ties** at every geometry/router/sparsity.
   **The 1.6x FP8 asymmetry is real**: 1.604x at s=0.80 and 1.550x at s=0.90 (n=36,000 paired
   cells/arm), reproduced from Phase 1's archived fp32-vs-fp64 pair. Phase 2's non-reproduction was
   a category error — it compared tie *counts* and per-arm flip rates, which are correctly
   symmetric because fp32 noise is arm-independent, rather than the **Jaccard shift** the claim is
   about, which is asymmetric because FP8's true deficit is 2.4x smaller and so is contaminated
   relatively more. The trap #8 fp64 mandate was therefore correct and remains in force.
6. **Development-set scope.** 10 prompts, 1 seed. No benchmark-wide claim is licensed.
7. **End-to-end video quality — CLOSED (Phase 5).** 70 generations, 0 failures, 10 prompts at
   sparsity 0.90. Routing precision has **no detectable effect on generated video**: NVFP4-router
   vs BF16-router gives Wilcoxon **p = 0.557**, and across the full 24-test routing family
   **0 tests survive Holm-Bonferroni** (3 raw hits vs 1.2 expected by chance). The same tests find
   sparsity significant on **8/8** metrics at the n=10 exact floor (p = 0.00195), so the instrument
   is not simply insensitive. The three SPARSE-FP4-* arms agree to 0.07 dB PSNR / 0.003 SSIM /
   0.005 LPIPS — about 3% of the between-prompt spread.
   Sparsity-to-routing effect ratio, reported across all instruments rather than the flattering
   ones: **9,245x** at the attention output (the physically meaningful figure, n=34,560 paired
   cells), 13.0x–45.0x on VBench quality dimensions, 5.4x on free-running pixel MAE.
8. **Paired pixel metrics are SATURATED in multi-step diffusion — a methodological finding, and a
   trap for anyone repeating this.** A calibration arm injecting a *known* attention perturbation
   showed 1e-6 already produces pixel MAE 0.0186 while 1e-1 produces 0.0317: five orders of
   magnitude compressed into 1.7x, non-monotone. PSNR/SSIM/LPIPS in a 50-step free-running sampler
   therefore measure **whether the trajectory decorrelated, not by how much attention differed**.
   The 5.4x pixel ratio above is consequently *not* an effect size. VBench is the instrument for
   the quality claim. This negative control is reusable and arguably publishable on its own.
9. **Dense NVFP4's end-to-end gain is 1.055x, not 1.28x.** The 1.28x is the attention kernel in
   isolation. Never present the kernel figure as an end-to-end speedup.

## Paper viability

**GO** (upgraded from BORDERLINE-to-GO once the geometry control landed).

The empirical story is pre-registered, well-controlled, mechanistically explained with a
falsifiable contrast control, and — critically — **confirmed at the geometry FastVideo actually
deploys**, where the effect is *stronger* than at the geometry we happened to start with. The
one threat that could have reduced this to an artifact has been closed in the favorable
direction.

Remaining gap for a STRONG GO: a measured native sparse-NVFP4 kernel number (H4, the true stretch
goal) — but see the Amdahl bound above, which caps end-to-end gain at ~1.35x at this
configuration and makes H4 a modest-ceiling systems contribution rather than the headline.

Recommended order: **final report now** (all three hypotheses are resolved and the geometry and
end-to-end contingencies are discharged) -> only then consider the Phase 4 kernel, with the 1.35x
ceiling understood up front.

### Claim the geometry control licenses

> Measured on Wan2.1-T2V-1.3B at 480x832x81, single-step attention-output error against dense
> BF16, at three block geometries including VSA's deployed 64-token `(4,4,4)` spatio-temporal
> cubes: NVFP4-induced routing swaps land on near-degenerate top-k boundaries and are **27–76x
> cheaper than an equal-magnitude random perturbation**, so sparse-attention routers can be run
> at the compute precision. A numerical result with no latency claim; the cube arm was executed
> on a research block-sparse kernel rather than through VSA's own scorer.

Must **not** say: that H3 was re-tested at cube geometry; anything about VSA end-to-end quality
or VSA's own scorer; or that cube geometry is generally better for sparse attention.
