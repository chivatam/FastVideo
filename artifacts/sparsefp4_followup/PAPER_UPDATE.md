# Paper update protocol

Follow-up validation of the SparseFP4 routing study. Every number below carries its
source path, run ID and *n*. Nothing here strengthens a claim beyond what the data
supports; two claims are weakened and one is newly marked untestable.

Run IDs (all on one 8xB200 host, commit `6e886a9c`):

| Phase | Run ID | Records |
|---|---|---|
| F1 scorer arithmetic | `20260816-215059-6e886a9c-f1-full` | 1,555,200 |
| F2 real VSA selector | `20260816-215059-6e886a9c-f2-full` | 1,166,400 |
| F3A seeds 2026/3407 | `20260816-223631-6e886a9c-f3a-seeds` | 1,036,800 |
| F3B 720p | `20260816-225059-6e886a9c-f3b-720p` | 259,200 |

Total: 4,017,600 paired measurement records, every one validated (F1 11/11 checks, F2
17/17, F3A 11/11, F3B 11/11) with complete lattices and zero holes.

---

## 1. One-sentence new headline result

On a deployed dynamic sparse-attention selector, routing precision is not the binding
constraint: NVFP4 routing changes the selected block set by 3.1% in Jaccard terms —
touching 74% of query blocks, about one swapped block each — yet costs only 0.11% of the
sparsification error the method already accepts, and replacing the selector with an fp64
one does not reduce output error at all; it slightly increases it.

## 2. Should the title change?

**Yes, if the current title frames the contribution as a precision-decoupled routing
*fix*.** The follow-up removes the premise of that framing: there is nothing to fix,
because higher-precision routing does not help (§4, rescue arm). A title promising a
routing-precision remedy would now overclaim.

Recommended reframing: the paper's contribution is a *negative-result characterization*
— an attribution of where block-sparse attention error actually comes from, with the
routing-precision channel bounded and shown to be non-binding. Suggested direction:
"Routing precision is not the bottleneck in block-sparse video attention".

If the existing title is already descriptive rather than remedial, keep it.

## 3. Exact abstract edits

**Remove** any sentence asserting or implying that higher-precision routing recovers
quality. **Replace** with the bounded, signed result.

Replace:

> Low-precision routing degrades sparse-attention quality, and restoring routing
> precision recovers it.

With:

> Low-precision routing does change which blocks a dynamic sparse attention mechanism
> selects — on Wan2.1's real VSA selector, NVFP4 Q/K moves the selected block set by 3.1%
> (Jaccard 0.969), altering at least one block in 74% of query blocks — but the resulting
> output error is only 0.11% of the sparsification error the method already accepts, and
> an equal-count random mask perturbation is 225x more damaging. Restoring the selector
> to fp64 does not recover quality: it is marginally *worse* than the shipped bf16
> selector (+0.13% of sparsification error, better in only 9.9% of measurement cells).
> Routing precision is therefore not the binding constraint on block-sparse attention
> quality at these operating points.

**Add** the scope sentence the follow-up now licenses:

> Measured across 4.0M paired records on a real `VIDEO_SPARSE_ATTN` trajectory, three
> seeds, two token counts (32.8k and 75.6k), and three sparsities (0.80/0.90/0.95).

## 4. Exact method-section additions

Four additions, each of which the original method section lacks.

**4a. Scorer arithmetic is a separate axis from routing representation.** Study 1
varied only the Q/K *representation* while computing block scores in fp64 — an
arithmetic precision no deployed kernel uses. Add the 2x6 factorial: representation
(exact BF16, NVFP4) x scorer arithmetic (fp64, fp32, bf16 with fp32 accumulation, bf16
with bf16 accumulation, FP8-E4M3, NVFP4-like), 12 arms measured as a side channel on
one shared trajectory so all arms are exactly paired within a cell.

**4b. The real VSA selector, and what it actually is.** Document the mapping
(`artifacts/sparsefp4_followup/VSA_GATE_MAP.md`): `gate_compress` is **not** the
selector — it is a multiplicative weight on the compression-branch output. The real
selector is `fused_block_mean` (bf16 in, fp32 accumulate, bf16 out) → bf16 matmul with
fp32 accumulation → `fused_topk_mask` (fp32 bisection threshold, ties broken toward the
lower key-block index). The probe subclasses `VideoSparseAttentionImpl`, so the model
follows a genuine VSA trajectory and the measurement uses VSA's own kernels rather than
a reimplementation.

**4c. Two guards that changed the results.** State both, because each silently
invalidated a first pass:

- The denoising loop runs under `torch.autocast(bfloat16)`, which downcasts fp32
  matmul inputs. Without an explicit guard, the "fp32 scorer" arm is not fp32 and the
  arithmetic ladder collapses to a single point. Measurements therefore run inside a
  `declared_precision_arithmetic()` guard that disables autocast, and TF32 is disabled
  locally for fp32 arms; every record carries the ambient state so the distinction is
  auditable.
- The FP8 scorer arms use a native `torch._scaled_mm` dot where the block count permits
  it. At 720p the key-block count is 1182 (not a multiple of 16), so no native kernel
  exists and those arms fall back to an fp32 dot on FP8-rounded inputs. The fallback is
  recorded in `native_or_simulated` and in `score_semantics`
  (`..._FALLBACK_no_native_fp8_gemm`), never silently substituted.

**4d. Uncertainty on ratio-of-medians statistics.** The isolation ratio is a ratio of
medians with no closed-form interval, so intervals are 95% percentile bootstraps
(4000 resamples, seed 20260816) resampling **prompts**, not cells: cells within a prompt
share a denoising trajectory and are not independent. The same shared implementation
(`configs/sparsefp4_stats.py`) produces both the gate verdicts and the figure error
bars, so they cannot disagree.

## 5. Exact result-section additions

**5a. Scorer arithmetic precision is nearly free, and fp32 is exactly free.**
At sparsity 0.90 (`tables/f1_full/table1_arm_headline.csv`, n=1,555,200):

| Scorer arithmetic | Jaccard (exact Q/K) | Jaccard (NVFP4 Q/K) | Damage share (NVFP4 Q/K) | Isolation |
|---|---|---|---|---|
| fp64 | 1.000000 | 0.9751 | reference | 21.1 |
| fp32 | 1.000000 | 0.9751 | 1.374e-03 | 20.9 |
| bf16 (fp32 acc) | 0.9907 | 0.9729 | 1.606e-03 | 23.5 |
| bf16 (bf16 acc) | 0.9568 | 0.9491 | 2.447e-03 | 45.9 |
| FP8-E4M3 (native dot) | 0.9598 | 0.9529 | 2.144e-03 | 25.0 |
| NVFP4-like | 0.8874 | 0.8849 | 7.000e-03 | 12.2 |

The fp32 arm's masks are **bit-identical** to fp64's, so its damage is exactly zero, not
merely small. Damage grows monotonically down the ladder but the worst arm still sits at
0.70% of sparsification error, below the 1% revision threshold and above the 0.1%
strong-survival bound. Matched-random isolation holds everywhere (12.2x–490x, all
bootstrap intervals entirely above 10x).

**5b. Study 1 reproduces independently.** F1's R1 arm re-derives study 1's condition
through a different implementation (`tables/f1_full/table6_r1_vs_study1_baseline.csv`):

| Quantity | Sparsity | Follow-up | Study 1 (frozen) |
|---|---|---|---|
| Mask Jaccard | 0.80 / 0.90 / 0.95 | 0.9812 / 0.9751 / 0.9646 | 0.9807 / 0.9738 / 0.9611 |
| Isolation ratio | 0.80 / 0.90 / 0.95 | 29.5 / 21.1 / 14.4 | 27.0 / 21.7 / 10.1 |
| Wrong-mask excess | 0.90 | 4.057e-05 | 3.326e-05 |

**5c. The result transfers to the real VSA selector.** At sparsity 0.90
(`tables/f2_full/summary.json`, n=1,166,400, genuine `VIDEO_SPARSE_ATTN` trajectory):

| Intervention | Jaccard vs deployed | Damage share | Isolation [95% CI] |
|---|---|---|---|
| FP8 routing representation | 0.9869 | 4.530e-04 | 310 [216, 495] |
| NVFP4 routing representation | 0.9687 | 1.118e-03 | 225 [124, 707] |
| Degraded selector arithmetic (bf16 acc) | 0.9622 | 8.752e-04 | 47.7 [41.4, 55.1] |
| NVFP4 repr. + fp64 selector | 0.9687 | 1.752e-03 | 9.97 [8.17, 12.04] |

**5d. `gate_compress` is provably outside the selection path.** Quantizing it to NVFP4
leaves the mask bit-identical to the deployed mask in **129,600 / 129,600 records** at
every sparsity. This is a falsification test of the gate map, not a description: had the
map been wrong, this arm would have had to differ.

**5e. The high-precision rescue does not rescue.** Routing VSA at fp64 while executing
the identical sparse kernel yields a signed excess of **+1.303e-03** of sparsification
error at sparsity 0.90 (+9.05e-04 at 0.80, +1.513e-03 at 0.95) — i.e. *worse* than the
shipped bf16 selector — and beats it in only **9.9%** of cells. The bf16 selector is not
leaving quality on the table.

**5f. Seed robustness.** `SEED_ROBUST` across seeds 1234/2026/3407
(`tables/f3a/summary.json`, n=1,036,800 new records). Worst-arm damage share is
7.000e-03 / 6.991e-03 / 6.959e-03 and minimum isolation 12.18 / 11.88 / 11.62; the set
of arms damaging in a majority of cells is identical for all three seeds.

**5g. Token-count generalization.** `GENERALIZES_ON_TOKEN_COUNT` at 2.31x tokens
(32,760 → 75,600; `tables/f3b/summary.json`, n=259,200). Damage share **decreases** at
the larger token count (ratios 0.56–0.82 across arms), so the effect does not grow with
sequence length. Labelled **proxy generalization**: this configuration uses the
controlled scorer, not VSA.

**5h. Incidental finding: VSA's top-k kernel can exceed its own budget.**
`fused_topk_mask` returns `topk + 1` blocks when the k-th and (k+1)-th block scores tie
exactly. Its 32-iteration fp32 bisection converges toward the k-th value from below and
never lands on it, so both tied scores test as strictly above the threshold and the
tie-fill branch is skipped. Rate is ~1 row in 7,488 at affected cells (5,376 selector
rows of 1,166,400 records), scale-invariant, and it affects the **shipped** `V0` path,
not only the probe. Reproduced deterministically by `configs/f2_kernel_topk_bug.py`.

## 6. Figures and tables to replace or add

| Artifact | Action | Path |
|---|---|---|
| Figure A — scorer arithmetic ladder, 3 panels | **add** (new axis) | `figures/figureA_scorer_arithmetic.{png,pdf}` |
| Figure B — real VSA selector + gate invariant | **add** (new experiment) | `figures/figureB_vsa_selector.{png,pdf}` |
| Figure C — generalization by seed and configuration | **add** | `figures/figureC_generalization.{png,pdf}` |
| Table A — claim boundary matrix (13 rows) | **add** | `tables/f5/tableA_claim_boundary_matrix.{csv,md}` |
| Table B — claim before/after ledger (8 rows) | **add** | `tables/f5/tableB_claim_before_after.{csv,md}` |
| Study 1's mask-overlap-vs-sparsity figure | **keep**, annotate as reproduced | — |
| Any figure captioned as showing routing-precision *recovery* | **replace** — the rescue is negative | — |

## 7. Limitations removed

- *"Only measured on a controlled proxy scorer, not a deployed selector."* Removed: F2
  measures VSA's own `fused_block_mean` / `fused_topk_mask` on a genuine VSA trajectory.
- *"Single seed."* Removed: three seeds, pre-declared before execution
  (`configs/f3a_seeds.json`), all four robustness criteria hold per seed.
- *"Single token count / resolution."* Removed for the proxy scorer: 2.31x tokens, with
  damage share decreasing.
- *"Scorer arithmetic precision untested."* Removed: full 2x6 factorial.

## 8. New limitations introduced

- **One decision threshold is unresolved.** The `NVFP4 repr. + fp64 selector` arm's
  isolation ratio is 9.04 / 9.97 / 10.49 at sparsities 0.80 / 0.90 / 0.95, and its
  bootstrap interval spans the 10x criterion at all three. The F2 verdict is therefore
  reported as `INDETERMINATE_ISOLATION_THRESHOLD` at 0.80/0.90 rather than as a
  revision. Resolving it requires more prompts, not more cells.
- **NVFP4 scorer arithmetic is simulated.** No native NVFP4 block-dot exists. Fidelity
  against the native quantizer is quantified (median 9.80e-03, p90 1.07e-02, max
  1.10e-02 relative disagreement; 99.95% of elements bit-identical; only 1.39e-03 after
  block pooling), and simulated arms are barred from any latency claim.
- **FP8 dot is emulated at 720p.** Key-block count 1182 is not a multiple of 16, so
  arms R6/R7 fall back to an fp32 dot on FP8-rounded inputs at that configuration.
- **Second configuration is proxy, not VSA.** F3B varies token count under the
  controlled scorer; VSA generalization across token counts is untested.
- **Damage exceeds the 0.1% strong-survival bound.** Every arm below fp32 does, which is
  why F1's verdict is `PARTIAL_SUPPORT` rather than strong survival.
- **No end-to-end perceptual quality measurement.** All damage figures are attention-
  output relative-L2 against a dense reference, not video quality.

## 9. Claims that must be weakened

1. **High-precision routing as a remedy — reverse it.** The fp64 selector is worse than
   the deployed bf16 one (§5e). Any "precision-decoupled routing recovers quality"
   claim must become "routing precision is not the binding constraint".
2. **"Low-precision routing degrades quality" — quantify or drop the verb.** It changes
   *decisions* (Jaccard 0.969; at least one swapped block in 74% of query blocks) but the
   quality cost is 0.11% of an error the method already accepts. Say "perturbs selection"
   and give the bounded share.
3. **Any speed claim — remove entirely.** No latency was measured and the NVFP4 arms
   are simulated (§8). Speed requires a native fused sparse-NVFP4 kernel that does not
   exist here.
4. **Strong-survival framing — downgrade to partial.** F1 is `PARTIAL_SUPPORT` at all
   three sparsities.

## 10. Claims that may now be strengthened

1. **The matched-random contrast is a real boundary effect**, not an artifact: isolation
   holds at 12x–490x in F1 and 47x–310x in F2 for the degradation arms, with bootstrap
   intervals entirely above 10x in every case except the one noted in §8.
2. **fp32 scorer arithmetic is exactly free** — bit-identical masks, not approximately
   equal. This is a stronger statement than "small effect" and it is directly useful:
   an implementer can drop the scorer to fp32 with provably zero routing change.
3. **The mechanism is stable**, across three seeds (three significant figures of
   agreement) and across a 2.31x token-count change (magnitude decreasing).
4. **Study 1's numbers reproduce** through an independent implementation at all three
   sparsities (§5b).
5. **The gate mapping is verified by falsification**, in 129,600/129,600 records.

## 11. Every new number with source, run ID and n

| Number | Value | Source | Run ID | n |
|---|---|---|---|---|
| F1 Jaccard, fp32 scorer, exact Q/K, sp 0.90 | 1.000000 (bit-identical) | `tables/f1_full/table1_arm_headline.csv` | f1-full | 129,600 cells |
| F1 Jaccard, NVFP4-like scorer, NVFP4 Q/K, sp 0.90 | 0.8849 | same | f1-full | 129,600 |
| F1 worst damage share, sp 0.90 | 7.000e-03 [6.609e-03, 7.473e-03] | `raw/f4_gates.json` | f1-full | 129,600 |
| F1 isolation range, sp 0.90 | 12.2 – 490 | `tables/f1_full/table1_arm_headline.csv` | f1-full | 1,555,200 total |
| F1 R1 vs study 1, sp 0.80/0.90/0.95 | 0.9812/0.9751/0.9646 vs 0.9807/0.9738/0.9611 | `tables/f1_full/table6_r1_vs_study1_baseline.csv` | f1-full | 129,600 per cell group |
| F2 Jaccard, NVFP4 routing, sp 0.90 | 0.9687 | `tables/f2_full/table1_vsa_arm_headline.csv` | f2-full | 129,600 |
| F2 damage share, NVFP4 routing, sp 0.90 | 1.118e-03 [1.059e-03, 1.183e-03] | `raw/f4_gates.json` | f2-full | 129,600 |
| F2 isolation, NVFP4 routing, sp 0.90 | 225 [124, 707] | `raw/f4_gates.json` | f2-full | 129,600 |
| F2 unresolved arm isolation, sp 0.80/0.90/0.95 | 9.04 [6.62, 11.99] / 9.97 [8.17, 12.04] / 10.49 [8.68, 12.45] | `raw/f4_gates.json` | f2-full | 129,600 each |
| F2 fp64 rescue signed excess, sp 0.80/0.90/0.95 | +9.05e-04 / +1.303e-03 / +1.513e-03 | `tables/f2_full/summary.json` | f2-full | 129,600 each |
| F2 fp64 rescue win rate, sp 0.90 | 9.9% of cells | same | f2-full | 129,600 |
| F2 gate invariant | 129,600 / 129,600 masks bit-identical | `raw/f2_full_validation.json` | f2-full | 129,600 |
| F2 kernel budget deviation | 5,376 selector rows; worst cell 6.68e-04 of rows | `raw/f2_full_validation.json` | f2-full | 1,166,400 |
| F3A damage share by seed | 7.000e-03 / 6.991e-03 / 6.959e-03 | `tables/f3a/table2_seed_verdicts.csv` | f1-full + f3a-seeds | 129,600 per seed |
| F3A min isolation by seed | 12.18 / 11.88 / 11.62 | same | f1-full + f3a-seeds | 129,600 per seed |
| F3B token counts | 32,760 → 75,600 (2.3077x) | `tables/f3b/table2_f3c_required_metrics.csv` | f3b-720p | 259,200 |
| F3B damage share ratio vs baseline | 0.56 – 0.82 across arms | `tables/f3b/table3_paired_arm_comparison.csv` | f3b-720p | 64,800 per arm |
| F4.5 native-vs-simulated NVFP4 | median 9.80e-03, p90 1.07e-02, max 1.10e-02 | `raw/f4_representation_fidelity.json` | f45-capture | 30 real tensors |
| F4.5 elements bit-identical | median 99.95% | same | f45-capture | 30 tensors |
| F4.5 pooled-vector disagreement | median 1.39e-03, max 2.84e-03 | same | f45-capture | 30 tensors |
| Total validated records | 4,017,600 | `raw/f{1,2,3a,3b}*_validation.json` | all | — |

Bootstrap intervals: 4000 resamples, seed 20260816, resampling prompts
(`raw/f4_gates.json`, `configs/sparsefp4_stats.py`).
