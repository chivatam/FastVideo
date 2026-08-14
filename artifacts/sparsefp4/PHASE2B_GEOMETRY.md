# Phase 2B — the geometry generalization control

**Status:** complete. **Verdict: the near-tie mechanism generalizes to VSA's
deployed cube geometry, and gets *stronger* there.**

Every number in Phases 1 and 2 was measured at raster-order **128x64** blocks.
FastVideo's deployed sparse backend, VSA, does not use that geometry: it uses
**(4,4,4) spatio-temporal cubes of 64 tokens with `block_q == block_k`**, and it
re-orders tokens into tile-contiguous order first, which is a different
token-to-block assignment and not merely a different block size
(`STATUS.md` trap 3). This control re-ran the decisive subset — `C`, `D`,
`C_rand` plus the mechanism records — at VSA's real geometry, and at a
`64x64-raster` intermediate that separates *block size* from *token ordering*.

The headline result is that the mechanism is not a raster artifact. At VSA's cube
geometry the equal-magnitude random contrast costs **76x / 45x / 36x** more
output error than letting NVFP4's quantization pick the same number of blocks, at
sparsity 0.80 / 0.90 / 0.95 — against **27x / 22x / 10x** at 128x64. Swapped
blocks carry **11.1x / 8.1x / 6.9x** less attention mass than blocks both masks
agree on, against 5.7x / 4.7x / 3.8x at 128x64. Both quantities move in the
direction that makes the negative result *more* robust for the deployed path, not
less.

Phase 2's §7 tie-count discrepancy with Phase 1 is also resolved, and it was a
counting-convention difference, not a measurement disagreement (§6).

---

## 1. What ran

| Item | Value |
|---|---|
| Branch / base commit | `exp/sparsefp4-mask-stability` / `8208536cd1db7a1d32b68aaa6a679953ae23ab8b` |
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 480x832x81, 50 steps, guidance 3.0, seed 1234, `sp_size=1` |
| Attention the model consumed | **dense BF16 (configuration A), byte for byte** — every arm shares one denoising trajectory, so all comparisons are exactly paired |
| Hardware | 8x NVIDIA B200 (sm_100), `FASTVIDEO_FA4=1` |
| Prompts / layers / timesteps / heads / CFG | 10 x 17 x 5 x 12 x 2, identical to Phase 2 |
| Sparsity | `0.80`, `0.90`, `0.95` |
| **Block-score dtype** | **`fp64` everywhere** (`STATUS.md` trap 8), enforced as a hard gate in the analysis verifier |
| Paired cells per (geometry, sparsity) | **20,400** = 10 x 17 x 12 x 5 x 2 |
| Verification | **PASS**, zero failures — `artifacts/sparsefp4/tables/phase2b_geometry/verification.json` |

### 1.1 The three geometry arms

| Arm | Run ID | Records | `n_q x n_k` blocks | Token order | Padded `seq_len` | Native / simulated |
|---|---|---|---|---|---|---|
| `128x64-raster` | `20260814-025500-8208536-p2-main` (Phase 2, re-aggregated) | 615,380 | 256 x 512 | raster | 32,768 | native compute, native router |
| `64x64-raster` | `20260814-035500-8208536-p2b-64x64-raster` | 375,680 | 512 x 512 | raster | 32,768 | native compute, native router |
| `64x64-cube` | `20260814-032500-8208536-p2b-64x64-cube` | 375,680 | 624 x 624 | **VSA (4,4,4) tile-contiguous** | **39,936** | native compute, native router |

`seq_len` is 32,760 real tokens in all three, asserted constant. The
`64x64-raster` arm exists solely to separate the two confounded factors: it
changes the block size from 128x64 to 64x64 while holding raster token order, so
any residual difference at `64x64-cube` is attributable to the re-tiling.

### 1.2 Arms measured

| Config | Attention compute | Mask source | Isolates | Labels |
|---|---|---|---|---|
| **A** | dense BF16 (FA4) | — | reference | native |
| **C** | sparse BF16 (Triton) | BF16 scores | sparsification only | native compute, native router |
| **C_null** | sparse BF16 | BF16 scores, independently re-derived | **null control — must be an exact identity** | native compute, native router |
| **D** | sparse BF16 | NVFP4 scores | + NVFP4 wrong-mask | native compute, native router |
| **C_rand** | sparse BF16 | BF16 scores, *N* blocks swapped at random | + equal-magnitude **random** wrong-mask | native compute, **synthetic control** |

`B`/`B_sim`/`D8`/`E`/`F8`/`F16` were not re-run: Phase 2 established the
quantization and H3 arms at 128x64, and this control is about the mask geometry,
not the compute precision. **No latency claim is made anywhere in this report** —
sparse NVFP4 compute has no native kernel in this repository, and even the BF16
sparse arms here are diagnostic.

**The null control is new and stronger than Phase 1's.** Phase 1's bf16-vs-bf16
control was a scorer identity. `C_null` re-derives the BF16 mask from an
independent second quantizer call and pushes it through the block-sparse kernel,
so it gates the *whole executed path* at each geometry. It is exact:
**61,200 paired cells per geometry, 0 deviations in `rel_l2`, 0 deviations from
Jaccard 1.0.** Per the trap-8 correction this is reported as an
arithmetic-identity check and does **not** certify scorer resolution.

---

## 2. Exact commands

```bash
source artifacts/sparsefp4/configs/env.sh    # FV_PYTHON, FASTVIDEO_FA4=1, CUDA_HOME, HF_HOME

# Correctness gate — must PASS before any measurement (see section 3)
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_selftest.py \
  --out artifacts/sparsefp4/raw/phase2b_selftest.json

# The two new geometry arms, 10 prompts sharded over 8 B200s, nohup per shard
bash artifacts/sparsefp4/configs/phase2b_launch.sh 20260814-035500-8208536-p2b-64x64-raster 64x64-raster
bash artifacts/sparsefp4/configs/phase2b_launch.sh 20260814-032500-8208536-p2b-64x64-cube  64x64-cube
#   -> ARMS="A C C_null D C_rand", --score-dtype float64, --no-activation-stats,
#      --tie-diagnostic-geometries "128x64-raster 64x64-raster 64x64-cube",
#      raw on /mnt/scratch, logs under artifacts/sparsefp4/logs/<run_id>/

# Three-geometry comparison (verification gate runs first and gates the rest)
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_geometry_analyze.py \
  --raw 128x64-raster=artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \
  --raw 64x64-raster=artifacts/sparsefp4/raw/20260814-035500-8208536-p2b-64x64-raster \
  --raw 64x64-cube=artifacts/sparsefp4/raw/20260814-032500-8208536-p2b-64x64-cube \
  --out-tables artifacts/sparsefp4/tables/phase2b_geometry \
  --out-figures artifacts/sparsefp4/figures/phase2b_geometry

# Phase-1-vs-Phase-2 tie-count reconciliation, from archived records (no GPU)
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_tie_reconcile.py \
  --fp32-raw   artifacts/sparsefp4/raw/20260814-013449-8208536-p1-stage1 \
  --fp64-raw   artifacts/sparsefp4/raw/20260814-015113-8208536-p1-stage1-fp64score \
  --phase2-raw artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \
  --out-tables artifacts/sparsefp4/tables/phase2b_geometry

# Figures (each PNG's plotted values are the CSV written next to it)
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_figures.py \
  --figures artifacts/sparsefp4/figures/phase2b_geometry
```

---

## 3. How the padding was handled, and how it was checked

The cube geometry is the crux, and its padding is the part that could silently
invalidate everything. VSA's own utilities were used rather than a hand-rolled
tiling: `get_tile_partition_indices`, `construct_variable_block_sizes`,
`get_non_pad_index` and `get_reverse_tile_partition_indices`
(`fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py`, re-exported through
`fastvideo/attention/backends/video_sparse_attn.py`). At Wan's DiT token grid the
result is:

```
latent 21 x 60 x 104, patch (1,2,2) -> DiT grid (21, 30, 52) = 32,760 tokens
tiles: ceil(21/4) x ceil(30/4) x ceil(52/4) = 6 x 8 x 13 = 624
padded length: 624 x 64 = 39,936  ->  7,176 pad slots
tile token counts: {8, 16, 32, 64}   (boundary tiles are short; none is empty)
```

Four separate mechanisms keep the 7,176 pad slots inert, each **checked** rather
than assumed:

1. **Pad slots hold zeros.** Tokens are gathered in tile order and scattered into
   a zero-filled buffer (`to_block_layout`, byte-identical in structure to
   `VideoSparseAttentionImpl.tile`). Gate check: `pad_slots_all_zero = true`.
2. **Pad slots cannot pollute a mean-pooled router score.** Pooling divides by
   each block's *true* token count, not by 64 (`pool_geometry_blocks`). Gate
   check: pooled block vector vs an independently computed mean over that block's
   real tokens only — max abs difference **0.0** at cube geometry (and 3.0e-08 at
   raster, i.e. fp32 rounding).
3. **No block is all-pad, so top-k cannot select a pad-only block as
   "important".** The smallest cube tile at this grid holds 8 real tokens. Gate
   check: `all_pad_blocks = 0`, recorded per geometry rather than argued.
4. **Pad slots cannot enter the softmax.** The block-sparse kernel masks padded
   columns per key block from `variable_block_sizes`
   (`block_sparse_attn_triton.py:310-321`). Gate check: **all 7,176 pad slots of
   V were perturbed by +100.0 and the output was bit-identical** — and the same
   check passed for every never-selected block.

Pad *query* rows are computed by the kernel and then dropped by
`from_block_layout`, which is VSA's own single combined gather
(`untile_combined_index = non_pad_index[reverse_tile_partition_indices]`), so they
never reach an error metric. The round trip raster → tile layout → raster is
**bit-identical** at all three geometries, and `n_valid_tokens = 32,760` in each.

### 3.1 The correctness gate

`artifacts/sparsefp4/raw/phase2b_selftest.json` — **PASS, 0 failures.** The gate
the brief requires (a 64x64 cube-order mask executed on the kernel matches a
masked dense reference) is `geometry_mask_on_kernel_vs_masked_reference`, run at
Wan's real 32,760-token shape against an independent naive fp32 masked-softmax
reference that shares no code with the kernel:

| Geometry | s | k | rel-L2 vs masked reference | cosine | verdict |
|---|---|---|---|---|---|
| `128x64-raster` | 0.80 | 103 | 2.478e-03 | 0.9999970 | PASS |
| `128x64-raster` | 0.95 | 26 | 2.418e-03 | 0.9999971 | PASS |
| `64x64-raster` | 0.80 | 103 | 2.478e-03 | 0.9999970 | PASS |
| `64x64-raster` | 0.95 | 26 | 2.419e-03 | 0.9999971 | PASS |
| **`64x64-cube`** | **0.80** | **125** | **2.470e-03** | **0.9999970** | **PASS** |
| **`64x64-cube`** | **0.95** | **32** | **2.385e-03** | **0.9999973** | **PASS** |

The residual ~2.5e-03 is the BF16-vs-fp32 accumulation difference between the
Triton kernel and the fp32 reference, and it is identical across geometries — it
is not a geometry-dependent error. Attention-mass rows still integrate to 1.0
(max deviation 6.0e-08) at every geometry, so the mass measurements in §5 are
sound in the padded layout.

### 3.2 One real budget confound, measured not hidden

Sparsity is defined on the **block** axis. Cube tiles are variable-size, so an
equal block budget is **not** an equal token budget:

| Geometry | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| retained *block* fraction (all arms, by construction) | 0.2005 / 0.2003 | 0.1016 / 0.1010 | 0.0508 / 0.0513 |
| retained **token** fraction, `64x64-raster` | **0.2012** | **0.1016** | **0.0508** |
| retained **token** fraction, `64x64-cube` | **0.1912** | **0.0952** | **0.0478** |

The cube arm retains **5-6% fewer tokens** at the same nominal sparsity, because
top-k on mean-pooled scores mildly prefers the short boundary tiles. This is a
property of VSA's geometry, not of this harness, and it biases the cube arm
*against* itself on absolute error. It cannot explain the mechanism results,
which are all paired *within* a geometry against that geometry's own `C`.
`retained_token_fraction` is recorded per row; it is absent for the
`128x64-raster` arm because Phase 2's records predate the field (the raster value
is 0.2012 / 0.1016 / 0.0508 by construction, identical to `64x64-raster`).

---

## 4. The three-geometry comparison

All medians, with `n`. Full table with IQR:
`artifacts/sparsefp4/tables/phase2b_geometry/table1_three_geometry_headline.csv`.
**Every row is `native` compute with a `native` router except `C_rand`, whose
router is a `synthetic_control`; no row carries a latency claim.**

### 4.1 The quantities Phase 2 reported, side by side

`n = 20,400` paired cells for every rel-L2 / Jaccard cell;
`n` for the mass and margin columns is the number of sampled query-block
observations that had at least one swap, given per row.

| Geometry | s | mask Jaccard vs BF16 (median, IQR) | frac q-blocks changed | swaps per q-block | mass of swapped-out block | mass of agreed block | **agreed / swapped** | score margin of swap | `D - C` | `C_rand - C` | **`C_rand`/`D`** | frac cells random worse | n (cells) | n (q-blocks) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `128x64-raster` | 0.80 | 0.97973 (0.0139) | 0.789 | 1.428 | 0.001436 | 0.008221 | **5.72x** | 0.00484 | 3.10e-05 | 8.36e-04 | **27.0x** | 63.2% | 20,400 | 27,257 |
| `128x64-raster` | 0.90 | 0.97229 (0.0188) | 0.656 | 0.997 | 0.002833 | 0.013197 | **4.66x** | 0.00468 | 3.33e-05 | 7.20e-04 | **21.7x** | 61.6% | 20,400 | 22,674 |
| `128x64-raster` | 0.95 | 0.96198 (0.0225) | 0.504 | 0.662 | 0.004628 | 0.017734 | **3.83x** | 0.00423 | 9.52e-05 | 9.59e-04 | **10.1x** | 61.6% | 20,400 | 17,430 |
| `64x64-raster` | 0.80 | 0.97861 (0.0144) | 0.770 | 1.301 | 0.001322 | 0.008245 | **6.24x** | 0.00501 | 2.03e-05 | 9.57e-04 | **47.1x** | 64.7% | 20,400 | 27,876 |
| `64x64-raster` | 0.90 | 0.97149 (0.0189) | 0.615 | 0.872 | 0.002702 | 0.013324 | **4.93x** | 0.00478 | 3.69e-05 | 8.60e-04 | **23.3x** | 64.1% | 20,400 | 23,013 |
| `64x64-raster` | 0.95 | 0.96183 (0.0229) | 0.459 | 0.582 | 0.004352 | 0.018534 | **4.26x** | 0.00432 | 5.04e-05 | 1.22e-03 | **24.2x** | 65.8% | 20,400 | 17,428 |
| **`64x64-cube`** | 0.80 | 0.97937 (0.0096) | 0.830 | 1.498 | **0.000627** | 0.006929 | **11.05x** | 0.00492 | 1.84e-05 | 1.39e-03 | **75.6x** | 67.7% | 20,400 | 29,597 |
| **`64x64-cube`** | 0.90 | 0.97226 (0.0141) | 0.688 | 1.036 | **0.001440** | 0.011673 | **8.11x** | 0.00437 | 2.90e-05 | 1.29e-03 | **44.6x** | 67.7% | 20,400 | 25,250 |
| **`64x64-cube`** | 0.95 | 0.96448 (0.0186) | 0.510 | 0.682 | **0.002447** | 0.016824 | **6.88x** | 0.00418 | 4.04e-05 | 1.44e-03 | **35.7x** | 69.8% | 20,400 | 19,899 |

![Isolation ratio at three geometries](figures/phase2b_geometry/fig2_random_over_quantization_by_geometry.png)

![Block mass and margin at three geometries](figures/phase2b_geometry/fig4_block_mass_by_geometry.png)

![Mask stability at three geometries](figures/phase2b_geometry/fig3_mask_jaccard_by_geometry.png)

### 4.2 Reading the table

**Mask stability is essentially geometry-invariant.** Median Jaccard agrees to
within 0.003 across all three geometries at every sparsity (0.9786-0.9794 at 0.80;
0.9715-0.9723 at 0.90; 0.9618-0.9645 at 0.95), and the cube arm is the *most*
stable at 0.95. Churn is likewise flat: 0.46-0.83 of query blocks touched, ~0.6-1.5
blocks swapped per query block. Phase 1's H1 finding therefore transfers to the
deployed geometry as a magnitude, even though the block indices are not comparable
between geometries. Note the cube arm's Jaccard IQR is **narrower** at every
sparsity (0.0096 vs 0.0139 at 0.80), i.e. cube geometry makes routing *more*
uniformly stable across cells, not less.

**Absolute sparsification error is lower at cube geometry** (`C` = 0.0846 vs
0.1058 at 0.80), which is the interesting engineering side-effect: grouping
spatio-temporally coherent tokens into a block makes a fixed block budget capture
more of the attention distribution. Total mass retained by the BF16 top-k at
sparsity 0.90 is 0.725 at cube vs 0.676 at 128x64. Some of that is the mildly
different token budget of §3.2, but it points the same way.

**The mechanism strengthens.** Both of the mechanism's load-bearing quantities
improve monotonically as the geometry moves toward VSA's:

| Quantity, s=0.90 | 128x64-raster | 64x64-raster | 64x64-cube |
|---|---|---|---|
| agreed / swapped-out mass ratio | 4.66x | 4.93x | **8.11x** |
| `C_rand` / `D` excess-error ratio | 21.7x | 23.3x | **44.6x** |
| wrong-mask excess `D - C` | 3.33e-05 | 3.69e-05 | **2.90e-05** |
| `D - C` as a share of `C` | 0.019% | 0.023% | **0.021%** |

**Where the swaps go is unchanged, and it is the boundary.** The normalized score
margin of the swapped pair is 0.42%-0.50% of the score spread at *every* geometry
— the swaps are near-ties everywhere. And the swapped-out block still sits far
closer to the excluded population than to the retained one: at cube geometry,
s=0.90, a swapped block carries 0.00144 against 0.01167 for an agreed block and
0.00049 for an average excluded block.

**The wrong-mask term stays three-to-four orders below the sparsification term.**
At cube geometry `D - C` is 2.9e-05 against `C` = 0.137, i.e. **0.021%**. The
per-query-block wrong-mask excess is **exactly 0.0 at the median** in all three
geometries, and its largest regional median anywhere in this study is 7.6e-05
(`128x64-raster`, `broad` region).

---

## 5. Verdict: does the mechanism generalize?

**Yes — and it is stronger at VSA's deployed cube geometry than at the raster
geometry the study was built on.**

Both pre-stated criteria are met with margin:

1. **The `C_rand`/`D` ratio stays large.** It is **75.6x / 44.6x / 35.7x** at cube
   geometry against 27.0x / 21.7x / 10.1x at 128x64 — larger at every sparsity,
   and most emphatically at 0.95, where the 128x64 arm was weakest (10.1x → 35.7x).
   The fraction of cells where the random perturbation is worse also rises
   (61.6% → 69.8%).
2. **Swapped-block mass stays far below agreed-block mass.** The ratio *widens*
   from 3.8-5.7x to **6.9-11.1x**. A swapped block at cube geometry carries about
   half the mass it did at 128x64 (0.00144 vs 0.00283 at s=0.90) while agreed
   blocks are only slightly lighter, so the gap grows on both ends.

The mechanistic reading is that cube tiles group spatio-temporally coherent
tokens, which **separates** the block scores rather than clustering them: coherent
tiles are either clearly relevant or clearly not, so the population of genuinely
near-tied blocks is pushed further into the tail of the mass distribution. This is
one of the two outcomes the brief flagged as possible, and it is the favorable
one. It was checked, not assumed.

There is **no material weakening anywhere** in the cube results. The one number
that could be read as a caution is that cube geometry has slightly *more* churn
(1.036 vs 0.997 swaps per query block at s=0.90, and 0.830 vs 0.789 of query blocks
touched at s=0.80) — but that is the same "more churn, less consequence" pattern
Phase 2 §6 found across layers: the extra swaps land on even less important
blocks, so `D - C` is *lower* at cube geometry despite more of them.

### 5.1 Block size or token ordering — which mattered more?

**Token ordering, on the mass-based measures — but the answer depends on which
quantity you ask about, and the honest version says so.** The `64x64-raster`
intermediate is what makes this separable; without it the cube result would be
uninterpretable.

The two **mass** quantities — the direct measurement of the mechanism — are
consistently and heavily ordering-dominated, at every sparsity:

| Quantity | s | 128x64→64x64 raster (**block size**) | 64x64 raster→cube (**token ordering**) |
|---|---|---|---|
| agreed / swapped mass ratio | 0.80 | 5.72 → 6.24 (**+9%**) | 6.24 → 11.05 (**+77%**) |
| | 0.90 | 4.66 → 4.93 (**+6%**) | 4.93 → 8.11 (**+64%**) |
| | 0.95 | 3.83 → 4.26 (**+11%**) | 4.26 → 6.88 (**+61%**) |
| mass of a swapped-out block | 0.80 | 0.001436 → 0.001322 (−8%) | 0.001322 → 0.000627 (**−53%**) |
| | 0.90 | 0.002833 → 0.002702 (−5%) | 0.002702 → 0.001440 (**−47%**) |
| | 0.95 | 0.004628 → 0.004352 (−6%) | 0.004352 → 0.002447 (**−44%**) |

Ordering is roughly **6-8x more influential than block size** on both, in the same
direction, at all three sparsities. Halving the query-block size does almost
nothing to where the swaps land; re-tiling into spatio-temporal cubes does.

The **`C_rand`/`D` error ratio** tells a messier story and I am not going to
smooth it:

| s | 128x64→64x64 raster (block size) | 64x64 raster→cube (ordering) |
|---|---|---|
| 0.80 | 27.0 → 47.1 (**+74%**) | 47.1 → 75.6 (**+61%**) |
| 0.90 | 21.7 → 23.3 (**+8%**) | 23.3 → 44.6 (**+91%**) |
| 0.95 | 10.1 → 24.2 (**+140%**) | 24.2 → 35.7 (**+48%**) |

Here block size matters as much as ordering at s=0.80 and *more* at s=0.95. The
likely reason is arithmetic rather than mechanistic: this ratio is a quotient of
two medians that are themselves ~1e-05 and ~1e-03, so it inherits far more
sampling noise than the mass measurements do, and the 128x64 s=0.95 denominator
(`D - C` = 9.5e-05) is the single largest wrong-mask excess in the whole study and
therefore the least stable divisor. The numerator, `C_rand - C`, is far steadier
across geometries (7.2e-04 to 1.4e-03).

**The defensible statement:** token ordering dominates on the quantities measured
directly and with high `n` (block mass, `n` = 17k-30k query-block observations per
cell); on the derived error ratio both factors contribute and their split is not
resolved by this data. Both factors point the same way in every one of the 18
cells, so the *direction* of the conclusion does not depend on the attribution.

Two smaller confirmations that this is not a budget artifact: the `64x64-raster`
arm has **identical** `n_k_blocks` (512) to `128x64-raster` and therefore identical
`k` (103/52/26), so its mass and error differences cannot come from a different
budget; and the cube arm reaches *lower* error while retaining *fewer* tokens
(§3.2).

---

## 6. Resolving the Phase 1 / Phase 2 tie-count contradiction

Phase 1 §7.1 reported **~104-115 exact boundary ties per cell** with fp32 block
scores and concluded that fp32 "penalizes the FP8 arm 1.6x harder than the NVFP4
arm". Phase 2 §7, on real Wan Q/K at the same 128x64 geometry, measured **~1,400
ties per cell** and near-identical counts across routers (1429/1430/1430), and
recorded that it could not reproduce the asymmetry.

Both phases are right about what they measured. The apparent contradiction is two
separate bookkeeping problems, and neither is a measurement disagreement.

### 6.1 The ~13x scale gap is the counting denominator

Read from the code rather than the reports:

- Phase 1 (`routing_probe_attn.compare_masks`) computes
  `boundary_ties = (candidate_margin_raw == 0).sum(dim=-1)` where
  `candidate_margin_raw` is `s_(k) - s_(k+1)` per query block. The reduction is
  over `n_q_blocks` **only**, so the emitted value is **per (cell, head)** —
  out of 256 query blocks.
- Phase 2 (`precision_sparse_attn._emit_score_resolution_row`) computed
  `(ordered[..., k-1] == ordered[..., k]).sum()` over the whole
  `[head, query_block]` grid, so the emitted value is **per cell, summed over all
  12 heads** — out of 12 x 256 = 3,072.

Recomputed from the archived records
(`table6_tie_count_denominator.csv`, s=0.80, NVFP4 router, fp32 scores):

| Source | as reported | counting unit | rescaled to the other unit | tie **rate** |
|---|---|---|---|---|
| Phase 1 Stage 1 | **115.0** | per (cell, head), of 256 q-blocks | x12 heads = **1,380** | **0.449** of query blocks |
| Phase 2 main | **1,430.0** | per cell, of 3,072 (head, q-block) pairs | ÷12 heads = **119.2** | **0.465** of pairs |

The two figures agree to **4%** once expressed on the same denominator. This is
one quantity measured twice, not two findings. Confirmed independently by the new
`tie_diagnostic` record, which emits both denominators at every geometry from the
same scores: at `128x64-raster`, s=0.80, `n = 1,700` cells, the per-head median is
**110-111** (matching Phase 1's 104-115 range) and the per-cell median is
**1,429-1,430** (matching Phase 2 exactly).

Block geometry is a second-order contributor and moves the count the other way
from the direction that would have explained the gap — more query blocks means
more ties, not fewer:

| Diagnostic geometry | `n_q_blocks` | ties per (cell, head), fp32 | ties per cell, fp32 | tie rate | ties, fp64 |
|---|---|---|---|---|---|
| `128x64-raster` | 256 | 110-111 | 1,429-1,430 | 0.430-0.434 | **0** |
| `64x64-raster` | 512 | 220-222 | 2,850-2,855 | 0.430-0.434 | **0** |
| `64x64-cube` | 624 | 292-294 | 3,634-3,651 | 0.468-0.470 | **0** |

`n = 1,700` cells per row. Table:
`artifacts/sparsefp4/tables/phase2b_geometry/table8_tie_diagnostic_by_geometry.csv`.

Two things follow. First, **the tie *rate* — 43% of query blocks at raster, 47% at
cube — is essentially geometry- and router-invariant**, exactly as expected if
ties come from the score's large common-mode offset rather than from anything the
router precision does. Second, **fp64 produces exactly zero ties at every
geometry, every router and every sparsity**, so trap 8's correction holds
unchanged for the cube geometry and the Phase 2B numbers inherit it.

### 6.2 The 1.6x FP8 asymmetry is real, and Phase 2 tested a different quantity

Phase 1's asymmetry claim is **not** about tie counts. It is about how much the
*measured median Jaccard* moves when the scorer goes fp32 → fp64 — Phase 1's own
table quotes "+0.0048 vs +0.0031 at sp 0.90". Phase 2 checked whether the *tie
counts* and the *per-arm fp32-vs-fp64 flip rates* differed across routers, found
they did not, and reported a failure to reproduce. Those are different quantities,
so the non-reproduction was never evidence against the claim.

Recomputed as an exactly paired per-cell comparison between Phase 1's fp32 Stage-1
run and its fp64 Stage-1 control, `n = 36,000` paired cells per arm per sparsity
(`table7_fp32_vs_fp64_jaccard_shift.csv`):

| s | router | median Jaccard, fp32 scorer | median Jaccard, fp64 scorer | shift | **FP8 / NVFP4 shift ratio** | n |
|---|---|---|---|---|---|---|
| 0.80 | bf16 (null) | 1.000000 | 1.000000 | 0.000000 | — | 36,000 |
| 0.80 | fp8_e4m3 | 0.992142 | 0.995006 | **+0.002864** | **1.604x** | 36,000 |
| 0.80 | nvfp4 | 0.979802 | 0.981587 | **+0.001785** | | 36,000 |
| 0.90 | fp8_e4m3 | 0.988646 | 0.993411 | **+0.004765** | **1.550x** | 36,000 |
| 0.90 | nvfp4 | 0.972732 | 0.975807 | **+0.003074** | | 36,000 |

**The 1.6x is real** (1.604x at 0.80, 1.550x at 0.90), reproduced to three
significant figures from Phase 1's archived data, and it is not in conflict with
Phase 2's tie-count symmetry. Both are expected simultaneously: fp32's arithmetic
noise is a property of the score magnitude and is therefore arm-independent (equal
tie counts, equal per-arm flip rates — Phase 2 correct), but its *effect on the
overlap measured against the BF16 reference mask* is larger for the arm whose
genuine perturbation is smaller, because a smaller true disagreement leaves more
room for noise-induced flips to land on fresh blocks rather than on blocks already
in disagreement. FP8's true deficit at s=0.90 is 0.0109 against NVFP4's 0.0262 —
roughly 2.4x smaller — which is the right order to produce a ~1.6x larger relative
contamination.

### 6.3 The statement to carry into the final report

> fp32 block scores manufacture exact top-k boundary ties in **~43-47% of query
> blocks** at every geometry and for every router precision, and fp64 removes all
> of them. Phase 1 reported this count per `(cell, head)` (~110 of 256 query
> blocks) and Phase 2 reported it per cell summed over 12 heads (~1,430 of 3,072);
> the two agree to 4% on a common denominator and are one measurement, not two.
> The tie *count* is router-independent, but the resulting *bias* is not: moving
> the scorer from fp32 to fp64 raises the measured mask overlap of the FP8 router
> **1.6x more** than the NVFP4 router (+0.0048 vs +0.0031 at sparsity 0.90,
> `n = 36,000` paired cells each), so an fp32 scorer would have biased the
> FP8-vs-NVFP4 router comparison against H3. Phase 2's inability to reproduce the
> asymmetry was a category error — it compared tie counts and per-arm flip rates,
> not the Jaccard shift the claim is about. All Phase 2 and Phase 2B numbers use
> fp64 scores, so neither is affected.

Nothing here is left unresolved. `STATUS.md` trap 8 should be read as two claims
with different scopes — a router-independent tie count and a router-dependent
Jaccard bias — and Phase 2 §7's deviation note is superseded by §6.2 above.

---

## 7. Threats to validity

| Threat | Status |
|---|---|
| **Cube geometry is scored and executed by this harness, not by VSA's own kernel** | Real, and the honest limit of this control. The *token-to-block assignment*, *tile ordering*, *ragged tile sizes* and *padded layout* are VSA's own utilities, and the executed mask is verified against an independent masked-dense reference at Wan's real shape (§3.1). But the arms run on the Triton `block_sparse_attn` kernel with a mean-pooled research scorer, **not** through `VideoSparseAttentionImpl` with VSA's own coarse scorer and gating. A VSA-native arm could differ; what is established is that the *geometry* is not what makes the mechanism work. |
| **Equal block budget is not an equal token budget at cube geometry** | Real, quantified in §3.2 (cube retains 5-6% fewer tokens). Biases the cube arm against itself on absolute error; cannot affect the paired within-geometry mechanism ratios. |
| **`C_rand`/`D` ratio is a quotient of two small medians** | Real. The numerator is stable across geometries; the denominator (~1e-05 to 1e-04) is not, which is why §5.1 declines to attribute that ratio's change to block size vs ordering. The mass-based measures, with `n` = 17k-30k per cell, carry the attribution instead. |
| **`C_rand` is synthetic** | By design, and labeled `router_native_or_simulated = "synthetic_control"` in every raw record. Its role is to hold the perturbation *magnitude* fixed and vary only *where* it lands. |
| **Single model, single resolution, single seed** | Real, unchanged from Phase 2. Wan2.1-T2V-1.3B at 480x832x81, seed 1234, 10-prompt development set. |
| **Error measured at the attention output, single-step** | Deliberate and unchanged: the trajectory is dense BF16 so all 20,400 cells are exactly paired. These are single-step errors and do not capture accumulation over 50 steps. |
| **Mechanism records sampled on 6 layers x 2 timesteps x 12 query blocks** | Same sampling lattice as Phase 2, so the geometries are compared like for like. 17k-30k query-block observations per (geometry, sparsity). |
| **No latency claim** | None is made. Sparse NVFP4 has no native kernel here; the BF16 sparse arms are diagnostic. Every raw row carries `numerical_only` and `native_latency_claim_allowed`. |

## 8. Exclusions

Recorded per the no-silent-drops rule.

| Exclusion | Count | Reason |
|---|---|---|
| Records excluded from analysis | **0** | Nothing filtered. All 1,366,740 records across the three arms were loaded and aggregated. `null_metric_exclusions = 0` in every arm. |
| Arms not re-run at the new geometries | `B`, `B_sim`, `D8`, `E`, `F8`, `F16` | Deliberate scoping per Phase 2 §10's own recommendation: this control is about mask geometry, not compute or router precision, and Phase 2 established those arms at 128x64. Consequence: **no H3 verdict is restated at cube geometry**, and §5's claim is about the mechanism, not about H3 directly. |
| Activation/saturation rows not re-emitted | — | `--no-activation-stats`. Phase 2 §6 already closed the saturation confound and those statistics are geometry-independent (they are properties of Q/K, not of blocking). |
| **First `64x64-raster` run superseded** | `20260814-032500-8208536-p2b-64x64-raster` (10 complete shards) | **Run-hygiene exclusion, not a result exclusion.** Its last two shards started after a mypy-driven type-annotation edit to `precision_sparse_attn.py` landed, so that run spans two code states. Rather than mix code versions within one run — the same call Phase 2 made for its first launch — all 10 prompts were re-run under the final code as `20260814-035500-8208536-p2b-64x64-raster`, and only the re-run appears in this report. The superseded run remains on scratch at `/mnt/scratch/sparsefp4/20260814-032500-8208536-p2b-64x64-raster`. The edit was annotation-only (a dict comprehension rewritten to satisfy mypy's index check; no arithmetic changed) and both runs wrote identical shard record counts, so the exclusion is hygiene rather than result-shopping. Verified rather than asserted: re-analyzing the superseded run reproduces the reported run's headline numbers **to every printed digit** (`C_rand`/`D` = 47.077 / 23.298 / 24.196; agreed/swapped mass = 6.2375 / 4.9305 / 4.2589), and its verification gate also reports PASS. |
| Per-row churn columns for the `128x64-raster` arm | — | `frac_query_blocks_changed` and `blocks_swapped_per_query_block` are emitted per error row only from Phase 2B onward. For the Phase 2 arm they are recomputed from that run's mechanism records and the row carries `churn_source = "mechanism_sample"`, which is a 12-query-block deterministic lattice rather than the full set. The Phase 2B arms carry `churn_source = "error_rows"`. **The three geometries' churn numbers are therefore not measured identically**, and this is why §4.2 reads churn as "flat" rather than quoting a precise ordering. |
| Uncompressed raw JSONL kept off the repo volume | 1.1 GB | Root volume has 3.6 GB free. Gzipped, integrity-verified (`gzip -t`) and line-count-verified copies (375,680 records each, 64 MB total) are under `artifacts/sparsefp4/raw/`; the uncompressed originals are on `/mnt/scratch`, which is **ephemeral instance store**. |

All 10 shards of both reported runs wrote **exactly identical record counts
(37,568 each; 375,680 per arm)**, verified by decompressing and re-counting every
archived line — itself a check that no layer/timestep/head cell fell off the
research backend. Note that `run_summary_p*.json` reports 37,563-37,568 for some
shards: the driver counts lines before the writer's final flush, so those figures
undercount by up to five lines. The archived shards are the authoritative count
and every table in this report was regenerated from them.

## 9. What this means for the study's headline claim

The claim survives, and it now covers the deployed geometry. **Required scoping, in
one line:**

> Measured on Wan2.1-T2V-1.3B at 480x832x81, single-step attention-output error
> against dense BF16, at three block geometries including VSA's deployed
> 64-token (4,4,4) spatio-temporal cubes: NVFP4-induced routing swaps land on
> near-degenerate top-k boundaries and are 27-76x cheaper than an
> equal-magnitude random perturbation, so sparse-attention routers can be run at
> the compute precision — a numerical result with no latency claim, and with the
> cube arm executed on a research block-sparse kernel rather than through VSA's
> own scorer.

Concretely, what may and may not be said:

- **May say:** the mechanism holds at 128x64 raster, 64x64 raster **and** VSA's
  64-token cube geometry, and is *strongest* at the cube geometry
  (`C_rand`/`D` = 36-76x, agreed/swapped mass = 6.9-11.1x).
- **May say:** the result is not an artifact of the raster diagnostic geometry.
  This was the first item on Phase 2's own list of threats and it is now closed.
- **May say:** token ordering, not block size, is what makes cube geometry
  favorable — on the mass measurements. The error-ratio attribution is not
  resolved (§5.1).
- **Must not say:** that H3 was re-tested at cube geometry. It was not; the
  NVFP4-compute arms were deliberately not re-run.
- **Must not say:** anything about VSA end-to-end quality, VSA's own scorer, or
  any latency at any geometry.
- **Must not say:** that cube geometry is *better* for sparse attention generally.
  Its lower `C` here is at a 5-6% smaller retained token budget and on a single
  model at a single resolution.

## 10. Artifacts

| Kind | Path |
|---|---|
| Raw JSONL, gzipped, integrity-verified | `artifacts/sparsefp4/raw/20260814-035500-8208536-p2b-64x64-raster/`, `.../20260814-032500-8208536-p2b-64x64-cube/` |
| Per-run configs / summaries / verification | same directories, `phase2_config_p*.json`, `run_summary_p*.json`, `verification.json` |
| Correctness gate (29 checks, PASS) | `artifacts/sparsefp4/raw/phase2b_selftest.json` |
| Tables (CSV + Markdown) | `artifacts/sparsefp4/tables/phase2b_geometry/table1`-`table8`, `verification.json`, `summary.json`, `tie_reconciliation_summary.json` |
| Figure values (CSV, archival) + PNGs | `artifacts/sparsefp4/figures/phase2b_geometry/fig1`-`fig5` |
| Per-shard logs | `artifacts/sparsefp4/logs/20260814-0*-p2b-*/gpu*_p*.log`, `artifacts/sparsefp4/logs/phase2b/` |
| Geometry abstraction + kernel numerics | `fastvideo/attention/backends/sparsefp4_numerics.py` (`BlockGeometry`, `raster_geometry`, `cube_geometry`, `to_block_layout`, `from_block_layout`, `pool_geometry_blocks`, `retained_token_fraction`) |
| Backend | `fastvideo/attention/backends/precision_sparse_attn.py` (geometry is a run parameter; `C_null` arm; `tie_diagnostic` record) |
| Driver / launcher / analysis / figures / reconciliation | `artifacts/sparsefp4/configs/phase2_run.py`, `phase2b_launch.sh`, `phase2b_geometry_analyze.py`, `phase2b_figures.py`, `phase2b_tie_reconcile.py` |

### 10.1 Code touched

| File | Change |
|---|---|
| `fastvideo/attention/backends/sparsefp4_numerics.py` | Additive: `BlockGeometry` and the three geometry constructors, layout round-trip, geometry-aware masked-mean pooling, `retained_token_fraction`. `masked_reference` and `block_attention_mass` now take a validity mask / geometry instead of a scalar `valid_len`, so they work in the cube padded layout; `masked_reference` also no longer produces NaN for an all-masked query row (it matches the kernel's zero). No metric definition changed. |
| `fastvideo/attention/backends/precision_sparse_attn.py` | Geometry is a config parameter; arms are selectable; added the `C_null` executed-path null control, the `tie_diagnostic` record with both tie denominators, and per-row `frac_query_blocks_changed` / `blocks_swapped_per_query_block` / `retained_token_fraction`. |
| `artifacts/sparsefp4/configs/phase2_selftest.py` | Four new gate groups: layout round-trip and pad-slot invariants, cube/raster mask-on-kernel vs masked reference at Wan's shape, geometry-aware mass normalization, tie-denominator reconciliation. `--geometry-only` flag. |
| `artifacts/sparsefp4/configs/phase2_run.py` | `--geometry`, `--arms`, `--tie-diagnostic-geometries`, `--no-activation-stats`; block sizes derived from the geometry name so the two cannot disagree. |
| `artifacts/sparsefp4/configs/phase2b_launch.sh` | **New.** 8-GPU sharding for one geometry arm, `nohup` per shard with per-shard logs. |
| `artifacts/sparsefp4/configs/phase2b_geometry_analyze.py` | **New.** Cross-geometry verification gate and tables 1-5 and 8. |
| `artifacts/sparsefp4/configs/phase2b_tie_reconcile.py` | **New.** GPU-free reconciliation of the Phase 1 / Phase 2 tie counts and the FP8 asymmetry from archived records. |
| `artifacts/sparsefp4/configs/phase2b_figures.py` | **New.** Renders the three figures from their value CSVs. |

`pre-commit run --files` passes on every source file touched. Nothing was
committed. `STATUS.md`, `PHASE0.md`, `PHASE1.md`, `PHASE2.md`, `CODEBASE_MAP.md`,
`SKILL.md`, `EXPERIMENT_SPEC.md`, `REPORT_TEMPLATE.md` and
`experiment_config.yaml` were not modified.
