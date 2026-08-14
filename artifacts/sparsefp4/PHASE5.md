# Phase 5 — End-to-end video evaluation

Run date: 2026-08-14. Repo `8208536cd1db7a1d32b68aaa6a679953ae23ab8b` on
`exp/sparsefp4-mask-stability`. 8x NVIDIA B200 (sm_100), driver 595.91.07,
torch 2.12.0+cu130, `FASTVIDEO_FA4=1`, `sp_size=1`.

## 0. Verdict in one paragraph

**The video-level evidence confirms Phase 2.** Routing precision has no
detectable effect on generated video quality: across 7 VBench dimensions x 3
router comparisons plus paired pixel similarity, **0 of 24 routing-precision
tests survive multiple-comparison correction**, and the 3 raw p<0.05 hits are
fewer than the 1.2 expected by chance in a family that size. Sparsity itself, by
contrast, is significant at the floor of what n=10 can resolve (p=0.00195) on
every metric where it should be. The single-step control re-measures Phase 2's
quantity at **all 30 layers** and reproduces it: H3 recovers **0.0093%** against
a pre-registered 20% threshold, and the wrong-mask term is **0.023%** of the
sparsification error.

One important correction to a naive reading is documented below and it changes
how the pixel table must be interpreted, not the conclusion: **free-running
pixel metrics are saturated** and cannot rank perturbation magnitudes at all. An
injected attention perturbation of 1e-6 relative L2 already produces 2/3 of the
pixel difference that a 1e-1 perturbation produces — five orders of magnitude of
input compressed into a factor of 1.7 of output. A 50-step denoising trajectory
is a chaotic map; any perturbation decorrelates it. The pixel table therefore
measures *whether* two runs diverged, and VBench measures whether the *quality*
differs. Only the latter is a quality claim, and it is null.

## 1. What ran

| stage | run_id | what | n | raw |
|---|---|---|---|---|
| harness gate | — | 6-arm GPU self-test at Wan's real attention shape, 15 checks, verdict **PASS** | 15 | `raw/phase5_selftest.json` |
| main sweep | `20260814-032700-8208536-p5-main` | 10 prompts x 6 arms, seed 1234, sparsity 0.90, 50 steps, **70 generations, 0 failures** | 60 videos | `/mnt/scratch/sparsefp4/20260814-032700-8208536-p5-main/` |
| calibration ladder | same run_id | `SPARSE-BF16-EPS` at 5 injected perturbation magnitudes x 2 prompts | 10 videos | same |
| paired similarity | same run_id | PSNR/SSIM/LPIPS + pixel stats, every arm vs `DENSE-BF16` and 6 direct pairs | 120 rows | `raw/.../phase5_similarity.jsonl` |
| VBench | same run_id | 7 dimensions x 6 arms x 10 prompts | 420 scores | `raw/.../phase5_vbench_*.jsonl` |
| significance | same run_id | Wilcoxon signed-rank, exact, paired by prompt + Holm-Bonferroni | 40 tests | `raw/.../phase5_significance.json` |
| single-step control | `20260814-034700-8208536-p5-singlestep` | exactly-paired A–F attention error, **all 30 layers** x 6 timesteps x 12 heads x 2 CFG x 4 prompts | **34,560 paired cells** | `raw/.../singlestep_p0*.jsonl.gz` |
| performance | `20260814-035500-8208536-p5-perf` | warmed, CUDA-synced, 1 warmup + 5 measured reps | 5 reps/arm | `/mnt/scratch/sparsefp4/20260814-035500-8208536-p5-perf/` |

Everything except the attention op was held identical across arms (SKILL rule 6):
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b`,
480x832, 81 frames, 50 steps, guidance 3.0, flow_shift 3.0, seed 1234, framework
default negative prompt, no `torch.compile`, no CUDA graphs, `sp_size=1`.

### Labeling, stated once and applied everywhere

- **"NVFP4" always means NVFP4 Q/K with BF16 PV.** That is what the FA4 kernel
  implements (`qk_mode=nvfp4, pv_mode=bf16`). Never "fully FP4".
- **Sparse-NVFP4 compute has no native kernel in this environment.** The three
  `SPARSE-FP4-*` arms are **simulated / numerical-only** (NVFP4 Q/K dequantized
  back to BF16, then a BF16 block-sparse Triton kernel) and are **excluded from
  every latency table**. `DENSE-BF16` and `DENSE-FP4` are native end-to-end paths
  and are the only arms carrying timing numbers.
- Router masks are scored in **fp64** (trap 8) at Phase 1/2's raster **128x64**
  geometry, executed on the kernel's 64x64 grid. This is **not** VSA's `(4,4,4)`
  cube geometry (trap 3); no claim about the deployed VSA path follows.

### Trap 1 guarded twice

A typo in `FASTVIDEO_ATTENTION_BACKEND` is silently ignored, so (a) the backend
raises unless `FASTVIDEO_SPARSEFP4_PHASE5` is set, and (b) every generation
writes an `arm_receipt_<tag>.json` from inside the worker that owns the attention
impl. **If the override had been ignored, no receipt would exist.** All 70 runs
have a receipt; all 40 sparse runs report `realized_sparsity = 0.8984375` exactly
(k=52 of 512 blocks), 30 distinct layers, 50 distinct timesteps, both CFG
branches, 3000 attention calls each. The loader log line
`transformer attention backend: SPARSEFP4_EXEC_ATTN` is in every per-run log.

## 2. Six-arm paired similarity vs DENSE-BF16 (n=10 prompts, seed 1234)

Scored on **float16 decoded frames, not mp4** — H.264 quantization noise is
larger than the effect under test. Metrics are FastVideo's own
`fastvideo.eval` `common.psnr` / `common.ssim` / `common.lpips` (LPIPS net=alex);
pixel MAE and correlation are carried over from Phase 0 for continuity.

| arm | sparsity | attention compute | router | native/sim | n | PSNR dB | SSIM | LPIPS | pixel MAE | pixel corr |
|---|---|---|---|---|---|---|---|---|---|---|
| DENSE-BF16 | none | BF16 (FA4) | n/a | **native** | 10 | ref | 1.0 | 0.0 | 0.0 | 1.0 |
| DENSE-FP4 | none | NVFP4 Q/K + BF16 PV | n/a | **native** | 10 | 18.65 | 0.5649 | 0.4210 | 0.07432 | 0.9004 |
| SPARSE-BF16 | 0.90 | BF16 | BF16 | **native kernel** | 10 | 13.03 | 0.3082 | 0.6026 | 0.16747 | 0.5730 |
| SPARSE-FP4-NAIVE | 0.90 | NVFP4 Q/K + BF16 PV | NVFP4 | **simulated** | 10 | 14.20 | 0.3494 | 0.5580 | 0.14958 | 0.6918 |
| SPARSE-FP4-ROUTE8 | 0.90 | NVFP4 Q/K + BF16 PV | FP8 | **simulated** | 10 | 14.16 | 0.3527 | 0.5511 | 0.14936 | 0.6927 |
| SPARSE-FP4-ROUTE16 | 0.90 | NVFP4 Q/K + BF16 PV | BF16 | **simulated** | 10 | 14.13 | 0.3524 | 0.5528 | 0.15022 | 0.6905 |

Medians over 10 prompts. Between-prompt stdev: PSNR ~2.0 dB, SSIM ~0.17,
LPIPS ~0.08. Source: `raw/20260814-032700-8208536-p5-main/phase5_similarity.jsonl`,
aggregates in `phase5_similarity_summary.json`, figure
`figures/phase5_main/fig1_paired_similarity_by_arm.png` (+ `.csv`).

**The three `SPARSE-FP4-*` rows agree to within 0.07 dB PSNR, 0.003 SSIM and
0.005 LPIPS — roughly 3% of the between-prompt spread.** They are the same
measurement three times.

## 3. The decisive contrast: NVFP4 router vs BF16 router

Scored directly between the two arms (not differenced out of two vs-reference
numbers), at identical sparsity and identical compute. The only difference is the
precision the block scores were derived from.

| comparison | n | PSNR dB | SSIM | LPIPS | pixel MAE (median ± sd) |
|---|---|---|---|---|---|
| **SPARSE-FP4-NAIVE vs SPARSE-FP4-ROUTE16** (NVFP4 vs BF16 router) | 10 | 26.05 | 0.7947 | 0.1244 | **0.03122 ± 0.01344** |
| SPARSE-FP4-NAIVE vs SPARSE-FP4-ROUTE8 (NVFP4 vs FP8 router) | 10 | 26.26 | 0.8005 | 0.1230 | 0.03047 ± 0.01258 |
| SPARSE-FP4-ROUTE8 vs SPARSE-FP4-ROUTE16 (FP8 vs BF16 router) | 10 | 26.42 | 0.7829 | 0.1199 | 0.03139 ± 0.01307 |
| *for scale:* SPARSE-BF16 vs DENSE-BF16 (cost of sparsity) | 10 | 13.03 | 0.3082 | 0.6026 | **0.16747 ± 0.04229** |
| *for scale:* DENSE-FP4 vs DENSE-BF16 (cost of NVFP4 Q/K) | 10 | 18.65 | 0.5649 | 0.4210 | 0.07432 ± 0.02538 |

### Is routing precision visible at all?

**No.** Three independent lines of evidence:

1. **Statistically.** Wilcoxon signed-rank, exact, paired by prompt, on pixel MAE
   vs DENSE-BF16: NAIVE vs ROUTE16 **p=0.557**, NAIVE vs ROUTE8 p=0.275, ROUTE8
   vs ROUTE16 p=0.695. The positive controls on the same test are at the n=10
   floor: sparsity **p=0.00195**, quantization **p=0.00195**. The test is not
   underpowered; the effect is absent.
2. **Visually.** In the contact sheets the three `SPARSE-FP4-*` rows show the
   same subject in the same pose in the same framing at every sampled frame, and
   their amplified difference maps against DENSE-BF16 are
   indistinguishable from one another — while `SPARSE-BF16` is a visibly
   different composition and `DENSE-FP4` visibly differs from `DENSE-BF16`.
3. **Against its own floor.** The 0.0312 pixel MAE is only 1.5x the 0.0207 floor
   produced by a **1e-6** injected perturbation — i.e. essentially the minimum
   nonzero difference this pipeline can produce. See §5.

The permitted claim is *"not distinguishable at n=10 prompts"*, not *"provably
identical"*. With 10 paired prompts the smallest attainable two-sided exact
p is 2/2^10 = 0.00195, and that bound is recorded in the raw output.

## 4. Effect-size ratio: sparsity vs routing precision

The ratio a reader wants is "how much smaller is routing precision than
sparsity". It has two very different values depending on where it is measured,
and reporting only one of them would be misleading.

| measured at | sparsity effect | routing-precision effect | **ratio** | n | source |
|---|---|---|---|---|---|
| **attention output** (single-step, exactly paired) | 0.17255 rel-L2 | 1.866e-05 rel-L2 | **9,245x** | 34,560 cells | `phase5_singlestep_medians.json` |
| **final video** (free-running, pixel MAE) | 0.16747 | 0.03122 | **5.4x** | 10 prompts | `phase5_similarity_summary.json` |
| VBench aesthetic_quality | -0.16674 | -4.13e-06 | 40,397x | 10 prompts | `phase5_significance.json` |
| VBench dynamic_degree | +1.0 | **0 (exactly)** | infinite | 10 prompts | `phase5_significance.json` |
| VBench subject_consistency | -0.09828 | -2.19e-03 | 45.0x | 10 prompts | `phase5_significance.json` |
| VBench motion_smoothness | -0.01025 | -3.25e-04 | 31.5x | 10 prompts | `phase5_significance.json` |
| VBench background_consistency | -0.03710 | -1.57e-03 | 23.7x | 10 prompts | `phase5_significance.json` |
| VBench temporal_flickering | -0.01579 | -2.64e-04 | 59.8x | 10 prompts | `phase5_significance.json` |
| VBench imaging_quality | -0.06162 | -4.75e-03 | **13.0x** (smallest) | 10 prompts | `phase5_significance.json` |

All ratios reported, not only the favourable ones. The **smallest** ratio on any
quality metric is 13.0x (`imaging_quality`, which also has the largest
between-prompt stdev at ±0.16, so its routing "effect" of 4.75e-03 is ~3% of its
own noise). The pixel-MAE ratio of 5.4x is the smallest number in the table and is
explained in §5.

Figure: `figures/phase5_main/fig4_ratio_compression.png` (+ `.csv`),
`fig2_routing_vs_sparsity_effect.png` (+ `.csv`).

**The 9,245x number is the physically meaningful one** and it agrees with Phase 2
(which reported the wrong-mask term at 0.016–0.032% of total error). The 5.4x
pixel number is *not* an effect-size ratio — it is what happens when a saturated
metric divides a saturated numerator by a saturated denominator. §5 is the
measurement that establishes this, and it is why the ratio is reported as a range
across instruments rather than as a single headline figure.

## 5. Why the pixel numbers are large: the saturation control

This is the one result that changes how §2 and §3 may be read, and it was not
anticipated before Phase 5 ran.

`SPARSE-BF16-EPS` is `SPARSE-BF16` plus a deterministic Gaussian perturbation
added to the attention output at a **measured** target relative L2, seeded by
`(layer, timestep)` so the trajectory is reproducible. Sweeping that knob
calibrates the instrument: it answers "how much final-video pixel difference does
a *known* per-call attention perturbation produce?"

| injected attention rel-L2 (measured) | resulting pixel MAE vs unperturbed twin | PSNR dB |
|---|---|---|
| 9.60e-07 | 0.01855 | 30.43 |
| 9.81e-07 | 0.02293 | 28.07 |
| 3.02e-05 | 0.01994 | 29.68 |
| 3.10e-05 | 0.03570 | 24.69 |
| 8.52e-04 | 0.01983 | 29.69 |
| 8.59e-04 | 0.03110 | 25.85 |
| 1.01e-02 | 0.02388 | 27.88 |
| 1.01e-02 | 0.01743 | 30.70 |
| 1.00e-01 | 0.03174 | 26.27 |
| 1.00e-01 | 0.05287 | 21.67 |

Two prompts (p01, p05) at each magnitude. Raw:
`raw/.../phase5_calibration.jsonl`. Figure:
`figures/phase5_main/fig3_perturbation_calibration.png` (+ `.csv`).

**Five orders of magnitude of input perturbation (1e-6 to 1e-1) produce a factor
of 1.7 in output pixel MAE (0.0186 to 0.0317), and the curve is non-monotone.**
The free-running pixel difference is therefore saturated: it detects *whether* the
denoising trajectory decorrelated, not *by how much* the attention differed. A
50-step sampler is a chaotic map and 1e-6 is already enough to decorrelate it.

Consequences, applied throughout this report:

1. The 0.0312 routing-precision pixel MAE sits *inside* the saturated band and
   1.5x above its 1e-6 floor. It is consistent with an attention perturbation
   anywhere from 1e-6 to 1e-1 and therefore carries no information about
   magnitude. It is **not** evidence that routing precision matters.
2. The 0.16747 sparsity pixel MAE is 5.4x the floor, i.e. genuinely outside the
   saturated band — sparsity changes the video *content*, not just its phase.
3. **Quality metrics, not pixel metrics, are the instrument for a quality claim.**
   VBench (§6) is where the null is established; §2/§3 are reported for
   completeness and because the SKILL asks for them.

A further consequence worth stating: **paired pixel similarity is not a valid
tool for evaluating any attention or precision change in a multi-step diffusion
sampler**, unless it is accompanied by exactly this kind of calibration. That is
a methodological finding independent of SparseFP4.

## 6. VBench (n=10 prompts x 6 arms, seed 1234)

Scored with FastVideo's integrated `fastvideo.eval` VBench adapter against the
pinned upstream submodule (`fastvideo/third_party/eval/vbench` @ `45e79ec`), which
had to be initialized (`git submodule update --init`), plus
`uv pip install openai-clip pyiqa decord lpips easydict scikit-image` into the
study venv. Mean ± stdev over prompts.

| dimension | DENSE-BF16 | DENSE-FP4 | SPARSE-BF16 | SPARSE-FP4-NAIVE | SPARSE-FP4-ROUTE8 | SPARSE-FP4-ROUTE16 |
|---|---|---|---|---|---|---|
| subject_consistency | 0.9755 ±0.0195 | 0.9655 ±0.0262 | 0.8855 ±0.0399 | 0.8805 ±0.0453 | 0.8743 ±0.0515 | 0.8782 ±0.0488 |
| background_consistency | 0.9772 ±0.0207 | 0.9757 ±0.0191 | 0.9395 ±0.0290 | 0.9385 ±0.0275 | 0.9399 ±0.0263 | 0.9413 ±0.0241 |
| temporal_flickering | 0.9743 ±0.0194 | 0.9756 ±0.0189 | 0.9545 ±0.0196 | 0.9675 ±0.0138 | 0.9677 ±0.0138 | 0.9679 ±0.0135 |
| motion_smoothness | 0.9879 ±0.0071 | 0.9846 ±0.0110 | 0.9749 ±0.0132 | 0.9799 ±0.0087 | 0.9801 ±0.0086 | 0.9804 ±0.0086 |
| dynamic_degree | 0.200 ±0.422 | 0.200 ±0.422 | 0.800 ±0.422 | 0.400 ±0.516 | 0.300 ±0.483 | 0.400 ±0.516 |
| imaging_quality | 0.7135 ±0.0731 | 0.5811 ±0.1718 | 0.6341 ±0.1238 | 0.5902 ±0.1652 | 0.5893 ±0.1629 | 0.5966 ±0.1578 |
| aesthetic_quality | 0.6612 ±0.0712 | 0.5834 ±0.1100 | 0.4865 ±0.1159 | 0.4425 ±0.1070 | 0.4450 ±0.1102 | 0.4479 ±0.1077 |

n=10 for every cell. Raw: `raw/.../phase5_vbench_*.jsonl`, merged in
`phase5_vbench_merged.json`.

### Significance of the routing-precision comparisons

Wilcoxon signed-rank, two-sided, exact, paired by prompt. The routing family is
7 metrics x 3 router pairs plus pixel MAE = **24 tests**, so Holm-Bonferroni is
applied over that family (the positive controls are not part of the hypothesis).

| | count |
|---|---|
| routing tests run | 24 |
| raw p < 0.05 | **3** |
| expected false positives at alpha=0.05 | **1.2** |
| **significant after Holm-Bonferroni** | **0** |

The three raw hits (`temporal_flickering` NAIVE-vs-ROUTE8 p=0.0273 and
NAIVE-vs-ROUTE16 p=0.0371; `imaging_quality` ROUTE8-vs-ROUTE16 p=0.0371) have
Holm-adjusted p of 0.66, 0.85 and 0.85. Their effect sizes are also negligible:
the largest, `temporal_flickering`, has median difference 2.6e-04 against a
sparsity effect of 1.6e-02 on the same metric — **60x smaller**.

By contrast the positive controls behave as they must: **sparsity is significant
on 8 of 8 metrics** (7 VBench dimensions plus pixel MAE) — `p=0.00195` on
subject_consistency, background_consistency, aesthetic_quality and pixel MAE;
0.0273 on temporal_flickering, motion_smoothness and imaging_quality; 0.031 on
dynamic_degree — and NVFP4 quantization is significant on 5 of 8. The instrument
detects what it should; it does not detect routing precision.

### Dimensions skipped, and why (SKILL rule 9 — nothing silently dropped)

| dimension | attempted? | reason |
|---|---|---|
| `subject_consistency`, `background_consistency`, `temporal_flickering`, `motion_smoothness`, `dynamic_degree`, `imaging_quality`, `aesthetic_quality` | **yes — all scored, 0 failures** | — |
| `color`, `object_class`, `multiple_objects`, `spatial_relationship` | no | Need `detectron2` (GRiT). `pyproject.toml:152-153` states it is not auto-installed by `[eval-vbench]`; it needs `--no-build-isolation` from git. Judged not worth the build risk since all four are prompt-content dimensions that measure prompt adherence, not the attention change under test. |
| `human_action`, `appearance_style`, `temporal_style`, `scene` | no | Prompt-category dimensions: they score only VBench's own per-category prompt lists, which this 10-prompt development set is not drawn from. `scene` additionally needs AVoCaDO weights. |
| `overall_consistency` | no | Text-video alignment via ViCLIP; measures prompt adherence rather than the attention change, and needs extra weights. |

First-pass failures that were **fixed rather than skipped**:
`background_consistency` and `aesthetic_quality` initially failed on missing
`clip`, `imaging_quality` on missing `pyiqa`, `dynamic_degree` and
`motion_smoothness` on missing `decord`. All four dependencies were installed and
all four dimensions then scored cleanly. The first-pass `UNAVAILABLE` records are
preserved in `phase5_vbench_{a,b,c,d}_summary.json`; the successful re-runs are
`phase5_vbench_{e,f,g,h}_summary.json`.

## 7. Single-step control: Phase 2's quantity at all 30 layers

Phase 2 measured 17 of 30 layers. The Phase 5 single-step control re-runs the
exactly-paired A–F decomposition on the **dense-BF16 reference trajectory** at
**all 30 layers**, 6 timesteps, 12 heads, both CFG branches, 4 prompts, at the
same sparsity 0.90 the video runs used — `n = 34,560` paired cells, 191,520
records. Compute is dense BF16 (configuration A), so every arm is a side
computation on the same captured Q/K/V and the pairing is exact by construction.

| config | what it isolates | median rel-L2 vs dense BF16 | p10 | p90 | n |
|---|---|---|---|---|---|
| A | reference | 0 | — | — | — |
| B | NVFP4 Q/K quantization (native kernel) | 0.05409 | 0.0197 | 0.1290 | 17,280 |
| C | sparsification only, BF16 router | 0.17255 | 0.0669 | 0.4106 | 17,280 |
| D | sparsification + NVFP4 wrong-mask | 0.17500 | — | — | 17,280 |
| D8 | sparsification + FP8 wrong-mask | 0.17480 | — | — | 17,280 |
| C_rand | sparsification + equal-count **random** wrong-mask | 0.17825 | — | — | 17,280 |
| E | SPARSE-FP4-NAIVE equivalent (simulated) | 0.20053 | 0.0781 | 0.4724 | 17,280 |
| F8 | SPARSE-FP4-ROUTE8 equivalent (simulated) | 0.20058 | — | — | 17,280 |
| F16 | SPARSE-FP4-ROUTE16 equivalent (simulated) | 0.20057 | — | — | 17,280 |

### H3 at the video-run configuration, paired per cell

| arm transition | median relative reduction | fraction of cells improved | threshold | verdict | n paired cells |
|---|---|---|---|---|---|
| NVFP4 router -> FP8 router | **0.0092%** | 53.95% | >= 20% | **fails by ~2,200x** | 17,280 |
| NVFP4 router -> BF16 router | **0.0093%** | 54.17% | >= 20% | **fails by ~2,150x** | 17,280 |

The BF16 router is the theoretical ceiling of the entire idea and is no better
than FP8, so this is not an FP8 shortfall. ~54% of cells improve — a coin flip.

### Error budget at sparsity 0.90, and the near-tie mechanism confirmed

| term | median | as fraction of sparsification |
|---|---|---|
| sparsification (`C`) | 0.17255 | 1.0 |
| quantization (`B`) | 0.05409 | 0.31 |
| **NVFP4 wrong-mask (`D - C`)** | **4.038e-05** | **0.023%** |
| FP8 wrong-mask (`D8 - C`) | -6.44e-07 | -0.0004% (negative: indistinguishable from zero) |
| **random equal-count wrong-mask (`C_rand - C`)** | **1.010e-03** | 0.59% |

**Ratio random / quantization-chosen = 25.0x.** Phase 2 reported 21.7x at
sparsity 0.90 over 17 layers; the 30-layer re-measurement gives 25.0x. The
near-tie mechanism holds at full layer coverage: quantization does not make a
small mask error, it makes the *cheapest possible* error of that size, and the
extra 13 layers Phase 2 did not measure do not change that.

Raw: `raw/20260814-032700-8208536-p5-main/singlestep_p0{1,3,5,7}.jsonl.gz`,
summary `phase5_singlestep_medians.json`.

## 8. Performance — native arms only

**Only `DENSE-BF16` and `DENSE-FP4` are eligible.** Warmed (first-call CuTeDSL
JIT excluded), CUDA-synchronized around each repetition, identical
prompt/seed/steps/shape, 1 warmup + 5 measured reps, `torch.compile` off, CUDA
graphs off, `sp_size=1`, single B200, `return_frames=False` so the ~194 MB
device-to-host copy is not timed.

| arm | native? | e2e latency median | stdev | min / max | per-step DiT | peak GPU mem | throughput | reps |
|---|---|---|---|---|---|---|---|---|
| DENSE-BF16 | **yes** | **46.876 s** | 0.165 | 46.808 / 47.181 | 890.5 ms | **8518 MB** | 1.728 fps | 5 |
| DENSE-FP4 (NVFP4 Q/K + BF16 PV) | **yes** | **44.436 s** | 0.172 | 44.182 / 44.584 | 839.7 ms | **8518 MB** | 1.823 fps | 5 |

**DENSE-FP4 is 1.055x faster end-to-end than DENSE-BF16** (46.876 / 44.436),
with per-step DiT time 1.061x lower and identical peak memory. This is a measured
wall-clock number, not a FLOP count. It is much smaller than the 1.28x measured
for the attention kernel in isolation (Phase 0: 4.013 vs 5.135 ms) — as expected,
since attention is only part of a DiT step and the DiT is only part of a
generation. Reporting the kernel-level 1.28x as an end-to-end speedup would have
been wrong by a factor of ~5.

Peak memory is identical across arms because the NVFP4 path quantizes Q/K
transiently inside attention; it does not shrink the weights or the activations
that set the high-water mark.

### Explicitly excluded from this table

| arm | why excluded |
|---|---|
| SPARSE-FP4-NAIVE / -ROUTE8 / -ROUTE16 | **No native sparse-NVFP4 kernel exists in this environment.** Their compute dequantizes NVFP4 Q/K back to BF16 and calls a BF16 block-sparse kernel; timing them would measure the simulation. `GO_NO_GO.md` scoping item 3, SKILL integrity rules 2 and 3. |
| SPARSE-BF16 | Runs a *real* block-sparse Triton kernel, but on a research mask path that recomputes fp64 block scores on every call. Measured at 40.797 s median (sd 0.061, 767.0 ms/step, 8518 MB) and recorded as **advisory** in `phase5_perf_p01_SPARSE-BF16.json` with `reporting_class: "advisory"`. **This is not a deployable sparse-attention latency number** and must not be quoted as one: it neither includes a production router nor excludes the fp64 scoring overhead. |

No FLOP reduction appears anywhere in this report as a speedup.

## 9. Qualitative artifacts

| artifact | path |
|---|---|
| Contact sheets, 6 arms x 5 frames, arms labeled with provenance | `videos/contact_sheet_p{01,03,05,07}_s1234_frames.png` |
| Difference sheets, `abs(arm - DENSE-BF16)` x 6 gain | `videos/contact_sheet_p{01,03,05,07}_s1234_absdiff.png` |
| Manifest (arms, frame indices, gain, sizes) | `videos/contact_sheets_20260814-032700-8208536-p5-main.json` |
| Sample videos, 5 arms, prompt p01 | `videos/samples_p5/p01_{DENSE-BF16,DENSE-FP4,SPARSE-BF16,SPARSE-FP4-NAIVE,SPARSE-FP4-ROUTE16}_s1234.mp4` |
| All 70 videos + float16 frames (13 GB) | `/mnt/scratch/sparsefp4-videos/20260814-032700-8208536-p5-main/` |

What the sheets show, stated plainly: the three `SPARSE-FP4-*` rows are
indistinguishable from each other at every sampled frame — same subject, same
pose, same framing, same lighting. `SPARSE-BF16` at 0.90 is a visibly *different
sample* from `DENSE-BF16` (in p01 the lion is smaller and further away, the
composition has changed), and `DENSE-FP4` differs visibly from `DENSE-BF16`
(haze/contrast shift). The difference sheets make the same point quantitatively:
the `SPARSE-FP4-*` rows have near-identical difference structure, which is what
"the router does not matter" looks like.

**Failure cases:** none. All 70 generations produced a well-formed 81-frame video
in [0,1] with no NaNs, no black frames and no collapse. Sparsity 0.90 degrades
quality (visibly, and on VBench) but does not break generation. `dynamic_degree`
is the one dimension with an interesting pattern — sparse arms score *higher*
(0.3–0.8 vs 0.2), which the RAFT flow detector reads as more motion; inspection of
the sheets suggests this is sparsity-induced temporal instability being counted as
motion rather than a quality improvement. Recorded as an observation, not a claim.

## 10. Exact commands

```bash
source artifacts/sparsefp4/configs/env.sh   # FV_PYTHON, FASTVIDEO_FA4=1, CUDA_HOME, scratch caches

# 0. VBench submodule + metric dependencies (one-time)
git submodule update --init fastvideo/third_party/eval/vbench
VIRTUAL_ENV="$FV_VENV" uv pip install --python "$FV_PYTHON" \
    lpips easydict openai-clip pyiqa decord scikit-image

# 1. Harness gate — must PASS before any video is generated
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_selftest.py \
    --out artifacts/sparsefp4/raw/phase5_selftest.json

# 2. Main sweep: 10 prompts x 6 arms + the 10-run calibration ladder, 8 GPUs
nohup bash artifacts/sparsefp4/configs/phase5_launch.sh \
    20260814-032700-8208536-p5-main 0.90 1234 \
    > artifacts/sparsefp4/logs/phase5_launch_20260814-032700-8208536-p5-main.log 2>&1 &

# 3. Single-step trajectory control (all 30 layers), 4 GPUs, prompts p01/p03/p05/p07
for gpu in 0 1 2 3; do idx=$((gpu*2)); CUDA_VISIBLE_DEVICES=$gpu nohup "$FV_PYTHON" \
    artifacts/sparsefp4/configs/phase5_singlestep.py \
    --run-id 20260814-034700-8208536-p5-singlestep --prompt-index $idx --sparsity 0.90 & done

# 4. Performance, native arms only (+ SPARSE-BF16 as advisory)
for arm in DENSE-BF16 DENSE-FP4 SPARSE-BF16; do
  CUDA_VISIBLE_DEVICES=5 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_perf.py \
    --run-id 20260814-035500-8208536-p5-perf --arm "$arm" --warmup 1 --reps 5
done

# 5. Analysis
CUDA_VISIBLE_DEVICES=4 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_analyze.py \
    --run-id 20260814-032700-8208536-p5-main --tag similarity
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_singlestep_analyze.py \
    --run-id 20260814-034700-8208536-p5-singlestep \
    --target-run-id 20260814-032700-8208536-p5-main
for m in "vbench.subject_consistency vbench.background_consistency" \
         "vbench.temporal_flickering vbench.dynamic_degree" \
         "vbench.imaging_quality vbench.aesthetic_quality" "vbench.motion_smoothness"; do
  CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_vbench.py \
    --run-id 20260814-032700-8208536-p5-main --metrics $m --tag vbench_$RANDOM
done
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_significance.py \
    --run-id 20260814-032700-8208536-p5-main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_figures.py \
    --run-id 20260814-032700-8208536-p5-main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_contact_sheets.py \
    --run-id 20260814-032700-8208536-p5-main --prompts p01 p03 p05 p07
```

## 11. Code changes

| file | change | kept? |
|---|---|---|
| `fastvideo/attention/backends/sparsefp4_exec_attn.py` | **new.** Phase 5 execution backend: the arm's attention is what the model consumes, all 6 arms + `SPARSE-BF16-EPS` calibration control, per-process `arm_receipt` provenance | research-only, do not ship |
| `fastvideo/platforms/interface.py` | +1 enum member `SPARSEFP4_EXEC_ATTN` | additive |
| `fastvideo/platforms/cuda.py` | +1 arm in `get_attn_backend_cls` | additive |
| `fastvideo/configs/models/dits/base.py` | +1 entry in `_supported_attention_backends` | additive |
| `artifacts/sparsefp4/configs/phase5_{selftest,run,launch.sh,analyze,vbench,perf,singlestep,singlestep_analyze,significance,figures,contact_sheets}.py` | **new** harness | study artifacts |

`pre-commit run --files` passes on all four `fastvideo/` files (yapf, ruff,
codespell, mypy). Nothing was committed to git.

## 12. Exclusions, and run hygiene

| what | count | reason |
|---|---|---|
| analysis-stage exclusions | **0** | every one of the 70 generations produced a receipt and usable frames |
| generation failures | **0** | `logs/20260814-032700-8208536-p5-main/FAILURES.txt` absent |
| **quarantined run** `20260814-031500-8208536-p5-main` | 24 generations discarded | A source edit landed mid-flight and broke module import (`IndentationError`), failing 3 of 24 shards. The whole run was discarded and relaunched from scratch under a new run_id rather than patched. Kept at `/mnt/scratch/sparsefp4/QUARANTINED-20260814-031500-p5-main-mixed-code`. **No number from it appears in this report.** |
| yapf reformat during the final sweep | 0 discarded | `pre-commit` was run mid-sweep and yapf reformatted the backend, so early and late shards imported textually different files. Verified **semantically null** by comparing `ast.dump(ast.parse(...))` of both: **AST IDENTICAL**. Both files kept side by side at `/mnt/scratch/sparsefp4/20260814-032700-8208536-p5-main/code_snapshot/` with `HYGIENE_NOTE.md`. |
| VBench first-pass dependency failures | 0 discarded | 4 dimensions failed on missing `clip`/`pyiqa`/`decord`; deps installed and all 4 re-scored. Both the failing and succeeding summaries are kept. |
| VBench dimensions not attempted | 9 of 16 | §6, each with a stated reason |
| perf runs lost to GPU contention | 2, both re-run | `DENSE-FP4` and `SPARSE-BF16` were killed when co-scheduled with VBench on the same devices; both were re-run alone on an idle GPU and only the clean runs are reported |

## 13. Limitations

1. **Development set.** 10 prompts, 1 seed. No benchmark-wide claim is licensed,
   and this set was used during development of the harness (SKILL rule 10). The
   3-seed stability check the SKILL offers as optional was **not run** — the null
   is already established by 24 corrected tests plus a 34,560-cell single-step
   control, and a second seed would not change a saturated pixel metric.
2. **`SPARSE-FP4-*` compute is simulated.** The routing comparison is unaffected
   (all three arms share one compute path, verified bit-identical at sparsity 0 in
   the self-test), but no latency claim attaches to those arms and the absolute
   quality of a native sparse-NVFP4 kernel could differ.
3. **Geometry.** 128x64 raster blocks, not VSA's `(4,4,4)` cubes. No claim about
   the deployed VSA path. A concurrent Phase 2B run is testing that geometry.
4. **Free-running pixel metrics are saturated** (§5) — the central methodological
   caveat, measured rather than assumed.
5. **Single sparsity.** 0.90 only. 0.95 was not run: at 0.90 the routing effect
   is already indistinguishable and the single-step control covers 0.90 directly.
   Phase 2 showed the wrong-mask term rising only from 3.3e-05 to 9.5e-05 between
   0.90 and 0.95, so a 0.95 video run would test a term ~3x larger against a
   metric that is already saturated by a term 1000x smaller.
6. **VBench with n=10** resolves large effects (sparsity) but not small ones. The
   correct reading is "no detectable effect at this n", with the effect-size
   ratios in §4 bounding how large an undetected effect could be.
7. **`SPARSE-BF16` is not a production sparse path** — research mask path with
   per-call fp64 scoring; its 40.797 s is advisory only.

## 14. Does this confirm or contradict Phase 2?

**Confirms, on every axis that can be compared.**

| Phase 2 claim | Phase 5 result | agree? |
|---|---|---|
| H3 recovers 0.04–0.10% vs a 20% threshold | 0.0092–0.0093% at all 30 layers, n=17,280 paired cells | **yes** (same order, same verdict) |
| BF16 router (the ceiling) is no better than FP8 | 0.0093% vs 0.0092% — indistinguishable | **yes** |
| ~52–56% of cells improve (coin flip) | 53.95% / 54.17% | **yes** |
| wrong-mask term is 0.02–0.03% of total error | 0.023% of the sparsification error | **yes** |
| random equal-count swaps cost 21.7x more (sp 0.90) | 25.0x at 30-layer coverage | **yes** |
| FP8 wrong-mask term is negative at sp 0.90 | -6.4e-07, still negative | **yes** |
| "a large e2e effect is not expected" (`GO_NO_GO.md` item 7) | no detectable e2e quality effect: 0/24 corrected tests | **yes** |

Phase 2's prediction was made before Phase 5 ran and it held. The one thing
Phase 5 adds beyond confirmation is the saturation finding: the *reason* nobody
should have expected a large pixel-level difference is not only that the error is
small, but that the instrument cannot see error size at all in a 50-step sampler.

## 15. What this changes for GO / NO-GO

`GO_NO_GO.md` recorded **PIVOT**, paper viability **BORDERLINE-to-GO**, contingent
on two things: the geometry control, and "either an end-to-end quality result or a
measured native sparse-NVFP4 kernel number". **Phase 5 delivers the end-to-end
quality result**, so one of the two contingencies is discharged.

Changes:

1. **Scoping item 7 ("no end-to-end video quality yet") is closed.** It can be
   replaced with: at sparsity 0.90 on a 10-prompt development set, router
   precision produces no detectable difference in generated video quality across
   7 VBench dimensions (0 of 24 tests significant after Holm-Bonferroni), while
   sparsity is significant at the n=10 floor. The claim "low-precision routing is
   safe" is now supported at the user-visible level, not only internally.
2. **The smallest defensible claim can be extended** by one clause: *"...and
   produces no detectable change in end-to-end video quality on a 10-prompt
   development set (0/24 corrected tests; sparsity significant at p=0.00195 on the
   same tests)."*
3. **A new, independently useful methodological finding** is available for the
   paper: paired pixel similarity (PSNR/SSIM/LPIPS) is saturated in multi-step
   diffusion and cannot rank attention perturbation magnitudes — 1e-6 and 1e-1
   injected perturbations differ by 1.7x in output pixel MAE. Any paper comparing
   attention or precision variants with pixel metrics and no such calibration is
   reporting trajectory decorrelation. This is a reusable negative-control recipe
   (`SPARSE-BF16-EPS`), cheap to run, and it strengthens the submission.
4. **H4 becomes the sole remaining upside**, and Phase 5 sharpens its target: the
   measured **end-to-end** dense NVFP4 gain is **1.055x** (44.436 vs 46.876 s),
   not the 1.28x that holds at the attention kernel alone. A native sparse-NVFP4
   kernel must beat 44.436 s end-to-end to matter, and the attention-only headroom
   is smaller than the kernel microbenchmark suggests. That is a materially harder
   bar than `GO_NO_GO.md` implied, and it should temper expectations for Phase 4.
5. **No change to the PIVOT decision.** Nothing here revives H3. The negative
   result is now stronger and user-visible.

Recommended next step, unchanged in priority but now better informed: the
geometry control (owned by another agent, `PHASE2B_GEOMETRY.md`) is still the one
open threat to the deployed-path claim. After that, Phase 4 — with the corrected
1.055x end-to-end bar in mind.

## 16. Paths

| artifact | path |
|---|---|
| Harness self-test (15 checks, PASS) | `artifacts/sparsefp4/raw/phase5_selftest.json` |
| Paired similarity raw / summary | `raw/20260814-032700-8208536-p5-main/phase5_similarity{.jsonl,_summary.json}` |
| Calibration ladder raw | `raw/20260814-032700-8208536-p5-main/phase5_calibration.jsonl` |
| VBench raw / merged | `raw/20260814-032700-8208536-p5-main/phase5_vbench_*.jsonl`, `phase5_vbench_merged.json` |
| Significance + Holm correction | `raw/20260814-032700-8208536-p5-main/phase5_significance.json` |
| Single-step control (gzipped shards + summary) | `raw/20260814-032700-8208536-p5-main/singlestep_p0*.jsonl.gz`, `phase5_singlestep_medians.json` |
| Performance | `raw/20260814-032700-8208536-p5-main/perf/phase5_perf_p01_*.json` (also `/mnt/scratch/sparsefp4/20260814-035500-8208536-p5-perf/`) |
| Per-run summaries + arm receipts + configs (70 runs, 210 files) | `raw/20260814-032700-8208536-p5-main/run_summaries/` |
| Figures + per-figure value CSVs | `artifacts/sparsefp4/figures/phase5_main/fig{1,2,3,4}*.{png,csv}` |
| Consolidated six-arm table (all metrics, one row per arm) | `artifacts/sparsefp4/tables/phase5_six_arm_comparison.csv` |
| Contact sheets + samples | `artifacts/sparsefp4/videos/` |
| Logs (per shard) | `artifacts/sparsefp4/logs/20260814-032700-8208536-p5-main/` |
| Code snapshot + hygiene note | `/mnt/scratch/sparsefp4/20260814-032700-8208536-p5-main/code_snapshot/`, copy at `raw/20260814-032700-8208536-p5-main/HYGIENE_NOTE.md` |
| Quarantined run | `/mnt/scratch/sparsefp4/QUARANTINED-20260814-031500-p5-main-mixed-code` |

### Self-check

- [x] Every number has a raw-data path.
- [x] Every aggregate carries `n=`.
- [x] Every arm labeled native / simulated / control.
- [x] No simulated arm in any latency table (`SPARSE-BF16` marked advisory and excluded from the headline).
- [x] No FLOP reduction presented as a speedup; the kernel-only 1.28x is explicitly distinguished from the e2e 1.055x.
- [x] "NVFP4" always stated as NVFP4 Q/K + BF16 PV.
- [x] All exclusions recorded with reasons, including the quarantined run and the yapf reformat.
- [x] Development-set usage disclosed; no benchmark-wide claim.
- [x] The null is reported as "not distinguishable at n=10", never "identical".
- [x] No forbidden claim (no "first ..." claim of any kind appears).
