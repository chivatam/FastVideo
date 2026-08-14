# SparseFP4 Video Attention — Phase 1: sparse-mask stability under low-precision routing

Status: **COMPLETE**. Stages 1 and 2 both ran to completion, both verified, both analyzed.

Headline: **H1 is supported in direction but the effect is small; H2 is supported
only weakly and only along the layer/head axes, not the timestep axis. The
pre-registered PIVOT condition of `SKILL.md` is met.** Numbers, distributions and
exclusions below.

---

## 1. What ran

| Run id | Stage | Scope | Sparsities | Records | Verdict |
|---|---|---|---|---|---|
| `20260814-013449-8208536-p1-stage1` | 1 | 1 prompt (p01) x seed 1234 x **30 layers x 12 heads x all 50 timesteps** x 2 CFG branches | 0.80, 0.90 | **288,000** | PASS |
| `20260814-014229-8208536-p1-stage2` | 2 | **10 prompts** (p01–p10) x seed 1234 x 30 layers x 12 heads x 10 timesteps x 2 CFG branches, sharded 1 process/GPU over 8 B200s | 0.50, 0.70, 0.80, 0.90, 0.95 | **1,116,000** | PASS |
| `20260814-015113-8208536-p1-stage1-fp64score` | 1 (confounder control) | identical to Stage 1, block-score matmul in **fp64** instead of fp32 | 0.80, 0.90 | **288,000** | PASS |
| `20260814-014803-8208536-p1-ctrl-fp64score` | control pilot | 5 timesteps, fp64 scorer, used to size the control above | 0.80, 0.90 | 28,800 | PASS |

Total **1,720,800** raw records. All four run directories carry a
`verification.json` produced by `phase1_verify_run.py`; all four report
`"verdict": "PASS"` with an empty `failures` list.

Every arm was measured on a **dense BF16 attention pass-through** — the probe
computes routing metrics on the side and returns the same dense FA output the
`FLASH_ATTN` baseline returns, so all arms share one denoising trajectory by
construction and the comparison is exactly paired.

- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b`, 480x832, 81 frames, 50 steps, guidance 3.0, `FlowUniPCMultistepScheduler(shift=3.0)`, seed 1234.
- Geometry: `seq_len` 32760 (constant in every record), 30 layers, 12 heads, `head_dim` 128, `block_q` 128, `block_k` 64 → `n_q_blocks` 256, `n_k_blocks` 512, `ragged_tail` 56, `softmax_scale` 0.0883883, `force_retain_diagonal` false.
- Code: branch `exp/sparsefp4-mask-stability`, commit `8208536cd1db7a1d32b68aaa6a679953ae23ab8b` plus the uncommitted probe backend and its 3 registrations.
- Hardware: 8x NVIDIA B200 (sm_100). Total measurement cost **42.6 GPU-minutes** serial-equivalent; Stage 2 wall-clock ~7 min across 8 GPUs.

### Raw data location

Archival records live under `artifacts/sparsefp4/raw/<run_id>/*.jsonl.gz`
(**99 MB** total, gzip -6, 24x compression, verified with `gzip -t` and by
decompressing and re-counting every line). The uncompressed originals remain at
`/mnt/scratch/sparsefp4/<run_id>/*.jsonl` (**2.0 GB**) because the root volume has
only 4.7 GB free; `/mnt/scratch` is ephemeral instance store, so the gzipped
copies under `artifacts/` are the durable ones. `scripts/analyze_masks.py` and
`phase1_deepdive.py` both read `.jsonl.gz` directly, and every table and figure
in this report was regenerated from the archived gzipped copies.

Retained raw metric data is 99 MB against the `EXPERIMENT_SPEC.md` §11 cap of
5 GiB. No tensor dumps were written.

---

## 2. Exact commands

```bash
source artifacts/sparsefp4/configs/env.sh   # FV_PYTHON, FASTVIDEO_FA4=1, CUDA_HOME, HF_HOME

# Stage 1 — full timestep axis, one prompt
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_probe_run.py \
  --run-id 20260814-013449-8208536-p1-stage1 --prompt-index 0 \
  --sparsities 0.80 0.90 --routing-precisions bf16 fp8_e4m3 nvfp4 nvfp4_sim \
  --steps 50 --measure-timestep-stride 1 --null-control-layer-stride 1 \
  --null-control-timestep-stride 1 --spearman-timestep-stride 10 \
  --raw-root /mnt/scratch/sparsefp4

# Stage 2 — 10 prompts, full sparsity sweep, one process per GPU, sp_size=1
bash artifacts/sparsefp4/configs/phase1_stage2_launch.sh 20260814-014229-8208536-p1-stage2
#   GPU->prompt map 0:[p01,p09] 1:[p02,p10] 2..7:[p03..p08]; nohup per GPU;
#   MEASURE_TIMESTEP_STRIDE=5 NULL_LAYER_STRIDE=5 NULL_TIMESTEP_STRIDE=10
#   SPARSITIES="0.50 0.70 0.80 0.90 0.95" STEPS=50

# fp64 scorer confounder control (see section 7)
CUDA_VISIBLE_DEVICES=2 "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_probe_run.py \
  --run-id 20260814-015113-8208536-p1-stage1-fp64score --prompt-index 0 \
  --sparsities 0.80 0.90 --steps 50 --stage 1-fp64score --score-dtype float64 \
  --measure-timestep-stride 1 --raw-root /mnt/scratch/sparsefp4

# Verification gate (must PASS before any aggregate is quoted)
"$FV_PYTHON" artifacts/sparsefp4/configs/phase1_verify_run.py \
  --raw-dir /mnt/scratch/sparsefp4/<run_id> --expect-layers 30 --expect-heads 12 \
  --expect-timesteps 50 --out /mnt/scratch/sparsefp4/<run_id>/verification.json

# Analysis
"$FV_PYTHON" .agents/skills/sparsefp4-video-attention/scripts/analyze_masks.py \
  --raw artifacts/sparsefp4/raw/<run_id> --out-tables artifacts/sparsefp4/tables/<tag> \
  --out-figures artifacts/sparsefp4/figures/<tag> --format both \
  --sparsity 0.80 --sparsity 0.90
"$FV_PYTHON" artifacts/sparsefp4/configs/phase1_deepdive.py \
  --raw artifacts/sparsefp4/raw/<run_id> --out-tables artifacts/sparsefp4/tables/<tag>
```

---

## 3. Null control and harness integrity

The BF16-routing-vs-BF16-reference arm is an identity by construction and was
kept in **every** run as a live gate.

| Run | Null-control records | Deviations from Jaccard 1.0 |
|---|---|---|
| Stage 1 | 72,000 | **0** |
| Stage 2 | 36,000 | **0** |
| Stage 1 fp64-score | 72,000 | **0** |
| fp64 pilot | 7,200 | **0** |

Across **187,200** null-control records: `jaccard == 1.0`, `recall == 1.0`,
`frac_query_blocks_changed == 0.0`, `spearman_rho == 1.0` in every single one. No
run ever deviated, so no run was stopped for debugging.

Additional integrity checks, all passing on every run:

- **Backend identity.** `attention_backend` is `ROUTING_PROBE_ATTN` in 100% of records, and the record lattice is *complete* — all 30 layers x 12 heads x every measured timestep x both CFG branches. Because a record only exists when `RoutingProbeAttentionImpl.forward` actually executes, a complete lattice is positive proof that every DiT self-attention layer resolved onto the probe and none silently fell back (trap 1 of `STATUS.md`). Observed non-null record counts equal the full lattice product exactly: 216,000 / 216,000 (Stage 1) and 1,080,000 / 1,080,000 (Stage 2).
- **Schema invariants** (`EXPERIMENT_SPEC.md` §6.4): 0 violations in 1,720,800 records.
- **Equal budget across arms**: `k_disagreements_across_arms = 0`. `k` is geometry-only and identical across all four precision arms for every cell, so precision changed *which* blocks were picked and never *how many*.
- **`seq_len` constant** at 32760 within every run, so the prompt-length confound of §9.3(3) does not arise (Wan self-attention carries video tokens only).
- **0 malformed lines** written or read.
- **Provenance labelling** is per-arm and machine-checked: `bf16 → native`, `nvfp4 → native`, `fp8_e4m3 → simulated`, `nvfp4_sim → simulated`.

---

## 4. Headline: mask overlap by sparsity x routing precision

Stage 2 (10 prompts, full sparsity sweep). Median, IQR and `n` for every cell.
`n` counts paired `(prompt, layer, head, timestep, cfg_branch)` observations.

| sparsity | retained | routing precision | native/simulated | median | IQR | p10 | min | n |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 | bf16 (control) | native | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 7,200 |
| 0.50 | 0.50 | fp8_e4m3 | simulated | 0.995858 | 0.001762 | 0.992460 | 0.923908 | 72,000 |
| 0.50 | 0.50 | nvfp4 | **native** | 0.988169 | 0.005968 | 0.979159 | 0.916818 | 72,000 |
| 0.50 | 0.50 | nvfp4_sim | simulated | 0.988138 | 0.005998 | 0.979132 | 0.917042 | 72,000 |
| 0.70 | 0.30 | bf16 (control) | native | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 7,200 |
| 0.70 | 0.30 | fp8_e4m3 | simulated | 0.993830 | 0.002873 | 0.988400 | 0.888258 | 72,000 |
| 0.70 | 0.30 | nvfp4 | **native** | 0.983597 | 0.009124 | 0.969969 | 0.892700 | 72,000 |
| 0.70 | 0.30 | nvfp4_sim | simulated | 0.983548 | 0.009174 | 0.969964 | 0.891157 | 72,000 |
| 0.80 | 0.20 | bf16 (control) | native | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 7,200 |
| 0.80 | 0.20 | fp8_e4m3 | simulated | 0.992368 | 0.004213 | 0.984496 | 0.855138 | 72,000 |
| 0.80 | 0.20 | nvfp4 | **native** | 0.980695 | 0.011067 | 0.963585 | 0.857883 | 72,000 |
| 0.80 | 0.20 | nvfp4_sim | simulated | 0.980620 | 0.011067 | 0.963512 | 0.857686 | 72,000 |
| 0.90 | 0.10 | bf16 (control) | native | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 7,200 |
| 0.90 | 0.10 | fp8_e4m3 | simulated | 0.989092 | 0.007866 | 0.975367 | 0.794191 | 72,000 |
| 0.90 | 0.10 | nvfp4 | **native** | 0.973756 | 0.017231 | 0.949048 | 0.801963 | 72,000 |
| 0.90 | 0.10 | nvfp4_sim | simulated | 0.973756 | 0.017083 | 0.949048 | 0.797947 | 72,000 |
| 0.95 | 0.05 | bf16 (control) | native | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 7,200 |
| 0.95 | 0.05 | fp8_e4m3 | simulated | 0.982723 | 0.013556 | 0.961108 | 0.757592 | 72,000 |
| 0.95 | 0.05 | nvfp4 | **native** | 0.961108 | 0.023636 | 0.929835 | 0.745378 | 72,000 |
| 0.95 | 0.05 | nvfp4_sim | simulated | 0.961108 | 0.023636 | 0.930115 | 0.742865 | 72,000 |

Stage 1 (single prompt, all 50 timesteps) agrees to within 0.001: nvfp4 median
0.979803 @ 0.80 (n=36,000) and 0.972733 @ 0.90 (n=36,000).

**Recall and Jaccard are one measurement, not two.** Masks are equal-sized, so
precision == recall and `jaccard = recall / (2 - recall)`. Recall is in the CSVs
for readers who prefer it; it is never cited as corroboration.

### Tail shape — the quantity the PIVOT test actually turns on

`SKILL.md` asks whether overlap is "> 0.95 almost everywhere". That is a statement
about the *fraction of cells below a threshold*, so here it is directly, for the
native NVFP4 arm (n = 72,000 per row):

| sparsity | frac records < 0.99 | < 0.95 | < 0.90 | < 0.80 | p1 | min |
|---|---|---|---|---|---|---|
| 0.50 | 0.826 | 0.00042 | 0 | 0 | 0.9656 | 0.9168 |
| 0.70 | 0.977 | 0.00894 | 0.00019 | 0 | 0.9508 | 0.8927 |
| 0.80 | 0.986 | 0.02901 | 0.00040 | 0 | 0.9396 | 0.8579 |
| 0.90 | 0.988 | 0.10560 | 0.00688 | 0 | 0.9068 | 0.8020 |
| 0.95 | 0.994 | 0.29158 | 0.02378 | 0.00121 | 0.8789 | 0.7454 |

At the pre-registered operating points (0.80 / 0.90) **97.1% / 89.4% of all cells
sit above 0.95** and 99.96% / 99.3% sit above 0.90. Only at 0.95 sparsity — beyond
the range any deployed video sparse-attention method uses — does the sub-0.95
population reach 29%. Nothing anywhere in 1.7M records falls below 0.74.

### The one framing where the effect looks large

`frac_query_blocks_changed` — the fraction of the 256 query blocks in a cell whose
selected set changed *at all* — is dramatic even though Jaccard is not
(native NVFP4, median over cells):

| sparsity | frac_query_blocks_changed (median) | p90 | median frac of *decisions* changed |
|---|---|---|---|
| 0.50 | 0.875 | — | 0.0060 |
| 0.80 | 0.730 | 0.941 (Stage 1) | 0.0102 |
| 0.90 | 0.578 | 0.871 (Stage 1) | 0.0138 |
| 0.95 | 0.465 | — | 0.0202 |

So at 80% sparsity roughly **three quarters of query blocks have at least one
swapped key block, while only ~1% of individual key-block decisions change.** The
disruption is **diffuse, not concentrated**: many blocks each lose one marginal
key block. That distinction matters for Phase 2 — an error model that assumes a
few catastrophically mis-routed blocks is the wrong model here.

### Global score ordering is essentially untouched

Spearman rho between BF16 and candidate block scores over all 512 key blocks
(sampled every 10th timestep, sp 0.90, n = 3,600 per arm):

| arm | median rho | min rho |
|---|---|---|
| bf16 (control) | 1.000000 | 1.000000 |
| fp8_e4m3 | 0.999963 | 0.994403 |
| nvfp4 (native) | 0.999738 | 0.991596 |

NVFP4 preserves the *ranking* of key blocks almost perfectly. What moves is only
the position of the top-k cut line among near-tied candidates.

---

## 5. The timestep trend — the key open question from the 2-step smoke run

The prior 2-step smoke could not see this axis at all. It is now measured at every
one of the 50 denoising steps (Stage 1, n = 720 per step per cell).

**Verdict: overlap does not collapse at any timestep band.** Native NVFP4 at
sparsity 0.90, median Jaccard across all 50 steps:

| step band | mean of per-step medians |
|---|---|
| steps 0–4 (highest noise) | 0.968301 |
| steps 10–39 (middle) | 0.973435 |
| steps 45–49 (lowest noise) | 0.973113 |

- Worst step: **step 0**, median 0.964726.
- Best step: step 38, median 0.974195.
- **Total spread over the entire trajectory: 0.009469** — under one percentage point.

The shape is a monotone rise over roughly the first ten steps, then a flat plateau
for the remaining forty. Step 0 is the single most sensitive timestep at every
sparsity and in every precision arm, and it is the *only* timestep effect present:

| arm | sparsity | step-0 median | best-step median | spread over 50 steps |
|---|---|---|---|---|
| fp8_e4m3 | 0.80 | 0.989550 | 0.992444 | 0.002894 |
| fp8_e4m3 | 0.90 | 0.984496 | 0.989241 | 0.004745 |
| nvfp4 | 0.80 | 0.975316 | 0.980844 | 0.005528 |
| nvfp4 | 0.90 | 0.964726 | 0.974195 | 0.009469 |

Stage 2 reproduces this across all 10 prompts and out to 0.95 sparsity: step 0 is
always the worst step, and the spread never exceeds 0.0115.

**No `(timestep)` cell is affected** under the pre-registered `EXPERIMENT_SPEC.md`
§5.5 threshold (median Jaccard < 0.90): 0 of 50 at every sparsity in Stage 1, and 0
of 10 at every sparsity in Stage 2. Even at the far looser 0.95 threshold, 0 of 50
timestep cells qualify.

So the timestep axis, which was the main thing a 2-step sample could have been
hiding, turns out to carry **almost no structure**. This is a negative finding and
it is a load-bearing one: it removes "schedule the router precision by timestep"
from the list of viable Phase 2/3 methods.

Figure: `artifacts/sparsefp4/figures/main/fig4_overlap_vs_timestep.png`
(+ `.csv`), and per-step values in
`artifacts/sparsefp4/tables/stage1/agg_by_timestep.csv`.

---

## 6. H2 localization — per-layer, per-head, per-(layer, head) rankings

### Affected-cell counts (native NVFP4, `EXPERIMENT_SPEC.md` §5.5 threshold = median Jaccard < 0.90, cells need n >= 20)

Stage 2, 10 prompts:

| granularity | sparsity | eligible cells | affected (<0.90) | <0.95 | frac <0.95 | worst cell | worst median |
|---|---|---|---|---|---|---|---|
| layer | 0.80 | 30 | **0** | 0 | 0.000 | layer 0 | 0.963695 |
| layer | 0.90 | 30 | **0** | 0 | 0.000 | layer 0 | 0.952336 |
| layer | 0.95 | 30 | **0** | 3 | 0.100 | layer 0 | 0.934040 |
| head | 0.80 | 12 | **0** | 0 | 0.000 | head 2 | 0.978614 |
| head | 0.90 | 12 | **0** | 0 | 0.000 | head 2 | 0.973025 |
| head | 0.95 | 12 | **0** | 0 | 0.000 | head 6 | 0.958223 |
| timestep | 0.80–0.95 | 10 | **0** | 0 | 0.000 | step 0 | 0.952192 |
| layer x timestep | 0.90 | 300 | **0** | 7 | 0.023 | L28/step0 | 0.930814 |
| layer x timestep | 0.95 | 300 | **0** | 35 | 0.117 | L28/step0 | 0.918156 |
| layer x head | 0.80 | 360 | **0** | 8 | 0.022 | L1H1 | 0.937969 |
| layer x head | 0.90 | 360 | **2** | 24 | 0.067 | L28H11 | 0.868089 |
| layer x head | 0.95 | 360 | **5** | 100 | 0.278 | L28H11 | 0.811773 |
| layer x head x timestep | 0.90 | 3,600 | **25** | 322 | 0.089 | L28H4/step0 | 0.861298 |
| layer x head x timestep | 0.95 | 3,600 | **59** | 1,026 | 0.285 | L28H4/step0 | 0.792862 |

Stage 1 (single prompt, all 50 timesteps) agrees: 0 affected `(layer, timestep)`
cells out of 1,500 at either sparsity, and 3 affected `(layer, head)` cells out of
360 at 0.90.

**Reading this honestly:** at the two pre-registered operating points, *nothing* is
affected at layer, head, or timestep granularity. Localization only appears when
you intersect layer with head — and even then it is 2 cells out of 360 at sparsity
0.90. The pre-registered threshold "selects nothing" at coarse granularity, which
§5.5 anticipated and instructed to report rather than re-tune. The full
distributions are in the tables; the threshold was not moved.

### The sensitive regions, named

The structure that does exist is real, reproducible across all 10 prompts, and
concentrated at the **ends of the network**:

- **Layers.** The most sensitive layers are **28, 0, 29, 1, 27, 2** — i.e. the first two and last three blocks. Native NVFP4 @ 0.90, Stage 1 per-layer medians: layer 28 = 0.940666, layer 0 = 0.949048, layer 29 = 0.952694, layer 1 = 0.957215, layer 2 = 0.961758, layer 27 = 0.964001. The most stable layers are the middle band **24, 18, 9, 13, 10, 11, 6, 7** (0.976–0.982). Total per-layer spread: **0.041** at 0.90, **0.039** at 0.95.
- **Heads.** Essentially unstructured. Per-head medians span only **0.0035** at sparsity 0.90 (worst head 2 at 0.973025, best head 1 at 0.975221) and **0.0061** at 0.95. Heads are ~12x less discriminative than layers. Any claim that specific heads are routing-sensitive is **not supported** by this data.
- **`(layer, head)` cells.** This is where the signal concentrates. At 0.95 sparsity the worst cells are **L28H11 (0.8118), L0H9 (0.8657), L29H9 (0.8821), L23H8 (0.8837), L0H3 (0.8993)**, versus a best of L5H3 = 0.9873 — a spread of **0.176**. At 0.90 the two affected cells are **L28H11 (0.8681)** and **L0H9**. Stage 1 additionally flags **L25H7 (0.8408 @ 0.90)**.
- **`(layer, head, timestep)` cells.** Worst is **L28H4 @ step 0** (0.8613 @ 0.90, 0.7929 @ 0.95). The affected set is dominated by `step 0` combined with a late layer.

So H2's honest form is: *sensitivity is concentrated in a handful of
`(layer, head)` pairs at the first and last few transformer blocks, most acutely at
the highest-noise timestep, but the effect is absent at any single-axis
granularity and involves ~0.5–1.4% of cells at the pre-registered operating
points.*

Figures: `fig2_layer_timestep_jaccard_s0.{80,90}_nvfp4.png` (layer x timestep
heatmap, all 50 steps) and `fig3_head_jaccard_box_s0.{80,90}_nvfp4.png`, with
value CSVs alongside. Rankings: `tables/stage2/ranked_cells.csv` (6,180 rows),
`agg_by_layer_head.csv`, `agg_by_layer.csv`, `agg_by_head.csv`.

### Content generality

Stage 2's 10-prompt spread is narrow. Native NVFP4 @ 0.90: per-prompt medians run
0.968648 (p07, aerial drone) to 0.975954 (p04/p06/p08, near-static subjects),
n = 7,200 each. Motion-heavy and camera-moving prompts are consistently ~0.005–0.007
less stable than static ones — a real but tiny content effect. This is a
**development-set** statement on 10 prompts and a **single seed**; it is not a
benchmark claim, and seed-robustness is untested.

CFG branches are statistically indistinguishable (negative 0.972440 vs positive
0.972879 @ 0.90, n = 18,000 each), so pooling them is safe.

---

## 7. Confounders found

### 7.1 fp32 block-score resolution inflates the measured instability (material; quantified; corrected)

This was discovered during Phase 1 and is the most important methodological finding
of the phase.

**Symptom.** Every cell reported ~104–115 exact ties at the top-k selection boundary
(`boundary_ties`), including the BF16-vs-BF16 null control. Ties at the boundary are
expected from *quantization* (`EXPERIMENT_SPEC.md` §1.3 predicts NVFP4 creates them)
but not from BF16 compared against itself.

**Diagnosis.** The pooled block scores carry a large common-mode component. Measured
from the records: median raw boundary margin `s_(k) - s_(k+1)` is exactly
**0.03125** = 2^-5, the second most common value is 2^-6, the third 2^-4 — the
margins are quantized onto a **power-of-two grid**, the signature of catastrophic
cancellation in a float subtraction. The implied score magnitude is
`0.03125 / 2^-24 ≈ 5.2e5`, while the discriminative spread `s_(1) - s_(n)` is only
**≈ 14.2**. So the informative part of the block score is ~4.5 decimal orders of
magnitude smaller than the score itself, and fp32 has ~7 digits — leaving barely
2 digits to resolve the top-k cut. A control on synthetic zero-mean Gaussian Q/K
produced **0 ties**, confirming the cause is the real data's mean offset (the
block-mean pooling of `EXPERIMENT_SPEC.md` §3.1 does not remove it), not TF32
(`allow_tf32` is `False` and `float32_matmul_precision` is `highest`, both verified).

**Correction.** A full Stage-1 arm was re-run with the score matmul in **fp64**,
everything else byte-identical (run `...-p1-stage1-fp64score`, 288,000 records,
verified PASS, null control clean):

| sparsity | arm | fp32 median | fp64 median | delta | fp32 boundary ties | fp64 ties | fp32 frac<0.95 | fp64 frac<0.95 |
|---|---|---|---|---|---|---|---|---|
| 0.80 | fp8_e4m3 | 0.992143 | 0.995006 | **+0.002863** | 115 | **0** | 0.000583 | **0.000000** |
| 0.80 | nvfp4 | 0.979803 | 0.981588 | **+0.001785** | 115 | **0** | 0.040472 | 0.034944 |
| 0.90 | fp8_e4m3 | 0.988647 | 0.993411 | **+0.004764** | 104 | **0** | 0.009639 | 0.002778 |
| 0.90 | nvfp4 | 0.972733 | 0.975807 | **+0.003074** | 105 | **0** | 0.139944 | 0.111750 |
| — | bf16 (null) | 1.000000 | 1.000000 | 0.000000 | 104–115 | **0** | 0 | 0 |

fp64 eliminates boundary ties entirely and raises every candidate arm's median.

**Consequences.**

1. **All headline numbers in this report are the fp32, pre-registered ones, and they are therefore *conservative* — they overstate NVFP4 routing instability** by roughly +0.002 (sp 0.80) to +0.003 (sp 0.90) of Jaccard. The true effect is *smaller* than reported, which strengthens the PIVOT conclusion rather than weakening it.
2. The **fp8_e4m3 arm is hit ~1.6x harder** than NVFP4 (+0.0048 vs +0.0031 at sp 0.90), because its genuine perturbation is small enough to be comparable to the numerical noise floor. **Any H3 measurement that compares an FP8 router against an NVFP4 router in fp32 would be partly measuring float resolution, not precision.** Phase 2 must use fp64 (or mean-centred) block scores.
3. The null control **still passed** throughout, because both sides of the identity hit the same grid the same way. This is a case where a passing null control does *not* certify numerical adequacy — worth recording as a lesson.

The `--score-dtype` knob is additive and defaults to `float32` (the pre-registered
value); the fp64 run is labelled `stage: 1-fp64score` and every record carries a
`score_dtype` field, so the arms can never be silently pooled.

### 7.2 NVFP4 e2m1 saturation is heavy (mechanism, not a bug)

Native NVFP4 Q/K saturates at the e2m1 max (±6.0) in **10.55%** of Q elements and
**10.54%** of K elements (median over cells, sp 0.90). FP8 e4m3 saturates in
0.0001%. This ~10% saturation is the mechanistic origin of the NVFP4 routing
perturbation and matches the independent Phase-0 quantizer probe (0.111). It is
recorded per-record as `sat_frac_q` / `sat_frac_k`, not inferred.

### 7.3 The margin mechanism is confirmed but weak

Binning cells by BF16 reference decision-margin decile (`EXPERIMENT_SPEC.md` §5.3
asks for the relationship, not an assertion). Stage 2, native NVFP4:

| reference-margin decile | median margin | median frac decisions changed @ sp 0.90 | @ sp 0.95 |
|---|---|---|---|
| 1 (smallest) | 0.000000 | 0.01758 | 0.02945 |
| 4 | 0.001305 | 0.01735 | 0.02148 |
| 7 | 0.002151 | 0.01149 | 0.01638 |
| 10 (largest) | 0.003525 | 0.00714 | 0.01322 |

The direction is as predicted — instability concentrates where the BF16 top-k
boundary is nearly tied — and the effect is a **~2.5x** ratio from top to bottom
decile at sp 0.90. But deciles 1–3 all have a median margin of exactly 0.0, which
is the §7.1 artifact, so the low-margin end of this relationship is partly
numerical. The monotone decline across deciles 4–10 (0.0174 → 0.0071), where
margins are resolved, is the trustworthy part.

### 7.4 Simulated NVFP4 is a faithful stand-in for native (cross-check passes)

Paired per-cell, native `nvfp4` minus simulated `nvfp4_sim` Jaccard, Stage 2:

| sparsity | n paired cells | frac identical | mean diff | median diff | max abs diff |
|---|---|---|---|---|---|
| 0.50 | 72,000 | 0.074 | +2.2e-05 | +3.0e-05 | 0.0017 |
| 0.80 | 72,000 | 0.082 | +2.4e-05 | 0.0 | 0.0028 |
| 0.90 | 72,000 | 0.095 | +3.4e-06 | 0.0 | 0.0046 |
| 0.95 | 72,000 | 0.107 | −1.1e-05 | 0.0 | 0.0087 |

The two arms agree to a mean of **3.4e-06** at sparsity 0.90 — three orders of
magnitude below the effect being measured — with no systematic sign. Keeping both
arms was worth it: it independently validates the deterministic e2m1 round-trip
against flashinfer's real `fp4_quantize_sm100`, so Phase 2 may use the simulated
quantizer where the native packed layout cannot be read back. **Labelling still
differs and must stay differentiated**: `nvfp4` is `native`, `nvfp4_sim` is
`simulated`, and neither may appear in a latency table.

### 7.5 Confounders checked and found inert

- **Token ordering / block geometry.** All records use the raster-order 128x64 diagnostic geometry (`scorer = diag_mean_pool_dot`). The VSA 64-token-cube scorer was **not** run, so no result here is conflated with VSA's tile ordering (trap 3). Any VSA-integrated arm will need its own records at 64x64 — those two geometries are different partitions of tokens and are not comparable.
- **Ragged final block.** `ragged_tail = 56` of 64 in every record, constant across arms; `n_k_blocks = 512` identical for all arms, so it cannot bias a comparison. Not excluded, and no headline number depends on it.
- **`softmax_scale` ordering.** Applied only to block scores, after quantization, as `EXPERIMENT_SPEC.md` §9.3(6) requires; recorded as 0.0883883 in every record.
- **Capture point.** Every arm derives from one BF16 capture at `AttentionImpl.forward` — post-RMSNorm, post-RoPE, post-SP-layout — so the forbidden "early BF16 vs late low-precision" comparison cannot occur by construction. Limitation: the simulated arms model "quantize the final Q/K", not "propagate low precision through Q/K-norm and RoPE".
- **Determinism.** Single seed 1234, deterministic quantizers, stable index-ascending tie-break. Run-to-run repeatability was not separately quantified; the fp64 arm reproduced the fp32 arm's *structure* (same worst layers, same worst step) which is indirect evidence of stability.

---

## 8. Exclusions

Every exclusion, with count and reason. **No run, shard, or outlier was dropped.**

| Exclusion | Count | Reason |
|---|---|---|
| Records excluded from analysis | **0** | Nothing was filtered. All 1,720,800 records were loaded and aggregated. |
| Malformed lines | **0** | None produced or encountered. |
| Records failing schema invariants | **0** | None. |
| Failed / crashed shards | **0** | All 8 Stage-2 GPU shards and all 13 prompt-runs completed; `grep "SHARD FAILED"` returns nothing. |
| `spearman_rho` = null | 129,600 of 144,000 in the sp-0.90 subset | **By design**, not a failure: Spearman is emitted only every 10th timestep (`--spearman-timestep-stride 10`) because tie-corrected ranking over 512 key blocks x 12 heads is the most expensive metric. n is reported wherever rho is quoted (n = 3,600). |
| Null control subsampled in Stage 2 | layers every 5th, timesteps every 10th | The bf16-vs-bf16 identity is invariant by construction; full enumeration would have added ~40% output volume to assert one identity. 36,000 control records were still written and all 36,000 pass. Stage 1 ran the control at **full** density (72,000 records, every layer, every timestep). |
| Stage 2 timestep decimation | steps {0,5,...,45} of 50 | Deliberate, deterministic, recorded in every shard's `probe_config_*.json` as `measure_timestep_stride: 5`. The full 10-prompt x 50-step x 5-sparsity enumeration projects ~8 GB of JSONL, past the `EXPERIMENT_SPEC.md` §11 5 GiB cap. **Stage 1 covers the timestep axis at full 50-step density**, so no axis is left unmeasured — the two stages are complementary by design, as §11 rule 2 requires (reduce enumerated cells by a recorded deterministic rule, never truncate output). |
| `(layer, head, timestep)` cells flagged `insufficient_n` | 18,000 of 18,000 in Stage 1 | At n = 4 per cell (2 CFG branches x 1 prompt x 1 seed) this granularity is below the §10.3 n>=20 bar in Stage 1 and is **excluded from all claims there**. Stage 2 reaches n = 20 per cell (10 prompts x 2 branches) and is the only place `(layer, head, timestep)` claims are made. |
| Uncompressed raw JSONL kept off the repo volume | 2.0 GB | Root volume has 4.7 GB free. Gzipped copies (99 MB, integrity-verified, line-count-verified) are under `artifacts/sparsefp4/raw/`; uncompressed originals are at `/mnt/scratch/sparsefp4/<run_id>/` which is **ephemeral instance store** and will not survive an instance stop. |

---

## 9. Hypothesis verdicts

### H1 — "NVFP4 Q/K changes the top block selection relative to BF16, especially at high sparsity"

**SUPPORTED IN DIRECTION, BUT THE EFFECT IS SMALL.**

The direction is unambiguous and the ordering is monotone and clean in every one of
the 20 sparsity x precision cells, across 1.4M records, 10 prompts, and both CFG
branches:

```
bf16 (1.0000)  >  fp8_e4m3  >  nvfp4 ≈ nvfp4_sim        at every sparsity
median Jaccard falls monotonically with sparsity for every arm
```

- Native NVFP4 median Jaccard: **0.9882** @ sp 0.50 → **0.9807** @ 0.80 → **0.9738** @ 0.90 → **0.9611** @ 0.95 (n = 72,000 each).
- The "especially at high sparsity" clause holds: the median gap from 1.0 grows **3.3x** from sparsity 0.50 (0.0118) to 0.95 (0.0389), and IQR grows **4.0x** (0.0060 → 0.0236).
- FP8 sits cleanly between BF16 and NVFP4 everywhere (0.9891 @ 0.90), closing **58%** of NVFP4's median deficit, so the effect scales with router precision as predicted.
- The effect is not numerical noise: the null control is exactly 1.0 on 187,200 records, and the fp64 control confirms the *ordering* survives removal of the float-resolution artifact.

**But the magnitude is far below what the hypothesis needs to be interesting.** At
the two pre-registered operating points, **97.1% (sp 0.80) and 89.4% (sp 0.90) of
all cells have Jaccard above 0.95**, and 99.96% / 99.3% are above 0.90. Spearman rho
is 0.9997. And the true effect is *smaller still* — the fp64 control shows the
reported fp32 numbers overstate instability by +0.002 to +0.003.

So H1 is real but weak: **NVFP4 perturbs routing measurably and systematically, and
the perturbation is too small to break the mask.**

### H2 — "routing instability is concentrated in particular heads, layers, and/or diffusion timesteps rather than uniform"

**WEAKLY SUPPORTED ON THE LAYER AXIS, UNSUPPORTED ON THE HEAD AND TIMESTEP AXES.**

Judged against the pre-registered §5.5 definition (median Jaccard < 0.90, n >= 20):

| axis | affected cells @ sp 0.90 | affected @ sp 0.95 | verdict |
|---|---|---|---|
| timestep | **0 / 50** (Stage 1), 0 / 10 (Stage 2) | 0 / 10 | **UNSUPPORTED.** Total spread over the full 50-step trajectory is 0.0095. |
| head | **0 / 12** | 0 / 12 | **UNSUPPORTED.** Per-head spread is 0.0035 — heads are interchangeable. |
| layer | **0 / 30** | 0 / 30 (3 below 0.95) | **NOT AFFECTED at threshold**, but real structure: spread 0.041, monotone edge-vs-middle pattern. |
| layer x head | **2 / 360** | 5 / 360 | **WEAKLY SUPPORTED.** |
| layer x head x timestep | **25 / 3,600** (0.7%) | 59 / 3,600 (1.6%) | **WEAKLY SUPPORTED.** |

**Named sensitive regions** (reproducible across all 10 prompts):

- **Layers 28, 0, 29, 1, 27, 2** — the first two and last three transformer blocks. Layer 28 is worst (median 0.9407 @ sp 0.90). Middle layers 6–24 are the most stable (0.976–0.982). This edge-vs-middle pattern is the single clearest piece of structure in the whole phase.
- **Timestep 0** (the highest-noise step) is the worst step at every sparsity in every arm — but only by 0.0095 of Jaccard.
- **Specific `(layer, head)` cells: L28H11 (0.8681 @ 0.90, 0.8118 @ 0.95), L0H9 (0.8657 @ 0.95), L29H9 (0.8821), L23H8 (0.8837), L25H7 (0.8408 @ 0.90 in Stage 1), L0H3 (0.8993), L12H1, L27H2.**
- **Worst triple: L28H4 at timestep 0** (0.8613 @ sp 0.90, 0.7929 @ 0.95).
- **Heads alone are not a sensitivity axis** — this is a clean negative and it kills any per-head router-precision scheme.

The honest summary: structure exists, it is reproducible, it is located at the
network's first and last blocks in specific head pairings and at the highest-noise
step — but it involves **0.7% of cells at sparsity 0.90** and no coarse axis crosses
the pre-registered threshold. That is "strongly structured" only in the weakest
sense; it is not the "substantial layer/head/timestep regions with visibly lower
overlap" that `SKILL.md`'s GO criterion A asks for.

### H3 — "precision-decoupled routing recovers quality"

**UNTESTED** (Phase 2). Phase 1 provides its input: FP8 routing recovers
**58%** of NVFP4's median Jaccard deficit at sparsity 0.90 (deficit 0.0262 → 0.0109)
and **60%** at 0.80 (0.0193 → 0.0076). Whether that translates into attention-output
error recovery is exactly the Phase-2 question — but note that a 0.026 Jaccard
deficit is a small budget from which to recover 20% of a *downstream error*.

### H4 — "native sparse-NVFP4 gives wall-clock benefit"

**UNTESTED**, unchanged from Phase 0, and now the most promising branch. Phase 0
already measured the bar: native NVFP4 attention at the real Wan shape runs
**4.013 ms** vs **5.135 ms** for native BF16 FA4 (**1.28x**, kernel-only, warmed,
CUDA-synced, median of 20).

---

## 10. GO / PIVOT read

**My read: PIVOT.** The pre-registered PIVOT condition of `SKILL.md` is met almost
verbatim:

> "If BF16 and NVFP4 masks are nearly identical almost everywhere (e.g. Jaccard > 0.95 at 80–90% sparsity) **and** higher-precision routing does not improve output error, do not force the hypothesis."

The first clause is satisfied: median Jaccard is **0.9807 at sp 0.80** and **0.9738
at sp 0.90**, with 97.1% / 89.4% of individual cells above 0.95 — and the fp64
control says even these are pessimistic. The second clause is Phase 2's to test, so
the PIVOT is not yet final; but the routing evidence alone will not carry a paper.

Reasons to call it now rather than hope Phase 2 rescues it:

1. **The effect size is not merely small, it is small in the *right* direction to be uninteresting.** Spearman rho of 0.9997 means NVFP4 preserves the block ranking almost perfectly; only the cut line among near-ties moves. Blocks swapped at a near-tied boundary are by definition the blocks whose contributions are most similar, so the *output* error from swapping them is second-order. A 0.026 median Jaccard deficit made of near-tie swaps is a weak generator of downstream error.
2. **The timestep axis — the main thing the 2-step smoke could have been hiding — is flat.** 0.0095 spread over 50 steps. This removes the most attractive method ("spend router precision only where it matters in the trajectory") and it is a genuinely new negative result, not a null from insufficient data (n = 720 per step, 50 steps, full enumeration).
3. **The head axis is flat too** (0.0035 spread), removing per-head schemes.
4. **What structure remains is 0.7% of cells** and sits at layers 0/1/27/28/29 — plausibly explainable by those blocks' activation statistics rather than by anything specific to routing. A method targeting 0.7% of cells cannot deliver a headline quality number.
5. **The one measurement that looked alarming resolves to a confounder in our favour.** The heavy `frac_query_blocks_changed` (73% of query blocks touched at sp 0.80) sounds severe, but pairs with ~1% of decisions changed — diffuse single-block churn at the boundary, plus a numerical artifact inflating the tail.

**Which pivot.** Of the three `SKILL.md` names, the evidence points at a combination
of (3) and (1):

- **Primary — pivot (3), the negative result:** *NVFP4 routing is unexpectedly stable in video DiTs, so block-sparse routing can be computed entirely in low precision.* This is a genuinely useful systems finding: it says a sparse-NVFP4 kernel does **not** need a higher-precision router side-path, which is a real simplification with a real cost saving, and it is backed by 1.7M paired measurements across 10 prompts, 5 sparsities, 30 layers, 12 heads and 50 timesteps with an exact null control. Quantified claim: median mask Jaccard >= 0.974 at sparsity <= 0.90 with block ranking preserved at rho = 0.9997.
- **Secondary — pivot (1), systems composability:** whether sparse tile skipping composes with native NVFP4 in **wall-clock** terms on Blackwell. This is where the remaining upside is, the dense-NVFP4 bar is already measured at 4.013 ms, and `STATUS.md` records that the native FP4 kernel is editable Python/CuTeDSL with the toolchain present.

A `BORDERLINE` rubric entry would also be defensible if Phase 2 finds that the
0.7% of affected `(layer, head)` cells produce disproportionate output error. I would
not bet on it, but it is cheap to check and Phase 2 should check it before the pivot
is locked.

**What would change my mind:** Phase 2 showing that `err(D) - err(C)` (wrong-mask
error at BF16 compute) is a large fraction of `err(B)` (pure quantization error).
If wrong-mask error turns out to dominate quantization error despite the tiny mask
difference, H1's small magnitude would not matter and GO would be back on.

---

## 11. What Phase 2 should target

Concrete, and mostly determined by Phase 1's numbers.

1. **Use fp64 (or mean-centred) block scores. Non-negotiable.** §7.1 shows fp32 scores put +0.003 of artificial instability on NVFP4 and +0.005 on FP8. Since H3 *is* the FP8-vs-NVFP4 router comparison, running it in fp32 would measure float resolution as if it were precision. Either compute scores in fp64 or subtract the per-head block-mean before the matmul (cheaper, removes the common mode, does not change top-k ordering) — and verify boundary ties go to 0 before trusting any H3 number.
2. **Regions to run A–F on** (`EXPERIMENT_SPEC.md` §7.3 wants >= 3 affected and >= 3 unaffected):
   - *Affected:* **L28H11**, **L0H9**, **L29H9**, **L23H8**, **L25H7**, each at **timestep 0** and at a mid-trajectory step for contrast.
   - *Unaffected controls:* **L5H3**, **L6H6**, **L10H9**, **L13H5**, **L11H7** — middle-band layers with medians 0.979–0.987.
   - Sparsities **0.80, 0.90, and 0.95**. Include 0.95: it is the only setting where the effect reaches 29% of cells below 0.95, so it is the best chance of seeing H3 at all. Report it as out-of-deployment-range.
3. **Expect a small H3 effect and pre-commit to reporting it as such.** FP8 routing closes 58% of the Jaccard deficit at sp 0.90, but the absolute deficit is 0.026 and the swapped blocks are near-ties. The pre-registered bar is a >= 20% relative reduction in median `rel_l2`; report the paired per-cell difference distribution and the **fraction of cells that improve**, not just the median (§8.2 rule 1). A 20% median reduction where only 55% of cells improve is a different result and both must be visible.
4. **Test the near-tie hypothesis directly** — this is the highest-value new measurement. For a swapped block pair, measure how much the *attention output* actually changes as a function of the score gap between the dropped and added block. If output error scales with that gap (which rho = 0.9997 and the margin-decile trend both predict), that single plot explains the whole phase and justifies the negative result far better than another aggregate table.
5. **Do not build a per-timestep or per-head router-precision schedule.** Both axes are measured flat: 0.0095 and 0.0035 of spread. If any precision scheduling is worth trying it is **per-layer** (spread 0.041, edge-vs-middle), i.e. higher-precision routing in layers {0, 1, 27, 28, 29} only — 5 of 30 layers, ~17% of the router cost. That is the only Phase-1-supported method proposal, and Phase 1 predicts its ceiling is small.
6. **Check whether layers 0/1/27/28/29 are special for a boring reason** before claiming routing sensitivity. Measure their Q/K activation range and e2m1 saturation directly; the ~10.5% saturation is the mechanism, and if the edge layers simply have wider activations, the correct framing is "NVFP4 saturation tracks activation range", not "routing is layer-sensitive". This is a cheap, decisive control.
7. **Carry the geometry caveat.** Every Phase-1 number is the raster-order **128x64** diagnostic scorer. VSA's deployed router is a **64-token (4,4,4) spatio-temporal cube** with `block_q == block_k`. These are different partitions of tokens and Jaccard is not transferable between them. If Phase 3 integrates VSA, re-measure at 64x64 — the probe already supports it via `run_vsa_scorer`, and it was deliberately left off here so no result is conflated.
8. **Scope discipline.** Do not spend time on a fused kernel for the H3 method. If Phase 2 confirms the PIVOT, the remaining engineering value is in pivot (1) — measuring whether sparse tile skipping composes with native NVFP4 wall-clock — and that work does not need a precision-decoupled router at all.

---

## 12. Artifact index

| What | Path |
|---|---|
| Raw records (archival, gzipped, verified) | `artifacts/sparsefp4/raw/<run_id>/*.jsonl.gz` (99 MB, 1,720,800 records) |
| Raw records (uncompressed, **ephemeral**) | `/mnt/scratch/sparsefp4/<run_id>/*.jsonl` (2.0 GB) |
| Per-run verification verdicts | `artifacts/sparsefp4/raw/<run_id>/verification.json` |
| Resolved run configs | `artifacts/sparsefp4/raw/<run_id>/probe_config_*.json`, `run_summary_*.json` |
| Stage 1 tables (50 timesteps) | `artifacts/sparsefp4/tables/stage1/`, `tables/main_stage1/` |
| Stage 2 tables (10 prompts, 5 sparsities) | `artifacts/sparsefp4/tables/stage2/`, `tables/main_stage2/` |
| fp64 confounder-control tables | `artifacts/sparsefp4/tables/stage1_fp64score/` |
| Figure 1 — mask overlap vs sparsity (+CSV) | `artifacts/sparsefp4/figures/main/fig1_mask_overlap_vs_sparsity.{png,csv}` |
| Figure 2 — layer x timestep heatmap, sp 0.80 / 0.90 (+CSV) | `artifacts/sparsefp4/figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.{80,90}_nvfp4.{png,csv}` |
| Figure 3 — per-head boxplots (+CSV) | `artifacts/sparsefp4/figures/main/fig3_head_jaccard_box_s0.{80,90}_nvfp4.{png,csv}` |
| Figure 4 — overlap vs timestep (+CSV) | `artifacts/sparsefp4/figures/main/fig4_overlap_vs_timestep.{png,csv}` |
| Tail / below-threshold fractions | `tables/*/tail_by_sparsity_precision.csv` |
| Affected-cell counts, all granularities | `tables/*/affected_cell_counts.csv` |
| Per-layer / per-head / per-cell rankings | `tables/*/ranked_cells.csv`, `agg_by_layer_head.csv`, `agg_by_layer.csv`, `agg_by_head.csv`, `agg_by_timestep.csv` |
| Margin-decile mechanism | `tables/*/margin_decile_vs_changed.csv` |
| Native-vs-simulated NVFP4 cross-check | `tables/*/native_vs_simulated_nvfp4.csv` |
| Shard logs (Stage 2, one per GPU) | `artifacts/sparsefp4/logs/stage2_<run_id>_gpu{0..7}.log` |

Every figure ships the CSV of its exact plotted values alongside the PNG; the CSV
is the archival artifact.

### Code touched in Phase 1

| File | Change |
|---|---|
| `fastvideo/attention/backends/routing_probe_attn.py` | Additive: record the scheduler's own timestep value (not just the loop index) per §9.3(4); `measure_timestep_stride` for deterministic timestep decimation; `score_dtype` for the fp64 confounder control. No change to any metric definition. |
| `.agents/skills/sparsefp4-video-attention/scripts/analyze_masks.py` | Added `agg_by_timestep`, `agg_by_layer_head`, `agg_by_prompt`, `agg_by_cfg_branch` tables; Figure 4 (overlap vs timestep); timestep-trend and per-layer/head/(layer,head) ranking sections in `summary.md`; medians for `frac_query_blocks_changed`, `boundary_ties`, `spearman_rho`, `sat_frac_q`; readable figure scales (Figure 1 dual-panel, Figure 2 data-driven colour range). Self-test extended and passing. |
| `artifacts/sparsefp4/configs/phase1_probe_run.py` | `--measure-timestep-stride`, `--score-dtype`. |
| `artifacts/sparsefp4/configs/phase1_stage2_launch.sh` | **New.** 8-GPU Stage-2 sharding, one process per GPU at `sp_size=1`, `nohup` per shard with per-shard logs. |
| `artifacts/sparsefp4/configs/phase1_verify_run.py` | **New.** Pre-analysis gate: lattice completeness (proves no layer fell off the probe backend), null-control check, §6.4 invariants, cross-arm `k` equality, backend/`seq_len` constancy. |
| `artifacts/sparsefp4/configs/phase1_deepdive.py` | **New.** Tail quantiles and below-threshold fractions, margin-decile mechanism, paired native-vs-simulated diff, affected-cell counts at 6 granularities, ranked cells. |

The 3 one-line backend registrations (`platforms/interface.py`, `platforms/cuda.py`,
`configs/models/dits/base.py`) are unchanged from Phase 0. `pre-commit run --files`
passes on every source file touched. Nothing was committed.
