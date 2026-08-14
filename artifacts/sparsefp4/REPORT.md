# SparseFP4 Video Attention — Final Report

**Study question.** Does NVFP4 quantization perturb dynamic block-sparse attention
routing in a Video DiT, and can a cheap higher-precision router recover the lost
quality while the expensive attention compute stays low precision?

**Answer.** The perturbation is real and the recovery hypothesis is **falsified**,
with a measured mechanism: quantization moves the top-k decision only where the
decision is nearly degenerate, so the swaps it makes are 27–76x cheaper than an
equal-count random perturbation. Low-precision routing is safe.

Repo `/home/ec2-user/FastVideo` @ `8208536cd1db7a1d32b68aaa6a679953ae23ab8b`,
branch `exp/sparsefp4-mask-stability`. Date 2026-08-14.
Decision spine: [`GO_NO_GO.md`](GO_NO_GO.md). Phase detail:
[`PHASE0.md`](PHASE0.md), [`PHASE1.md`](PHASE1.md), [`PHASE2.md`](PHASE2.md),
[`PHASE2B_GEOMETRY.md`](PHASE2B_GEOMETRY.md), [`PHASE5.md`](PHASE5.md).

**Terminology, binding on every table below.** "NVFP4" means **NVFP4 Q/K with BF16
PV** — the FA4 kernel is `qk_mode=nvfp4, pv_mode=bf16`
([`env.json`](env.json), `attention_stack.probe.attn_qat_infer_receipt`). This
study contains no fully-FP4 attention. Sparse-NVFP4 *compute* has no native kernel
in this environment; every such arm is simulated, marked `numerical_only`, and
appears in no latency table.

---

## 1. Executive conclusion

- **NVFP4 Q/K routing perturbs the sparse mask measurably but weakly.** Median
  BF16↔NVFP4 mask Jaccard is **0.9807** at sparsity 0.80 and **0.9738** at 0.90
  (native NVFP4 router, n = 72,000 cells per row); 97.1% / 89.4% of cells sit
  above 0.95, and block-score Spearman rho is **0.9997** (n = 36,000). The
  ordering `bf16 (1.0) > fp8 > nvfp4` is monotone in all 20 sparsity × precision
  cells. Source: `figures/main/fig1_mask_overlap_vs_sparsity.csv`,
  `tables/stage2/tail_by_sparsity_precision.csv`.
- **The instability is not localized, so it is not schedulable.** At the
  pre-registered threshold (median Jaccard < 0.90, n ≥ 20) the affected-cell count
  is **0 of 50 timesteps, 0 of 12 heads, 0 of 30 layers** at sparsity 0.90; only
  `(layer, head)` shows 2 of 360. Whole-trajectory timestep spread is **0.0095**
  and per-head spread **0.0035**. The residual edge-layer structure is *not*
  explained by NVFP4 saturation: saturation is flat across all 17 measured layers
  (Q 0.0958–0.1098) and correlates with wrong-mask error at Spearman **−0.25** —
  the wrong sign. Source: `tables/stage2/affected_cell_counts.csv`,
  `tables/phase2_main/table8_saturation_vs_layer_sensitivity.csv`.
- **H3 is falsified by 200–2,200x, and the BF16 router — the theoretical ceiling —
  is no better than FP8.** Replacing the NVFP4 router with FP8 or BF16 reduces
  attention-output relative L2 by **0.037%–0.104%** against a pre-registered
  **≥ 20%** threshold (n = 20,400 exactly paired cells per sparsity, 17 layers);
  the 30-layer re-measurement gives **0.0092%/0.0093%** (n = 17,280). Only
  52–56% of cells improve — a coin flip. The wrong-mask term H3 proposes to remove
  is **0.016%–0.032%** of total error. Source:
  `tables/phase2_main/table3_h3_paired.csv`, `table4_error_attribution.csv`,
  `raw/20260814-032700-8208536-p5-main/phase5_singlestep_medians.json`.
- **The null has a mechanism, established by an equal-magnitude contrast control.**
  `C_rand` changes exactly as many blocks from the same baseline mask under the
  same budget but picks them at random. It costs **27.0x / 21.7x / 10.1x** more
  output error at sparsity 0.80 / 0.90 / 0.95 (128x64 raster) and **75.6x /
  44.6x / 35.7x** at VSA's deployed 64-token `(4,4,4)` cube geometry. So the
  finding is not "too few errors to matter" — it is **quantization errs only where
  erring is nearly free**: a swapped block carries 3.8–11.1x less attention mass
  than a block both masks keep, and the swapped pair's score gap is 0.42–0.50% of
  the score spread. Source:
  `tables/phase2_main/table7_random_perturbation_contrast.csv`,
  `tables/phase2b_geometry/table1_three_geometry_headline.csv`.
- **Native vs simulated, stated plainly.** Dense BF16 and dense NVFP4 (Q/K only)
  are **native** end-to-end paths with measured latency. Every sparse-NVFP4
  compute arm is **simulated** (bounded against its native twin at a median
  relative disagreement of 9.6e-8) and carries no latency claim. The only measured
  performance numbers are the dense attention-kernel microbenchmark
  (**1.28x**, kernel in isolation) and the dense end-to-end generation
  (**1.055x**, 44.436 s vs 46.876 s). These are different quantities.
- **Paper verdict: GO** — as a negative result with a measured mechanism, confirmed
  at the deployed block geometry and at the video level, plus a reusable
  methodological control. **Pivot chosen: SKILL options (3) negative result +
  (1) systems composability.**
- **Most surprising observation (a methodological finding in its own right):
  paired pixel metrics are saturated in multi-step diffusion.** A calibration arm
  injecting a *known* attention perturbation shows 1e-6 already produces pixel MAE
  0.0186 while 1e-1 produces 0.0317 — five orders of magnitude of input compressed
  into 1.7x of output, non-monotone. PSNR/SSIM/LPIPS in a 50-step free-running
  sampler measure *whether* the trajectory decorrelated, not *by how much*
  attention differed. Source:
  `figures/phase5_main/fig3_perturbation_calibration.csv`.

---

## 2. Setup

| Item | Value | Source |
|---|---|---|
| GPU(s) | 8x NVIDIA B200, sm_100 `(10, 0)`, 183,359 MiB each; 1 GPU per run, `sp_size=1` | `env/nvidia-smi.txt`, `env.json` |
| Driver / CUDA / nvcc | driver 595.91.07 / CUDA 13.0 / nvcc release 13.0, V13.0.88 | `env/nvcc.txt`, `env.json` |
| Torch version | 2.12.0+cu130 (`sm_100` in `get_arch_list()`) | `env.json` |
| FastVideo commit | `8208536cd1db7a1d32b68aaa6a679953ae23ab8b` (branch `exp/sparsefp4-mask-stability`) | `env.json` |
| Sparse dep commits (VSA / SpargeAttention / FA4) | VSA: in-repo `fastvideo-kernel 0.3.2` (`vsa_utils`, `VSA_TILE_SIZE=(4,4,4)`). SpargeAttention: **n/a, not used.** FA4: `hao-ai-lab/flash-attention-fp4` @ `fix/cutlass-dsl-4.5`, commit `940bf7e511375ec160bc2d7188bef35915ded1e3`, with `nvidia-cutlass-dsl 4.5.3`, `quack-kernels 0.5.0`, `flashinfer-python 0.6.17`, `apache-tvm-ffi 0.1.13.post3` | `env.json` |
| Model id | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `configs/`, `env.json` |
| Model revision (pinned) | `0fad780a534b6463e45facd96134c9f345acfa5b` | `env.json` |
| Scheduler / steps / guidance | `FlowUniPCMultistepScheduler(shift=3.0)` / **50** steps / guidance **3.0** (framework defaults, resolved from `SamplingParam.from_pretrained`, not assumed) | `raw/*/probe_config_*.json`, `raw/*/phase2_config_*.json`, `raw/.../run_summaries/phase5_config_*.json` |
| Resolution / frames | 480x832 / 81 | `configs/` |
| `seq_len` / layers / heads / head_dim | 32,760 (asserted constant in every record) / 30 / 12 / 128 | `raw/*/verification.json` |
| Attention backend(s) exercised | `FLASH_ATTN` (native BF16 and native NVFP4 via `FASTVIDEO_NVFP4_FA4=1`), `ROUTING_PROBE_ATTN` (Phase 1), `PRECISION_SPARSE_ATTN` (Phases 2/2B), `SPARSEFP4_EXEC_ATTN` (Phase 5), plus `TORCH_SDPA` as a probe reference | `raw/*/probe_config_*.json`, `raw/.../perf/arm_receipt_*.json` |
| Compile / CUDA graphs | **off for every arm** (`torch_compile: false`, `cuda_graphs: false`, `compile_mode: null`) | `raw/.../perf/phase5_perf_p01_*.json` |
| Determinism flags | `allow_tf32 = False`, `float32_matmul_precision = highest`, deterministic quantizers, index-ascending tie-break; `use_fsdp_inference=False` for all arms (required by the FP4 path, held symmetric) | `env.json`, `PHASE1.md` §7.1/§7.5, `PHASE0.md` §8 |
| Prompt set | `.agents/skills/sparsefp4-video-attention/assets/prompts.txt` — 10-prompt **development** set (`p01`–`p10`) | — |
| Seeds | **1234** for Phases 1/2/2B/5 (single seed). 1024 for the Phase 0 smoke only. | `raw/*/probe_config_*.json`, `raw/.../phase5_config_*.json` |
| Native NVFP4 available? | **Yes, native, not simulated.** Emitted PTX contains `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` (4 occurrences in the NVFP4 log, **0** in the BF16 log); tensors are genuinely `torch.float4_e2m1fn_x2` with a `uint8` scale-factor tensor; framework receipt `qk_mode=nvfp4(per-16-e4m3-sf) pv_mode=bf16` | `logs/phase0_nvfp4_kernel_probe.log`, `logs/phase0_smoke_nvfp4.log`, `raw/phase0_nvfp4_kernel_probe.json` |

**Unusual things about this environment, because they affect reproduction.** The
root volume had ~9 GiB free at study start, so an unmounted 3.5 TB instance-store
NVMe was formatted and mounted as `/mnt/scratch` to hold the venv (6.6 GB), the
27 GB model and the CUDA toolkit — **that mount is ephemeral and does not survive
an instance stop/start**, and the `/usr/local/cuda-13.0` bind mount is not in
`fstab`. `nvcc` was absent at start and had to be installed: the NVFP4 **Q/K
quantizer** is a flashinfer JIT module (`fp4_quantize_sm100`), so a CUDA toolkit
plus `ninja` is a *runtime* requirement of the NVFP4 path, not just a
kernel-development one. No FlashAttention-2 is installed — the FP4 fork replaces
it — so `FASTVIDEO_FA4=1` is mandatory for *every* attention path including BF16.
Python **3.12** is required in practice (only cp312 wheels exist for
`fastvideo-kernel 0.3.2`; the docs' "3.10 or 3.11" is stale), and
`nvidia-cutlass-dsl` must stay at 4.5.3 because the 4.6 line removes
`cute.make_fragment`. Activation is one line and is not optional:
`source artifacts/sparsefp4/configs/env.sh`, then use `"$FV_PYTHON"`
(`/mnt/scratch/fv-venv/bin/python`, CPython 3.12.13). Full detail:
[`PHASE0.md`](PHASE0.md) §3–§4.

---

## 3. Code changes by file

Verified against `git status --short` and `git diff --stat` at report time: **3
tracked files modified (18 insertions, 8 deletions), 4 new untracked files (2,429
lines)**. Nothing was committed.

| File | Change | Why | Kept or reverted? |
|---|---|---|---|
| `fastvideo/platforms/interface.py` | +3 `AttentionBackendEnum` members: `ROUTING_PROBE_ATTN`, `PRECISION_SPARSE_ATTN`, `SPARSEFP4_EXEC_ATTN` | a research backend needs an enum member to be selectable | **kept** (additive, 3 lines) |
| `fastvideo/platforms/cuda.py` | +3 branches in `get_attn_backend_cls` mapping those enums to their classes | the string→class map lives here, **not** in `selector.py` as `attention/AGENTS.md:100-108` states | **kept** (additive, 9 lines) |
| `fastvideo/configs/models/dits/base.py` | +3 entries in `_supported_attention_backends`; the tuple was also re-wrapped by yapf | Wan's DiT config gates which backends it will accept; without the entry the override is refused | **kept** (additive; the reflow is formatter-only) |
| `fastvideo/attention/backends/routing_probe_attn.py` | **new, 630 lines.** Dense BF16 pass-through that computes routing metrics on the side: block scorer, 4 router-precision arms, per-record Jaccard/recall/margins/ties/saturation, `--measure-timestep-stride`, `--score-dtype` | Phase 1 needs post-RoPE Q/K at the backend boundary without changing the trajectory | **research-only — do not ship** |
| `fastvideo/attention/backends/sparsefp4_numerics.py` | **new, 493 lines.** Shared numerics: `BlockGeometry` + `raster_geometry` / `cube_geometry`, `to_block_layout` / `from_block_layout`, geometry-aware masked-mean pooling, `retained_token_fraction`, `masked_reference`, `block_attention_mass` | Phases 2/2B need one definition of blocking, pooling and the reference so raster and cube arms cannot silently diverge | **research-only — do not ship** |
| `fastvideo/attention/backends/precision_sparse_attn.py` | **new, 861 lines.** A–F error decomposition plus `B_sim`, `C_rand`, `C_null`; geometry as a run parameter; `tie_diagnostic` record emitting both tie denominators; per-row churn and `retained_token_fraction` | Phase 2/2B: exact per-cell pairing of all arms against one dense-BF16 trajectory | **research-only — do not ship** |
| `fastvideo/attention/backends/sparsefp4_exec_attn.py` | **new, 445 lines.** Phase 5 execution backend — the arm's attention is what the model *consumes*; all 6 end-to-end arms plus the `SPARSE-BF16-EPS` calibration control; per-process `arm_receipt` provenance; **raises unless `FASTVIDEO_SPARSEFP4_PHASE5` is set** | end-to-end video needs the arm in the real denoising loop, and trap 1 needs a positive receipt that the override was honored | **research-only — do not ship** |
| `.agents/skills/sparsefp4-video-attention/scripts/analyze_masks.py` | extended: `agg_by_timestep` / `agg_by_layer_head` / `agg_by_prompt` / `agg_by_cfg_branch` tables, Figure 4, readable figure scales, extra medians; self-test extended and passing | the shipped analyzer had no timestep axis, which is the axis the study most needed | kept in the skill (GPU-free, self-tested) |
| `artifacts/sparsefp4/configs/*.py`, `*.sh` (30 files) | **new** study harness: env capture, smoke, kernel probe, per-phase runners/launchers/verifiers/analyzers/figure renderers, self-tests | drivers and analysis, deliberately outside `fastvideo/` | study artifacts |

**Research-only code that must not ship as a feature.** All four new
`fastvideo/attention/backends/*.py` files. They recompute fp64 block scores on
every call, emit JSONL from inside the attention hot path, and in Phase 2/2B
deliberately return the *dense BF16* result while measuring alternatives on the
side. `precision_sparse_attn.py` and `sparsefp4_exec_attn.py` also implement
simulated NVFP4 by dequantizing back to BF16, which is a measurement device, not
an implementation. The only pieces with plausible upstream value are the
`BlockGeometry` abstraction and the padding gate-checks in
`sparsefp4_numerics.py`. `pre-commit run --files` (yapf, ruff, codespell, mypy)
passes on every touched `fastvideo/` file.

Codebase orientation used to pick these seams — including why the hook must sit at
`AttentionImpl.forward` (RoPE is applied inside the attention layer at
`layer.py:130-132`, so anything captured in `wanvideo.py` is pre-RoPE and would be
the exact comparison the SKILL forbids) — is in
[`CODEBASE_MAP.md`](CODEBASE_MAP.md) §A, §D.3, §D.7.

---

## 4. Exact commands

In execution order. Every block assumes `cd /home/ec2-user/FastVideo` first.

```bash
# ---- 0. environment (once) -------------------------------------------------
# scripts/check_env.py does not exist in this repo (SKILL discrepancy); the repo
# ships collect_env.py at its root, and env.json is written by a purpose-built script.
python collect_env.py > artifacts/sparsefp4/env/collect_env.txt
nvidia-smi -q          > artifacts/sparsefp4/env/nvidia-smi.txt
/usr/local/cuda-13.0/bin/nvcc --version >> artifacts/sparsefp4/env/nvcc.txt
source artifacts/sparsefp4/configs/env.sh   # FV_PYTHON, FASTVIDEO_FA4=1, CUDA_HOME, HF_HOME, TMPDIR
"$FV_PYTHON" -m pip freeze > artifacts/sparsefp4/env/pip-freeze.txt
"$FV_PYTHON" artifacts/sparsefp4/configs/write_env_json.py --out artifacts/sparsefp4/env.json
```

```bash
# ---- 0b. dense smoke tests + native-NVFP4 evidence + kernel microbenchmark --
source artifacts/sparsefp4/configs/env.sh
for mode in bf16 nvfp4; do
  CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase0_smoke.py \
    --mode $mode --steps 4 --out-dir artifacts/sparsefp4/videos \
    --metrics-json artifacts/sparsefp4/raw/phase0_smoke_$mode.json \
    > artifacts/sparsefp4/logs/phase0_smoke_$mode.log 2>&1
done
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase0_nvfp4_kernel_probe.py \
  --out artifacts/sparsefp4/raw/phase0_nvfp4_kernel_probe.json \
  > artifacts/sparsefp4/logs/phase0_nvfp4_kernel_probe.log 2>&1
```

```bash
# ---- 1. Phase 1 stage 1: 1 prompt x 1 seed, all 30 layers x all 50 timesteps
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_probe_run.py \
  --run-id 20260814-013449-8208536-p1-stage1 --prompt-index 0 \
  --sparsities 0.80 0.90 --routing-precisions bf16 fp8_e4m3 nvfp4 nvfp4_sim \
  --steps 50 --measure-timestep-stride 1 --null-control-layer-stride 1 \
  --null-control-timestep-stride 1 --spearman-timestep-stride 10 \
  --raw-root /mnt/scratch/sparsefp4

# ---- 2. Phase 1 stage 2: 10 prompts, full sparsity sweep, 1 process per GPU ---
bash artifacts/sparsefp4/configs/phase1_stage2_launch.sh 20260814-014229-8208536-p1-stage2
#   GPU->prompt map 0:[p01,p09] 1:[p02,p10] 2..7:[p03..p08]; sp_size=1; nohup per GPU
#   MEASURE_TIMESTEP_STRIDE=5 NULL_LAYER_STRIDE=5 NULL_TIMESTEP_STRIDE=10
#   SPARSITIES="0.50 0.70 0.80 0.90 0.95" STEPS=50

# ---- 2b. fp64-scorer confounder control (trap 8) -----------------------------
CUDA_VISIBLE_DEVICES=2 "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_probe_run.py \
  --run-id 20260814-015113-8208536-p1-stage1-fp64score --prompt-index 0 \
  --sparsities 0.80 0.90 --steps 50 --stage 1-fp64score --score-dtype float64 \
  --measure-timestep-stride 1 --raw-root /mnt/scratch/sparsefp4

# ---- verification gate: must PASS before any aggregate is quoted -------------
"$FV_PYTHON" artifacts/sparsefp4/configs/phase1_verify_run.py \
  --raw-dir /mnt/scratch/sparsefp4/<run_id> --expect-layers 30 --expect-heads 12 \
  --expect-timesteps 50 --out /mnt/scratch/sparsefp4/<run_id>/verification.json
```

```bash
# ---- 3. Phase 2: A-F error decomposition + H3 (fp64 scores throughout) --------
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_selftest.py --out artifacts/sparsefp4/raw/phase2_selftest.json
bash artifacts/sparsefp4/configs/phase2_launch.sh 20260814-025500-8208536-p2-main
#   ARMS="A B B_sim C C_rand D D8 E F8 F16", --score-dtype float64,
#   17 layers x 12 heads x 5 timesteps x 2 CFG x 3 sparsities x 10 prompts

# ---- 3b. Phase 2B: the geometry control (C / D / C_rand / C_null only) --------
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_selftest.py \
  --out artifacts/sparsefp4/raw/phase2b_selftest.json
bash artifacts/sparsefp4/configs/phase2b_launch.sh 20260814-035500-8208536-p2b-64x64-raster 64x64-raster
bash artifacts/sparsefp4/configs/phase2b_launch.sh 20260814-032500-8208536-p2b-64x64-cube  64x64-cube
#   -> ARMS="A C C_null D C_rand", --score-dtype float64, --no-activation-stats,
#      --tie-diagnostic-geometries "128x64-raster 64x64-raster 64x64-cube"
```

```bash
# ---- 4. Phase 5: end-to-end video, VBench, single-step control, performance ---
git submodule update --init fastvideo/third_party/eval/vbench
VIRTUAL_ENV="$FV_VENV" uv pip install --python "$FV_PYTHON" \
    lpips easydict openai-clip pyiqa decord scikit-image
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_selftest.py \
    --out artifacts/sparsefp4/raw/phase5_selftest.json
nohup bash artifacts/sparsefp4/configs/phase5_launch.sh \
    20260814-032700-8208536-p5-main 0.90 1234 \
    > artifacts/sparsefp4/logs/phase5_launch_20260814-032700-8208536-p5-main.log 2>&1 &
for gpu in 0 1 2 3; do idx=$((gpu*2)); CUDA_VISIBLE_DEVICES=$gpu nohup "$FV_PYTHON" \
    artifacts/sparsefp4/configs/phase5_singlestep.py \
    --run-id 20260814-034700-8208536-p5-singlestep --prompt-index $idx --sparsity 0.90 & done
for arm in DENSE-BF16 DENSE-FP4 SPARSE-BF16; do
  CUDA_VISIBLE_DEVICES=5 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_perf.py \
    --run-id 20260814-035500-8208536-p5-perf --arm "$arm" --warmup 1 --reps 5
done
```

```bash
# ---- 5. Analysis (GPU-free except the VBench scorers) ------------------------
"$FV_PYTHON" .agents/skills/sparsefp4-video-attention/scripts/analyze_masks.py \
  --raw artifacts/sparsefp4/raw/<run_id> --out-tables artifacts/sparsefp4/tables/<tag> \
  --out-figures artifacts/sparsefp4/figures/<tag> --format both --sparsity 0.80 --sparsity 0.90
"$FV_PYTHON" artifacts/sparsefp4/configs/phase1_deepdive.py \
  --raw artifacts/sparsefp4/raw/<run_id> --out-tables artifacts/sparsefp4/tables/<tag>
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_analyze.py \
  --raw artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \
  --out-tables artifacts/sparsefp4/tables/phase2_main \
  --out-figures artifacts/sparsefp4/figures/phase2_main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2_figures.py \
  --tables artifacts/sparsefp4/tables/phase2_main --figures artifacts/sparsefp4/figures/phase2_main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_geometry_analyze.py \
  --raw 128x64-raster=artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \
  --raw 64x64-raster=artifacts/sparsefp4/raw/20260814-035500-8208536-p2b-64x64-raster \
  --raw 64x64-cube=artifacts/sparsefp4/raw/20260814-032500-8208536-p2b-64x64-cube \
  --out-tables artifacts/sparsefp4/tables/phase2b_geometry \
  --out-figures artifacts/sparsefp4/figures/phase2b_geometry
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_tie_reconcile.py \
  --fp32-raw   artifacts/sparsefp4/raw/20260814-013449-8208536-p1-stage1 \
  --fp64-raw   artifacts/sparsefp4/raw/20260814-015113-8208536-p1-stage1-fp64score \
  --phase2-raw artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \
  --out-tables artifacts/sparsefp4/tables/phase2b_geometry
"$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_figures.py --figures artifacts/sparsefp4/figures/phase2b_geometry
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
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_significance.py --run-id 20260814-032700-8208536-p5-main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_figures.py --run-id 20260814-032700-8208536-p5-main
"$FV_PYTHON" artifacts/sparsefp4/configs/phase5_contact_sheets.py \
    --run-id 20260814-032700-8208536-p5-main --prompts p01 p03 p05 p07
```

Phase 3 (a VSA-native sparse integration) and Phase 4 (a native sparse-NVFP4 FA4
kernel) were **not run**; see §14.

---

## 5. Hypotheses

| ID | Hypothesis | Verdict | Key evidence (number + n) | Raw path |
|---|---|---|---|---|
| H1 | NVFP4 Q/K changes top-block selection vs BF16 | **supported in direction only; effect small** | Median mask Jaccard **0.9882 → 0.9807 → 0.9738 → 0.9611** at sparsity 0.50 / 0.80 / 0.90 / 0.95, **n = 72,000 cells per row**; monotone in sparsity and in router precision (`bf16 1.0 > fp8 > nvfp4`) across all 20 cells. Median gap from 1.0 grows 3.3x and IQR 4.0x from sparsity 0.50→0.95. But 97.1% / 89.4% of cells exceed 0.95 at 0.80 / 0.90 and Spearman rho of block scores is **0.9997** (n = 36,000), so the ranking is essentially preserved and only the top-k cut line moves. Null control exact at 1.0 over **187,200** records. | `figures/main/fig1_mask_overlap_vs_sparsity.csv`; `tables/stage2/tail_by_sparsity_precision.csv`; `raw/20260814-014229-8208536-p1-stage2/*.jsonl.gz` |
| H2 | Instability is localized by head/layer/timestep | **unsupported** | Affected cells (median Jaccard < 0.90, n ≥ 20) at sparsity 0.90: **timesteps 0/50, heads 0/12, layers 0/30**; `(layer,head)` 2/360; `(layer,head,timestep)` 25/3,600 (0.7%). Full-trajectory timestep spread **0.00947** (worst step 0 at 0.9647, best step 38 at 0.9742, n = 720/step); per-head spread **0.0035**; per-layer spread 0.041 with an edge-vs-middle shape. The edge-layer residue is **explained away**: NVFP4 saturation is flat over all 17 layers (Q 0.0958–0.1098, K 0.0940–0.1099) and its Spearman correlation with per-layer wrong-mask excess is **−0.25** (n = 17), the wrong sign; worst layer's wrong-mask excess is **2.04e-04**, two orders below threshold. Edge layers churn more but on *less* important blocks (dropped/agreed mass 0.13 affected vs 0.32 unaffected). | `tables/stage2/affected_cell_counts.csv`; `tables/stage1/agg_by_timestep.csv`; `tables/phase2_main/table8_saturation_vs_layer_sensitivity.csv` |
| H3 | Higher-precision routing reduces sparse-attention error | **FALSIFIED** | Median relative reduction in attention-output rel-L2 from swapping the NVFP4 router for FP8 / BF16: **0.050% / 0.051%** (s=0.80), **0.037% / 0.055%** (s=0.90), **0.104% / 0.073%** (s=0.95), **n = 20,400 exactly paired cells per sparsity** — against a pre-registered **≥ 20%** threshold, i.e. short by 200–500x. Independently re-measured at **all 30 layers**: **0.0092% / 0.0093%**, n = 17,280. Only **52.2–56.5%** of cells improve. The **BF16 router is the theoretical ceiling of the idea and is no better than FP8**. The whole wrong-mask term is 0.016%–0.032% of combined error. | `tables/phase2_main/table3_h3_paired.csv`, `table4_error_attribution.csv`; `raw/20260814-032700-8208536-p5-main/phase5_singlestep_medians.json` |
| H4 | Native sparse-NVFP4 gives wall-clock benefit | **untested** (no native sparse-NVFP4 kernel was built) | Bar to beat is measured, not assumed: dense NVFP4 attention **kernel in isolation** 4.013 ms vs native BF16 FA4 5.135 ms = **1.28x** (warmed, CUDA-synced, median of 20, 3 warmup); dense **end-to-end** 44.436 s vs 46.876 s = **1.055x** (n = 5 reps, identical 8,518 MB peak). Amdahl bound below caps end-to-end gain at ~1.35x at this configuration. | `raw/phase0_nvfp4_kernel_probe.json`; `raw/20260814-032700-8208536-p5-main/perf/phase5_perf_p01_{DENSE-BF16,DENSE-FP4}.json` |

Pre-registered thresholds were fixed before measurement in
`.agents/skills/sparsefp4-video-attention/references/EXPERIMENT_SPEC.md` and were
not moved after seeing data. The H2 threshold "selects nothing" at coarse
granularity; §5.5 of the spec anticipated that and instructed reporting the full
distribution rather than re-tuning, which is what §6.2 below does.

---

## 6. Mask-stability results (Phase 1)

**Definitions** (`EXPERIMENT_SPEC.md` §1). `sparsity` is the fraction of key
blocks *dropped*; `retained_fraction = 1 − sparsity`; `k` is the per-query-block
retained count, derived from geometry only and **identical across all router
precisions at every cell** (`k_disagreements_across_arms = 0`), so precision
changed *which* blocks were selected and never *how many*. Tie-break is
index-ascending and deterministic. `force_retain_diagonal = false` — no diagonal
or local blocks were force-retained, so no overlap metric here is inflated by
guaranteed-common blocks. Because masks are equal-sized, **precision equals recall
and `jaccard = recall / (2 − recall)`: recall and Jaccard are one measurement, not
two.** Recall is present in every CSV for readers who prefer it and is never cited
as corroboration.

Runs backing this section: `20260814-013449-…-p1-stage1` (1 prompt, 30 layers ×
12 heads × **all 50 timesteps**, 288,000 records),
`20260814-014229-…-p1-stage2` (10 prompts, 5 sparsities, 1,116,000 records),
`20260814-015113-…-p1-stage1-fp64score` (fp64-scorer control, 288,000 records),
and a 28,800-record fp64 pilot. Total **1,720,800** records; all four carry
`verification.json` with `"verdict": "PASS"` and an empty `failures` list. Every
arm was measured on a **dense BF16 pass-through**, so all arms share one denoising
trajectory by construction and comparisons are exactly paired.

### 6.1 Mask overlap vs sparsity

![Figure 1 — mask overlap vs sparsity](figures/main/fig1_mask_overlap_vs_sparsity.png)

**Figure 1 — BF16↔candidate mask Jaccard vs sparsity**, one line per routing
precision, with median and IQR dispersion.
Plotted values: [`figures/main/fig1_mask_overlap_vs_sparsity.csv`](figures/main/fig1_mask_overlap_vs_sparsity.csv)

| sparsity | retained | routing_precision | native/simulated | jaccard median | jaccard IQR | recall median | n |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 | bf16 (null control) | native | **1.000000** | 0.000000 | 1.000000 | 7,200 |
| 0.50 | 0.50 | fp8_e4m3 | simulated | 0.995858 | 0.001762 | 0.997925 | 72,000 |
| 0.50 | 0.50 | nvfp4 | **native** | 0.988169 | 0.005968 | 0.994049 | 72,000 |
| 0.50 | 0.50 | nvfp4_sim | simulated | 0.988138 | 0.005998 | 0.994034 | 72,000 |
| 0.70 | 0.30 | bf16 (null control) | native | **1.000000** | 0.000000 | 1.000000 | 7,200 |
| 0.70 | 0.30 | fp8_e4m3 | simulated | 0.993830 | 0.002873 | 0.996905 | 72,000 |
| 0.70 | 0.30 | nvfp4 | **native** | 0.983597 | 0.009124 | 0.991731 | 72,000 |
| 0.70 | 0.30 | nvfp4_sim | simulated | 0.983548 | 0.009174 | 0.991706 | 72,000 |
| 0.80 | 0.20 | bf16 (null control) | native | **1.000000** | 0.000000 | 1.000000 | 7,200 |
| 0.80 | 0.20 | fp8_e4m3 | simulated | 0.992368 | 0.004213 | 0.996170 | 72,000 |
| 0.80 | 0.20 | nvfp4 | **native** | **0.980695** | 0.011067 | 0.990253 | 72,000 |
| 0.80 | 0.20 | nvfp4_sim | simulated | 0.980620 | 0.011067 | 0.990215 | 72,000 |
| 0.90 | 0.10 | bf16 (null control) | native | **1.000000** | 0.000000 | 1.000000 | 7,200 |
| 0.90 | 0.10 | fp8_e4m3 | simulated | 0.989092 | 0.007866 | 0.994516 | 72,000 |
| 0.90 | 0.10 | nvfp4 | **native** | **0.973756** | 0.017231 | 0.986704 | 72,000 |
| 0.90 | 0.10 | nvfp4_sim | simulated | 0.973756 | 0.017083 | 0.986704 | 72,000 |
| 0.95 | 0.05 | bf16 (null control) | native | **1.000000** | 0.000000 | 1.000000 | 7,200 |
| 0.95 | 0.05 | fp8_e4m3 | simulated | 0.982723 | 0.013556 | 0.991286 | 72,000 |
| 0.95 | 0.05 | nvfp4 | **native** | 0.961108 | 0.023636 | 0.980168 | 72,000 |
| 0.95 | 0.05 | nvfp4_sim | simulated | 0.961108 | 0.023636 | 0.980168 | 72,000 |

Source: [`tables/main_stage2/agg_by_sparsity_precision.csv`](tables/main_stage2/agg_by_sparsity_precision.csv).
Stage 1 (single prompt, all 50 timesteps) agrees to within 0.001: nvfp4 median
0.979803 @ 0.80 and 0.972733 @ 0.90, n = 36,000 each
(`tables/stage1/agg_by_sparsity_precision.csv`).

**Tail shape — the quantity the PIVOT test actually turns on.** The SKILL asks
whether overlap is "> 0.95 almost everywhere", which is a statement about the
fraction of cells below a threshold. Native NVFP4 arm, n = 72,000 per row
(`tables/stage2/tail_by_sparsity_precision.csv`):

| sparsity | frac < 0.99 | frac < 0.95 | frac < 0.90 | frac < 0.80 | p1 | min |
|---|---|---|---|---|---|---|
| 0.50 | 0.6925 | 0.00042 | 0 | 0 | 0.9656 | 0.9168 |
| 0.70 | 0.9217 | 0.00894 | 0.00019 | 0 | 0.9508 | 0.8927 |
| **0.80** | 0.9851 | **0.02901** | 0.00040 | 0 | 0.9396 | 0.8579 |
| **0.90** | 0.9905 | **0.10560** | 0.00688 | 0 | 0.9068 | 0.8020 |
| 0.95 | 0.9980 | 0.29158 | 0.02378 | 0.00121 | 0.8789 | 0.7454 |

At the two pre-registered operating points **97.1% / 89.4% of cells exceed 0.95**
and 99.96% / 99.3% exceed 0.90. Nothing in 1.7M records falls below 0.7454.

**The one framing where the effect looks large, and why it is not.**
`frac_query_blocks_changed` — the fraction of the 256 query blocks whose selected
set changed *at all* — is dramatic while the fraction of individual key-block
decisions changed is ~1%. Native NVFP4, medians over cells, n = 72,000
(`figures/main/fig1_mask_overlap_vs_sparsity.csv`, plus Stage-1 p90 from
`tables/stage1/tail_by_sparsity_precision.csv`):

| sparsity | frac_query_blocks_changed (median) | p90 (Stage 1) | median frac of *decisions* changed |
|---|---|---|---|
| 0.50 | 0.875 | — | 0.0060 |
| 0.80 | 0.730 | 0.926 | 0.0102 |
| 0.90 | 0.578 | 0.832 | 0.0138 |
| 0.95 | 0.465 | 0.703 | 0.0202 |

So at 80% sparsity roughly three quarters of query blocks lose at least one key
block while only ~1% of key-block decisions change. **The disruption is diffuse,
not concentrated** — many blocks each losing one marginal key block. An error
model assuming a few catastrophically mis-routed blocks is the wrong model here,
which is exactly what §7's mechanism confirms.

**Global score ordering is essentially untouched.** Spearman rho between BF16 and
candidate block scores over all 512 key blocks, sampled every 10th timestep at
sparsity 0.90 (n = 3,600 per arm): bf16 control **1.000000**, fp8_e4m3
**0.999963** (min 0.994403), native nvfp4 **0.999738** (min 0.991596). NVFP4
preserves the *ranking*; what moves is only the position of the cut line among
near-tied candidates.

### 6.2 Localization (H2)

![Figure 2a — layer x timestep Jaccard, sparsity 0.80](figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.80_nvfp4.png)

![Figure 2b — layer x timestep Jaccard, sparsity 0.90](figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.90_nvfp4.png)

**Figure 2 — layer × timestep heatmap** of BF16↔NVFP4 mask Jaccard, native NVFP4
router, all 30 layers × all 50 denoising steps (1,500 cells per panel, n = 24 per
cell), one panel per sparsity.
Plotted values:
[`figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.80_nvfp4.csv`](figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.80_nvfp4.csv),
[`…_s0.90_nvfp4.csv`](figures/main_stage1_timesteps/fig2_layer_timestep_jaccard_s0.90_nvfp4.csv)

| Region | sparsity | jaccard median | IQR | n | affected? (median < 0.90) |
|---|---|---|---|---|---|
| most affected `(layer, timestep)` = L0 / step 47 | 0.80 | 0.944399 | 0.015351 | 24 | **no** |
| least affected `(layer, timestep)` = L13 / step 16 | 0.80 | 0.988050 | 0.005055 | 24 | no |
| most affected `(layer, timestep)` = L28 / step 1 | 0.90 | 0.926623 | 0.030464 | 24 | **no** |
| least affected `(layer, timestep)` = L24 / step 11 | 0.90 | 0.988275 | 0.016263 | 24 | no |

**Zero of 1,500 `(layer, timestep)` cells are affected at either sparsity.** The
10-prompt Stage-2 aggregation (n = 240 per cell) agrees: 0 of 300 affected at
0.90, with the worst cell L28/step0 at 0.930814
(`tables/stage2/affected_cell_counts.csv`).

Affected-cell counts at every granularity, native NVFP4, Stage 2
(`tables/stage2/affected_cell_counts.csv`):

| granularity | sparsity | eligible cells | affected (< 0.90) | < 0.95 | worst cell | worst median |
|---|---|---|---|---|---|---|
| layer | 0.80 / 0.90 / 0.95 | 30 | **0 / 0 / 0** | 0 / 0 / 3 | layer 0 | 0.9637 / 0.9523 / 0.9340 |
| head | 0.80 / 0.90 / 0.95 | 12 | **0 / 0 / 0** | 0 / 0 / 0 | head 2 (0.80/0.90), head 6 (0.95) | 0.9786 / 0.9730 / 0.9582 |
| timestep | 0.80 / 0.90 / 0.95 | 10 | **0 / 0 / 0** | 0 / 0 / 0 | step 0 | 0.9752 / 0.9644 / 0.9522 |
| layer × timestep | 0.90 / 0.95 | 300 | **0 / 0** | 7 / 35 | L28/step0 | 0.9308 / 0.9182 |
| layer × head | 0.80 / 0.90 / 0.95 | 360 | 0 / **2** / **5** | 8 / 24 / 100 | L1H1 (0.80), L28H11 | 0.9380 / 0.8681 / 0.8118 |
| layer × head × timestep | 0.90 / 0.95 | 3,600 | **25** / **59** | 322 / 1,026 | L28H4/step0 | 0.8613 / 0.7929 |

**Named sensitive regions, reproducible across all 10 prompts.** Most sensitive
layers are **28, 0, 29, 1, 27, 2** — the first two and last three transformer
blocks (Stage-1 per-layer medians @ 0.90: L28 0.9407, L0 0.9490, L29 0.9527,
L1 0.9572, L2 0.9618, L27 0.9640); most stable is the middle band 6–24
(0.976–0.982); per-layer spread **0.041**. Worst `(layer, head)` cells at 0.95 are
L28H11 (0.8118), L0H9 (0.8657), L29H9 (0.8821), L23H8 (0.8837), L0H3 (0.8993)
against a best of L5H3 = 0.9873. Worst triple is L28H4 at step 0.
Rankings: `tables/stage2/ranked_cells.csv` (6,180 rows), `agg_by_layer.csv`,
`agg_by_head.csv`, `agg_by_layer_head.csv`.

**Timesteps and heads are flat, which is a load-bearing negative.** Over the full
50-step trajectory at sparsity 0.90 the per-step median moves from **0.964726**
(step 0, the highest-noise step and the worst step at every sparsity in every arm)
to **0.974195** (step 38) — a total spread of **0.00947**, n = 720 per step
(`tables/stage1/agg_by_timestep.csv`, figure
[`figures/main/fig4_overlap_vs_timestep.png`](figures/main/fig4_overlap_vs_timestep.png)
+ `.csv`). The shape is a monotone rise over the first ~10 steps then a flat
plateau. Per-head spread is **0.0035** at 0.90
(`tables/stage2/agg_by_head.csv`), and the head-level distribution is in
[`figures/main/fig3_head_jaccard_box_s0.90_nvfp4.png`](figures/main/fig3_head_jaccard_box_s0.90_nvfp4.png)
(+ `.csv`, and an s0.80 panel alongside). Together these two negatives remove
"schedule router precision by timestep" and "by head" from the method space. The
only Phase-1-supported scheduling axis is per-layer, and its ceiling is small.

**Is the structure reproducible?** Yes, within the development set. Stage 2's
per-prompt medians at 0.90 span **0.968648** (p07, aerial drone) to **0.975954**
(p04/p06/p08, near-static subjects), n = 7,200 each — motion-heavy prompts are
consistently 0.005–0.007 less stable
(`tables/stage2/agg_by_prompt.csv`). CFG branches are indistinguishable
(0.972440 negative vs 0.972879 positive, n = 18,000 each,
`tables/stage2/agg_by_cfg_branch.csv`), so pooling them is safe. This is a
**10-prompt, single-seed** statement; seed robustness was not measured.

### 6.3 Decision margins and mechanism

Binning cells by BF16 reference decision-margin decile, Stage 2, native NVFP4
(`tables/stage2/margin_decile_vs_changed.csv`):

| reference-margin decile | median margin | median frac decisions changed @ 0.90 | @ 0.95 |
|---|---|---|---|
| 1 (smallest) | 0.000000 | 0.01758 | 0.02945 |
| 4 | 0.001305 | 0.01735 | 0.02148 |
| 7 | 0.002151 | 0.01149 | 0.01638 |
| 10 (largest) | 0.003525 | 0.00714 | 0.01322 |

The direction is as predicted — instability concentrates where the BF16 top-k
boundary is nearly tied — with a **~2.5x** ratio from bottom to top decile at
0.90. **Caveat, stated rather than buried:** deciles 1–3 have a median margin of
exactly 0.0, which is the fp32 artifact of §10; only the monotone decline across
deciles 4–10 (0.0174 → 0.0071), where margins are resolved, is trustworthy. The
mechanism is established properly in §7.5 with fp64 scores and an exact
attention-mass measurement.

**Boundary tie counts.** With fp32 scores the raw boundary margin `s_(k) − s_(k+1)`
lands on a power-of-two grid — recomputed directly from the archived Stage-1
records, the most common values at sparsity 0.80 are exactly `0.0` (2,446 of
5,000 sampled records), `0.015625` = 2⁻⁶, `0.03125` = 2⁻⁵, `0.0078125` = 2⁻⁷ —
the signature of catastrophic cancellation. Median fp32 boundary ties are **~104–115
per (cell, head)** of 256 query blocks; fp64 gives **exactly 0** at every geometry,
router and sparsity. Full treatment in §10 and §11.

**Saturation fractions.** Native NVFP4 Q/K saturates at the e2m1 maximum (±6.0) in
**10.55%** of Q elements and **10.54%** of K elements (median over cells, sparsity
0.90), against **0.0001%** for FP8-E4M3 — recorded per record as
`sat_frac_q` / `sat_frac_k`, not inferred, and matching the independent Phase-0
quantizer probe (0.111, `raw/phase1_quantizer_probe.json`). This ~10% saturation
is the mechanistic origin of the NVFP4 routing perturbation; §7.6 shows it does
*not* explain the layer ranking.

### 6.4 Null control

The BF16-routing-vs-BF16-reference arm is an identity by construction and was kept
live in **every** run as a gate. Across **187,200** null-control records —
72,000 (Stage 1) + 36,000 (Stage 2) + 72,000 (fp64 Stage 1) + 7,200 (fp64 pilot) —
every single record has `jaccard == 1.0`, `recall == 1.0`,
`frac_query_blocks_changed == 0.0` and `spearman_rho == 1.0`. **Zero deviations.**

Phase 2B added a stronger control: `C_null` re-derives the BF16 mask from an
independent second quantizer call and pushes it through the block-sparse kernel,
gating the *whole executed path* rather than just the scorer. It is exact —
**61,200 paired cells per geometry, 0 deviations in `rel_l2`, 0 deviations from
Jaccard 1.0** (`tables/phase2b_geometry/verification.json`).

Additional integrity checks, all passing on every run
(`raw/*/verification.json`): `attention_backend` is the research backend in 100%
of records and the record lattice is **complete** (216,000/216,000 Stage 1;
1,080,000/1,080,000 Stage 2) — because a record exists only when the probe's
`forward` actually executes, a complete lattice is positive proof no DiT
self-attention layer silently fell back to the default backend (trap 1);
0 schema-invariant violations in 1,720,800 records; `k_disagreements_across_arms
= 0`; `seq_len` constant at 32,760; 0 malformed lines.

**A passing null control is not a certificate of numerical adequacy.** This
control passed the entire time the fp32 scorer was manufacturing ~1,400 boundary
ties per cell, because both sides of the identity hit the same float grid the same
way. That is the single most transferable lesson of the study; see §10.

---

## 7. Numerical-error decomposition (Phase 2)

All errors are relative L2 `||x − A||₂ / ||A||₂` per head-cell against **A (dense
BF16)**, the sole numerical reference. C–F use an **identical retained fraction**
(`k = 103 / 52 / 26` at sparsity 0.80 / 0.90 / 0.95, asserted; zero cells
disagree). Differences between rows are **attributions, not an exact additive
decomposition** — quantization and sparsification compose *sub-additively* here
(`E − B − C` is consistently negative, median −0.034 to −0.043, i.e. the two
errors partially cancel).

Run `20260814-025500-8208536-p2-main`: 615,380 records over 10 shards, 10 prompts
× 17 layers × 12 heads × 5 timesteps × 2 CFG branches × 3 sparsities,
**fp64 block scores throughout** (gated in the analysis verifier),
`verification.json` `PASS` with zero failures. **n = 20,400 exactly paired cells
per sparsity.** Every arm shares one dense-BF16 trajectory, so the per-cell
difference is available and is the correct statistic.

| ID | Sparse? | Attention compute | Mask source | native/simulated | rel_L2 median | rel_L2 IQR | cosine median | max_abs median | n | Raw path |
|---|---|---|---|---|---|---|---|---|---|---|
| A | no | BF16 (FA4) | n/a | **native** | 0 (reference) | — | 1 | 0 | 20,400 | `raw/20260814-025500-8208536-p2-main/` |
| B | no | NVFP4 Q/K + BF16 PV | n/a | **native** | 0.052039 | 0.060701 | 0.998648 | 0.390625 | 20,400 | same |
| B_sim | no | simulated NVFP4 Q/K + BF16 PV | n/a | simulated (**simulation control**) | 0.052039 | 0.060701 | 0.998648 | 0.390625 | 20,400 | same |
| C (s=0.80) | yes | BF16 (Triton block-sparse) | BF16 | **native** | 0.105807 | 0.174669 | 0.995376 | 0.807617 | 20,400 | same |
| D (s=0.80) | yes | BF16 | NVFP4 | **native** compute, **native** router | 0.105715 | 0.174540 | 0.995368 | 0.810547 | 20,400 | same |
| D8 (s=0.80) | yes | BF16 | FP8-E4M3 | native compute, **simulated** router | 0.105735 | 0.174688 | 0.995377 | 0.806641 | 20,400 | same |
| C_rand (s=0.80) | yes | BF16 | BF16 with *N* blocks swapped at random | native compute, **synthetic control** | 0.114549 | 0.151940 | 0.994210 | 0.973633 | 20,400 | same |
| E (s=0.80) | yes | NVFP4 Q/K + BF16 PV | NVFP4 | **simulated — numerical only** | 0.155321 | 0.155017 | 0.989399 | 0.987793 | 20,400 | same |
| F8 (s=0.80) | yes | NVFP4 Q/K + BF16 PV | FP8-E4M3 | **simulated — numerical only** | 0.155243 | 0.154962 | 0.989394 | 0.984375 | 20,400 | same |
| F16 (s=0.80) | yes | NVFP4 Q/K + BF16 PV | BF16 | **simulated — numerical only** | 0.155242 | 0.154914 | 0.989394 | 0.984375 | 20,400 | same |
| C (s=0.90) | yes | BF16 | BF16 | native | 0.176219 | 0.239374 | 0.987105 | 1.24219 | 20,400 | same |
| D (s=0.90) | yes | BF16 | NVFP4 | native / native | 0.176463 | 0.239199 | 0.987088 | 1.24072 | 20,400 | same |
| D8 (s=0.90) | yes | BF16 | FP8-E4M3 | native / simulated | 0.176196 | 0.239569 | 0.987088 | 1.24219 | 20,400 | same |
| C_rand (s=0.90) | yes | BF16 | random-matched | native / synthetic | 0.179674 | 0.216538 | 0.986228 | 1.33301 | 20,400 | same |
| E (s=0.90) | yes | NVFP4 Q/K + BF16 PV | NVFP4 | simulated — numerical only | 0.212640 | 0.200925 | 0.980489 | 1.33203 | 20,400 | same |
| F8 (s=0.90) | yes | NVFP4 Q/K + BF16 PV | FP8-E4M3 | simulated — numerical only | 0.212561 | 0.200819 | 0.980485 | 1.33301 | 20,400 | same |
| F16 (s=0.90) | yes | NVFP4 Q/K + BF16 PV | BF16 | simulated — numerical only | 0.212524 | 0.200858 | 0.980482 | 1.33521 | 20,400 | same |
| C (s=0.95) | yes | BF16 | BF16 | native | 0.274005 | 0.292882 | 0.968633 | 1.71094 | 20,400 | same |
| D (s=0.95) | yes | BF16 | NVFP4 | native / native | 0.274499 | 0.292149 | 0.968657 | 1.71289 | 20,400 | same |
| D8 (s=0.95) | yes | BF16 | FP8-E4M3 | native / simulated | 0.274136 | 0.292583 | 0.968618 | 1.70898 | 20,400 | same |
| C_rand (s=0.95) | yes | BF16 | random-matched | native / synthetic | 0.276121 | 0.274343 | 0.968243 | 1.75977 | 20,400 | same |
| E (s=0.95) | yes | NVFP4 Q/K + BF16 PV | NVFP4 | simulated — numerical only | 0.296144 | 0.264134 | 0.962821 | 1.75830 | 20,400 | same |
| F8 (s=0.95) | yes | NVFP4 Q/K + BF16 PV | FP8-E4M3 | simulated — numerical only | 0.295838 | 0.264084 | 0.962887 | 1.76025 | 20,400 | same |
| F16 (s=0.95) | yes | NVFP4 Q/K + BF16 PV | BF16 | simulated — numerical only | 0.295929 | 0.264026 | 0.962896 | 1.76172 | 20,400 | same |

Source: [`tables/phase2_main/table1_af_decomposition.csv`](tables/phase2_main/table1_af_decomposition.csv)
(also `.md`). Per-region split: `table2_af_by_region.csv`. Cosine stays ≥ 0.9628
everywhere, so these are magnitude-scale errors, not directional collapse.

![A–F error decomposition](figures/phase2_main/fig1_af_decomposition.png)

Plotted values: [`figures/phase2_main/fig1_af_rel_l2_by_sparsity.csv`](figures/phase2_main/fig1_af_rel_l2_by_sparsity.csv)

**Simulation fidelity, measured on real Q/K rather than assumed.** `B` (native) vs
`B_sim` (simulated) agree to a median relative difference of **9.6e-8**
(p90 = 3.9e-7, max abs difference 1.7e-6, 34% of cells bit-identical, n = 20,400)
— about five orders of magnitude tighter than any effect size discussed here. That
is what licenses interpreting the `E`/`F` rows numerically. It licenses **no**
latency claim: every such row carries `numerical_only = true`.

### 7.1 Attributions

Source: [`tables/phase2_main/table4_error_attribution.csv`](tables/phase2_main/table4_error_attribution.csv),
n = 20,400 per column.

| Term | s=0.80 | s=0.90 | s=0.95 |
|---|---|---|---|
| quantization ≈ `err(B)` | 0.052039 | 0.052039 | 0.052039 |
| sparsification ≈ `err(C)` | 0.105807 | 0.176219 | 0.274005 |
| **wrong-mask, NVFP4 ≈ `err(D) − err(C)`** | **3.096e-05** | **3.326e-05** | **9.520e-05** |
| wrong-mask, FP8 ≈ `err(D8) − err(C)` | 7.86e-08 | **−3.99e-07** | 1.29e-06 |
| random wrong-mask ≈ `err(C_rand) − err(C)` | 8.359e-04 | 7.205e-04 | 9.593e-04 |
| **router-recoverable ≈ `err(E) − err(F16)`** | **6.63e-06** | **8.42e-06** | **4.81e-05** |
| residual `E − B − C` (sub-additivity) | −0.033656 | −0.039007 | −0.042975 |
| share of `E` from quantization | 33.50% | 24.47% | 17.57% |
| share of `E` from sparsification | 68.12% | 82.87% | 92.52% |
| **share of `E` from wrong-mask** | **0.0199%** | **0.0156%** | **0.0321%** |

The wrong-mask term — the *entire* quantity H3 proposes to remove — is
**0.016%–0.032%** of combined error, three to four orders of magnitude below both
quantization and sparsification. At sparsity 0.90 the **FP8 router's wrong-mask
term is negative**, i.e. indistinguishable from zero.

**Post-residual hidden-state error: not measured.** Errors were measured at the
attention output, which is deliberate — it is the quantity H3 is about and the
*most favourable* place to look for a router effect, since downstream residual
mixing can only dilute it. The pre-registered gate was a ≥ 20% reduction at the
attention output. The end-to-end consequence is measured instead, at the video
level, in §8.

### 7.2 H3 test at equal budget

**Pre-registered support threshold: ≥ 20% relative reduction in median `rel_l2` in
affected regions, with the full paired distribution reported.** The comparison is
paired per cell at `(prompt, layer, head, timestep, cfg_branch, sparsity)`, not a
difference of independently pooled medians.

| Router precision | Attention compute | sparsity | rel_L2 p10 / p50 / p90 (arm `E`) | reduction vs NVFP4 router | frac cells improved | n | native/simulated |
|---|---|---|---|---|---|---|---|
| NVFP4 | NVFP4 Q/K + BF16 PV | 0.80 | 0.06341 / 0.15532 / 0.35214 | — | — | 20,400 | simulated |
| FP8 | NVFP4 Q/K + BF16 PV | 0.80 | 0.06340 / 0.15524 / 0.35229 | **0.050%** | 55.5% | 20,400 | simulated |
| BF16 | NVFP4 Q/K + BF16 PV | 0.80 | 0.06339 / 0.15524 / 0.35220 | **0.051%** | 55.4% | 20,400 | simulated |
| NVFP4 | NVFP4 Q/K + BF16 PV | 0.90 | 0.08561 / 0.21264 / 0.49813 | — | — | 20,400 | simulated |
| FP8 | NVFP4 Q/K + BF16 PV | 0.90 | 0.08547 / 0.21256 / 0.49802 | **0.037%** | 52.9% | 20,400 | simulated |
| BF16 | NVFP4 Q/K + BF16 PV | 0.90 | 0.08550 / 0.21252 / 0.49798 | **0.055%** | 52.2% | 20,400 | simulated |
| NVFP4 | NVFP4 Q/K + BF16 PV | 0.95 | 0.11672 / 0.29614 / 0.66364 | — | — | 20,400 | simulated |
| FP8 | NVFP4 Q/K + BF16 PV | 0.95 | 0.11675 / 0.29584 / 0.66323 | **0.104%** | 54.0% | 20,400 | simulated |
| BF16 | NVFP4 Q/K + BF16 PV | 0.95 | 0.11682 / 0.29593 / 0.66312 | **0.073%** | 54.7% | 20,400 | simulated |

Source: [`tables/phase2_main/table3_h3_paired.csv`](tables/phase2_main/table3_h3_paired.csv),
region `all`. Paired-difference quantiles (p10 / p50 / p90) for `E→F16` at 0.90:
**−5.39e-04 / +8.42e-06 / +6.74e-04** — the distribution straddles zero
symmetrically, and p10 is negative in every row.

![H3 verdict: measured reduction vs threshold, with the per-cell distribution](figures/phase2_main/fig2_h3_verdict.png)

Per-cell reduction ECDF values: [`figures/phase2_main/fig2_h3_paired_reduction_ecdf.csv`](figures/phase2_main/fig2_h3_paired_reduction_ecdf.csv)

- **Affected regions** (edge layers, `region = affected`, n = 7,200 per row):
  `E→F16` reduction **0.036%** @ 0.80, **0.322%** @ 0.90, **0.192%** @ 0.95;
  frac improved 58.6% / 54.6% / 57.7%. Still 60–550x below threshold. The
  *strongest* single number anywhere in the H3 family is 0.322%.
- **Unaffected regions (control**, mid-stack layers, n = 6,000 per row): reduction
  is **negative** in all three — −0.078% @ 0.80, −0.080% @ 0.90, −0.056% @ 0.95,
  i.e. a higher-precision router made the median cell slightly *worse*.
- **Broad region** (n = 7,200): −0.024% to +0.111%. No region reaches 1%.
- Distribution figure/CSV: `figures/phase2_main/fig2_h3_verdict.png`,
  `figures/phase2_main/fig2_h3_paired_reduction_ecdf.csv`.
- Null control: `F16` vs itself is exactly 0 across all **61,200** rows
  (`h3_null_control_max_abs_diff: 0.0` in `tables/phase2_main/summary.json`).
  Per §10's correction this is reported as an **arithmetic-identity check only**
  and does *not* certify scorer resolution.
- **Verdict: unsupported — falsified.** Three things make this a strong negative
  rather than a weak one. (i) The effect is 200–500x below threshold, so no amount
  of additional power changes it. (ii) The sign is barely better than a coin flip:
  a *perfect* router makes the output worse in ~45% of cells. (iii) **`F16` and
  `F8` are indistinguishable**, so H3 does not merely fail at FP8 — it fails at
  infinite router precision, which means the ceiling on precision-decoupled
  routing is itself ~0.05%.

### 7.3 Why H3 fails: the near-tie mechanism, measured

The prediction was that quantization flips exactly the blocks sitting at a
near-degenerate top-k boundary. This was tested directly by computing the **exact
dense attention mass** each key block contributes to each sampled query block
(softmax over the full key axis, then summed within block; verified to integrate
to 1.0 to within 6.0e-08 in the self-test). Measured at **34,560 query-block
observations per sparsity** over 6 layers × 2 timesteps.

| Quantity (median, sparsity 0.90, all regions) | Value | n |
|---|---|---|
| blocks swapped per query block (mean) | 0.997 of `k` = 52 | 34,560 |
| query blocks with at least one swap | 65.6% | 34,560 |
| **attention mass of a swapped-out block** | **0.002833** | 34,560 |
| **attention mass of a block both masks agree to keep** | **0.013197** | 34,560 |
| attention mass of an average *excluded* block | 0.000704 | 34,560 |
| attention mass of a **random**-dropped block (control) | 0.005251 | 34,560 |
| total mass retained by the BF16 top-k | 0.676 | 34,560 |
| normalized score gap of the swapped pair | **0.00468** of the score spread | 34,560 |

Source: [`tables/phase2_main/table5_margin_mechanism.csv`](tables/phase2_main/table5_margin_mechanism.csv).

![Mechanism: block mass and the random contrast control](figures/phase2_main/fig3_mechanism.png)

Four independent implications, all confirmed:

1. **Swapped blocks are near-worthless.** A swapped block carries **4.66x less**
   mass than an agreed block and sits far closer to the *excluded* population
   (0.000704) than to the retained one. Total displaced mass is 3.43e-03 against
   0.676 retained.
2. **The swaps happen at a vanishing margin.** The reference-score gap between
   dropped and added block is **0.47% of the score spread** — a near-tie by any
   measure. NVFP4's quantization error is larger than that margin and smaller than
   essentially every other margin.
3. **Output error is concentrated where the margin is not the driver.** Binning
   query blocks by the score gap of their swapped pair, the wrong-mask excess is
   **exactly 0.0 in 8 of 10 deciles** and never exceeds 1.0e-05 in any decile,
   while sparsification error `C` varies by more than 10x across the same deciles
   (0.187 → 0.018). Dropped-block mass falls monotonically as the gap widens
   (0.0038 → 0.0005), exactly as a boundary-effect account predicts. Source:
   [`tables/phase2_main/table6_error_vs_score_gap_decile.csv`](tables/phase2_main/table6_error_vs_score_gap_decile.csv),
   figure values `figures/phase2_main/fig3_output_error_vs_score_gap.csv`.
4. **The equal-magnitude contrast control isolates the mechanism.** This is the
   decisive test and the study's core evidence.

| Excess rel-L2 over `C` (median) | s=0.80 | s=0.90 | s=0.95 | n |
|---|---|---|---|---|
| quantization-chosen swaps (`D − C`) | 3.096e-05 | 3.326e-05 | 9.520e-05 | 20,400 |
| **random swaps of equal count (`C_rand − C`)** | **8.359e-04** | **7.205e-04** | **9.593e-04** | 20,400 |
| **ratio random / quantization** | **27.0x** | **21.7x** | **10.1x** | 20,400 |
| cells where random is worse | 63.2% | 61.6% | 61.6% | 20,400 |
| same ratio, `affected` region only | 125.4x | 82.5x | 14.5x | 7,200 |
| same ratio, `unaffected` region only | 5.96x | 4.61x | 1.34x | 6,000 |

Source: [`tables/phase2_main/table7_random_perturbation_contrast.csv`](tables/phase2_main/table7_random_perturbation_contrast.csv),
figure values `figures/phase2_main/fig4_random_vs_quantization_excess.csv`.

Changing the same *number* of blocks at random costs **10x to 27x** more output
error than letting NVFP4's quantization choose them, and the mass measurement
agrees independently (random-dropped 0.005251 vs quantization-dropped 0.002833, a
**1.85x** ratio). Quantization is therefore not making a generic mask error of a
given magnitude — it is making the **cheapest possible** error of that magnitude.
That is a mechanistic explanation of the H3 failure, not a restatement of it.

**Interpretation caveat.** `C_rand` is a synthetic control, not a system anyone
would build. Its role is to hold the perturbation *magnitude* fixed and vary only
*where* it lands. It is labelled
`router_native_or_simulated = "synthetic_control"` in every raw record so it can
never be mistaken for a measured precision arm.

### 7.4 Are the "sensitive" edge layers real, or just wider activations?

The boring alternative to Phase 1's edge-layer structure is that those layers
simply have wider activations, so more elements clip at the e2m1 maximum. Both were
measured per layer, n = 3,600 per layer row, 17 layers
([`tables/phase2_main/table8_saturation_vs_layer_sensitivity.csv`](tables/phase2_main/table8_saturation_vs_layer_sensitivity.csv)).

![Saturation control](figures/phase2_main/fig5_saturation_control.png)

Plotted values: [`figures/phase2_main/fig5_saturation_vs_wrong_mask_excess.csv`](figures/phase2_main/fig5_saturation_vs_wrong_mask_excess.csv)

**The saturation explanation is rejected — and so is the sensitivity framing.**

- NVFP4 saturation is **flat across all 17 layers**: Q 0.09579–0.10975,
  K 0.09403–0.10991. Layer 0, Phase 1's most extreme outlier, has the *lowest*
  saturation of any layer measured (Q 0.09579).
- Spearman correlation between a layer's wrong-mask excess and its Q saturation
  fraction is **−0.2525** (n = 17) — the wrong sign for the saturation account.
  Correlation with Q absolute max is **+0.3554** and with intra-group dynamic
  range **+0.2230**: weakly positive, far from explanatory. (Recomputed from the
  17 rows of that CSV; the underlying per-layer values are the archived artifact.)
- More importantly, **the layer ranking does not matter**: the largest per-layer
  wrong-mask excess anywhere is **2.035e-04** (layer 2) and the largest *relative*
  excess **1.70e-03**. Even the worst layer in the network is two orders of
  magnitude below the H3 threshold. Edge layers do churn more (1.32 vs 0.72 swaps
  per query block at 0.90) but their swapped blocks are correspondingly *less*
  important (`dropped_over_agreed_ratio` 0.13 affected vs 0.32 unaffected). **More
  churn, less consequence.**
- One genuine asymmetry: layer 0 has an intra-group dynamic range of **12.18**,
  roughly 4x every other layer (3.0–4.4). That is a real property of layer 0's Q/K
  distribution and would matter for a *weight* or *activation* quantization study.
  It does not translate into routing error here.

### 7.5 Geometry generalization control (Phase 2B)

Every Phase 1/2 number is at raster-order **128x64** blocks. FastVideo's deployed
sparse backend, VSA, uses **(4,4,4) spatio-temporal cubes of 64 tokens with
`block_q == block_k`**, and re-orders tokens into tile-contiguous order first —
a different token-to-block assignment, not merely a different block size. Phase 2B
re-ran the decisive subset (`C`, `D`, `C_rand`, `C_null` and the mechanism records)
at VSA's real geometry plus a `64x64-raster` intermediate that separates block size
from token ordering. n = 20,400 paired cells per (geometry, sparsity);
`verification.json` `PASS`, zero failures.

| Geometry | s | Jaccard median (IQR) | mass swapped-out | mass agreed | **agreed/swapped** | `C` | `D − C` | `C_rand − C` | **`C_rand`/`D`** | frac cells random worse | n cells | n q-blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128x64-raster | 0.80 | 0.97973 (0.0139) | 0.001436 | 0.008221 | **5.72x** | 0.105807 | 3.096e-05 | 8.359e-04 | **27.0x** | 63.2% | 20,400 | 27,257 |
| 128x64-raster | 0.90 | 0.97229 (0.0188) | 0.002833 | 0.013197 | **4.66x** | 0.176219 | 3.326e-05 | 7.205e-04 | **21.7x** | 61.6% | 20,400 | 22,674 |
| 128x64-raster | 0.95 | 0.96198 (0.0225) | 0.004628 | 0.017734 | **3.83x** | 0.274005 | 9.520e-05 | 9.593e-04 | **10.1x** | 61.6% | 20,400 | 17,430 |
| 64x64-raster | 0.80 | 0.97861 (0.0144) | 0.001322 | 0.008245 | **6.24x** | 0.097907 | 2.032e-05 | 9.567e-04 | **47.1x** | 64.7% | 20,400 | 27,876 |
| 64x64-raster | 0.90 | 0.97149 (0.0189) | 0.002702 | 0.013324 | **4.93x** | 0.159627 | 3.692e-05 | 8.602e-04 | **23.3x** | 64.1% | 20,400 | 23,013 |
| 64x64-raster | 0.95 | 0.96183 (0.0229) | 0.004352 | 0.018534 | **4.26x** | 0.242475 | 5.037e-05 | 1.219e-03 | **24.2x** | 65.8% | 20,400 | 17,428 |
| **64x64-cube (VSA deployed)** | 0.80 | 0.97937 (**0.0096**) | **0.000627** | 0.006929 | **11.05x** | 0.084608 | 1.844e-05 | 1.394e-03 | **75.6x** | 67.7% | 20,400 | 29,597 |
| **64x64-cube (VSA deployed)** | 0.90 | 0.97226 (**0.0141**) | **0.001440** | 0.011673 | **8.11x** | 0.136905 | 2.898e-05 | 1.292e-03 | **44.6x** | 67.7% | 20,400 | 25,250 |
| **64x64-cube (VSA deployed)** | 0.95 | 0.96448 (**0.0186**) | **0.002447** | 0.016824 | **6.88x** | 0.200024 | 4.038e-05 | 1.442e-03 | **35.7x** | 69.8% | 20,400 | 19,899 |

Source: [`tables/phase2b_geometry/table1_three_geometry_headline.csv`](tables/phase2b_geometry/table1_three_geometry_headline.csv);
per-region excess in `table3_paired_excess_by_geometry.csv`; mask stability in
`table4_mask_stability_by_geometry.csv`; mechanism in
`table5_mechanism_by_geometry.csv`.

![Isolation ratio at three geometries](figures/phase2b_geometry/fig2_random_over_quantization_by_geometry.png)

![Block mass and margin at three geometries](figures/phase2b_geometry/fig4_block_mass_by_geometry.png)

![Mask stability at three geometries](figures/phase2b_geometry/fig3_mask_jaccard_by_geometry.png)

Reading it:

- **Mask stability is essentially geometry-invariant.** Median Jaccard agrees to
  within 0.003 across all three geometries at every sparsity, and the cube arm is
  the *most* stable at 0.95. Its IQR is **25% narrower** at 0.80 (0.0096 vs
  0.0139), i.e. cube geometry makes routing *more* uniformly stable across cells.
- **The mechanism strengthens at the deployed geometry.** `C_rand`/`D` rises to
  **75.6x / 44.6x / 35.7x**, most emphatically at 0.95 where 128x64 was weakest
  (10.1x → 35.7x); the agreed/swapped mass gap widens from 3.8–5.7x to
  **6.9–11.1x**; the fraction of cells where random is worse rises 61.6% → 69.8%.
  All 18 cells point the same way.
- **Where the swaps go is unchanged and it is the boundary.** The normalized score
  margin of the swapped pair is 0.42%–0.50% of the spread at *every* geometry.
- **The wrong-mask term stays 3–4 orders below sparsification.** At cube geometry
  `D − C` = 2.90e-05 against `C` = 0.1369, i.e. **0.021%**. Per-query-block
  wrong-mask excess is **exactly 0.0 at the median** in all three geometries.
- **Block size or token ordering?** The two mass measures (high `n`: 17k–30k
  query-block observations per cell) are ordering-dominated by roughly 6–8x: e.g.
  the agreed/swapped ratio at 0.90 goes 4.66 → 4.93 (**+6%**, block size) then
  4.93 → 8.11 (**+64%**, ordering). The `C_rand`/`D` error ratio is messier —
  block size contributes as much at 0.80 and more at 0.95 — because it is a
  quotient of two medians of order 1e-05 and 1e-03 and inherits far more sampling
  noise. **Defensible statement:** ordering dominates on the directly measured
  high-`n` quantities; on the derived error ratio the split is not resolved by this
  data. Both factors point the same way in all 18 cells, so the conclusion's
  *direction* does not depend on the attribution.
- **Padding was gate-checked four ways, not argued.** Cube tiling gives
  6×8×13 = 624 tiles → padded length 39,936 → 7,176 pad slots (tile token counts
  {8,16,32,64}, none empty). Checks: pad slots hold zeros (`pad_slots_all_zero`);
  pooling divides by the *true* token count (max abs difference **0.0** vs an
  independent per-block mean); `all_pad_blocks = 0` so top-k cannot select a
  pad-only block; and **perturbing all 7,176 pad V slots by +100 left the output
  bit-identical**. The raster→tile→raster round trip is bit-identical at all three
  geometries, `n_valid_tokens = 32,760` in each. Correctness gate:
  a cube-order mask executed on the kernel matches an independent naive fp32
  masked-softmax reference at Wan's real 32,760-token shape to rel-L2 2.470e-03,
  cosine 0.9999970 — **identical to the raster arms**, so the residual is the
  BF16-vs-fp32 accumulation difference, not a geometry-dependent error
  (`raw/phase2b_selftest.json`, 29 checks, `PASS`).
- **One real budget confound, measured rather than hidden.** Sparsity is defined on
  the block axis and cube tiles are variable-size, so the cube arm retains **5–6%
  fewer tokens** at equal nominal sparsity (0.1912 vs 0.2012 at s=0.80). This
  biases the cube arm **against itself** on absolute error and cannot explain the
  mechanism ratios, which are all paired *within* a geometry against that
  geometry's own `C`.

**Two limits this control does NOT license.** (i) **H3 was not re-tested at cube
geometry** — the NVFP4-compute arms (`B`, `B_sim`, `D8`, `E`, `F8`, `F16`) were
deliberately not re-run, so **H3's falsification remains a 128x64 result**.
(ii) The cube arm ran on the **research Triton block-sparse kernel with a
mean-pooled research scorer, not through `VideoSparseAttentionImpl` with VSA's own
coarse scorer and gating** — so **nothing may be claimed about VSA end-to-end
quality**. What is established is that the *geometry* is not what makes the
mechanism work.
