# SparseFP4 Video Attention — STATUS

Last updated: 2026-08-14 (Phases 0–2 complete; H3 falsified)

## Run identity

- Repo: `/home/ec2-user/FastVideo`
- Branch: `exp/sparsefp4-mask-stability`
- Base commit: `8208536cd1db7a1d32b68aaa6a679953ae23ab8b` (main)
- Hardware: 8x NVIDIA B200 (sm_100, 183 GiB each), driver 595.91.07
- **Interpreter every phase must use:** `source artifacts/sparsefp4/configs/env.sh` then `"$FV_PYTHON"`
  (`/mnt/scratch/fv-venv/bin/python`, CPython 3.12.13, torch 2.12.0+cu130)
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b`
- Geometry: 30 layers x 12 heads x head_dim 128, seq_len 32760 @ 480x832x81, 50 steps, guidance 3.0

## What ran

**Phase 0 — PASS.** See [`PHASE0.md`](PHASE0.md) and [`CODEBASE_MAP.md`](CODEBASE_MAP.md).

**Phase 1 — PASS.** See [`PHASE1.md`](PHASE1.md).

**Phase 2 — PASS, H3 falsified.** See [`PHASE2.md`](PHASE2.md). Run
`20260814-025500-8208536-p2-main`: 615,380 records, 10 prompts x 17 layers x 12 heads x 5 timesteps
x 2 CFG branches x 3 sparsities, configs A–F (plus `B_sim` simulation control and `C_rand`
equal-magnitude random contrast control), fp64 block scores throughout, verification `PASS` with
zero failures. Recommendation: **PIVOT** away from precision-decoupled routing.

- Storage: root volume had only ~9 GiB free; an unmounted 3.5 TB instance-store NVMe was
  formatted and mounted as `/mnt/scratch` to hold the venv, the 27 GB model, and the CUDA toolkit.
- Environment: FastVideo installed editable against torch 2.12.0+cu130; `sm_100` confirmed in
  `torch.cuda.get_arch_list()` and by a real BF16 matmul (rel-err 1.7e-3). CUDA Toolkit 13.0.88
  installed (was absent at start), so Phase 4 is tooling-ready.
- Smoke tests: dense BF16 **PASS**, dense NVFP4 **PASS**.
- Attention-stack survey completed with file:line citations for every claim, including a ranked
  list of Phase 1 hook points and a recommended Phase 3 integration point.
- Protocol pre-registered before measurement: `EXPERIMENT_SPEC.md`, `REPORT_TEMPLATE.md`,
  `experiment_config.yaml`, a 10-prompt development set, and a GPU-free `analyze_masks.py`
  (self-test passes on both a numpy and a stdlib-only interpreter).

## What passed / failed

- PASS — **native NVFP4 attention is real, not simulated.** Evidence: emitted PTX contains
  `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` (4 occurrences in the NVFP4
  log, **0** in the BF16 log); genuinely FP4-typed tensors (`torch.float4_e2m1fn_x2` with a `uint8`
  scale-factor tensor); framework receipt `qk_mode=nvfp4(per-16-e4m3-sf) pv_mode=bf16`.
- PASS — a **real perturbation exists to study.** At the true Wan2.1 attention shape
  (B=1, seq 32760, 12 heads, dim 128) native NVFP4 vs BF16 SDPA gives cosine **0.99050**,
  rel-L2 **0.13783**. The path is not numerically null, so H1 has something to detect.
- PASS — measured attention-kernel microbenchmark (warmed, CUDA-synced, median of 20):
  native NVFP4 **4.013 ms** vs native BF16 FA4 **5.135 ms** = **1.28x**. Scope is the attention
  kernel only; this is neither an end-to-end speedup nor a FLOP-count claim.
- FAILED then FIXED — `nvcc` and a CUDA toolkit were absent; installed 13.0.88 to match torch cu130.
- CONSTRAINT — `FASTVIDEO_FA4=1` is mandatory: the flash-attention-fp4 fork ships no compiled FA2,
  so every dense attention path raises `ImportError` without it.
- REPRODUCIBILITY — the native FP4 attention path is **not** in upstream FlashAttention-4 (the pin at
  `pyproject.toml:127`). It requires the out-of-tree `hao-ai-lab/flash-attention-fp4` fork plus
  `nvidia-cutlass-dsl` (4.4.2, or >=4.5.2 depending on branch), `flashinfer-python`,
  `CUTE_DSL_ENABLE_TVM_FFI=1`, sm_100a/sm_103a, and no FSDP. Python **3.12** is required in practice
  (3.11 has no prebuilt `fastvideo-kernel` wheel and falls back to a source build); the docs' "3.10 or
  3.11" is stale. `quack-kernels==0.5.0` is the only release pinning cutlass-dsl >=4.5.2 without
  dragging in the unsupported 4.6 line. Pin all of these in the report's setup section.

## Important labeling correction

The FA4 kernel is `qk_mode=nvfp4, pv_mode=bf16`. **"Dense NVFP4" means NVFP4 Q/K with BF16 PV**,
not fully-FP4 attention. Every table and claim must say so. This also means the SKILL's Phase 4
ladder (NVFP4 QK + BF16 PV first) already describes the *existing* kernel's configuration.

## Traps and corrections (read before trusting any measurement)

These came out of the survey and must be honored by every later phase.

1. **A typo in `FASTVIDEO_ATTENTION_BACKEND` is silently ignored** (`selector.py:23-34`) — it falls
   through to auto-selection instead of erroring. A misspelled backend therefore yields a
   confidently-wrong measurement of the *default* path. **Every run must assert the resolved backend
   via `transformer.config._resolved_attention_backend`** and record it in the raw records.
   The string→class mapping lives in `platforms/cuda.py:112-288`, *not* in `selector.py` as
   `attention/AGENTS.md:100-108` claims.
2. **Corrected path**: the NVFP4 round-trip quant/dequant pair is at
   `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/nvfp4_utils.py:12-133` and `:136-239`.
   There is **no dequantizer inside `fastvideo/`** — only the forward quantizer
   `_nvfp4_quantize_for_fa4` (`flash_attn.py:138-141`).
3. **Block geometry conflict with the pre-registered spec.** VSA's blocks are 3D spatio-temporal
   cubes, `VSA_TILE_SIZE = (4, 4, 4)` = 64 tokens (`fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py:16`),
   and its query-block size **equals** its key-block size. The SKILL's default 128x64 geometry is
   therefore **not expressible in VSA**; it *is* native to SLA's `get_block_map` (`sla.py:78-110`).
   Consequence: the controlled 128x64 diagnostic and VSA's deployed 64-cube router are different
   geometries and must be reported as such, not conflated. Needs an amendment-log entry.
4. **RoPE is applied inside the attention layer** (`layer.py:130-132`), not in `wanvideo.py`.
   Anything captured at the model file is **pre-RoPE** — precisely the comparison the SKILL forbids.
5. **Phase 3 gap**: VSA-H3 is the only *backend* accepting an external block mask
   (`video_sparse_attn_h3.py:312`), but Wan cannot reach `VIDEO_SPARSE_ATTN_H3` / `BSA_ATTN` /
   `NABLA_ATTN` — they are absent from the DiT supported tuple (`configs/models/dits/base.py:22-29`).
   The real seam is one layer down, at the kernel functions `block_sparse_attn`,
   `block_sparse_attn_from_indices`, `block_sparse_attn_256_bshd`.
6. **`/mnt/scratch` is ephemeral instance store.** An instance stop/start wipes the venv, the 27 GB
   model, and the CUDA toolkit; `PHASE0.md` §3 is the rebuild recipe. The `/usr/local/cuda-13.0`
   bind mount is not in `fstab` and must be re-created after a reboot.
7. **No FlashAttention-2 is installed** — the FP4 fork replaces it, so `FASTVIDEO_FA4=1` must stay
   set for *any* attention path, BF16 included.
9. **Tie-count metrics must state their denominator.** Phase 1 counted `boundary_ties` **per
   (cell, head)** (115 of 256); Phase 2 counted **per cell** over the whole head x q-block grid
   (1,430 of 3,072). Those differ by 12x and looked like a contradiction for hours. They agree to
   4% once rescaled. Always emit both denominators.
8. **fp32 block scores are NOT safe for the H3 test — use fp64 (or mean-centred) scores.**
   Discovered in Phase 1. Block scores have magnitude ~5e5 while the *discriminative* spread between
   competing blocks is ~14, so fp32 lands margins on a power-of-two grid and manufactures **~110 exact
   boundary ties per cell**. This inflates apparent instability by +0.002-0.003 and — critically —
   **penalizes the FP8 arm 1.6x harder than the NVFP4 arm**. Since H3 is precisely a comparison of the
   FP8 and NVFP4 routers, running it in fp32 would partly measure float resolution rather than
   quantization precision, biasing the result *against* H3.
   **The null control cannot catch this**: bf16-vs-bf16 passed the entire time because both sides of
   the identity hit the same grid. A passing null control does not certify scorer resolution.

## Exact next action

Phase 2 is **complete** — see [`PHASE2.md`](PHASE2.md). H3 is falsified with a measured mechanism.
Next: `GO_NO_GO.md` synthesizing Phases 0–2, then the highest-value follow-up is confirming the
mechanism at VSA's real 64x64 spatio-temporal-cube geometry (configs `C`/`D`/`C_rand` only), since
geometry is the one threat that could change the conclusion for the deployed path.

## Strongest current result

**The near-tie mechanism (Phase 2, complete, 10 prompts, `n = 20,400` paired cells per sparsity).**
Quantization-induced routing swaps land almost exclusively on near-degenerate top-k boundaries, so
they are nearly free:

| quantity (median, all regions) | sp 0.80 | sp 0.90 | sp 0.95 |
|---|---|---|---|
| attention mass of *swapped-out* blocks | 0.00144 | 0.00283 | 0.00463 |
| attention mass of blocks both masks *agree* on | 0.00822 | 0.01320 | 0.01773 |
| mass dropped by an equal-size **random** swap | 0.00324 | 0.00525 | 0.00813 |
| score gap of swapped blocks (normalized) | 0.00484 | 0.00468 | 0.00423 |
| blocks swapped per query block (mean) | 1.43 | 1.00 | 0.66 |

The decisive number is the **equal-magnitude random contrast**: changing the same *number* of blocks
at random costs **27x / 22x / 10x** more output error than letting NVFP4's quantization choose them
(`D - C` = 3.1e-05 vs `C_rand - C` = 8.4e-04 at sp 0.80). Quantization does not make a small mask
error — it makes the *cheapest possible* error of that size. It perturbs only the harmless boundary.

Consequently the wrong-mask term is **0.02%–0.03% of total error**, while sparsification is 68% at
sp 0.80 rising to **93% at sp 0.95**. H3's measured effect is **0.04%–0.10%** against a
pre-registered **≥20%** threshold, and a *BF16* router — the theoretical ceiling of the whole idea —
is no better than FP8.

**Confirmed at the deployed geometry (Phase 2B).** Re-running `C`/`D`/`C_rand` at VSA's real
64-token `(4,4,4)` cubes *strengthens* the mechanism: the random-contrast ratio rises to
**75.6x / 44.6x / 35.7x** (from 27.0/21.7/10.1 at 128x64 raster), the agreed/swapped mass gap
widens to 6.9–11.1x, and cube Jaccard IQR is 25% narrower. Coherent cube tiles *separate* block
scores, pushing near-ties further into the mass tail. Token ordering drives this more than block
size. See [`PHASE2B_GEOMETRY.md`](PHASE2B_GEOMETRY.md) for the padding gate-checks and the two
limits it does not license (H3 was not re-tested at cube geometry; VSA's own scorer was not used).

## Current blocker

None. Phase 2 complete and verified (`PASS`, zero failures, 615,380 records).

## Run hygiene note

The first Phase 2 launch (`20260814-023500`) mixed records written before and after a code change and
was **quarantined**, not used: `/mnt/scratch/sparsefp4/QUARANTINED-20260814-023500-p2-main-mixed-code`.
One shard also crashed on a stale local variable from that mid-flight edit. All 10 prompts were
re-run from scratch as `20260814-025500` under a clean code state; no quarantined number appears in
`PHASE2.md`. Its medians agree with the final run's to ~3 significant figures, so the exclusion was
hygiene, not result-shopping. Recorded per SKILL rule 9 (no silent discarding).

## Hypothesis state

| Hypothesis | State |
|---|---|
| H1 — NVFP4 Q/K perturbs top-block routing vs BF16 | **SUPPORTED IN DIRECTION ONLY** — monotone in
  sparsity and precision, null control exact at 1.0, but the effect is small: median Jaccard 0.9807
  @sp0.80 and 0.9738 @sp0.90 (n=72,000/cell), with 97%/89% of cells above 0.95. Spearman rho 0.9997.
  Meets the pre-registered PIVOT condition. |
| H2 — instability is localized in heads/layers/timesteps | **WEAK; LAYERS ONLY, AND INCONSEQUENTIAL.**
  Affected cells (<0.90): timesteps 0/50, heads 0/12, layers 0/30, layer x head 2/360, layer x head x
  timestep 25/3600 (0.7%). Residual structure sits at the network edges (layers 0,1,2,27,28,29).
  **The saturation confound is now CLOSED and rejected** (Phase 2 §6): NVFP4 saturation is flat across
  all 17 measured layers (0.096–0.110) and correlates with wrong-mask error at Spearman **-0.25** —
  the wrong sign. But the localization does not matter: the worst layer's wrong-mask excess is
  2.0e-04, and edge layers' extra churn lands on *even less* important blocks
  (dropped/agreed mass ratio 0.13 affected vs 0.32 unaffected). More churn, less consequence. |
| H3 — higher-precision router recovers sparse-attention error | **FALSIFIED.** Measured relative
  reduction **0.04%–0.10%** vs a **≥20%** pre-registered threshold, over n=20,400 exactly paired
  cells at each of sp {0.80, 0.90, 0.95}; only 52–56% of cells improve at all, and a BF16 router (the
  ceiling) matches FP8. Mechanism measured, not assumed (see above). fp64 scores used throughout
  (trap #8), enforced by a hard gate in `phase2_analyze.verify`. |
| H4 — native sparse-NVFP4 gives wall-clock benefit | UNTESTED, but **available**: native FP4 kernel
  exists to extend, its source is editable Python/CuTeDSL (not a C++ recompile), CUDA 13.0.88 present,
  and the dense-NVFP4 bar to beat is measured at 4.013 ms |
