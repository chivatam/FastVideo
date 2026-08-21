# VSA Phase 1 — Local Cross-Query KV Reuse Characterization

**Question:** if PR [#1719](https://github.com/hao-ai-lab/FastVideo/pull/1719)'s
sm_100a CTA (which already owns two adjacent VSA query blocks, `mtile0 = 2p`,
`mtile1 = 2p+1`) loaded the *union* of their selected KV blocks once instead of
walking each row independently, how much KV-load reuse do **real** Wan VSA
masks expose?

**Answer: ~1.56x median theoretical KV-load reuse at the primary 720p
workload (1.47x at 480p). GO.**

## Workload

- Model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (the deployed VSA Wan2.1-T2V-1.3B),
  bf16, B200 (sm_100), 1 GPU, 3-step DMD schedule, no CFG.
- VSA sparsity 0.9, tile (4,4,4) = 64 tokens.
- Primary: 720p — latent (21,45,80) → tiles (6,12,20) = 1440 blocks, K=144.
- Secondary: 480p — latent (21,30,52) → tiles (6,8,13) = 624 blocks, K=63.
- Captured: every VSA layer (30) × every denoising step (3) × all 12 heads,
  3 prompts × 1 seed at 720p, 1 prompt at 480p → 360 layer/step shards,
  4320 (layer, step, head) samples.

## Primary table (720p, all pairs; medians over 2.3M–4.4M pairs)

| Pairing strategy | Median overlap/K | Median Jaccard | Median union/K | Median KV reuse | P10 | P90 |
|---|---|---|---|---|---|---|
| PR1719 current (2p, 2p+1) | 0.715 | 0.557 | 1.285 | **1.557** | 1.274 | 1.811 |
| horizontal (w+1) | 0.715 | 0.557 | 1.285 | 1.557 | 1.274 | 1.811 |
| temporal (t+1) | 0.688 | 0.524 | 1.313 | 1.524 | 1.079 | 1.823 |
| vertical (h+1) | 0.618 | 0.447 | 1.382 | 1.447 | 1.180 | 1.756 |

PR#1719's pairing *is* the horizontal pairing at 720p (n_w = 20 is even, so
every (2p, 2p+1) pair is a w-neighbor) — the current static pairing is already
the best of the tested static pairings. No re-pairing win is on the table.
Interior-only pairs (excluding partial boundary tiles): 1.53x — boundary
tiles do not inflate the result.

## Group-size extension (aligned local groups)

| Group size | Geometry | Median union/K | Median theoretical KV reuse |
|---|---|---|---|
| 2 | 1x1x2 (current pairs) | 1.285 | **1.557** |
| 4 | 1x2x2 spatial / 1x1x4 w-run | 1.778 | **2.250** |
| 4 | 2x1x2 (2t × 2w) | 1.771 | 2.259 |
| 8 | 2x2x2 cube | 2.375 | **3.368** |
| 8 | 1x2x4 slab | 2.396 | 3.339 |

Reuse does **not** saturate at pairs — g=4 exposes 2.25x and g=8 3.37x.

## Consistency (Part F)

- By layer: median reuse 1.40x (L18) … 1.73x (L0); every layer ≥ 1.40x.
- By step: 1.61x (step 0) → 1.52x (step 2); early denoising slightly higher.
- By head: 1.48x … 1.62x. No weak pockets → a single non-adaptive kernel
  change is justified; no adaptive dispatch needed.
- 480p secondary: median reuse 1.47x — same story at the smaller geometry.

See `pair_strategy_summary.csv`, `group_size_summary.csv`,
`layer_summary.csv`, `timestep_summary.csv`, `head_summary.csv`,
`summary.json`, and `plots/` (`secondary_480p/` for the 480p replication).
The headline plot is `plots/reuse_factor_histogram.png` (histogram + CDF of
`2K/|union|` for the current PR#1719 pairing).

## How the data was produced

1. Opt-in instrumentation (`FASTVIDEO_VSA_CAPTURE_OVERLAP=/path`) in
   `fastvideo_kernel.vsa_capture`, hooked immediately after
   `fused_topk_mask` in `fastvideo_kernel.ops.video_sparse_attn`; the
   framework backend pushes layer/timestep/CFG-branch context. Production
   execution is untouched (the bool mask keeps flowing to the kernels;
   compact indices exist only in the capture path).
2. Real inference runs (per prompt/resolution, one process each):

   ```bash
   PYTHONPATH=fastvideo-kernel/python HF_HOME=... \
   python artifacts/vsa_overlap_phase1/run_capture.py \
       --resolution 720p --prompt-id p0 --seed 1024 \
       --capture-root /mnt/nvme/outputs/vsa_capture
   # repeated for p1, p2 and once with --resolution 480p
   ```

3. Analysis:

   ```bash
   python artifacts/vsa_overlap_phase1/analyze_vsa_overlap.py \
       --capture-root /mnt/nvme/outputs/vsa_capture --run-glob '720p_*' \
       --out artifacts/vsa_overlap_phase1
   ```

Raw captures (~2 GB of compact int16 top-k indices) live outside the repo
under the capture root; each run dir carries a `run_meta.json` with model,
prompt, seed, geometry, sparsity, and the flattening convention.

## Sanity checks (all enforced in code / `summary.json`)

- Row counts: 262/270 shards have exactly K selections per row; 8 shards
  contain rare `fused_topk_mask` tie rows (≠K) and are stored losslessly via
  a ragged encoding (`counts_ok=False`, `reconstruct_ok=True`). Analysis uses
  true per-row set sizes.
- Compact indices reconstruct the original bool mask exactly (`reconstruct_bad: 0`).
- Pair metrics are order-invariant and verified against a hand example
  (`test_vsa_overlap_phase1.py`).
- q_block_id → (t,h,w) mapping verified against `get_tile_partition_indices`
  for both geometries; boundary (partial) tiles identified via
  `variable_block_sizes` and reported separately.
- Capture on/off produces bit-identical generated frames
  (sha256 `494b29a0…` both ways at 480p/seed 1024).

Unit tests: `PYTHONPATH=fastvideo-kernel/python pytest artifacts/vsa_overlap_phase1/ -v`

## Go / No-Go

Median reuse 1.56x > 1.50x threshold → **GO: proceed to the local-union
execution prototype for the existing two-Q-block CTA.** The g=4 (2.25x) and
g=8 (3.37x) numbers say a wider local group is the natural follow-on once
the pair-level union path exists.

This phase measures *theoretical load reuse only*; no kernel speedup is
claimed. Achieved speedup depends on how much of the sm_100a kernel's time
is KV-load-bound.
