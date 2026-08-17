# DQVSA_NATIVE_SERVING_PROOF — trained checkpoints run the unmodified native P4 path

Claim: the DQ-VSA candidates (T2-c250, T3-c250, T3-c500) are evaluated and
served through EXACTLY the same native SparseFP4 serving operator as the
untrained P4 arm. Training changed model weights only; no serving code,
kernel, geometry, precision, or selector changed.

## 1. Code-path invariance (source proof)

- Serving backend: `SPARSEFP4_VSA256_FA4_ATTN`
  (`fastvideo/attention/backends/sparsefp4_vsa256_fa4.py`) — untouched by
  the DQ-VSA work (git history: training additions live in the separate
  `sparsefp4_qat_vsa256.py` backend and `wan_dqvsa_distillation_pipeline.py`).
- The trained checkpoints are plain diffusers-layout transformer weight
  directories symlink-assembled into a Wan pipeline
  (`/mnt/nvme/scratch/sparsefp4_native/t_serve/<cand>`); every other
  component (VAE, text encoder, scheduler, tokenizer, model_index) symlinks
  the canonical HF snapshot. A weights-only swap cannot alter the executed
  kernel path.
- `NATIVE_PROOF.md` (V2/V3, unchanged) documents the kernel itself: packed
  `float4_e2m1fn_x2` Q/K + per-16 E4M3 SFs, `flash_fwd_sm100_fp4`
  block-scaled MMA over retained-tile-only loops, BF16 PV, no BF16 Q/K
  materialization, profiler receipts.

## 2. Runtime receipts (from the paper-scale generation logs, `logs/t_final/`)

Every worker process logs the resolved backend and operator configuration.
Representative lines (identical across all shards of all candidates):

```text
INFO ... [cuda.py:118] Selected backend: AttentionBackendEnum.SPARSEFP4_VSA256_FA4_ATTN
INFO ... [sparsefp4_vsa256_fa4.py:227] sparsefp4 VSA256-FA4: fine=nvfp4,
    tiles=168 x 256 tokens, keep256=0.1012 (exact FA4 mapping, no coarsening)
```

- `fine=nvfp4`: native NVFP4 QK fine branch (same env/default as P4).
- `tiles=168 x 256 tokens`, `keep256=0.1012`: identical VSA256 geometry and
  exact 10% retention (the same keep fraction the untrained P4 logs show).
- Selector remains BF16 (selector code path unchanged; see backend source).
- 480p (168 tiles) and 720p (360 tiles) receipts both present in
  `logs/t_final/dqvsa-*.log`.

## 3. Performance invariance (wall-clock receipt)

`tables/dqvsa_final_performance.md` — trained checkpoints vs the untrained
P4 rows of `tables/c8_performance_v2.md`, same allocator config
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), same protocol (median
of 3 steady-state reps, first gen excluded as warmup/JIT). Latency equality
within noise is the operational proof that no compute path changed:
weights-only differences cannot change kernel selection, and measured E2E
confirms it.

## 4. What training-time fake quantization does NOT affect

Training used fake-quant NVFP4 (production quantizer round-trip) with the
autograd Triton fine kernel; that operator exists only in
`SPARSEFP4_QAT_VSA256_ATTN` and is never selected at serving. The final
claims rest exclusively on the native-kernel generations above.
