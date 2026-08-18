> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state: REPORT_V4.md (see REPORT_CANONICAL.md). Human-readable summaries: supplementary/w4a4_gate/.

# RUNTIME_BREAKDOWN_EAGER — D0 component decomposition (Phase 1)

D0 = B0 (DQ-VSA T3-c500) through the native P4 operator, eager FastVideo
stack, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 1x B200.
Method: CUDA-event pairs around disjoint module families (stream-ordered,
no per-module syncs; 3 warmup + 3 measured forwards;
`profile_components.py`, raw JSONs in `raw/` and
`/mnt/nvme/scratch/sparsefp4_native/w4a4_gate/`), cross-checked against
chrome kernel traces (`full_dqvsa/MODEL_FLOP_RUNTIME_BREAKDOWN.md`).
Wall here = one transformer forward (one CFG branch); E2E receipts imply
the same per-forward figure (450 ms at 480p), so GPU-time share == wall
share in this stack.

## Per-forward wall decomposition (ms, share of wall)

| Component | 480p B=1 | 480p B=2 | 480p B=4 | 720p B=1 | 720p B=2 | 720p B=4 |
|---|---|---|---|---|---|---|
| sparse attention total (selector+quant+FP4 kernel+glue) | 218.1 (47.4%) | 433.2 (48.4%) | 856.4 (48.7%) | 542.8 (50.6%) | 1096.4 (51.1%) | 2199.3 (51.3%) |
| norms (FP32LayerNorm/RMSNorm) | 159.1 (34.6%) | 307.9 (34.4%) | 603.2 (34.3%) | 352.7 (32.9%) | 701.2 (32.7%) | 1392.5 (32.5%) |
| FFN GEMMs | 30.9 (6.7%) | 61.7 (6.9%) | 124.0 (7.1%) | 71.3 (6.7%) | 144.2 (6.7%) | 291.4 (6.8%) |
| self QKV proj GEMMs | 8.3 (1.8%) | 16.1 (1.8%) | 32.3 (1.8%) | 18.3 (1.7%) | 37.0 (1.7%) | 74.7 (1.7%) |
| cross-attn proj GEMMs | 7.1 (1.5%) | 13.7 (1.5%) | 27.2 (1.5%) | 15.9 (1.5%) | 32.1 (1.5%) | 64.5 (1.5%) |
| gate_compress proj GEMM | 2.9 (0.6%) | 5.6 | 11.2 | 6.4 | 12.9 | 26.1 |
| self o_proj GEMM | 2.6 (0.6%) | 5.1 | 10.3 | 5.3 | 10.8 | 21.8 |
| cross-attn kernel | 3.1 (0.7%) | 6.1 | 12.3 | 6.9 | 14.0 | 28.3 |
| other linears (embed/head) | 0.2 | 0.4 | 0.9 | 0.5 | 1.0 | 2.1 |
| residual glue (modulation/residual/rope/patchify, unhooked) | 28.0 (6.1%) | 48.7 | 85.4 | 54.7 (5.1%) | 100.8 | 202.3 |
| **wall total** | **460.4** | **895.7** | **1757.4** | **1072.1** | **2144.4** | **4290.8** |
| **linear GEMMs total** | **51.9 (11.3%)** | **100.9 (11.3%)** | **202.6 (11.5%)** | **116.1 (10.8%)** | **234.7 (10.9%)** | **473.7 (11.0%)** |
| peak memory (transformer fwd only) | 3.3 GB | 6.4 GB | 12.4 GB | 7.3 GB | 14.3 GB | 28.3 GB |

## Kernel-trace cross-check (from `MODEL_FLOP_RUNTIME_BREAKDOWN.md`)

- The FP4 attention *kernel* itself is only 8.9% (480p) / 13.6% (720p) of
  GPU kernel time -> of the ~48-51% "sparse attention total" above,
  **~35-38 points are integration machinery**: BF16->NVFP4 quantize +
  packing, selector coarse branch, BHSD `.contiguous()` copies, tile
  scatter/gather (`index_elementwise`), mask packing.
- GEMM kernels (`nvjet_sm100_*`) = 11.2%/10.8% of kernel time — matches
  the module-event measurement exactly (GEMM modules launch ~only GEMMs).
- The GEMMs already sustain ~1.6 PFLOP/s BF16 (82 TFLOP / 51.9 ms at 480p)
  — near B200 peak; they are *efficient*, just small relative to overhead.

## FLOP share vs measured time share (the required contrast)

| Component | FLOP share 480p | measured 480p | FLOP share 720p | measured 720p |
|---|---|---|---|---|
| linear GEMMs (QKV+O+FFN+cross) | 78.2% | **11.3%** | 62.7% | **10.8%** |
| sparse self-attention QK+PV | 18.8% | 9% kernel (47% with machinery) | 34.9% | 14% kernel (51% with machinery) |
| norms/elementwise/glue | ~0% | ~41% | ~0% | ~38% |

## Verified answers (Phase 1)

1. The ~11% GEMM-time measurement is **reproduced** with an independent
   method (module CUDA events vs kernel-trace categorization) at both
   resolutions and B=1/2/4.
2. **Batching does not shift the bottleneck**: shares are flat in B
   (everything scales ~linearly; per-sample work already fills the GPU),
   so the throughput path (Gate C) cannot come from batching alone in this
   stack.
3. The non-GEMM time is dominated by two specific, attributable costs:
   (a) sparse-attention integration machinery (~35-38% of wall) and
   (b) unfused FP32 norms + modulation (~33-35% of wall) — both are
   serving-stack artifacts in principle addressable by fusion/compilation,
   NOT inherent arithmetic.
