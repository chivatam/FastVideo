# FULL_DQVSA_GATE_DECISION — final systems gate (with AMDAHL_ANALYSIS)

## Required figure/table: FLOP share vs eager time vs optimized time

Per transformer forward, 480p / 720p:

| Component | FLOP share | D0 eager time | D1 optimized time |
|---|---|---|---|
| QKV GEMMs | 13.3% / 10.6% | 1.8% / 1.7% | (in 23.7% / 21.9% GEMM total) |
| O GEMM | 4.4% / 3.5% | 0.6% / 0.5% | " |
| FFN GEMMs | 51.6% / 41.4% | 6.7% / 6.7% | " |
| cross-attn proj GEMMs | 9.0% / 7.1% | 1.5% / 1.5% | " |
| **linear GEMMs total** | **78.2% / 62.7%** | **11.3% / 10.8%** | **23.7% / 21.9%** |
| sparse attention (kernel) | 18.8% / 34.9% | 8.9% / 13.6% | 12.2% / 19.0% |
| sparse-attn machinery + tiling/rope/copies | ~0% | ~40% | ~38% |
| norms/elementwise/modulation | ~0% | ~35% | ~21% (fused) |
| quantization (attention QK) | ~0% | (in machinery) | (in machinery) |
| other | — | ~2% | ~1% |

(Receipts: `RUNTIME_BREAKDOWN_EAGER.md`, `RUNTIME_BREAKDOWN_OPTIMIZED.md`;
module-CUDA-event and kernel-trace methods agree.)

## Amdahl panel

Let p = measured linear-GEMM time share.

| Stack | p | infinite-GEMM ceiling 1/(1-p) | 2x-GEMM ceiling 1/((1-p)+p/2) | **measured W4A4 E2E effect** |
|---|---|---|---|---|
| D0 eager 480p | 0.113 | 1.13x | 1.06x | **0.84x-0.93x (slower)** |
| D0 eager 720p | 0.108 | 1.12x | 1.06x | **0.92x-0.98x (slower)** |
| D1 optimized 480p | 0.237 | 1.31x | 1.13x | **0.75x-0.87x (slower)** |
| D1 optimized 720p | 0.219 | 1.28x | 1.12x | **0.89x-0.96x (slower)** |

Measured result sits BELOW even the do-nothing line because W4A4's
activation-quantization + launch overhead exceeds the GEMM savings, and
o_proj/QKV GEMMs (K=1536, memory-bound) gain nothing or lose.

## The ten required answers

1. **Linear GEMMs as % of nominal FLOPs:** 78.2% (480p), 62.7% (720p).
2. **% of eager measured GPU time:** 11.3% / 10.8% (verified two ways,
   stable across batch 1/2/4).
3. **% after reasonable serving optimization (torch.compile blocks,
   1.34-1.39x faster, outputs equivalent):** 23.7% / 21.9%.
4. **Native W4A4 speedup on the GEMMs themselves:** FFN 1.5-2.0x including
   activation quant (2.3-3.2x pre-quantized); QKV 1.07x; o_proj 0.39-0.49x
   (slower).
5. **B=1 E2E speedup from W4A4:** none — measured **slowdowns** of 2.4-19%
   (eager) and 4.5-33% (compiled) across W1/W2/W3 at both resolutions.
6. **Throughput/concurrency improvement:** none — component shares are flat
   in batch (B=1/2/4), so throughput tracks latency, which regressed.
7. **Memory/capacity improvement:** none realized — ~2.2 GB weight savings
   potential, but measured peak grew (transient quant buffers) and capacity
   is compute-bound, not memory-bound (28.3 GB peak at 720p B=4 on 183 GB).
8. **PTQ quality loss:** not evaluated at video level (triage skipped —
   the gate fails on performance alone); per-GEMM NVFP4 error is at the
   arithmetic floor (rel-L2 0.134 per linear).
9. **Did any training gate trigger?** No. Gate A: failed (no >=10% E2E —
   sign is negative). Gate B: first clause borderline-met at D1
   (21.9-23.7%), second clause failed (no credible >=10% E2E path — the
   measured path is negative and the realistic ceiling is ~1.12-1.13x).
   Gate C: failed. Gate D: failed.
10. **Is W4A4 DQ distillation justified?** No.

## Decision

**STOP FULL-DQ-VSA — AMDAHL-LIMITED IN THIS SERVING STACK**

The negative systems result, stated for the paper: linear GEMMs dominate
nominal DiT FLOPs (63-78%) but are not the latency bottleneck in the
measured SparseFP4 serving regime — 11% of GPU time eager, 22-24% after
torch.compile — because sparse-attention integration machinery
(tiling/scatter/copies/quantize, ~38-40%) and norm/elementwise work
dominate. Native W4A4 NVFP4 GEMMs, despite genuine 1.5-2x FFN kernel
speedups, produce end-to-end slowdowns once per-call activation
quantization and integration overheads are paid. This is the second
independent bottleneck-migration example in this project (the first:
sparse+FP4 attention, where FP4's MMA advantage was consumed by
softmax/predicate/allocator costs), supporting the central systems
conclusion: **arithmetic-intensity reduction does not guarantee
proportional end-to-end acceleration in video DiTs, because each
optimization shifts the bottleneck composition.**

Caveats bounding the claim: single GPU (B200), single model (Wan2.1-1.3B),
flashinfer-0.6.17 cudnn `mm_fp4` path with unfused activation quant; a
serving stack with fused norm->quant epilogues (e.g. `rmsnorm_fp4quant`)
and CUDA-graph capture could reduce the overhead — but even the
infinite-GEMM-speed ceiling at D1 (1.28-1.31x) bounds the opportunity
below the cost of a new QAT/distillation effort for this system.

Per the stop condition: no further experiments. Next step is paper writing.
