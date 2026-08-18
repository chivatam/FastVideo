# W4A4_AMDAHL_ANALYSIS — supplementary

> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state:
> REPORT_V4.md. Appendix material only.

## FLOP share vs eager time vs optimized time (per transformer forward)

| Component | FLOP share (480p/720p) | D0 eager time | D1 optimized time |
|---|---|---|---|
| QKV GEMMs | 13.3% / 10.6% | 1.8% / 1.7% | (in GEMM total) |
| O GEMM | 4.4% / 3.5% | 0.6% / 0.5% | " |
| FFN GEMMs | 51.6% / 41.4% | 6.7% / 6.7% | " |
| cross-attn proj GEMMs | 9.0% / 7.1% | 1.5% / 1.5% | " |
| **linear GEMMs total** | **78.2% / 62.7%** | **11.3% / 10.8%** | **23.7% / 21.9%** |
| sparse attention (kernel) | 18.8% / 34.9% | 8.9% / 13.6% | 12.2% / 19.0% |
| sparse-attn machinery + tiling/rope/copies | ~0% | ~40% | ~38% |
| norms/elementwise/modulation | ~0% | ~35% | ~21% (fused) |

Receipts: `../../w4a4_gate/RUNTIME_BREAKDOWN_EAGER.md`,
`../../w4a4_gate/RUNTIME_BREAKDOWN_OPTIMIZED.md` (module CUDA events +
kernel-trace categorization agree; batch 1/2/4 shares flat).

## Amdahl panel (p = measured linear-GEMM time share)

| Stack | p | infinite-GEMM ceiling 1/(1-p) | 2x-GEMM ceiling 1/((1-p)+p/2) | measured W4A4 E2E |
|---|---|---|---|---|
| D0 eager 480p | 0.113 | 1.13x | 1.06x | 0.84-0.93x (slower) |
| D0 eager 720p | 0.108 | 1.12x | 1.06x | 0.92-0.98x (slower) |
| D1 optimized 480p | 0.237 | 1.31x | 1.13x | 0.75-0.87x (slower) |
| D1 optimized 720p | 0.219 | 1.28x | 1.12x | 0.89-0.96x (slower) |

The three columns distinguish theoretical ceiling, predicted realistic
ceiling, and the actual measured result — which sits below 1.0x because
per-call activation quantization and integration overheads exceed the GEMM
savings, and o_proj/QKV GEMMs (K=1536, memory-bound) gain nothing.

## Scope caveats (binding)

This bounds W4A4 value in THIS serving stack (eager/compiled FastVideo,
flashinfer-0.6.17 cudnn `mm_fp4`, unfused activation quant, B200,
Wan2.1-1.3B). It does NOT support universal claims such as "FLOPs never
correlate with latency", "all video DiTs are memory-bound", or "FP4 GEMMs
do not help video diffusion". Fused norm->quant epilogues and
CUDA-graph-safe attention could shift the calculus; the infinite-GEMM
ceiling (1.28-1.31x at D1) still bounds the opportunity.
