> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state: REPORT_V4.md (see REPORT_CANONICAL.md). Human-readable summaries: supplementary/w4a4_gate/.

# MODEL_FLOP_RUNTIME_BREAKDOWN — Wan2.1-1.3B at the P4/B0 operating point

Required first measurement for Full-DQ-VSA (do not assume the "~2/3 of DiT
FLOPs" folklore). Model: Wan2.1-T2V-1.3B (dim 1536, 12x128 heads, FFN 8960,
30 layers, text len 512). Operating point: native P4 serving (VSA256, exact
10% retained fine attention, native NVFP4 QK, BF16 PV, BF16 linears), B0 =
T3-c500 weights. Receipts: `flop_breakdown.py --analytic`, chrome traces
from `profile_dit_step.py` (standalone transformer forward, 2 warmup + 2
profiled reps, `prof_dqvsa/trace{480,720}.json`; per-forward totals match
the E2E DiT receipts: 445 ms/forward measured here vs 450 ms/forward
implied by `dqvsa_final_performance.md` at 480p).

## 1. Analytic FLOPs per transformer forward

### 480x832x81 (seq 32,760)

| Component | GFLOPs | share |
|---|---|---|
| FFN linears | 54,103 | 51.6% |
| self-attention (sparse QK+PV @10%) | 19,782 | 18.8% |
| QKV projections | 13,912 | 13.3% |
| cross-attn projections | 9,420 | 9.0% |
| O projection | 4,637 | 4.4% |
| cross-attn QK+PV | 3,092 | 2.9% |
| selector/coarse branch | 3 | 0.0% |
| **linear GEMMs total** | **82,073** | **78.2%** |

### 720x1280x81 (seq 75,600)

| Component | GFLOPs | share |
|---|---|---|
| FFN linears | 124,854 | 41.4% |
| self-attention (sparse QK+PV @10%) | 105,346 | 34.9% |
| QKV projections | 32,105 | 10.6% |
| cross-attn projections | 21,548 | 7.1% |
| O projection | 10,702 | 3.5% |
| cross-attn QK+PV | 7,135 | 2.4% |
| **linear GEMMs total** | **189,209** | **62.7%** |

The "~2/3" folklore holds only at 720p; at 480p linears are ~78% of
arithmetic because 10%-sparse attention has already collapsed the
attention share.

## 2. Measured CUDA kernel time (the number that governs E2E speed)

Categorized by kernel symbol (categorization receipts in
`flop_breakdown.py`; `nvjet_*` = cuBLAS GEMM, `flash_fwd_sm100_fp4` = FP4
attention):

### 480p (0.89 s GPU / 2 forwards = 445 ms/forward)

| Category | share |
|---|---|
| norm / elementwise / copies / index (incl. RoPE, modulation, tile scatter-gather, `.contiguous()` transposes) | **78.1%** |
| linear GEMMs (`nvjet_sm100_...`) | 11.2% |
| FP4 attention fine kernel | 8.9% |
| other | 1.9% |

### 720p (2.13 s GPU / 2 forwards)

| Category | share |
|---|---|
| norm / elementwise / copies / index | **74.0%** |
| FP4 attention fine kernel | 13.6% |
| linear GEMMs | 10.8% |
| other | 1.5% |

Cross-checks: FP4 attention 135 ms/forward at 720p = 4.5 ms/layer-call,
consistent with the kernel benchmarks (3.2-4.5 ms at 720p/10% with
mask_mod); GEMM 50 ms/forward at 480p over 82 TFLOP = ~1.6 PFLOP/s
sustained BF16 — near B200 peak, i.e. the GEMMs are already efficient.

## 3. Implication for Full-DQ-VSA (Amdahl, honest)

The deployment (eager FastVideo, no torch.compile/CUDA-graphs — the V3
canonical protocol) is **elementwise/memory-bound, not GEMM-bound**:
linear GEMMs are 78%/63% of FLOPs but only **~11% of measured GPU time**
at both resolutions.

Consequence: even an infinitely fast W4A4 linear path caps the E2E DiT
speedup at ~1.12x; a realistic ~2x GEMM kernel gain yields **~5-6% E2E**
— before paying activation-quantization overhead per linear call.

The FLOP story motivates W4A4 for arithmetic-bound deployments (compiled/
fused serving stacks, larger models, longer sequences); the measured story
says that in THIS serving stack the experiment cannot produce a material
E2E win. Kernel-level W4A4 speedups and the quality-composition science
(F1-F3 sensitivity, F4 recovery) remain measurable and reportable, but the
"REAL speed" bar from the experiment brief cannot be met E2E without first
removing the elementwise overhead (fusion/compile — out of scope per the
stop condition).

Caveat: the elementwise category includes attention-integration overheads
(BHSD `.contiguous()` copies, tile scatter/gather in the VSA256 backend)
alongside model-level norms/modulation/GELU; a fusion pass would attack
both. No FLOP-derived speedup claims are made anywhere.
