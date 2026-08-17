# Full-DQ-VSA — native W4A4 NVFP4 linear-GEMM tooling audit

Date: 2026-08-17. Environment: pinned V3 canonical venv
(`/mnt/nvme/scratch/fv-venv`, torch 2.12.0+cu130, 8x B200 sm_100, CUDA 13.0).

Question: can F1-F4 execute REAL Blackwell NVFP4 W4A4 GEMMs without adding
new dependencies or writing a research kernel?

## Answer: YES — flashinfer 0.6.17 (already pinned) ships the native path

Relevant surface (verified by import on this host):

| API | Role |
|---|---|
| `flashinfer.mm_fp4(a, b, a_descale, b_descale, alpha, block_size=16, use_nvfp4=True, backend=...)` | native W4A4 GEMM: packed `float4_e2m1fn_x2` A/B + per-16 E4M3 block scales + FP32 global alpha; backends `cudnn`/`trtllm`/`cutlass`/`cute-dsl`; SM100 route includes `Sm100BlockScaledPersistentDenseGemmKernel` (CuTe-DSL block-scaled tensor-core MMA) |
| `flashinfer.nvfp4_quantize` / `fp4_quantize` | production activation/weight quantizer — the SAME quantizer family already used by our FA4 attention path (`_nvfp4_quantize_for_fa4`) and validated by the C4 dequant oracle |
| `flashinfer.nvfp4_batched_quantize`, `nvfp4_block_scale_interleave` | batched activation quant + SF layout helpers |
| `flashinfer.rmsnorm_fp4quant`, `add_rmsnorm_fp4quant` | fused norm -> NVFP4-quantized activations (epilogue fusion for the pre-QKV / pre-FFN norms) |
| `flashinfer.silu_and_mul_nvfp4_quantize` | fused FFN gate activation -> NVFP4 (Wan FFN uses GELU, not SiLU — check applicability; plain `nvfp4_quantize` after activation is the fallback) |
| `flashinfer.prepare_bf16_fp4_weights`, `mm_bf16_fp4` | W4A16 variant (weights FP4, activations BF16) — useful ablation lever if A4 proves too damaging |
| `flashinfer.mm_nvfp4_svdquant`, `gemm_svdquant` | SVDQuant-style outlier-absorbing variant — out of scope for the first pass, noted as fallback |

NOT available / not needed:
- `nvidia-modelopt`, TensorRT: not installed; not required — the quantization
  representation used for training fake-quant can exactly mirror
  `nvfp4_quantize` + `mm_fp4` semantics (per-16 E4M3 scales, E2M1 elements,
  FP32 alpha), keeping train/serve parity without an adapter layer.
- transformer-engine, torchao: not installed.

## Native-proof plan (for `LINEAR_QUANTIZATION_NATIVE_PROOF.md` later)

1. Source proof: replacement `nn.Linear` forward calls
   `nvfp4_quantize(x)` + packed weight buffer + `mm_fp4` — no BF16
   dequantize before the GEMM.
2. Runtime proof: assert input dtypes at the GEMM boundary are
   `float4_e2m1fn_x2`/uint8 + `float8_e4m3fn` scales.
3. Profiler proof: kernel symbol capture (expect the SM100 block-scaled
   GEMM kernel from the selected backend, e.g.
   `Sm100BlockScaledPersistentDenseGemmKernel` for cute-dsl or the
   cudnn/cutlass equivalents).
4. Wall-clock: kernel-level W4A4 vs BF16 GEMM at Wan2.1-1.3B shapes
   (hidden 1536, FFN 8960, seq 32k-92k tokens), plus quantization overhead.

## Risks noted

- Activation quantization overhead per linear call (amortizable via the
  fused norm/activation quant epilogues; must be included in E2E numbers).
- Weight layout: `mm_fp4` wants column-major packed B + interleaved SF —
  one-time offline conversion per checkpoint, verify with a dequant oracle
  round-trip against the BF16 master weights.
- Wan FFN activation is GELU (`ffn.fc_in`/`fc_out` with gelu): the fused
  SiLU quant helper does not apply; use plain `nvfp4_quantize` after GELU.
- `mm_fp4` batching: DiT linears see (B*S, K) GEMMs — m is large (>=32k),
  well inside the persistent-kernel sweet spot.
