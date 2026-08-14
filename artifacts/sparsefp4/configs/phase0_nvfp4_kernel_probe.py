"""Phase 0 evidence: prove the NATIVE NVFP4 attention kernel runs on this machine.

This is a direct kernel-level probe, deliberately independent of the video
pipeline, so the "native NVFP4 available" verdict rests on a real kernel launch
rather than on an import succeeding.

What "native" means here (per the study's scientific-integrity rules):
Q and K are quantized to NVFP4 (E2M1, per-16-element E4M3 scale factors) and
multiplied on Blackwell's 5th-gen tensor cores via
`tcgen05.mma...kind::mxf4nvf4.block_scale`; V/PV stays BF16. There is NO
fake/simulated quantize-dequantize anywhere in this path.

Latency numbers here are a warmed microbenchmark of the ATTENTION KERNEL ONLY
(CuTeDSL JIT compile excluded via warmup, CUDA synchronized, median of N reps).
They are not end-to-end generation speedups and must not be reported as such.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from fastvideo.attention.backends.attn_qat_infer import (
    attn_qat_infer_receipt,
    is_attn_qat_infer_available,
)
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo.attention.utils.flash_attn_cute import flash_attn_fp4_func, flash_attn_func

# Wan2.1-T2V-1.3B DiT self-attention shape at the experiment defaults
# (480x832, 81 frames): 12 heads x 128 head_dim, seqlen = 21 latent frames
# x 30 x 52 patches = 32760.
WAN_SHAPE = {"batch": 1, "seqlen": 32760, "heads": 12, "head_dim": 128}


def timed(fn, warmup: int = 3, reps: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    cap = tuple(torch.cuda.get_device_capability())
    b, s, h, d = WAN_SHAPE["batch"], WAN_SHAPE["seqlen"], WAN_SHAPE["heads"], WAN_SHAPE["head_dim"]
    torch.manual_seed(0)
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    scale = d**-0.5

    qf, qsf = _nvfp4_quantize_for_fa4(q)
    kf, ksf = _nvfp4_quantize_for_fa4(k)
    qf, kf = qf[:, :s], kf[:, :s]

    def run_fp4() -> torch.Tensor:
        out = flash_attn_fp4_func(qf, kf, v, qsf, ksf, softmax_scale=scale, causal=False)
        return out[0] if isinstance(out, tuple) else out

    def run_bf16() -> torch.Tensor:
        out = flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
        return out[0] if isinstance(out, tuple) else out

    out_fp4 = run_fp4()
    out_bf16 = run_bf16()
    torch.cuda.synchronize()

    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1,
                                                                                                            2).float()

    def acc(x: torch.Tensor) -> dict[str, float]:
        xf = x.float()
        return {
            "cosine_similarity": round(F.cosine_similarity(xf.flatten(), ref.flatten(), dim=0).item(), 6),
            "relative_l2_error": round(((xf - ref).norm() / ref.norm()).item(), 6),
            "max_abs_error": round((xf - ref).abs().max().item(), 6),
        }

    fp4_ms, fp4_sd = timed(run_fp4, reps=args.reps)
    bf16_ms, bf16_sd = timed(run_bf16, reps=args.reps)

    result = {
        "verdict": "NATIVE NVFP4 ATTENTION AVAILABLE AND EXECUTED",
        "arch": f"sm_{cap[0]}{cap[1]}",
        "device": torch.cuda.get_device_name(0),
        "attn_qat_infer_receipt": attn_qat_infer_receipt(),
        "attn_qat_infer_available": is_attn_qat_infer_available(),
        "shape": WAN_SHAPE,
        "shape_note": "Wan2.1-T2V-1.3B DiT self-attention at 480x832x81 (seqlen 21*30*52=32760)",
        "nvfp4_quantized_tensors": {
            "q_dtype": str(qf.dtype),
            "q_shape": list(qf.shape),
            "scale_factor_dtype": str(qsf.dtype),
            "scale_factor_shape": list(qsf.shape),
        },
        "precision_labels": {
            "fp4_path": "native NVFP4 (E2M1 Q/K, per-16 E4M3 scale factors, BF16 V/PV) "
            "via tcgen05.mma.kind::mxf4nvf4.block_scale",
            "bf16_path": "native BF16 FA4 CuTe",
            "is_simulated_quantization": False,
        },
        "accuracy_vs_bf16_sdpa_reference": {
            "native_nvfp4": acc(out_fp4),
            "native_bf16_fa4": acc(out_bf16),
        },
        "attention_kernel_only_latency_ms": {
            "native_nvfp4_median":
            round(fp4_ms, 3),
            "native_nvfp4_stdev":
            round(fp4_sd, 3),
            "native_bf16_fa4_median":
            round(bf16_ms, 3),
            "native_bf16_fa4_stdev":
            round(bf16_sd, 3),
            "nvfp4_speedup_vs_bf16":
            round(bf16_ms / fp4_ms, 3),
            "reps":
            args.reps,
            "warmup":
            3,
            "measurement_note":
            "Measured attention-kernel wall-clock only, warmed (JIT compile "
            "excluded), CUDA-synchronized, median of reps. NOT an end-to-end "
            "generation speedup and NOT a theoretical FLOP count.",
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
