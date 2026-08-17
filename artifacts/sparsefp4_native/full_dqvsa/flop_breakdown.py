"""Full-DQ-VSA prerequisite: Wan2.1-1.3B FLOP + measured-runtime breakdown.

Two parts:

1. Analytic FLOP breakdown from the actual model shapes (dim 1536, 12 heads
   x 128, FFN 8960, 30 layers, text len 512) at the P4 operating point
   (VSA256, 10% retained fine attention + 1/256-pooled coarse branch),
   480p (seq 32760) and 720p (seq 75600).

2. Measured CUDA-kernel-time breakdown parsed from a torch-profiler chrome
   trace of the real denoising loop (FASTVIDEO_TORCH_PROFILER_DIR +
   FASTVIDEO_TORCH_PROFILE_REGIONS=inference_denoising), categorized by
   kernel symbol. Categorization is by substring and is approximate where
   GEMM kernels serve several call sites (noted in the output).

Usage:
  flop_breakdown.py --analytic
  flop_breakdown.py --trace <trace.json[.gz]> --label "P4 480p"
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

# Wan2.1-T2V-1.3B
DIM = 1536
HEADS, DHEAD = 12, 128
FFN = 8960
LAYERS = 30
TEXT = 512
KEEP = 0.10          # exact VSA256 retention
TILE = 256


def gflops_per_step(seq: int) -> dict[str, float]:
    """Forward GFLOPs for ONE transformer forward (one CFG branch)."""
    n_tiles = (seq + TILE - 1) // TILE
    f = defaultdict(float)
    # per layer
    qkv = 3 * 2 * seq * DIM * DIM
    o = 2 * seq * DIM * DIM
    fine_attn = 2 * (2 * seq * seq * DHEAD * HEADS) * KEEP        # QK + PV on retained
    coarse = 2 * (2 * n_tiles * n_tiles * DHEAD * HEADS) + 3 * 2 * n_tiles * DIM  # pooled attn
    xattn_q = 2 * seq * DIM * DIM
    xattn_kv = 2 * 2 * TEXT * DIM * DIM
    xattn_o = 2 * seq * DIM * DIM
    xattn = 2 * (2 * seq * TEXT * DHEAD * HEADS)
    ffn = 2 * 2 * seq * DIM * FFN
    f["qkv_proj"] = LAYERS * qkv
    f["o_proj"] = LAYERS * o
    f["self_attention (sparse QK+PV @10%)"] = LAYERS * fine_attn
    f["selector/coarse branch"] = LAYERS * coarse
    f["cross_attn projections"] = LAYERS * (xattn_q + xattn_kv + xattn_o)
    f["cross_attn QK+PV"] = LAYERS * xattn
    f["ffn"] = LAYERS * ffn
    return {k: v / 1e9 for k, v in f.items()}


def analytic() -> None:
    for label, seq in (("480x832x81 (seq 32760)", 32760), ("720x1280x81 (seq 75600)", 75600)):
        f = gflops_per_step(seq)
        total = sum(f.values())
        lin = f["qkv_proj"] + f["o_proj"] + f["cross_attn projections"] + f["ffn"]
        print(f"\n## Analytic FLOPs per transformer forward — {label}\n")
        print("| Component | GFLOPs | share |")
        print("|---|---|---|")
        for k, v in sorted(f.items(), key=lambda kv: -kv[1]):
            print(f"| {k} | {v:,.0f} | {100 * v / total:.1f}% |")
        print(f"| **total** | {total:,.0f} | 100% |")
        print(f"\nLinear GEMMs (QKV+O+cross-proj+FFN): {lin:,.0f} GFLOPs "
              f"= **{100 * lin / total:.1f}%** of forward FLOPs.")
        print(f"Sparse self-attention (10% retained): "
              f"{100 * f['self_attention (sparse QK+PV @10%)'] / total:.1f}%.")


CATEGORIES = [
    ("attention fine kernel (FP4 FA4)", ["flash", "fp4_fwd", "fmha", "sm100_fp4"]),
    ("quantize (NVFP4 Q/K)", ["quantize", "fp4quant", "nvfp4"]),
    ("selector/coarse + fused triton", ["triton"]),
    ("linear GEMMs (cublas/cutlass)", ["gemm", "nvjet", "cutlass", "sgemm", "hgemm",
                                       "gemvx", "matmul", "cublas"]),
    ("norm/elementwise", ["elementwise", "vectorized", "layer_norm", "rms", "norm",
                          "reduce_kernel", "softmax", "cat", "copy", "fill", "index",
                          "unrolled", "gelu", "sigmoid", "mul", "add", "div"]),
]


def parse_trace(path: Path, label: str) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    events = data["traceEvents"] if isinstance(data, dict) else data
    total = 0.0
    per_cat: dict[str, float] = defaultdict(float)
    per_kernel: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op", "Kernel"):
            continue
        dur = float(e.get("dur", 0.0))
        name = e.get("name", "").lower()
        total += dur
        per_kernel[e.get("name", "")] += dur
        for cat, keys in CATEGORIES:
            if any(k in name for k in keys):
                per_cat[cat] += dur
                break
        else:
            per_cat["other"] += dur
    print(f"\n## Measured CUDA kernel time — {label} (total {total / 1e6:.2f} s GPU time)\n")
    print("| Category | GPU s | share |")
    print("|---|---|---|")
    for cat, us in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        print(f"| {cat} | {us / 1e6:.2f} | {100 * us / total:.1f}% |")
    print("\nTop 15 kernels:\n")
    print("| Kernel | GPU s | share |")
    print("|---|---|---|")
    for name, us in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:15]:
        print(f"| `{name[:90]}` | {us / 1e6:.2f} | {100 * us / total:.1f}% |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic", action="store_true")
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.analytic:
        analytic()
    if args.trace:
        parse_trace(args.trace, args.label)


if __name__ == "__main__":
    main()
