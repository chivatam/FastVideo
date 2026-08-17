"""Micro-profile of the F1 per-cell measurement cost, so the sweep can be sized.

The F1 smoke run spent ~27 s per measured cell, which would put a 30-layer x
6-timestep x 2-CFG diagnostic at several hours per prompt. Before either accepting
that or optimizing blindly, this attributes the cost to the individual stages on
tensors of Wan's real shape.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_profile.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastvideo.attention.backends.routing_probe_attn import quantize_router_input  # noqa: E402
from fastvideo.attention.backends.sparsefp4_mask_metrics import (  # noqa: E402
    boundary_diagnostics, mask_comparison, spearman_rho)
from fastvideo.attention.backends.sparsefp4_numerics import (  # noqa: E402
    expand_query_axis, random_matched_mask, raster_geometry, sparse_bf16, to_block_layout, topk_block_mask)
from fastvideo.attention.backends.sparsefp4_scorer_precision import (  # noqa: E402
    pool_blocks_precision, quantize_pooled_fp8_e4m3, quantize_pooled_nvfp4, score_blocks_fp8_native,
    score_blocks_precision)

SEQ_LEN = 32760
HEADS = 12
DIM = 128


def timed(label: str, fn, repeats: int = 3, warmup: int = 2) -> tuple[str, float]:
    """Time ``fn`` after warmup.

    Warmup is not a formality here: the first ``block_sparse_attn`` call pays
    ~1.8 s of Triton JIT compilation, and an unwarmed measurement of it led to a
    per-cell estimate ~20x too high. The same applies to the NVFP4 quantization
    path.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / repeats
    print(f"  {label:52} {elapsed * 1000:9.2f} ms")
    return label, elapsed


def main() -> int:
    device = torch.device("cuda")
    torch.manual_seed(0)
    scale = DIM**-0.5
    query = torch.randn((1, SEQ_LEN, HEADS, DIM), device=device, dtype=torch.bfloat16)
    key = torch.randn((1, SEQ_LEN, HEADS, DIM), device=device, dtype=torch.bfloat16)
    value = torch.randn((1, SEQ_LEN, HEADS, DIM), device=device, dtype=torch.bfloat16)
    geometry = raster_geometry(SEQ_LEN, 128, device)
    laid_q = to_block_layout(query, geometry)
    laid_k = to_block_layout(key, geometry)
    laid_v = to_block_layout(value, geometry)
    print(f"shape seq={SEQ_LEN} heads={HEADS} dim={DIM} n_q={geometry.n_q_blocks} n_k={geometry.n_k_blocks}\n")

    timings: dict[str, float] = {}

    def add(label: str, fn, repeats: int = 3) -> None:
        name, elapsed = timed(label, fn, repeats)
        timings[name] = elapsed

    print("representation quantization (once per cell, per precision):")
    add("quantize_router_input(nvfp4) q+k", lambda:
        (quantize_router_input(query, "nvfp4"), quantize_router_input(key, "nvfp4")))

    print("\npooling (per arm):")
    add("pool fp64/native", lambda: pool_blocks_precision(laid_q, geometry.query_block_sizes, 128, "fp64", "native"))
    add("pool fp32/native", lambda: pool_blocks_precision(laid_q, geometry.query_block_sizes, 128, "fp32", "native"))
    add("pool bf16/native", lambda: pool_blocks_precision(laid_q, geometry.query_block_sizes, 128, "bf16", "native"))
    add("pool bf16/low (sequential, 128 steps)",
        lambda: pool_blocks_precision(laid_q, geometry.query_block_sizes, 128, "bf16", "low"))

    pooled_q, _ = pool_blocks_precision(laid_q, geometry.query_block_sizes, 128, "fp32", "native")
    pooled_k, _ = pool_blocks_precision(laid_k, geometry.key_block_sizes, 64, "fp32", "native")

    print("\nscoring (per arm):")
    add("score fp64/native", lambda: score_blocks_precision(pooled_q, pooled_k, scale, "fp64", "native"))
    add("score fp32/native", lambda: score_blocks_precision(pooled_q, pooled_k, scale, "fp32", "native"))
    add("score bf16/native", lambda: score_blocks_precision(pooled_q, pooled_k, scale, "bf16", "native"))
    add("score bf16/low (rank-1 loop, 128 steps)",
        lambda: score_blocks_precision(pooled_q, pooled_k, scale, "bf16", "low"))
    add(
        "quantize_pooled_fp8 + native fp8 gemm", lambda: score_blocks_fp8_native(
            quantize_pooled_fp8_e4m3(pooled_q)[0],
            quantize_pooled_fp8_e4m3(pooled_k)[0], scale))
    add("quantize_pooled_nvfp4 q+k", lambda: (quantize_pooled_nvfp4(pooled_q), quantize_pooled_nvfp4(pooled_k)))

    scores, _ = score_blocks_precision(pooled_q, pooled_k, scale, "fp64", "native")
    k = max(1, int(round(0.10 * geometry.n_k_blocks)))
    mask = topk_block_mask(scores, k)
    scores2, _ = score_blocks_precision(pooled_q, pooled_k, scale, "bf16", "native")
    mask2 = topk_block_mask(scores2, k)

    print("\nexecution (per arm, and per matched-random control):")
    add("sparse_bf16 kernel (full 12-head)",
        lambda: sparse_bf16(laid_q, laid_k, laid_v,
                            expand_query_axis(mask, 2).unsqueeze(0), geometry.key_block_sizes))

    print("\nmetrics (per head):")
    add("mask_comparison", lambda: mask_comparison(mask2[0], mask[0]))
    add("boundary_diagnostics", lambda: boundary_diagnostics(scores[0], mask2[0], mask[0], k))
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    add("random_matched_mask (all heads)", lambda: random_matched_mask(mask, mask2, generator))
    add("spearman_rho stride=32 (8 q-blocks)", lambda: spearman_rho(scores2[0][::32], scores[0][::32]))
    add("spearman_rho stride=128 (2 q-blocks)", lambda: spearman_rho(scores2[0][::128], scores[0][::128]))

    n_arms = 12
    per_cell = (timings["quantize_router_input(nvfp4) q+k"] + n_arms * 2 * timings["pool fp32/native"] +
                2 * timings["pool bf16/low (sequential, 128 steps)"] * 2 + n_arms * timings["score fp64/native"] +
                2 * timings["score bf16/low (rank-1 loop, 128 steps)"] +
                (2 * n_arms - 1) * timings["sparse_bf16 kernel (full 12-head)"] + HEADS *
                (timings["mask_comparison"] + timings["boundary_diagnostics"]) * n_arms +
                timings["spearman_rho stride=32 (8 q-blocks)"] * n_arms)
    print(f"\nestimated cost per measured cell (1 sparsity, {n_arms} arms): {per_cell:.2f} s")
    for cells, label in ((30 * 6 * 2, "diagnostic: 30 layers x 6 steps x 2 CFG"), ):
        print(f"  {label}: {cells} cells -> {cells * per_cell / 3600:.2f} h/prompt/sparsity")

    payload = {
        "shape": {
            "seq_len": SEQ_LEN,
            "heads": HEADS,
            "dim": DIM,
            "n_q_blocks": geometry.n_q_blocks,
            "n_k_blocks": geometry.n_k_blocks
        },
        "timings_ms": {
            label: value * 1000
            for label, value in timings.items()
        },
        "estimated_seconds_per_cell": per_cell,
    }
    out = REPO_ROOT / "artifacts/sparsefp4_followup/raw/f1_profile.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
