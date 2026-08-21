"""Phase-4 diagnostics: density/sequence sweeps + contiguous-index counterfactual.

Baseline PR#1719 kernel only. CUDA-event latency plus (optionally, via
--ncu-binary) per-point ncu metrics captured by re-execing profile points.

    python artifacts/vsa_latency_phase4/run_sweeps.py --out artifacts/vsa_latency_phase4/results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, "tests")

from bench_block_sparse_local_reuse_sm100a import baseline_meta_from_mask, cuda_time_ms, real_mask, synthetic_mask

from fastvideo_kernel import block_sparse_attn_sm100a as vsa
from fastvideo_kernel.vsa_utils import build_vsa_metadata

SHAPES = {
    "480p": (21, 30, 52),
    "720p": (21, 45, 80),
    "1080p_synth": (21, 68, 120),  # larger synthetic latent
}
HEADS = 12
HEAD_DIM = 128


def contiguous_mask(nb: int, topk: int, heads: int, device="cuda") -> torch.Tensor:
    """Every row selects K CONTIGUOUS ascending blocks (regular addresses)."""
    mask = torch.zeros(1, heads, nb, nb, dtype=torch.bool)
    for qb in range(nb):
        start = min(max(0, qb - topk // 2), nb - topk)
        mask[0, 0, qb, start:start + topk] = True
    mask[:, :, :, :] = mask[:, :1, :, :]
    return mask.to(device)


def bench_point(latent: tuple[int, int, int], topk: int, mask_kind: str, capture_root: str, seed: int = 0) -> dict:
    meta = build_vsa_metadata(latent, tile_size=(4, 4, 4), device="cuda")
    vbs = meta["variable_block_sizes"].to(torch.int32)
    nb = vbs.numel()
    S = nb * 64
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, HEADS, S, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    if mask_kind == "real":
        res = "720p" if latent == SHAPES["720p"] else "480p"
        mask, _ = real_mask(capture_root, res, HEADS)
    elif mask_kind == "contiguous":
        mask = contiguous_mask(nb, topk, HEADS)
    else:  # random VSA-like
        mask = synthetic_mask(nb, topk, HEADS, overlap=0.7, seed=seed)
    idx, num = baseline_meta_from_mask(mask)
    ms = cuda_time_ms(lambda: vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs, need_lse=False))
    return {"S_pad": S, "num_blocks": nb, "topk": topk, "ms": round(ms, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-root", default="/mnt/nvme/outputs/vsa_capture")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Part 11: density sweep at 720p and 480p
    rows = []
    for shape_name in ("480p", "720p", "1080p_synth"):
        latent = SHAPES[shape_name]
        nb = math.prod((math.ceil(latent[0] / 4), math.ceil(latent[1] / 4), math.ceil(latent[2] / 4)))
        for density in (0.05, 0.10, 0.25, 0.50):
            topk = max(1, math.ceil(density * nb))
            r = bench_point(latent, topk, "random", args.capture_root)
            r.update({"shape": shape_name, "density": density})
            rows.append(r)
            print(f"density {shape_name} d={density:.2f} K={topk}: {r['ms']} ms")
    with open(os.path.join(args.out, "density_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Part 6, Experiment C: real vs contiguous indices at deployed density
    rows_c = []
    for shape_name in ("480p", "720p"):
        latent = SHAPES[shape_name]
        nb = math.prod((math.ceil(latent[0] / 4), math.ceil(latent[1] / 4), math.ceil(latent[2] / 4)))
        topk = max(1, math.ceil(0.10 * nb))
        for kind in ("real", "random", "contiguous"):
            r = bench_point(latent, topk, kind, args.capture_root)
            r.update({"shape": shape_name, "mask": kind})
            rows_c.append(r)
            print(f"expC {shape_name} {kind}: {r['ms']} ms")
    with open(os.path.join(args.out, "diagnostic_microexperiments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_c[0]))
        w.writeheader()
        w.writerows(rows_c)


if __name__ == "__main__":
    main()
