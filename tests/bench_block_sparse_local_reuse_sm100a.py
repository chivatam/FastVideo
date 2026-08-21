# SPDX-License-Identifier: Apache-2.0
"""Benchmark: PR#1719 baseline vs local KV-reuse (B2 union / A shared-private).

CUDA-event timing at real Wan shapes, with real Phase-1 captured VSA masks
or controlled synthetic overlap.

    python tests/bench_block_sparse_local_reuse_sm100a.py \
        --mode baseline --mode b2 --mode shared-private \
        --resolution 720p --mask real --capture-root /mnt/nvme/outputs/vsa_capture

    python tests/bench_block_sparse_local_reuse_sm100a.py \
        --mode baseline --mode shared-private --resolution 720p \
        --mask synthetic --overlap 0.0 --overlap 0.3 --overlap 0.6 --overlap 0.9 --overlap 1.0
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import torch

from fastvideo_kernel import block_sparse_attn_sm100a as vsa
from fastvideo_kernel.pair_metadata import build_pair_metadata
from fastvideo_kernel.vsa_utils import build_vsa_metadata

RES = {
    "480p": {"latent": (21, 30, 52), "heads": 12},
    "720p": {"latent": (21, 45, 80), "heads": 12},
}
HEAD_DIM = 128


def cuda_time_ms(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    times.sort()
    return times[len(times) // 2]


def synthetic_mask(nb, topk, heads, overlap, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_shared = int(round(overlap * topk))
    mask = torch.zeros(1, heads, nb, nb, dtype=torch.bool)
    for h in range(heads):
        for p in range(nb // 2):
            perm = torch.randperm(nb, generator=g)
            shared = perm[:n_shared]
            pool = perm[n_shared:]
            mask[0, h, 2 * p, torch.cat([shared, pool[:topk - n_shared]])] = True
            mask[0, h, 2 * p + 1, torch.cat([shared, pool[topk - n_shared:2 * (topk - n_shared)]])] = True
    return mask.to(device)


def real_mask(capture_root, resolution, heads, device="cuda"):
    """One real Phase-1 captured mask (first shard of the middle layer/step)."""
    pref = "720p" if resolution == "720p" else "480p"
    files = sorted(glob.glob(os.path.join(capture_root, f"{pref}_*", "cap_step001_layer15_*.pt")))
    if not files:
        raise SystemExit(f"no capture shard under {capture_root} for {pref}")
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    idx = payload["indices"][0].long()  # [H, Nq, K]
    nk = payload["num_kv_blocks"]
    mask = torch.zeros(idx.shape[0], idx.shape[1], nk, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    assert idx.shape[0] >= heads
    return mask[:heads].unsqueeze(0).to(device), payload


def baseline_meta_from_mask(mask):
    B, H, Nq, Nk = mask.shape
    counts = mask.sum(-1).to(torch.int32).reshape(-1)
    width = max(1, int(counts.max()))
    ids = torch.arange(Nk, device=mask.device, dtype=torch.int32)
    key = torch.where(mask, ids, torch.full_like(ids, Nk)).reshape(-1, Nk)
    idx = key.sort(dim=-1).values[:, :width].contiguous()
    idx = torch.where(torch.arange(width, device=mask.device) < counts[:, None], idx,
                      torch.zeros_like(idx))
    return idx, counts


def structure_stats(mask):
    m0 = mask[:, :, 0::2, :]
    m1 = mask[:, :, 1::2, :]
    inter = (m0 & m1).sum(-1).float()
    s0 = m0.sum(-1).float()
    s1 = m1.sum(-1).float()
    union = s0 + s1 - inter
    st = torch.ceil(inter / 4)
    a_iters = st + torch.ceil(torch.maximum(s0 - inter, s1 - inter) / 4)
    return {
        "median_shared": inter.median().item(),
        "median_p0": (s0 - inter).median().item(),
        "median_p1": (s1 - inter).median().item(),
        "median_union": union.median().item(),
        "kv_blocks_baseline": (s0 + s1).sum().item(),
        "kv_blocks_union": union.sum().item(),
        "iters_baseline": torch.ceil(torch.maximum(s0, s1) / 4).sum().item(),
        "iters_b2": torch.ceil(union / 4).sum().item(),
        "iters_a": a_iters.sum().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", action="append", choices=["baseline", "b2", "shared-private"],
                    default=None)
    ap.add_argument("--resolution", action="append", choices=list(RES), default=None)
    ap.add_argument("--mask", choices=["real", "synthetic"], default="real")
    ap.add_argument("--overlap", action="append", type=float, default=None)
    ap.add_argument("--capture-root", default="/mnt/nvme/outputs/vsa_capture")
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    modes = args.mode or ["baseline", "b2", "shared-private"]
    resolutions = args.resolution or ["720p"]
    overlaps = args.overlap or [0.0, 0.3, 0.6, 0.9, 1.0]

    rows = []
    for res in resolutions:
        latent = RES[res]["latent"]
        heads = args.heads or RES[res]["heads"]
        meta = build_vsa_metadata(latent, tile_size=(4, 4, 4), device="cuda")
        vbs = meta["variable_block_sizes"].to(torch.int32)
        nb = vbs.numel()
        topk = max(1, math.ceil(0.1 * nb))
        S = nb * 64
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, heads, S, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))

        if args.mask == "real":
            mask, _ = real_mask(args.capture_root, res, heads)
            cases = [("real", mask)]
        else:
            cases = [(f"ov{ov:.1f}", synthetic_mask(nb, topk, heads, ov)) for ov in overlaps]

        for label, mask in cases:
            stats = structure_stats(mask)
            idx, num = baseline_meta_from_mask(mask)
            outs = {}
            for mode in modes:
                if mode == "baseline":
                    fn = lambda: vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs, need_lse=False)
                else:
                    m = build_pair_metadata(mask, vbs,
                                            mode="union" if mode == "b2" else "shared-private")
                    fn = (lambda m=m: vsa.block_sparse_attn_sm100a_pair(
                        q, k, v, m.q2k_idx, m.q2k_num, vbs, m.pair_shared_tiles,
                        m.block_thresholds, need_lse=False))
                outs[mode] = cuda_time_ms(fn)
            base = outs.get("baseline")
            for mode in modes:
                rows.append({"resolution": res, "case": label, "mode": mode, "heads": heads,
                             "S_pad": S, "topk": topk, "ms": round(outs[mode], 4),
                             "speedup_vs_baseline": round(base / outs[mode], 4) if base else "",
                             **{k2: round(v2, 1) for k2, v2 in stats.items()}})
            line = " | ".join(f"{m}={outs[m]:.3f}ms" for m in modes)
            print(f"{res} {label}: {line}"
                  + (f"  (A vs base {base / outs.get('shared-private', base):.3f}x)"
                     if base and "shared-private" in outs else ""))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
