"""One kernel invocation per run, for ncu profiling (Phase 3D).

    ncu --set basic python artifacts/vsa_local_reuse_phase3/profile_one.py --mode baseline
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import torch

sys.path.insert(0, "tests")

from bench_block_sparse_local_reuse_sm100a import RES, baseline_meta_from_mask, real_mask

from fastvideo_kernel import block_sparse_attn_sm100a as vsa
from fastvideo_kernel.pair_metadata import build_pair_metadata
from fastvideo_kernel.vsa_utils import build_vsa_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "b2", "shared-private"], required=True)
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--capture-root", default="/mnt/nvme/outputs/vsa_capture")
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    latent = RES[args.resolution]["latent"]
    heads = RES[args.resolution]["heads"]
    meta = build_vsa_metadata(latent, tile_size=(4, 4, 4), device="cuda")
    vbs = meta["variable_block_sizes"].to(torch.int32)
    nb = vbs.numel()
    S = nb * 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, S, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mask, _ = real_mask(args.capture_root, args.resolution, heads)

    fn: Callable[[], object]
    if args.mode == "baseline":
        idx, num = baseline_meta_from_mask(mask)
        fn = lambda: vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs, need_lse=False)
    else:
        m = build_pair_metadata(mask, vbs, mode="union" if args.mode == "b2" else "shared-private")
        fn = lambda: vsa.block_sparse_attn_sm100a_pair(
            q, k, v, m.q2k_idx, m.q2k_num, vbs, m.pair_shared_tiles, m.block_thresholds, need_lse=False)
    fn()  # warm/JIT outside the profiled region
    torch.cuda.synchronize()
    torch.cuda.profiler.start()  # type: ignore[no-untyped-call]
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
