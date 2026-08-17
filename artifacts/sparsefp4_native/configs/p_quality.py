"""P-arm quality scoring: paired PSNR/SSIM/LPIPS vs P0 on float16 frames.

Reuses fastvideo.eval's integrated metrics (same modules as the repo's own
regression tests). Pairing is by (prompt, seed); the reference arm is P0.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

ARMS = ("P0", "P1", "P2", "P2G", "P3", "P4G", "P4")
REFERENCE = "P0"


def load_frames(path: Path) -> torch.Tensor:
    """[B, C, T, H, W] float16 on disk -> (T, C, H, W) float32 (evaluator contract)."""
    arr = np.load(path)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 5:
        t = t[0]
    return t.permute(1, 0, 2, 3).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from fastvideo.eval import create_evaluator
    evaluator = create_evaluator(metrics=["common.psnr", "common.ssim", "common.lpips"],
                                 device=args.device)

    prompts = sorted({p.name.split("_")[0] for p in args.video_dir.glob("p*_P0_*.f16.npy")})
    rows = []
    for prompt in prompts:
        ref_path = args.video_dir / f"{prompt}_{REFERENCE}_s{args.seed}.f16.npy"
        if not ref_path.is_file():
            continue
        ref = load_frames(ref_path)
        for arm in ARMS:
            cand_path = args.video_dir / f"{prompt}_{arm}_s{args.seed}.f16.npy"
            if not cand_path.is_file():
                rows.append(dict(prompt=prompt, arm=arm, error="missing frames"))
                continue
            cand = load_frames(cand_path)
            scores = evaluator.evaluate(video=cand, reference=ref)
            diff = (cand - ref).abs()
            rows.append(dict(prompt=prompt, arm=arm,
                             psnr_db=scores["common.psnr"].score,
                             ssim=scores["common.ssim"].score,
                             lpips=scores["common.lpips"].score,
                             mean_abs_pixel=float(diff.mean()),
                             max_abs_pixel=float(diff.max())))
            print(rows[-1], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n| Arm | n | PSNR med | SSIM med | LPIPS med |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm and "psnr_db" in r]
        if not rs:
            continue
        med = lambda k: statistics.median(r[k] for r in rs)
        print(f"| {arm} | {len(rs)} | {med('psnr_db'):.2f} | {med('ssim'):.4f} "
              f"| {med('lpips'):.4f} |")


if __name__ == "__main__":
    main()
