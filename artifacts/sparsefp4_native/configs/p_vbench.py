"""VBench no-reference dimensions for the P-arm videos (study-1 protocol).

Scores each pXX_ARM_s1234.f16.npy with the repo-integrated VBench adapter,
one metric at a time so one unavailable dimension cannot take down the rest.
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from pathlib import Path

import numpy as np
import torch

ARMS = ("P0", "P1", "P2", "P2G", "P3", "P4G", "P4")
DEFAULT_METRICS = (
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.dynamic_degree",
    "vbench.imaging_quality",
    "vbench.aesthetic_quality",
)


def load_frames(path: Path) -> torch.Tensor:
    arr = np.load(path)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 5:
        t = t[0]
    return t.permute(1, 0, 2, 3).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from fastvideo.eval import create_evaluator

    files = sorted(args.video_dir.glob(f"p*_s{args.seed}.f16.npy"))
    cells = []
    for f in files:
        parts = f.name.split("_")
        arm = parts[1]
        if arm in ARMS:
            cells.append((parts[0], arm, f))

    rows: list[dict] = []
    unavailable: dict[str, str] = {}
    for metric in args.metrics:
        try:
            evaluator = create_evaluator(metrics=[metric], device=args.device)
        except Exception as err:  # noqa: BLE001
            unavailable[metric] = f"{type(err).__name__}: {err}"
            print(f"UNAVAILABLE {metric}: {err}", flush=True)
            continue
        failed = False
        for prompt, arm, path in cells:
            try:
                scores = evaluator.evaluate(video=load_frames(path), fps=16.0)
            except Exception as err:  # noqa: BLE001
                unavailable[metric] = f"{type(err).__name__}: {err}"
                print(f"UNAVAILABLE {metric} on {prompt}/{arm}: "
                      f"{traceback.format_exc()[-500:]}", flush=True)
                failed = True
                break
            rows.append(dict(metric=metric, prompt=prompt, arm=arm,
                             score=scores[metric].score))
        del evaluator
        torch.cuda.empty_cache()
        if not failed:
            print(f"scored {metric}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if unavailable:
            f.write(json.dumps({"unavailable": unavailable}) + "\n")

    print("\n| Metric | " + " | ".join(ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for metric in args.metrics:
        vals = []
        for arm in ARMS:
            rs = [r["score"] for r in rows if r["metric"] == metric and r["arm"] == arm]
            vals.append(f"{statistics.mean(rs):.4f}" if rs else "—")
        print(f"| {metric.split('.')[1]} | " + " | ".join(vals) + " |")
    print("VBENCH_DONE")


if __name__ == "__main__":
    main()
