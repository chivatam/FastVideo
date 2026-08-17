"""T-matrix gate scoring: dev-triage quality for trained checkpoints.

Scores the 10-dev-prompt generations of every (T-arm, gate-step) run dir
plus the untrained baselines (T0 = pq-s090 P4, teacher = pq-s090 P4G,
dense ref = pq-s090 P0):

  - 7 scorable VBench dimensions (no-reference), per video;
  - PSNR/SSIM/LPIPS paired against P0 (dense ref) and against P4G (the
    Stage-2 distillation teacher / BF16 twin).

Usage (shard by metric across GPUs; shard -1 = pixel metrics):
  t_gate_score.py --shard 0 --num-shards 7
  t_gate_score.py --pixel
  t_gate_score.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

VIDEO_ROOT = Path("/mnt/nvme/scratch/sparsefp4_native/videos")
OUT_DIR = Path("artifacts/sparsefp4_native/raw/quality")
TABLE = Path("artifacts/sparsefp4_native/tables/t_matrix_gates.md")
DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
T_ARMS = ("T1", "T2", "T3")
GATES = (100, 250, 500)
SEED = 1234
N_PROMPTS = 10


def systems() -> list[tuple[str, Path, str]]:
    """(label, video dir, arm-tag-in-filename) for every scored system."""
    out = [("P0-dense", VIDEO_ROOT / "pq-s090", "P0"),
           ("P4G-teacher", VIDEO_ROOT / "pq-s090", "P4G"),
           ("T0-untrained-P4", VIDEO_ROOT / "pq-s090", "P4")]
    for arm in T_ARMS:
        for gate in GATES:
            d = VIDEO_ROOT / f"tgate-{arm}-c{gate}"
            if d.is_dir():
                out.append((f"{arm}-c{gate}", d, "P4"))
    return out


def load_frames(path: Path) -> torch.Tensor:
    arr = np.load(path)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 5:
        t = t[0]
    return t.permute(1, 0, 2, 3).contiguous()


def video(dirpath: Path, tag: str, p: int) -> Path:
    return dirpath / f"p{p:02d}_{tag}_s{SEED}.f16.npy"


def score_vbench(shard: int, num_shards: int) -> None:
    from fastvideo.eval import create_evaluator
    rows = []
    out = OUT_DIR / f"t_gates_vbench.shard{shard}.jsonl"
    for mi, metric in enumerate(DIMS):
        if mi % num_shards != shard:
            continue
        ev = create_evaluator(metrics=[f"vbench.{metric}"], device="cuda")
        for label, d, tag in systems():
            for p in range(N_PROMPTS):
                f = video(d, tag, p)
                if not f.is_file():
                    continue
                s = ev.evaluate(video=load_frames(f), fps=16.0)
                rows.append(dict(metric=metric, system=label, prompt=p,
                                 score=s[f"vbench.{metric}"].score))
        del ev
        torch.cuda.empty_cache()
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"scored {metric}", flush=True)
    print("T_GATE_VBENCH_DONE", flush=True)


def score_pixel() -> None:
    from fastvideo.eval import create_evaluator
    ev = create_evaluator(metrics=["common.psnr", "common.ssim", "common.lpips"],
                          device="cuda")
    rows = []
    refs = {"P0": (VIDEO_ROOT / "pq-s090", "P0"), "P4G": (VIDEO_ROOT / "pq-s090", "P4G")}
    for label, d, tag in systems():
        if label == "P0-dense":
            continue
        for p in range(N_PROMPTS):
            f = video(d, tag, p)
            if not f.is_file():
                continue
            cand = load_frames(f)
            for ref_name, (rd, rt) in refs.items():
                if label == "P4G-teacher" and ref_name == "P4G":
                    continue
                s = ev.evaluate(video=cand, reference=load_frames(video(rd, rt, p)))
                rows.append(dict(system=label, prompt=p, ref=ref_name,
                                 psnr=s["common.psnr"].score,
                                 ssim=s["common.ssim"].score,
                                 lpips=s["common.lpips"].score))
    with open(OUT_DIR / "t_gates_pixel.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("T_GATE_PIXEL_DONE", flush=True)


def aggregate() -> None:
    vb = []
    for f in sorted(OUT_DIR.glob("t_gates_vbench.shard*.jsonl")):
        vb += [json.loads(line) for line in open(f)]
    px = [json.loads(line) for line in open(OUT_DIR / "t_gates_pixel.jsonl")]
    labels = [s[0] for s in systems()]
    lines = ["# T-matrix 100/250/500-step gates — 10-prompt dev triage "
             "(mean; NOT the paper protocol)", "",
             "T0 = untrained P4; teacher = P4G (BF16 twin). Primary recovery "
             "targets: imaging_quality and dynamic_degree toward P4G.", "",
             "| System | " + " | ".join(d[:14] for d in DIMS) +
             " | PSNR->P4G | LPIPS->P4G |",
             "|---|" + "---|" * (len(DIMS) + 2)]
    for label in labels:
        cells = []
        for dim in DIMS:
            xs = [r["score"] for r in vb if r["system"] == label and r["metric"] == dim]
            cells.append(f"{statistics.mean(xs):.4f}" if xs else "—")
        ps = [r["psnr"] for r in px if r["system"] == label and r["ref"] == "P4G"]
        ls = [r["lpips"] for r in px if r["system"] == label and r["ref"] == "P4G"]
        cells.append(f"{statistics.median(ps):.2f}" if ps else "—")
        cells.append(f"{statistics.median(ls):.4f}" if ls else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=7)
    ap.add_argument("--pixel", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
    elif args.pixel:
        score_pixel()
    else:
        score_vbench(args.shard, args.num_shards)


if __name__ == "__main__":
    main()
