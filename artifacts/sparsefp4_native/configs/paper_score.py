"""Paper-scale scoring: dimension-routed VBench + paired PSNR/SSIM/LPIPS.

- VBench dims are scored only on the prompts belonging to that dimension
  (official routing), per arm.
- Paired metrics are computed vs P0 on every prompt.
- P4-vs-P4G paired differences get a prompt-level bootstrap CI (10k resamples).

    paper_score.py --videos /mnt/.../paper_videos/paper-s090 --shard 0 --num-shards 4 --what vbench
    paper_score.py --videos ... --what paired
    paper_score.py --videos ... --what aggregate
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
ARMS = ("P0", "P1", "P2", "P4G", "P4")


def load_frames(path: Path) -> torch.Tensor:
    arr = np.load(path)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.dim() == 5:
        t = t[0]
    return t.permute(1, 0, 2, 3).contiguous()


def prompt_index():
    from fastvideo.eval.datasets.vbench import VBenchPromptDataset
    prompts = list(VBenchPromptDataset(dimensions=DIMS))
    return {i: p for i, p in enumerate(prompts)}


def score_vbench(videos: Path, out: Path, shard: int, num_shards: int, metrics=None):
    from fastvideo.eval import create_evaluator
    idx = prompt_index()
    rows = []
    for metric in (metrics or DIMS):
        name = f"vbench.{metric}"
        try:
            ev = create_evaluator(metrics=[name], device="cuda")
        except Exception as e:  # noqa: BLE001
            print(f"UNAVAILABLE {name}: {e}", flush=True)
            continue
        for i, entry in idx.items():
            if metric not in entry["dimensions"] or i % num_shards != shard:
                continue
            for arm in ARMS:
                f = videos / arm / f"v{i:04d}.f16.npy"
                if not f.is_file():
                    continue
                try:
                    s = ev.evaluate(video=load_frames(f), fps=16.0)
                    rows.append(dict(metric=metric, idx=i, arm=arm, score=s[name].score))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(metric=metric, idx=i, arm=arm, error=repr(e)[:200]))
            with open(out.with_suffix(f".shard{shard}.jsonl"), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        del ev
        torch.cuda.empty_cache()
        print(f"scored {metric} shard {shard}", flush=True)
    print("VB_SHARD_DONE", flush=True)


def score_paired(videos: Path, out: Path, shard: int, num_shards: int):
    from fastvideo.eval import create_evaluator
    ev = create_evaluator(metrics=["common.psnr", "common.ssim", "common.lpips"], device="cuda")
    idx = prompt_index()
    rows = []
    for i in idx:
        if i % num_shards != shard:
            continue
        ref_f = videos / "P0" / f"v{i:04d}.f16.npy"
        if not ref_f.is_file():
            continue
        ref = load_frames(ref_f)
        for arm in ARMS[1:]:
            f = videos / arm / f"v{i:04d}.f16.npy"
            if not f.is_file():
                continue
            s = ev.evaluate(video=load_frames(f), reference=ref)
            rows.append(dict(idx=i, arm=arm, psnr=s["common.psnr"].score,
                             ssim=s["common.ssim"].score, lpips=s["common.lpips"].score))
        with open(out.with_suffix(f".shard{shard}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("PAIRED_SHARD_DONE", flush=True)


def aggregate(videos: Path, out_dir: Path):
    seen = set()
    rows = []
    for f in sorted(out_dir.glob("paper_vbench*.shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            key = (r.get("metric"), r.get("idx"), r.get("arm"))
            if key in seen or "score" not in r:
                continue
            seen.add(key)
            rows.append(r)
    paired = []
    for f in out_dir.glob("paper_paired.shard*.jsonl"):
        paired += [json.loads(l) for l in open(f)]
    lines = ["# Paper-scale quality (326 VBench prompts, 7 dims, paired seed 1234)", ""]
    lines += ["## VBench (dimension-routed, mean)", "",
              "| Dim | n | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for metric in DIMS:
        ns = None
        vals = []
        for arm in ARMS:
            xs = [r["score"] for r in rows if r.get("metric") == metric and r["arm"] == arm
                  and "score" in r]
            ns = len(xs)
            vals.append(f"{statistics.mean(xs):.4f}" if xs else "—")
        lines.append(f"| {metric} | {ns} | " + " | ".join(vals) + " |")
    lines += ["", "## Paired vs P0 (median)", "",
              "| Arm | n | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for arm in ARMS[1:]:
        xs = [r for r in paired if r["arm"] == arm]
        if xs:
            lines.append(f"| {arm} | {len(xs)} | {statistics.median(r['psnr'] for r in xs):.2f} "
                         f"| {statistics.median(r['ssim'] for r in xs):.4f} "
                         f"| {statistics.median(r['lpips'] for r in xs):.4f} |")
    # P4 vs P4G paired bootstrap on VBench dims
    lines += ["", "## P4 - P4G paired differences (VBench, prompt-level bootstrap 10k, 95% CI)", "",
              "| Dim | mean diff | CI low | CI high | significant |", "|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for metric in DIMS:
        a = {r["idx"]: r["score"] for r in rows if r.get("metric") == metric and r["arm"] == "P4" and "score" in r}
        b = {r["idx"]: r["score"] for r in rows if r.get("metric") == metric and r["arm"] == "P4G" and "score" in r}
        common = sorted(set(a) & set(b))
        if len(common) < 5:
            continue
        d = np.array([a[i] - b[i] for i in common])
        boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {metric} | {d.mean():+.4f} | {lo:+.4f} | {hi:+.4f} | {sig} |")
    outp = Path("artifacts/sparsefp4_native/tables/paper_scale_quality.md")
    outp.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--what", choices=["vbench", "paired", "aggregate"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("artifacts/sparsefp4_native/raw/quality"))
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--out-tag", default="paper_vbench")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.what == "vbench":
        score_vbench(args.videos, args.out_dir / args.out_tag, args.shard,
                     args.num_shards, metrics=args.metrics)
    elif args.what == "paired":
        score_paired(args.videos, args.out_dir / "paper_paired", args.shard, args.num_shards)
    else:
        aggregate(args.videos, args.out_dir)


if __name__ == "__main__":
    main()
