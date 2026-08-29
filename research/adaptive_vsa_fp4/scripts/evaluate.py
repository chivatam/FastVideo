from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fastvideo.eval import create_evaluator
from fastvideo.eval.io import as_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--metrics",
        default="vbench.subject_consistency,vbench.motion_smoothness,vbench.dynamic_degree",
    )
    parser.add_argument("--fps", type=float, default=16.0)
    args = parser.parse_args()

    jobs = pd.read_parquet(args.jobs)
    jobs = jobs[(jobs["status"] == "ok") & jobs["video_path"].notna()]
    evaluator = create_evaluator(metrics=[value.strip() for value in args.metrics.split(",")], num_gpus=args.num_gpus)
    samples = [
        {
            "video": as_video(row.video_path),
            "video_path": row.video_path,
            "prompt": row.prompt,
            "text_prompt": row.prompt,
            "fps": args.fps,
        }
        for row in jobs.itertuples()
    ]
    results = evaluator.evaluate(samples=samples)
    evaluator.shutdown()
    rows = []
    for job, metric_results in zip(jobs.itertuples(), results, strict=True):
        for metric, result in metric_results.items():
            rows.append({
                "job_id": job.job_id,
                "prompt_id": job.prompt_id,
                "kernel_path": job.kernel_path,
                "sparsity": job.sparsity,
                "metric": metric,
                "score": result.score,
                "details": json.dumps(result.details, default=str) if result.details is not None else None,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"samples={len(samples)} metric_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
