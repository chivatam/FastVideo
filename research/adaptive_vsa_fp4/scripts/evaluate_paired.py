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
    parser.add_argument("--metrics", default="common.ssim,common.psnr")
    parser.add_argument("--fps", type=float, default=16.0)
    args = parser.parse_args()

    jobs = pd.read_parquet(args.jobs)
    jobs = jobs[(jobs["status"] == "ok") & jobs["video_path"].notna()].copy()
    dense = jobs[jobs["kernel_path"] == "dense_bf16_fa4"][
        ["prompt_id", "video_path"]
    ].rename(columns={"video_path": "reference_path"})
    if dense["prompt_id"].duplicated().any():
        raise ValueError("Expected one dense BF16 reference per prompt.")
    candidates = jobs[jobs["kernel_path"] != "dense_bf16_fa4"].merge(
        dense,
        on="prompt_id",
        how="inner",
        validate="many_to_one",
    )

    evaluator = create_evaluator(
        metrics=[value.strip() for value in args.metrics.split(",")],
        num_gpus=args.num_gpus,
    )
    samples = [
        {
            "video": as_video(row.video_path),
            "reference": as_video(row.reference_path),
            "video_path": row.video_path,
            "reference_path": row.reference_path,
            "prompt": row.prompt,
            "text_prompt": row.prompt,
            "fps": args.fps,
        }
        for row in candidates.itertuples()
    ]
    results = evaluator.evaluate(samples=samples)
    evaluator.shutdown()

    rows = []
    for job, metric_results in zip(candidates.itertuples(), results, strict=True):
        for metric, result in metric_results.items():
            rows.append(
                {
                    "job_id": job.job_id,
                    "prompt_id": job.prompt_id,
                    "kernel_path": job.kernel_path,
                    "sparsity": job.sparsity,
                    "metric": metric,
                    "score": result.score,
                    "details": (
                        json.dumps(result.details, default=str)
                        if result.details is not None
                        else None
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"samples={len(samples)} metric_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
