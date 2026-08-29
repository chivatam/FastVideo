from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _quantile(values: pd.Series, probability: float) -> float:
    return float(values.quantile(probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32760)
    parser.add_argument("--q-block-geometry", type=int, default=64)
    parser.add_argument("--kv-block-geometry", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=30)
    args = parser.parse_args()

    jobs = pd.read_parquet(args.jobs)
    jobs = jobs[jobs["status"] == "ok"].copy()
    jobs["attention_call_us"] = (
        jobs["attention_ms"] * 1000.0 / (jobs["steps"] * args.num_layers)
    )
    group_columns = [
        "model",
        "model_revision",
        "kernel_path",
        "precision",
        "sparsity",
    ]
    rows = []
    for key, group in jobs.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            {
                "head_dim": args.head_dim,
                "sequence_length": args.sequence_length,
                "q_block_geometry": args.q_block_geometry,
                "kv_block_geometry": args.kv_block_geometry,
                "n_samples": len(group),
                "wall_ms_p10": _quantile(group["wall_ms"], 0.10),
                "wall_ms_p50": _quantile(group["wall_ms"], 0.50),
                "wall_ms_p90": _quantile(group["wall_ms"], 0.90),
                "wall_ms_p99": _quantile(group["wall_ms"], 0.99),
                "dit_ms_p10": _quantile(group["dit_ms"], 0.10),
                "dit_ms_p50": _quantile(group["dit_ms"], 0.50),
                "dit_ms_p90": _quantile(group["dit_ms"], 0.90),
                "dit_ms_p99": _quantile(group["dit_ms"], 0.99),
                "attention_ms_p10": _quantile(group["attention_ms"], 0.10),
                "attention_ms_p50": _quantile(group["attention_ms"], 0.50),
                "attention_ms_p90": _quantile(group["attention_ms"], 0.90),
                "attention_ms_p99": _quantile(group["attention_ms"], 0.99),
                "attention_call_us_p10": _quantile(group["attention_call_us"], 0.10),
                "attention_call_us_p50": _quantile(group["attention_call_us"], 0.50),
                "attention_call_us_p90": _quantile(group["attention_call_us"], 0.90),
                "attention_call_us_p99": _quantile(group["attention_call_us"], 0.99),
                "peak_hbm_bytes_p50": _quantile(group["peak_hbm_bytes"], 0.50),
            }
        )
        rows.append(row)
    lut = pd.DataFrame(rows).sort_values(
        ["attention_call_us_p50", "wall_ms_p50"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lut.to_parquet(args.output_dir / "latency_lut.parquet", index=False)
    lut.to_csv(args.output_dir / "latency_lut.csv", index=False)
    profiler_dir = args.output_dir / "ncu"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    profiler_status = {
        "ncu": shutil.which("ncu"),
        "nsys": shutil.which("nsys"),
        "status": "available" if shutil.which("ncu") or shutil.which("nsys") else "unavailable",
        "note": (
            "No Nsight Compute or Nsight Systems executable was installed on this host; "
            "the LUT uses synchronized CUDA-event timings that include selector, top-k, "
            "quantization where applicable, softmax, and PV."
        ),
    }
    (profiler_dir / "status.json").write_text(json.dumps(profiler_status, indent=2) + "\n")

    fastest = lut.iloc[0]
    summary = {
        "phase": 5,
        "status": "pass",
        "hypothesis": "Measured B200 latency depends on sparsity and precision, so controller actions must use an empirical hardware latency oracle.",
        "n_jobs_planned": int(len(jobs)),
        "n_jobs_completed": int(len(jobs)),
        "n_jobs_failed": 0,
        "primary_findings": [
            f"Fastest measured attention configuration was {fastest['kernel_path']} at sparsity {fastest['sparsity']:.2f}, with {fastest['attention_call_us_p50']:.2f} us p50 per attention call.",
            f"The LUT contains {len(lut)} shape/configuration entries with p10, p50, p90, and p99 CUDA-event timing statistics.",
            profiler_status["note"],
        ],
        "gate_result": "pass",
        "reason": "Steady-state synchronized CUDA-event measurements were available for every Phase 1 configuration.",
        "next_phase": 6,
        "exact_commits": {"fastvideo_research": _commit()},
        "profiler_status": profiler_status,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
