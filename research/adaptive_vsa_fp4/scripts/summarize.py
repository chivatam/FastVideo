from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    args = parser.parse_args()

    jobs = pd.read_parquet(args.jobs)
    ok = jobs[jobs["status"] == "ok"]
    findings = []
    if not ok.empty:
        medians = ok.groupby(["kernel_path", "sparsity"], dropna=False)["wall_ms"].median().sort_values()
        findings.append(f"Fastest median configuration: {medians.index[0]} at {medians.iloc[0]:.1f} ms")
    summary = {
        "phase": args.phase,
        "status": "pass" if len(ok) == len(jobs) and len(jobs) else "partial",
        "hypothesis": "Phase-specific; see research protocol.",
        "n_jobs_planned": int(len(jobs)),
        "n_jobs_completed": int(len(ok)),
        "n_jobs_failed": int((jobs["status"] == "failed").sum()),
        "primary_findings": findings,
        "gate_result": "pending",
        "reason": "Automated summary generated; gate analysis follows.",
        "next_phase": args.phase + 1,
        "exact_commits": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
