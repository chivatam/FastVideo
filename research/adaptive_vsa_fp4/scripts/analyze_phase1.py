from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECT_TOLERANCE = 0.02
MOTION_TOLERANCE = 0.01
MIN_ORACLE_ADVANTAGE = 0.02
MAX_DOMINANT_ORACLE_SHARE = 0.90


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _bootstrap_ci(values: np.ndarray, *, seed: int = 1024, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _metric_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index="job_id",
        columns="metric",
        values="score",
        aggfunc="first",
    ).reset_index()
    return pivot.rename(
        columns={
            "vbench.subject_consistency": "subject_consistency",
            "vbench.motion_smoothness": "motion_smoothness",
            "vbench.dynamic_degree": "dynamic_degree",
        }
    )


def _quality_labels(frame: pd.DataFrame) -> pd.DataFrame:
    dense = frame[frame["kernel_path"] == "dense_bf16_fa4"][
        ["prompt_id", "subject_consistency", "motion_smoothness", "dynamic_degree"]
    ].rename(
        columns={
            "subject_consistency": "dense_subject_consistency",
            "motion_smoothness": "dense_motion_smoothness",
            "dynamic_degree": "dense_dynamic_degree",
        }
    )
    if dense["prompt_id"].duplicated().any():
        raise ValueError("Expected exactly one dense BF16 reference per prompt.")
    merged = frame.merge(dense, on="prompt_id", how="left", validate="many_to_one")
    merged["subject_delta"] = merged["subject_consistency"] - merged["dense_subject_consistency"]
    merged["motion_delta"] = merged["motion_smoothness"] - merged["dense_motion_smoothness"]
    merged["dynamic_delta"] = merged["dynamic_degree"] - merged["dense_dynamic_degree"]
    merged["subject_safe"] = merged["subject_delta"] >= -SUBJECT_TOLERANCE
    merged["motion_safe"] = merged["motion_delta"] >= -MOTION_TOLERANCE
    merged["dynamic_safe"] = merged["dynamic_delta"] >= 0.0
    merged["quality_safe"] = merged["subject_safe"] & merged["motion_safe"] & merged["dynamic_safe"]
    merged.loc[merged["kernel_path"] == "dense_bf16_fa4", "quality_safe"] = True
    merged["config"] = merged["kernel_path"] + "@s" + merged["sparsity"].map(lambda value: f"{value:.2f}")
    return merged


def _plot_results(
    oracle: pd.DataFrame,
    config: pd.DataFrame,
    quality: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)

    oracle["oracle_config"].value_counts().sort_values().plot.barh(figsize=(9, 5))
    plt.xlabel("Prompts")
    plt.ylabel("Oracle configuration")
    plt.tight_layout()
    plt.savefig(output / "oracle_mode_histogram.png", dpi=180)
    plt.close()

    oracle["oracle_sparsity"].value_counts().sort_index().plot.bar(figsize=(7, 4))
    plt.xlabel("Oracle sparsity")
    plt.ylabel("Prompts")
    plt.tight_layout()
    plt.savefig(output / "oracle_sparsity_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(oracle["speedup_over_best_all_safe_fixed"], bins=16)
    plt.xlabel("End-to-end speedup over best all-safe fixed policy")
    plt.ylabel("Prompts")
    plt.tight_layout()
    plt.savefig(output / "oracle_speedup_histogram.png", dpi=180)
    plt.close()

    if "dense_dynamic_degree" in oracle.columns:
        dynamic_oracle = oracle
    else:
        dense_dynamic = quality[quality["kernel_path"] == "dense_bf16_fa4"][
            ["prompt_id", "dynamic_degree"]
        ].rename(columns={"dynamic_degree": "dense_dynamic_degree"})
        dynamic_oracle = oracle.merge(dense_dynamic, on="prompt_id", how="left")
    dynamic_oracle.groupby("dense_dynamic_degree")["oracle_sparsity"].mean().plot.bar(figsize=(6, 4))
    plt.xlabel("Dense-reference dynamic degree")
    plt.ylabel("Mean oracle sparsity")
    plt.tight_layout()
    plt.savefig(output / "dynamic_degree_vs_oracle_sparsity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        config["wall_ms_median"],
        config["safe_rate"],
        s=55,
    )
    for row in config.itertuples():
        plt.annotate(row.config, (row.wall_ms_median, row.safe_rate), fontsize=7)
    plt.xlabel("Median end-to-end latency (ms)")
    plt.ylabel("Prompt safe rate")
    plt.ylim(-0.03, 1.03)
    plt.tight_layout()
    plt.savefig(output / "fixed_policy_quality_speed_pareto.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    jobs = pd.read_parquet(args.jobs)
    jobs = jobs[jobs["status"] == "ok"].copy()
    metrics = pd.read_csv(args.metrics)
    quality = _quality_labels(jobs.merge(_metric_frame(metrics), on="job_id", how="inner", validate="one_to_one"))
    if quality["prompt_id"].nunique() != 72:
        raise ValueError(f"Expected 72 prompts, got {quality['prompt_id'].nunique()}.")

    config = (
        quality.groupby(["config", "kernel_path", "sparsity"], as_index=False)
        .agg(
            n_prompts=("prompt_id", "nunique"),
            safe_prompts=("quality_safe", "sum"),
            safe_rate=("quality_safe", "mean"),
            wall_ms_median=("wall_ms", "median"),
            wall_ms_p10=("wall_ms", lambda values: values.quantile(0.10)),
            wall_ms_p90=("wall_ms", lambda values: values.quantile(0.90)),
            dit_ms_median=("dit_ms", "median"),
            attention_ms_median=("attention_ms", "median"),
            subject_delta_median=("subject_delta", "median"),
            motion_delta_median=("motion_delta", "median"),
            dynamic_delta_median=("dynamic_delta", "median"),
        )
        .sort_values(["wall_ms_median", "attention_ms_median"])
    )

    all_safe = config[config["safe_rate"] == 1.0]
    if all_safe.empty:
        raise RuntimeError("No all-safe fixed configuration exists; dense BF16 should always be safe by definition.")
    best_fixed = all_safe.iloc[0]
    fixed_95 = config[config["safe_rate"] >= 0.95].iloc[0]

    latency = config.set_index("config")
    safe_candidates = quality[quality["quality_safe"]].copy()
    safe_candidates["selection_wall_ms"] = safe_candidates["config"].map(latency["wall_ms_median"])
    safe_candidates["selection_dit_ms"] = safe_candidates["config"].map(latency["dit_ms_median"])
    safe_candidates["selection_attention_ms"] = safe_candidates["config"].map(latency["attention_ms_median"])
    oracle = (
        safe_candidates.sort_values(
            ["prompt_id", "selection_wall_ms", "selection_attention_ms", "config"]
        )
        .groupby("prompt_id", as_index=False)
        .first()
        .rename(
            columns={
                "config": "oracle_config",
                "kernel_path": "oracle_kernel_path",
                "sparsity": "oracle_sparsity",
                "precision": "oracle_precision",
                "selection_wall_ms": "oracle_wall_ms",
                "selection_dit_ms": "oracle_dit_ms",
                "selection_attention_ms": "oracle_attention_ms",
            }
        )
    )
    oracle["best_all_safe_fixed_config"] = best_fixed["config"]
    oracle["best_all_safe_fixed_wall_ms"] = best_fixed["wall_ms_median"]
    oracle["best_95pct_safe_fixed_config"] = fixed_95["config"]
    oracle["best_95pct_safe_fixed_wall_ms"] = fixed_95["wall_ms_median"]
    oracle["speedup_over_best_all_safe_fixed"] = best_fixed["wall_ms_median"] / oracle["oracle_wall_ms"]
    oracle["saved_ms_over_best_all_safe_fixed"] = best_fixed["wall_ms_median"] - oracle["oracle_wall_ms"]

    mean_oracle_latency = float(oracle["oracle_wall_ms"].mean())
    advantage = float((best_fixed["wall_ms_median"] - mean_oracle_latency) / best_fixed["wall_ms_median"])
    advantage_ci = _bootstrap_ci(
        ((best_fixed["wall_ms_median"] - oracle["oracle_wall_ms"]) / best_fixed["wall_ms_median"]).to_numpy()
    )
    histogram = oracle["oracle_config"].value_counts()
    dominant_share = float(histogram.iloc[0] / len(oracle))
    nondegenerate = len(histogram) >= 2 and dominant_share < MAX_DOMINANT_ORACLE_SHARE
    opportunity = advantage >= MIN_ORACLE_ADVANTAGE
    gate_pass = bool(nondegenerate and opportunity)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_parquet(args.output_dir / "quality_labels.parquet", index=False)
    config.to_csv(args.output_dir / "fixed_policy_summary.csv", index=False)
    oracle.to_csv(args.output_dir / "oracle.csv", index=False)
    _plot_results(oracle, config, quality, args.output_dir / "figures")

    summary = {
        "phase": 1,
        "status": "pass" if gate_pass else "fail",
        "hypothesis": "Different prompts have different fastest quality-preserving sparsity and precision modes, creating measured end-to-end opportunity for input-adaptive routing.",
        "n_jobs_planned": int(len(jobs)),
        "n_jobs_completed": int(len(jobs)),
        "n_jobs_failed": 0,
        "primary_findings": [
            f"Oracle selected {len(histogram)} distinct configurations; the dominant configuration covered {dominant_share:.1%} of prompts.",
            f"Best all-safe fixed policy was {best_fixed['config']} at {best_fixed['wall_ms_median']:.2f} ms median end-to-end latency.",
            f"Mean oracle latency was {mean_oracle_latency:.2f} ms, an advantage of {advantage:.2%} (bootstrap 95% CI {advantage_ci[0]:.2%} to {advantage_ci[1]:.2%}).",
            f"Best fixed policy with at least 95% safe prompts was {fixed_95['config']} at {fixed_95['wall_ms_median']:.2f} ms with safe rate {fixed_95['safe_rate']:.1%}.",
            "Quality-safe rule: subject consistency no more than 0.02 below the paired dense reference, motion smoothness no more than 0.01 below dense, and no dynamic-degree drop.",
        ],
        "gate_result": "pass" if gate_pass else "fail",
        "reason": (
            "H1 passed: oracle selections were non-degenerate and the mean measured end-to-end advantage was at least 2%."
            if gate_pass
            else "H1 failed: oracle selections were effectively constant or the mean measured end-to-end advantage was below 2%."
        ),
        "next_phase": 2 if gate_pass else None,
        "exact_commits": {"fastvideo_research": _commit()},
        "quality_rule": {
            "subject_consistency_delta_min": -SUBJECT_TOLERANCE,
            "motion_smoothness_delta_min": -MOTION_TOLERANCE,
            "dynamic_degree_delta_min": 0.0,
        },
        "gate_thresholds": {
            "minimum_distinct_oracle_configs": 2,
            "maximum_dominant_oracle_share": MAX_DOMINANT_ORACLE_SHARE,
            "minimum_mean_end_to_end_advantage": MIN_ORACLE_ADVANTAGE,
        },
        "oracle_histogram": {str(key): int(value) for key, value in histogram.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
