from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

VSA80_CONFIG = "vsa_bf16@s0.80"
NATIVE_K = 125
SELECTION_SALT = "br-vsa-census-v1"


def select_calibration_prompts(
    prompts: list[dict[str, Any]],
    labels: pd.DataFrame,
    *,
    per_stratum: int = 4,
) -> list[dict[str, Any]]:
    vsa80 = labels.loc[
        labels["config"].eq(VSA80_CONFIG),
        [
            "prompt_id",
            "quality_safe",
            "subject_delta",
            "motion_delta",
            "dynamic_delta",
        ],
    ].copy()
    if vsa80["prompt_id"].duplicated().any():
        raise ValueError("VSA80 labels contain duplicate prompt IDs")
    prompt_frame = pd.DataFrame(prompts)
    merged = prompt_frame.merge(
        vsa80,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(prompt_frame):
        raise ValueError("Some published prompts are missing VSA80 labels")
    merged["selection_hash"] = merged["prompt_id"].map(
        lambda value: hashlib.sha256(f"{SELECTION_SALT}:{value}".encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    for quality_safe, stratum in (
        (True, "vsa80_safe"),
        (False, "vsa80_unsafe"),
    ):
        candidates = (merged.loc[merged["quality_safe"].eq(quality_safe)].sort_values(["selection_hash",
                                                                                       "prompt_id"]).head(per_stratum))
        if len(candidates) != per_stratum:
            raise ValueError(f"Need {per_stratum} prompts in {stratum}, found {len(candidates)}")
        for rank, row in enumerate(
                candidates.to_dict(orient="records"),
                start=1,
        ):
            selected.append({
                "prompt_id": row["prompt_id"],
                "index": int(row["index"]),
                "prompt": row["prompt"],
                "sha256": row["sha256"],
                "dimensions": row["dimensions"],
                "n_samples": int(row["n_samples"]),
                "selection_stratum": stratum,
                "selection_rank": rank,
                "selection_hash": row["selection_hash"],
                "selection_method": (f"lowest SHA256({SELECTION_SALT}:prompt_id) "
                                     f"within {stratum}"),
                "vsa80_quality_safe": bool(row["quality_safe"]),
                "vsa80_subject_delta": float(row["subject_delta"]),
                "vsa80_motion_delta": float(row["motion_delta"]),
                "vsa80_dynamic_delta": float(row["dynamic_delta"]),
            })
    return sorted(selected, key=lambda row: row["prompt_id"])


def _spearman(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def _rank_stability(sensitivity: pd.DataFrame) -> pd.DataFrame:
    native = sensitivity.loc[sensitivity["K"].eq(NATIVE_K)].copy()
    unit_columns = ["step", "layer", "head"]
    rows: list[dict[str, Any]] = []
    prompts = sorted(native["prompt_id"].unique())
    for prompt_a, prompt_b in itertools.combinations(prompts, 2):
        left = native.loc[
            native["prompt_id"].eq(prompt_a),
            [*unit_columns, "relative_L2_error"],
        ].rename(columns={"relative_L2_error": "error_a"})
        right = native.loc[
            native["prompt_id"].eq(prompt_b),
            [*unit_columns, "relative_L2_error"],
        ].rename(columns={"relative_L2_error": "error_b"})
        paired = left.merge(
            right,
            on=unit_columns,
            validate="one_to_one",
        )
        group_specs: list[tuple[str, Any, pd.DataFrame]] = [("overall", "all", paired)]
        for group_type in ("step", "layer", "head"):
            for group_value, group in paired.groupby(group_type, sort=True):
                group_specs.append((group_type, int(group_value), group))
        for group_type, group_value, group in group_specs:
            rows.append({
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "group_type": group_type,
                "group_value": group_value,
                "n_units": len(group),
                "spearman": _spearman(
                    group["error_a"],
                    group["error_b"],
                ),
            })
    return pd.DataFrame(rows)


def _concentration(native_summary: pd.DataFrame) -> pd.DataFrame:
    ordered = native_summary.sort_values(
        "relative_L2_error_mean",
        ascending=False,
    ).reset_index(drop=True)
    total = float(ordered["relative_L2_error_mean"].sum())
    rows = []
    for percent in (5, 10, 20, 30, 100):
        count = min(
            len(ordered),
            max(1, math.ceil(len(ordered) * percent / 100.0)),
        )
        explained = float(ordered["relative_L2_error_mean"].iloc[:count].sum() / max(total, 1e-12))
        rows.append({
            "top_percent": percent,
            "unit_count": count,
            "error_fraction": explained,
        })
    return pd.DataFrame(rows)


def _classify_units(summary: pd.DataFrame) -> pd.DataFrame:
    curves = summary.pivot(
        index=["step", "layer", "head"],
        columns="K",
        values="relative_L2_error_mean",
    )
    candidate_k = sorted(int(value) for value in curves.columns)
    records = []
    for unit, row in curves.iterrows():
        low_error = float(row[candidate_k[0]])
        dense_support_error = float(row[candidate_k[-1]])
        target = dense_support_error + 0.10 * max(
            low_error - dense_support_error,
            0.0,
        )
        sufficient_k = candidate_k[-1]
        for exact_k in candidate_k:
            if float(row[exact_k]) <= target:
                sufficient_k = exact_k
                break
        if sufficient_k == 32:
            sensitivity_class = "very tolerant"
        elif sufficient_k == 64:
            sensitivity_class = "tolerant"
        elif sufficient_k in {96, 125}:
            sensitivity_class = "normal"
        elif sufficient_k in {192, 250}:
            sensitivity_class = "sensitive"
        else:
            sensitivity_class = "very sensitive"
        records.append({
            "step": unit[0],
            "layer": unit[1],
            "head": unit[2],
            "curve_sufficient_K": sufficient_k,
            "sensitivity_class": sensitivity_class,
            "classification_target_error": target,
        })
    return pd.DataFrame(records)


def _plot_sensitivity_heatmap(
    native_summary: pd.DataFrame,
    output: Path,
) -> None:
    steps = sorted(native_summary["step"].unique())
    maximum = float(native_summary["relative_L2_error_mean"].max())
    with PdfPages(output) as pdf:
        figure, axes = plt.subplots(
            len(steps),
            1,
            figsize=(12, 3.2 * len(steps)),
            constrained_layout=True,
        )
        if len(steps) == 1:
            axes = [axes]
        image = None
        for axis, step in zip(axes, steps, strict=True):
            matrix = (native_summary.loc[native_summary["step"].eq(step)].pivot(
                index="head", columns="layer", values="relative_L2_error_mean").sort_index())
            image = axis.imshow(
                matrix.to_numpy(),
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
            )
            axis.set_title(f"Step {step}: mean dense-relative error at K=125")
            axis.set_xlabel("Transformer layer")
            axis.set_ylabel("Head")
            axis.set_xticks(np.arange(matrix.shape[1])[::3])
            axis.set_xticklabels(matrix.columns.to_numpy()[::3])
            axis.set_yticks(np.arange(matrix.shape[0]))
            axis.set_yticklabels(matrix.index.to_numpy())
        assert image is not None
        figure.colorbar(
            image,
            ax=axes,
            label="Mean relative L2 error",
            shrink=0.85,
        )
        pdf.savefig(figure)
        plt.close(figure)


def _plot_concentration(
    native_summary: pd.DataFrame,
    output: Path,
) -> None:
    ordered = native_summary.sort_values(
        "relative_L2_error_mean",
        ascending=False,
    )
    cumulative = ordered["relative_L2_error_mean"].cumsum()
    cumulative = cumulative / cumulative.iloc[-1]
    x = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100.0
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        axis.plot(x, cumulative, linewidth=2.0, label="Observed K=125 error")
        axis.plot([0, 100], [0, 1], linestyle="--", color="gray", label="Uniform error")
        for percent in (5, 10, 20, 30):
            index = min(
                len(cumulative) - 1,
                max(0,
                    math.ceil(len(cumulative) * percent / 100.0) - 1),
            )
            axis.scatter(
                [percent],
                [float(cumulative.iloc[index])],
                zorder=3,
            )
            axis.annotate(
                f"{float(cumulative.iloc[index]):.1%}",
                (percent, float(cumulative.iloc[index])),
                xytext=(5, 5),
                textcoords="offset points",
            )
        axis.set(
            xlabel="Most-sensitive units included (%)",
            ylabel="Cumulative fraction of K=125 error",
            xlim=(0, 100),
            ylim=(0, 1.02),
            title="BR-VSA sensitivity concentration",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        pdf.savefig(figure)
        plt.close(figure)


def analyze(
    *,
    jobs_path: Path,
    stats_path: Path,
    prompt_ids_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    calibration = output_root / "calibration"
    figures = output_root / "figures"
    calibration.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    jobs = pd.read_parquet(jobs_path)
    stats = pd.read_parquet(stats_path)
    stats = stats.loc[stats["event_type"].eq("br_vsa_sensitivity")].copy()
    sensitivity = stats.merge(
        jobs[["job_id", "prompt_id", "prompt", "seed"]],
        on="job_id",
        validate="many_to_one",
    )
    timestep_order = {value: index for index, value in enumerate(sorted(sensitivity["timestep"].unique()))}
    sensitivity["step"] = sensitivity["timestep"].map(timestep_order)
    candidate_k = sorted(int(value) for value in sensitivity["K"].unique())
    prompt_ids = sorted(sensitivity["prompt_id"].unique())
    expected_rows = (len(prompt_ids) * sensitivity[["step", "layer", "head"]].drop_duplicates().shape[0] *
                     len(candidate_k))
    if len(sensitivity) != expected_rows:
        raise ValueError(f"Incomplete census: expected {expected_rows} rows, found {len(sensitivity)}")
    sensitivity.sort_values(
        ["prompt_id", "step", "layer", "head", "K"],
        inplace=True,
    )
    sensitivity.to_parquet(
        calibration / "sensitivity.parquet",
        index=False,
    )

    unit_columns = ["step", "layer", "head", "K"]
    summary = sensitivity.groupby(unit_columns, as_index=False).agg(
        relative_L2_error_mean=("relative_L2_error", "mean"),
        relative_L2_error_median=("relative_L2_error", "median"),
        relative_L2_error_p90=(
            "relative_L2_error",
            lambda values: values.quantile(0.9),
        ),
        relative_L2_error_std=("relative_L2_error", "std"),
        cosine_error_mean=("cosine_error", "mean"),
        cosine_error_median=("cosine_error", "median"),
        cosine_error_p90=("cosine_error", lambda values: values.quantile(0.9)),
        cosine_error_std=("cosine_error", "std"),
        max_absolute_error_mean=("max_absolute_error", "mean"),
        output_norm_mean=("output_norm", "mean"),
        dense_output_norm_mean=("dense_output_norm", "mean"),
    )
    classification = _classify_units(summary)
    summary = summary.merge(
        classification,
        on=["step", "layer", "head"],
        validate="many_to_one",
    )
    summary["previous_K"] = summary.groupby(["step", "layer", "head"])["K"].shift(1)
    summary["previous_error"] = summary.groupby(["step", "layer", "head"])["relative_L2_error_mean"].shift(1)
    summary["marginal_gain_per_block"] = (summary["previous_error"] -
                                          summary["relative_L2_error_mean"]) / (summary["K"] - summary["previous_K"])
    summary.to_csv(
        calibration / "sensitivity_summary.csv",
        index=False,
    )

    native_summary = summary.loc[summary["K"].eq(NATIVE_K)].copy()
    concentration = _concentration(native_summary)
    concentration.to_csv(
        calibration / "error_concentration.csv",
        index=False,
    )
    stability = _rank_stability(sensitivity)
    stability.to_csv(
        calibration / "rank_stability.csv",
        index=False,
    )

    top20 = float(concentration.loc[
        concentration["top_percent"].eq(20),
        "error_fraction",
    ].iloc[0])
    overall_stability = stability.loc[
        stability["group_type"].eq("overall"),
        "spearman",
    ]
    median_spearman = float(overall_stability.median())
    gate_pass = bool(top20 >= 0.50 and median_spearman >= 0.50)
    gate = {
        "candidate_K": candidate_k,
        "native_K": NATIVE_K,
        "prompt_count": len(prompt_ids),
        "unit_count": int(native_summary.shape[0]),
        "top20_error_fraction": top20,
        "median_pairwise_spearman": median_spearman,
        "heterogeneity_threshold": 0.50,
        "stability_threshold": 0.50,
        "stage0_pass": gate_pass,
        "decision": "GO" if gate_pass else "NO-GO",
        "timestep_to_step": {
            str(key): value
            for key, value in timestep_order.items()
        },
    }
    (calibration / "stage0_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    _plot_sensitivity_heatmap(
        native_summary,
        figures / "sensitivity_heatmap.pdf",
    )
    _plot_concentration(
        native_summary,
        figures / "error_concentration.pdf",
    )

    recorded_prompt_ids = {row["prompt_id"] for row in json.loads(prompt_ids_path.read_text())}
    if recorded_prompt_ids != set(prompt_ids):
        raise ValueError("Recorded calibration prompts and census prompts disagree")
    concentration_lines = "\n".join(f"- Top {int(row.top_percent)}%: {row.error_fraction:.2%}"
                                    for row in concentration.itertuples() if row.top_percent < 100)
    class_counts = (classification["sensitivity_class"].value_counts().reindex(
        [
            "very tolerant",
            "tolerant",
            "normal",
            "sensitive",
            "very sensitive",
        ],
        fill_value=0,
    ))
    class_lines = "\n".join(f"- {name}: {int(count)}" for name, count in class_counts.items())
    report = f"""# BR-VSA Stage 0 Sensitivity Census

## Protocol

- Prompts: {len(prompt_ids)} (4 frozen VSA80-safe and 4 frozen VSA80-unsafe)
- Candidate K: {candidate_k}
- Units: {len(native_summary)} `(step, layer, head)` units
- Primary metric: dense-relative output L2 error
- Sparse output scope: native exact fine attention plus the checkpoint coarse residual
- Replay: streamed from the same captured Q/K/V call; no per-K video regeneration

## Error concentration at K=125

{concentration_lines}

## Prompt rank stability

- Median pairwise overall Spearman: {median_spearman:.4f}
- Required diagnostic guideline: >= 0.5000

## Curve-based unit classes

The smallest K retaining 90% of the observed K32-to-K624 error reduction defines the class.

{class_lines}

## Stage 0 gate

- Heterogeneity: top 20% explain {top20:.2%} (guideline >= 50%)
- Stability: median Spearman {median_spearman:.4f} (guideline >= 0.5)
- Decision: **{"GO" if gate_pass else "NO-GO"}**

Selected prompt records are frozen in `prompt_ids.json`. The census contains no
quality-label tuning beyond the prescribed safe/unsafe stratification.
"""
    (calibration / "REPORT.md").write_text(report)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--prompts", type=Path, required=True)
    prepare_parser.add_argument("--labels", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--jobs", type=Path, required=True)
    analyze_parser.add_argument("--stats", type=Path, required=True)
    analyze_parser.add_argument("--prompt-ids", type=Path, required=True)
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        selected = select_calibration_prompts(
            json.loads(args.prompts.read_text()),
            pd.read_parquet(args.labels),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "selected": len(selected),
                    "prompt_ids": [row["prompt_id"] for row in selected],
                },
                indent=2,
            ))
    else:
        gate = analyze(
            jobs_path=args.jobs,
            stats_path=args.stats,
            prompt_ids_path=args.prompt_ids,
            output_root=args.output_root,
        )
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
