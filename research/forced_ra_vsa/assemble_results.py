from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "subject_consistency",
    "motion_smoothness",
    "dynamic_degree",
]
METRIC_RENAMES = {
    "vbench.subject_consistency": "subject_consistency",
    "vbench.motion_smoothness": "motion_smoothness",
    "vbench.dynamic_degree": "dynamic_degree",
}
BASE_CONFIGS = {
    "Dense BF16": "dense_bf16_fa4@s0.00",
    "VSA80": "vsa_bf16@s0.80",
    "VSA60": "vsa_bf16@s0.60",
    "VSA40": "vsa_bf16@s0.40",
}
FORCED_METHODS = {
    0.875: "Forced RA-VSA 12.5%",
    0.750: "Forced RA-VSA 25%",
}
TARGET_REPLACEMENT = {
    0.875: 0.125,
    0.750: 0.250,
}
METHOD_ORDER = [
    "Dense BF16",
    "VSA80",
    "Forced RA-VSA 12.5%",
    "Forced RA-VSA 25%",
    "Adaptive-K",
    "VSA60",
    "VSA40",
]
VSA80_WALL_MS = 9356.053424
DENSE_WALL_MS = 9586.638779
DECISION_STOP = "DECISION: STOP — FIXED-BUDGET BLOCK REALLOCATION DOES NOT RECOVER ENOUGH QUALITY"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
    ).strip()


def _metric_wide(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .pivot(index="job_id", columns="metric", values="score")
        .reset_index()
        .rename(columns=METRIC_RENAMES)
    )


def _add_quality_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["subject_delta"] = result["subject_consistency"] - result["dense_subject_consistency"]
    result["motion_delta"] = result["motion_smoothness"] - result["dense_motion_smoothness"]
    result["dynamic_delta"] = result["dynamic_degree"] - result["dense_dynamic_degree"]
    result["subject_safe"] = result["subject_delta"] >= -0.02
    result["motion_safe"] = result["motion_delta"] >= -0.01
    result["dynamic_safe"] = result["dynamic_delta"] >= 0.0
    result["quality_safe"] = result["subject_safe"] & result["motion_safe"] & result["dynamic_safe"]
    return result


def _load_base_quality(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet")
    vsa80 = labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    rows = []
    for method, config in BASE_CONFIGS.items():
        group = labels.loc[labels["config"].eq(config)].copy()
        group["method"] = method
        group = group.merge(vsa80, on="prompt_id", validate="one_to_one")
        group["original_failure"] = ~group["vsa80_safe"]
        group["repaired"] = group["original_failure"] & group["quality_safe"]
        group["new_failure"] = ~group["original_failure"] & ~group["quality_safe"]
        group["replacement_target"] = 0.0 if method == "VSA80" else np.nan
        group["replacement_actual"] = 0.0 if method == "VSA80" else np.nan
        rows.append(group)
    return pd.concat(rows, ignore_index=True), labels


def _load_adaptive(repo: Path, vsa80: pd.DataFrame) -> pd.DataFrame:
    adaptive = pd.read_csv(repo / "artifacts/adaptive_vsa_deadline_final/development_72/results.csv")
    adaptive = adaptive.loc[adaptive["method"].eq("Adaptive VSA")].copy()
    adaptive["method"] = "Adaptive-K"
    adaptive = adaptive.merge(
        vsa80[["prompt_id", "vsa80_safe"]],
        on="prompt_id",
        validate="one_to_one",
    )
    adaptive["subject_safe"] = adaptive["subject_delta"] >= -0.02
    adaptive["motion_safe"] = adaptive["motion_delta"] >= -0.01
    adaptive["dynamic_safe"] = adaptive["dynamic_delta"] >= 0.0
    adaptive["quality_safe"] = adaptive["subject_safe"] & adaptive["motion_safe"] & adaptive["dynamic_safe"]
    adaptive["original_failure"] = ~adaptive["vsa80_safe"]
    adaptive["repaired"] = adaptive["original_failure"] & adaptive["quality_safe"]
    adaptive["new_failure"] = ~adaptive["original_failure"] & ~adaptive["quality_safe"]
    adaptive["replacement_target"] = np.nan
    adaptive["replacement_actual"] = np.nan
    return adaptive


def _load_forced_quality(
    root: Path,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    jobs = pd.read_parquet(root / "development_runs/all_candidate_jobs.parquet")
    metrics = _metric_wide(root / "development_runs/vbench_metrics.csv")
    dense = labels.loc[
        labels["config"].eq(BASE_CONFIGS["Dense BF16"]),
        ["prompt_id", *METRICS],
    ].rename(columns={metric: f"dense_{metric}" for metric in METRICS})
    vsa80 = labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    forced = (
        jobs.merge(metrics, on="job_id", validate="one_to_one")
        .merge(dense, on="prompt_id", validate="many_to_one")
        .merge(vsa80, on="prompt_id", validate="many_to_one")
    )
    forced = _add_quality_labels(forced)
    forced["method"] = forced["ra_native_fraction"].map(FORCED_METHODS)
    forced["replacement_target"] = forced["ra_native_fraction"].map(TARGET_REPLACEMENT)
    forced["replacement_actual"] = forced["ra_native_fraction"].map({0.875: 16 / 125, 0.750: 31 / 125})
    forced["original_failure"] = ~forced["vsa80_safe"]
    forced["repaired"] = forced["original_failure"] & forced["quality_safe"]
    forced["new_failure"] = ~forced["original_failure"] & ~forced["quality_safe"]
    return forced


def _quality_export(
    base: pd.DataFrame,
    adaptive: pd.DataFrame,
    forced: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "method",
        "job_id",
        "prompt_id",
        "prompt",
        "seed",
        "replacement_target",
        "replacement_actual",
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "dense_subject_consistency",
        "dense_motion_smoothness",
        "dense_dynamic_degree",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
        "subject_safe",
        "motion_safe",
        "dynamic_safe",
        "quality_safe",
        "original_failure",
        "repaired",
        "new_failure",
        "wall_ms",
        "dit_ms",
        "attention_ms",
        "effective_sparsity",
        "video_path",
    ]
    frames = []
    for frame in (base, adaptive, forced):
        copy = frame.copy()
        for column in columns:
            if column not in copy:
                copy[column] = np.nan
        frames.append(copy[columns])
    result = pd.concat(frames, ignore_index=True)
    result["method"] = pd.Categorical(
        result["method"],
        METHOD_ORDER,
        ordered=True,
    )
    return result.sort_values(["method", "prompt_id"]).reset_index(drop=True)


def _load_traces(
    root: Path,
    forced: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for fraction, directory in (
        (0.875, "rho0125"),
        (0.750, "rho0250"),
    ):
        trace = pd.read_parquet(root / f"development_runs/{directory}/policy_trace.parquet")
        jobs = forced.loc[
            forced["ra_native_fraction"].eq(fraction),
            [
                "job_id",
                "prompt_id",
                "prompt",
                "quality_safe",
                "original_failure",
                "repaired",
                "new_failure",
                "subject_delta",
                "motion_delta",
                "dynamic_delta",
            ],
        ]
        trace = trace.merge(jobs, on="job_id", validate="many_to_one")
        trace["method"] = FORCED_METHODS[fraction]
        trace["replacement_target"] = TARGET_REPLACEMENT[fraction]
        trace["replacement_actual_expected"] = 16 / 125 if fraction == 0.875 else 31 / 125
        trace["native_k"] = trace["total_slots"]
        trace["ra_k"] = trace["selected_count_min"]
        trace["replacement_count"] = trace["replacement_count_min"]
        trace["rho_actual"] = trace["replacement_fraction_mean"]
        trace["example_risk_gap"] = trace.apply(
            lambda row: _array_mean(row["example_added_risk_scores"]) - _array_mean(row["example_removed_risk_scores"]),
            axis=1,
        )
        trace["example_heterogeneity_gap"] = trace.apply(
            lambda row: (
                _array_mean(row["example_added_key_heterogeneity"])
                - _array_mean(row["example_removed_key_heterogeneity"])
            ),
            axis=1,
        )
        frames.append(trace)
    return pd.concat(frames, ignore_index=True)


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    return np.asarray(value)


def _array_mean(value: Any) -> float:
    array = _array(value).astype(float)
    return float(array.mean()) if len(array) else math.nan


def _explode_block_replacements(trace: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "job_id",
        "prompt_id",
        "prompt",
        "method",
        "replacement_target",
        "replacement_fraction_mean",
        "layer",
        "timestep",
        "prefix",
        "quality_safe",
        "original_failure",
        "repaired",
        "new_failure",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
    ]
    frames = []
    for role, prefix in (
        ("dropped_native", "removed"),
        ("outside_topk_rescue", "added"),
    ):
        block_col = f"example_{prefix}_blocks"
        arrays = trace[block_col].map(_array)
        lengths = arrays.map(len).to_numpy()
        repeated = trace.loc[
            trace.index.repeat(lengths),
            metadata,
        ].reset_index(drop=True)
        repeated["role"] = role
        repeated["pair_index"] = np.concatenate([np.arange(length, dtype=np.int16) for length in lengths])
        repeated["block_index"] = np.concatenate(arrays.to_numpy()).astype(np.int32)
        for output, suffix in (
            ("native_rank", "native_ranks"),
            ("native_score", "native_scores"),
            ("residual_risk", "risk_scores"),
            ("key_heterogeneity", "key_heterogeneity"),
        ):
            values = trace[f"example_{prefix}_{suffix}"].map(_array)
            repeated[output] = np.concatenate(values.to_numpy()).astype(np.float64)
        frames.append(repeated)
    return pd.concat(frames, ignore_index=True)


def _load_clean_latency(
    root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    jobs = {}
    traces = {}
    for method, directory in (
        ("Forced RA-VSA 12.5%", "rho0125"),
        ("Forced RA-VSA 25%", "rho0250"),
    ):
        jobs[method] = pd.read_parquet(root / f"latency_runs/{directory}/jobs.parquet")
        traces[method] = pd.read_parquet(root / f"latency_runs/{directory}/policy_trace.parquet")
    return jobs, traces


def _trace_component_summary(trace: pd.DataFrame) -> dict[str, float]:
    component_columns = [
        "heterogeneity_ms",
        "rescue_selection_ms",
        "metadata_selector_ms",
        "fine_attention_ms",
    ]
    by_job = trace.groupby("job_id", as_index=False)[component_columns].sum()
    result = {f"median_{column}_per_video": float(by_job[column].median()) for column in component_columns}
    result.update({f"median_{column}_per_call": float(trace[column].median()) for column in component_columns})
    return result


def _summary_rows(
    quality: pd.DataFrame,
    forced_trace: pd.DataFrame,
    clean_jobs: dict[str, pd.DataFrame],
    clean_traces: dict[str, pd.DataFrame],
    native_trace: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_summary = (
        forced_trace.groupby("method", as_index=False)
        .agg(
            replacement_actual=("replacement_fraction_mean", "mean"),
            replacement_count_min=("replacement_count_min", "min"),
            replacement_count_max=("replacement_count_max", "max"),
            mask_jaccard=("mask_jaccard_mean", "mean"),
            selected_count_min=("selected_count_min", "min"),
            selected_count_max=("selected_count_max", "max"),
            rescue_key_heterogeneity=(
                "rescue_key_heterogeneity_mean",
                "mean",
            ),
            dropped_key_heterogeneity=(
                "removed_key_heterogeneity_mean",
                "mean",
            ),
            mean_example_risk_gap=("example_risk_gap", "mean"),
        )
        .set_index("method")
    )
    rows = []
    latency_rows = []
    for method in METHOD_ORDER:
        group = quality.loc[quality["method"].eq(method)]
        if group.empty:
            continue
        if method in clean_jobs:
            latency = clean_jobs[method]
            protocol = "clean 72-prompt no-save run; minimal invariant trace"
        elif method == "Adaptive-K":
            latency = group
            protocol = "reused frozen adaptive deadline run"
        else:
            latency = group
            protocol = "reused frozen 72-prompt baseline"
        replacement_target = (
            float(group["replacement_target"].dropna().iloc[0]) if group["replacement_target"].notna().any() else np.nan
        )
        row = {
            "method": method,
            "replacement_target": replacement_target,
            "replacement_actual": (
                float(trace_summary.loc[method, "replacement_actual"])
                if method in trace_summary.index
                else (0.0 if method == "VSA80" else np.nan)
            ),
            "unsafe": int((~group["quality_safe"]).sum()),
            "original_failures_repaired": int(group["repaired"].sum()),
            "new_failures": int(group["new_failure"].sum()),
            "subject_failures": int((~group["subject_safe"]).sum()),
            "motion_failures": int((~group["motion_safe"]).sum()),
            "dynamic_regressions": int((~group["dynamic_safe"]).sum()),
            "fine_k": (
                "125"
                if method
                in {
                    "VSA80",
                    "Forced RA-VSA 12.5%",
                    "Forced RA-VSA 25%",
                }
                else (
                    "dense"
                    if method == "Dense BF16"
                    else ("250" if method == "VSA60" else ("375" if method == "VSA40" else "variable"))
                )
            ),
            "median_e2e_ms": float(latency["wall_ms"].median()),
            "median_dit_ms": float(latency["dit_ms"].median()),
            "median_attention_ms": float(latency["attention_ms"].median()),
            "effective_sparsity": float(group["effective_sparsity"].mean()),
            "latency_protocol": protocol,
        }
        if method in trace_summary.index:
            for column in trace_summary.columns:
                row[column] = trace_summary.loc[method, column]
        rows.append(row)
        latency_row = {
            "method": method,
            "median_e2e_ms": row["median_e2e_ms"],
            "median_dit_ms": row["median_dit_ms"],
            "median_attention_ms": row["median_attention_ms"],
            "latency_protocol": protocol,
        }
        if method in clean_traces:
            latency_row.update(_trace_component_summary(clean_traces[method]))
        elif method == "VSA80":
            native_by_job = native_trace.groupby("job_id")["fine_attention_ms"].sum()
            latency_row.update(
                {
                    "median_fine_attention_ms_per_call": float(native_trace["fine_attention_ms"].median()),
                    "median_fine_attention_ms_per_video": float(native_by_job.median()),
                }
            )
        latency_rows.append(latency_row)
    summary = pd.DataFrame(rows)
    latency = pd.DataFrame(latency_rows)
    vsa60 = summary.loc[summary["method"].eq("VSA60")].iloc[0]
    vsa40 = summary.loc[summary["method"].eq("VSA40")].iloc[0]
    for index, row in summary.iterrows():
        if not row["method"].startswith("Forced"):
            continue
        same_k = int(row["selected_count_min"]) == 125 and int(row["selected_count_max"]) == 125
        pareto = all(
            (row["unsafe"] <= baseline["unsafe"] and row["median_e2e_ms"] < baseline["median_e2e_ms"])
            or (row["unsafe"] < baseline["unsafe"] and row["median_e2e_ms"] <= baseline["median_e2e_ms"])
            for baseline in (vsa60, vsa40)
        )
        passed = (
            row["unsafe"] <= 12
            and row["original_failures_repaired"] >= 12
            and row["new_failures"] <= 2
            and row["median_e2e_ms"] <= VSA80_WALL_MS * 1.02
            and row["median_e2e_ms"] < DENSE_WALL_MS
            and same_k
            and pareto
        )
        summary.loc[index, "same_k_invariant"] = same_k
        summary.loc[index, "pareto_beats_vsa60_vsa40"] = pareto
        summary.loc[index, "passes_all_gates"] = passed
    return summary, latency


def _prompt_association(trace: pd.DataFrame) -> pd.DataFrame:
    return trace.groupby(
        [
            "method",
            "prompt_id",
            "prompt",
            "original_failure",
            "repaired",
            "quality_safe",
        ],
        as_index=False,
    ).agg(
        risk_gap=("example_risk_gap", "mean"),
        heterogeneity_gap=("example_heterogeneity_gap", "mean"),
        rescue_heterogeneity=(
            "rescue_key_heterogeneity_mean",
            "mean",
        ),
        dropped_heterogeneity=(
            "removed_key_heterogeneity_mean",
            "mean",
        ),
    )


def _point_biserial(frame: pd.DataFrame, value: str, label: str) -> float:
    usable = frame[[value, label]].dropna()
    if usable[value].nunique() < 2 or usable[label].nunique() < 2:
        return math.nan
    return float(
        np.corrcoef(
            usable[value].to_numpy(float),
            usable[label].astype(float).to_numpy(),
        )[0, 1]
    )


def _plot_quality_latency(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {
        "Dense BF16": "0.35",
        "VSA80": "tab:red",
        "Forced RA-VSA 12.5%": "tab:purple",
        "Forced RA-VSA 25%": "tab:blue",
        "Adaptive-K": "tab:green",
        "VSA60": "tab:orange",
        "VSA40": "tab:brown",
    }
    for row in summary.itertuples():
        ax.scatter(
            row.median_e2e_ms,
            row.unsafe,
            color=colors[row.method],
            s=95 if row.method.startswith("Forced") else 65,
            zorder=3,
        )
        ax.annotate(
            row.method,
            (row.median_e2e_ms, row.unsafe),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(12, color="tab:red", linestyle="--", linewidth=1)
    ax.axvline(
        VSA80_WALL_MS * 1.02,
        color="tab:red",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("Median end-to-end latency (ms)")
    ax.set_ylabel("Unsafe prompts / 72 (lower is better)")
    ax.set_title("Forced replacement does not reach the quality gate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_replacement_ablation(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    names = ["VSA80", "Forced RA-VSA 12.5%", "Forced RA-VSA 25%"]
    frame = summary.set_index("method").loc[names]
    x = frame["replacement_actual"].to_numpy(float) * 100
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    ax.plot(x, frame["unsafe"], "o-", label="unsafe")
    ax.plot(
        x,
        frame["original_failures_repaired"],
        "s-",
        label="original failures repaired",
    )
    ax.plot(x, frame["new_failures"], "^-", label="new failures")
    ax.axhline(12, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual native-set replacement (%)")
    ax.set_ylabel("Prompts / 72")
    ax.set_xticks(x)
    ax.set_title("Fixed-K forced-replacement ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_native_vs_rescue(
    replacements: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.5))
    for ax, metric, label in (
        (axes[0], "key_heterogeneity", "Key heterogeneity $U_K$"),
        (axes[1], "residual_risk", "Residual risk $A_{coarse}U_K$"),
    ):
        data = []
        labels = []
        for method in ("Forced RA-VSA 12.5%", "Forced RA-VSA 25%"):
            for role in ("dropped_native", "outside_topk_rescue"):
                values = replacements.loc[
                    replacements["method"].eq(method) & replacements["role"].eq(role),
                    metric,
                ].to_numpy(float)
                data.append(values)
                short = "drop" if role == "dropped_native" else "rescue"
                ratio = "12.8%" if "12.5" in method else "24.8%"
                labels.append(f"{ratio}\n{short}")
        ax.boxplot(data, tick_labels=labels, showfliers=False, showmeans=True)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Traced native-tail blocks versus outside-Top-K rescues")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_bear_example(
    quality: pd.DataFrame,
    replacements: pd.DataFrame,
    path: Path,
) -> None:
    bear = quality.loc[
        quality["prompt"].str.contains(
            "a bear sniffing the air for scents of food",
            case=False,
            na=False,
        )
        & quality["method"].isin(
            [
                "Dense BF16",
                "VSA80",
                "Forced RA-VSA 12.5%",
                "Forced RA-VSA 25%",
            ]
        )
    ].copy()
    bear["method"] = bear["method"].astype(str)
    bear = bear.set_index("method").loc[
        [
            "Dense BF16",
            "VSA80",
            "Forced RA-VSA 12.5%",
            "Forced RA-VSA 25%",
        ]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(len(bear))
    width = 0.24
    for offset, metric, label in (
        (-width, "subject_consistency", "SC"),
        (0.0, "motion_smoothness", "MS"),
        (width, "dynamic_degree", "DD"),
    ):
        axes[0].bar(x + offset, bear[metric], width, label=label)
    axes[0].set_xticks(
        x,
        ["Dense", "VSA80", "RA 12.8%", "RA 24.8%"],
        rotation=20,
    )
    axes[0].set_ylim(0.75, 1.02)
    axes[0].set_ylabel("VBench score")
    axes[0].set_title("Bear prompt remains unsafe")
    axes[0].legend(fontsize=8)
    bear_blocks = replacements.loc[
        replacements["prompt"].str.contains(
            "a bear sniffing the air for scents of food",
            case=False,
            na=False,
        )
        & replacements["method"].eq("Forced RA-VSA 25%")
    ]
    for role, marker, color, label in (
        ("dropped_native", "x", "tab:red", "dropped native"),
        ("outside_topk_rescue", "o", "tab:blue", "outside-Top-K rescue"),
    ):
        group = bear_blocks.loc[bear_blocks["role"].eq(role)]
        sample = group.iloc[:: max(1, len(group) // 1500)]
        axes[1].scatter(
            sample["native_score"],
            sample["key_heterogeneity"],
            marker=marker,
            color=color,
            alpha=0.35,
            s=14,
            label=label,
        )
    axes[1].set_xlabel("Native coarse score")
    axes[1].set_ylabel("Key heterogeneity $U_K$")
    axes[1].set_title("31 of 125 blocks replaced per query")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Representative required example: bear sniffing for food")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])


def _write_reports(
    root: Path,
    repo: Path,
    summary: pd.DataFrame,
    latency: pd.DataFrame,
    trace: pd.DataFrame,
    prompt_association: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    development = root / "development_72"
    forced_summary = summary.loc[summary["method"].str.startswith("Forced")].copy()
    vsa80 = summary.loc[summary["method"].eq("VSA80")].iloc[0]
    vsa60 = summary.loc[summary["method"].eq("VSA60")].iloc[0]
    vsa40 = summary.loc[summary["method"].eq("VSA40")].iloc[0]
    r12 = forced_summary.loc[forced_summary["method"].eq("Forced RA-VSA 12.5%")].iloc[0]
    r25 = forced_summary.loc[forced_summary["method"].eq("Forced RA-VSA 25%")].iloc[0]
    association_rows = []
    for method, group in prompt_association.groupby("method"):
        failures = group.loc[group["original_failure"]]
        association_rows.append(
            {
                "method": method,
                "risk_gap_repair_point_biserial": _point_biserial(
                    failures,
                    "risk_gap",
                    "repaired",
                ),
                "heterogeneity_gap_repair_point_biserial": _point_biserial(
                    failures,
                    "heterogeneity_gap",
                    "repaired",
                ),
            }
        )
    association = pd.DataFrame(association_rows).set_index("method")
    bear = quality.loc[
        quality["prompt"].str.contains(
            "a bear sniffing the air for scents of food",
            case=False,
            na=False,
        )
        & quality["method"].isin(
            [
                "Dense BF16",
                "VSA80",
                "Forced RA-VSA 12.5%",
                "Forced RA-VSA 25%",
            ]
        )
    ].copy()
    bear["method"] = bear["method"].astype(str)
    bear_text = "; ".join(
        f"{row.method}: SC={row.subject_consistency:.4f}, "
        f"MS={row.motion_smoothness:.4f}, DD={row.dynamic_degree:.1f}, "
        f"{'safe' if row.quality_safe else 'unsafe'}"
        for row in bear.itertuples()
    )
    report_table = forced_summary[
        [
            "method",
            "replacement_actual",
            "unsafe",
            "original_failures_repaired",
            "new_failures",
            "subject_failures",
            "motion_failures",
            "dynamic_regressions",
            "median_e2e_ms",
            "median_attention_ms",
        ]
    ].copy()
    report_table["replacement_actual"] = report_table["replacement_actual"].map(lambda value: f"{value:.1%}")
    for column in ("median_e2e_ms", "median_attention_ms"):
        report_table[column] = report_table[column].map(lambda value: f"{value:.2f}")
    development.joinpath("REPORT.md").write_text(
        f"""# Forced-replacement RA-VSA — 72-prompt report

## Result

Both exact fixed-K counterfactuals fail the frozen quality gate.

{_format_table(report_table, list(report_table.columns))}

The 12.5% target becomes **16/125 = 12.8% actual** after integer rounding.
The 25% target becomes **31/125 = 24.8% actual**. Every traced decision has
K=125 and the rescue blocks are disjoint from the entire native VSA80 Top-K.

## Mechanistic result

- Mean rescue versus dropped-native U_K: 12.8% =
  {r12.rescue_key_heterogeneity:.6f} vs
  {r12.dropped_key_heterogeneity:.6f}; 24.8% =
  {r25.rescue_key_heterogeneity:.6f} vs
  {r25.dropped_key_heterogeneity:.6f}.
- Mean traced rescue-minus-dropped residual risk: 12.8% =
  {r12.mean_example_risk_gap:+.8f}; 24.8% =
  {r25.mean_example_risk_gap:+.8f}.
- Among the 24 original failures, the point-biserial association between
  prompt risk gap and repair is
  {association.loc["Forced RA-VSA 12.5%", "risk_gap_repair_point_biserial"]:+.3f}
  at 12.8% and
  {association.loc["Forced RA-VSA 25%", "risk_gap_repair_point_biserial"]:+.3f}
  at 24.8%.

The forced outside-Top-K blocks are slightly more heterogeneous in aggregate,
but their lower coarse importance means the product risk is not
systematically higher than the native tail being removed. Quality recovery
is correspondingly weak and inconsistent.

## Required bear example

{bear_text}. The prompt is not repaired; this is shown explicitly in
`figures/repaired_example.pdf`.

## Stop action

No additional replacement ratio, full 16-dimension VBench run, or Wan14B
transfer was launched. The optional U_V formula was not launched after the
two mandatory primary policies hit the experiment's terminal “both fail”
condition.

{DECISION_STOP}
"""
    )
    fine12 = latency.loc[latency["method"].eq("Forced RA-VSA 12.5%")].iloc[0]
    fine25 = latency.loc[latency["method"].eq("Forced RA-VSA 25%")].iloc[0]
    native_fine = latency.loc[latency["method"].eq("VSA80")].iloc[0]
    commit = _git(repo, "rev-parse", "HEAD")
    final = f"""# FINAL RESULT — Forced-Replacement RA-VSA

Date: 2026-08-29 UTC  
Code commit: `{commit}`  
Model revision: `25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef`

## Bottom line

Forcing 16 or 31 of VSA80's 125 exact-attention blocks to come from outside
the complete native Top-K does not recover enough quality. Both policies
leave 21/72 prompts unsafe, versus 24/72 for VSA80, while introducing more
than the allowed two new failures.

## Required answers

1. **Is forced replacement implemented exactly as specified?** Yes. Native
   ranks 1..K_native are retained, rescue candidates exclude all original
   native Top-K blocks, and runtime assertions enforce exact replacement.

2. **Is final K identical to native VSA80?** Yes. Every recorded decision has
   selected_count_min = selected_count_max = 125.

3. **What is the actual replacement ratio?** Native VSA80: 0/125 = 0%;
   12.5% target: 16/125 = {r12.replacement_actual:.1%}; 25% target:
   31/125 = {r25.replacement_actual:.1%}.

4. **How many VSA80 failures are repaired by 12.5% replacement?**
   {int(r12.original_failures_repaired)}/24.

5. **How many are repaired by 25% replacement?**
   {int(r25.original_failures_repaired)}/24.

6. **How many new failures are introduced?** {int(r12.new_failures)}/48 at
   12.8% actual and {int(r25.new_failures)}/48 at 24.8% actual.

7. **Does subject consistency improve?** Not enough. Subject-consistency
   failures are {int(r12.subject_failures)} and {int(r25.subject_failures)}
   versus {int(vsa80.subject_failures)} for VSA80; neither policy reaches
   the overall quality gate.

8. **Does motion smoothness improve?** Motion failures are
   {int(r12.motion_failures)} and {int(r25.motion_failures)}, versus
   {int(vsa80.motion_failures)} for VSA80. This component is not the dominant
   remaining failure mode.

9. **Are DD regressions reduced?** DD regressions are
   {int(r12.dynamic_regressions)} and {int(r25.dynamic_regressions)}, versus
   {int(vsa80.dynamic_regressions)} for VSA80; they are not consistently
   eliminated.

10. **What is E2E overhead?** Clean median E2E is
    {r12.median_e2e_ms:.2f} ms at 12.8%
    ({r12.median_e2e_ms / vsa80.median_e2e_ms - 1:+.2%} vs VSA80) and
    {r25.median_e2e_ms:.2f} ms at 24.8%
    ({r25.median_e2e_ms / vsa80.median_e2e_ms - 1:+.2%} vs VSA80).

11. **Is fine-attention kernel latency unchanged?** The fixed K and metadata
    width are unchanged. The native-compatible reference is
    {native_fine.median_fine_attention_ms_per_call:.4f} ms/call; measured
    forced-policy time is {fine12.median_fine_attention_ms_per_call:.4f}
    ms/call at 12.8%
    ({fine12.median_fine_attention_ms_per_call / native_fine.median_fine_attention_ms_per_call - 1:+.2%})
    and {fine25.median_fine_attention_ms_per_call:.4f} ms/call at 24.8%
    ({fine25.median_fine_attention_ms_per_call / native_fine.median_fine_attention_ms_per_call - 1:+.2%}).
    This is close but not numerically identical; most added attention time is
    heterogeneity and rescue selection.

12. **Are rescue blocks systematically more heterogeneous than dropped
    native blocks?** Slightly in aggregate: {r12.rescue_key_heterogeneity:.6f}
    vs {r12.dropped_key_heterogeneity:.6f} at 12.8%, and
    {r25.rescue_key_heterogeneity:.6f} vs
    {r25.dropped_key_heterogeneity:.6f} at 24.8%.

13. **Are quality improvements associated with replacing lower-risk native
    blocks with higher-risk outside-TopK blocks?** No robust association was
    established. Mean rescue-minus-dropped traced risk is
    {r12.mean_example_risk_gap:+.8f} and {r25.mean_example_risk_gap:+.8f};
    repair correlations are
    {association.loc["Forced RA-VSA 12.5%", "risk_gap_repair_point_biserial"]:+.3f}
    and
    {association.loc["Forced RA-VSA 25%", "risk_gap_repair_point_biserial"]:+.3f}.

14. **Does fixed-budget replacement beat fixed VSA60/VSA40 on quality versus
    speed?** No. Both forced policies have 21 unsafe prompts, compared with
    {int(vsa60.unsafe)} for VSA60 and {int(vsa40.unsafe)} for VSA40.

15. **Does this establish that native VSA quality loss is partly caused by
    block misallocation rather than only insufficient K?** It provides only
    weak, prompt-specific evidence: fixed-K reallocation repairs 6–7 of 24
    failures but creates 3–4 new failures, for a net reduction of only three
    unsafe prompts. Prior Adaptive-K repaired 23/24 by adding exact-attention
    capacity. The learned VSA selector's fixed-K block allocation therefore
    cannot be substantially improved by this simple coarse-residual
    rerouting; previously observed quality recovery primarily requires
    additional fine-attention compute.

## Literature framing

The already-collected primary-source table contains adaptive, variable-density,
clustering, calibration, and reuse methods. This experiment supports only the
bounded VSA-specific statement that we tested reassigning an already-trained
VSA checkpoint's fixed fine-attention budget toward high coarse-representation
risk. It does not support a universal novelty claim for error-aware sparse
attention or fixed-budget rerouting.

## Representative prompt

{bear_text}. The required bear example remains unsafe under both forced
policies and is not hidden.

{DECISION_STOP}
"""
    root.joinpath("FINAL_RESULT.md").write_text(final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/forced_ra_vsa"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.root.resolve()
    config_dir = root / "config"
    development_dir = root / "development_72"
    figures_dir = root / "figures"
    for directory in (config_dir, development_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base, labels = _load_base_quality(repo)
    vsa80 = base.loc[base["method"].eq("VSA80")]
    adaptive = _load_adaptive(repo, vsa80)
    forced = _load_forced_quality(root, labels)
    quality = _quality_export(base, adaptive, forced)
    trace = _load_traces(root, forced)
    replacements = _explode_block_replacements(trace)
    clean_jobs, clean_traces = _load_clean_latency(root)
    native_trace = pd.read_parquet(repo / "artifacts/ra_vsa_deadline/instrumentation_run/phase0/policy_trace.parquet")
    summary, latency = _summary_rows(
        quality,
        trace,
        clean_jobs,
        clean_traces,
        native_trace,
    )
    prompt_association = _prompt_association(trace)

    summary.to_csv(development_dir / "results.csv", index=False)
    latency.to_csv(development_dir / "latency.csv", index=False)
    quality.to_csv(development_dir / "quality.csv", index=False)
    trace.to_parquet(
        development_dir / "risk_scores.parquet",
        index=False,
    )
    replacements.to_parquet(
        development_dir / "block_replacements.parquet",
        index=False,
    )

    commit = _git(repo, "rev-parse", "HEAD")
    policy = {
        "date_utc": "2026-08-29",
        "code_commit": commit,
        "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "model_revision": "25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef",
        "hardware": "8x NVIDIA B200",
        "protocol": {
            "prompts": 72,
            "seed": 1024,
            "resolution": "480x832",
            "frames": 81,
            "steps": 3,
            "cfg": 3.0,
            "quality_thresholds": {
                "subject_consistency_delta_min": -0.02,
                "motion_smoothness_delta_min": -0.01,
                "dynamic_degree_delta_min": 0.0,
            },
        },
        "policy": {
            "native_sparsity": 0.8,
            "fine_k": 125,
            "risk_formula": "coarse_mass * U_K",
            "key_heterogeneity": "1 - mean cosine(K_token, mean(K_block))",
            "force_outside_entire_native_topk": True,
            "target_replacement_ratios": [0.125, 0.25],
            "actual_replacement_ratios": [16 / 125, 31 / 125],
            "native_slots": [109, 94],
            "rescue_slots": [16, 31],
            "trained_components": 0,
        },
        "terminal_action": (
            "Both mandatory primary ratios failed; no ratio search, full VBench, Wan14B, or U_V follow-up was launched."
        ),
        "decision": DECISION_STOP,
    }
    config_dir.joinpath("policy.json").write_text(json.dumps(policy, indent=2) + "\n")

    _plot_quality_latency(summary, figures_dir / "quality_latency.pdf")
    _plot_replacement_ablation(
        summary,
        figures_dir / "replacement_ratio_ablation.pdf",
    )
    _plot_native_vs_rescue(
        replacements,
        figures_dir / "native_vs_rescue_scores.pdf",
    )
    _plot_bear_example(
        quality,
        replacements,
        figures_dir / "repaired_example.pdf",
    )
    _write_reports(
        root,
        repo,
        summary,
        latency,
        trace,
        prompt_association,
        quality,
    )
    print(f"assembled={root}")


if __name__ == "__main__":
    main()
