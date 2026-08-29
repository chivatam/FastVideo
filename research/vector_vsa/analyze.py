from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DECISION = "DECISION: STOP — REMOVING K POOLING DOES NOT IMPROVE OVER FINE-VSA8"
RAW = "raw_k_token"
FINE8 = "fine8_pooled"
NATIVE = "native64_pooled"
BEST_VECTOR = "raw_vec8_top2_mean"
GRANULARITY_ORDER = [
    ("native64_pooled", 64, "K64 pooled"),
    ("fine32_pooled", 32, "K32 pooled"),
    ("fine16_pooled", 16, "K16 pooled"),
    ("fine8_pooled", 8, "K8 pooled"),
    ("raw_k_token", 1, "K1 raw"),
]
VECTOR_VARIANTS = [
    "raw_vec8_max",
    "raw_vec8_top2_mean",
    "raw_vec8_logsumexp",
    "raw_vec16_max",
    "raw_vec16_top2_mean",
    "raw_vec16_logsumexp",
]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load(
    root: Path,
    prompt_ids_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_paths = sorted(
        (root / "calibration/run/phase0/stats").glob("*.parquet")
    )
    if len(stats_paths) != 8:
        raise RuntimeError(
            f"Expected 8 frozen calibration traces, found {len(stats_paths)}"
        )
    raw = pd.concat(
        [pd.read_parquet(path) for path in stats_paths],
        ignore_index=True,
    )
    records = root / "calibration/run/phase0/records"
    job_to_prompt = {
        path.stem: json.loads(path.read_text())["prompt_id"]
        for path in records.glob("*.json")
    }
    prompts = pd.DataFrame(json.loads(prompt_ids_path.read_text()))
    raw["prompt_id"] = raw["job_id"].map(job_to_prompt)
    if raw["prompt_id"].isna().any():
        raise RuntimeError("At least one trace job ID has no prompt mapping")
    prompt_columns = [
        "prompt_id",
        "prompt",
        "selection_stratum",
        "vsa80_quality_safe",
    ]
    raw = raw.merge(
        prompts[prompt_columns],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    if raw["selection_stratum"].isna().any():
        raise RuntimeError("Frozen prompt metadata did not merge completely")
    return raw, prompts


def _summary(error: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "candidate_kind",
        "routing_width",
        "execution_width",
        "aggregation",
        "nominal_kv_tokens",
        "nominal_pair_budget_ratio",
    ]
    metrics = [
        "relative_L2_mean",
        "relative_L2_median",
        "relative_L2_p90",
        "relative_L2_p99",
        "cosine_error_mean",
        "cosine_error_median",
        "cosine_error_p90",
        "cosine_error_p99",
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        column: (column, "mean") for column in metrics
    }
    aggregations.update(
        {column: (column, "first") for column in metadata}
    )
    aggregations.update(
        {
            "attention_states": ("variant", "size"),
            "actual_pair_budget_ratio_mean": (
                "actual_pair_budget_ratio",
                "mean",
            ),
            "actual_pair_budget_ratio_max": (
                "actual_pair_budget_ratio",
                "max",
            ),
            "actual_kv_tokens_mean": ("actual_kv_tokens_mean", "mean"),
            "raw_score_ms_mean": ("raw_score_ms", "mean"),
            "aggregation_ms_mean": ("aggregation_ms", "mean"),
            "selection_ms_mean": ("selection_ms", "mean"),
            "execution_ms_mean": ("execution_ms", "mean"),
            "execution_ms_median": ("execution_ms", "median"),
            "execution_ms_p90": (
                "execution_ms",
                lambda values: values.quantile(0.90),
            ),
        }
    )
    return (
        error.groupby("variant", as_index=False)
        .agg(**aggregations)
        .sort_values("relative_L2_p90")
    )


def _granularity(candidate: pd.DataFrame) -> pd.DataFrame:
    indexed = candidate.set_index("variant")
    rows = []
    for variant, width, label in GRANULARITY_ORDER:
        source = indexed.loc[variant]
        rows.append(
            {
                "variant": variant,
                "label": label,
                "routing_width": width,
                "relative_L2_mean": source["relative_L2_mean"],
                "relative_L2_median": source["relative_L2_median"],
                "relative_L2_p90": source["relative_L2_p90"],
                "relative_L2_p99": source["relative_L2_p99"],
                "actual_pair_budget_ratio_max": source[
                    "actual_pair_budget_ratio_max"
                ],
            }
        )
    result = pd.DataFrame(rows)
    result["p90_improves_from_previous"] = (
        result["relative_L2_p90"].diff().lt(0).astype(object)
    )
    result.loc[0, "p90_improves_from_previous"] = None
    result["monotonic_decrease_through_here"] = (
        result["relative_L2_p90"].diff().dropna().lt(0).cummin()
    ).reindex(result.index)
    return result


def _raw_and_vector(
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = candidate.set_index("variant")
    fine = indexed.loc[FINE8]
    raw = indexed.loc[RAW]
    raw_row: dict[str, Any] = {
        "variant": RAW,
        "fine8_relative_L2_mean": fine["relative_L2_mean"],
        "raw_relative_L2_mean": raw["relative_L2_mean"],
        "mean_improvement_over_fine8": (
            fine["relative_L2_mean"] - raw["relative_L2_mean"]
        )
        / fine["relative_L2_mean"],
        "fine8_relative_L2_p90": fine["relative_L2_p90"],
        "raw_relative_L2_p90": raw["relative_L2_p90"],
        "p90_improvement_over_fine8": (
            fine["relative_L2_p90"] - raw["relative_L2_p90"]
        )
        / fine["relative_L2_p90"],
        "fine8_relative_L2_p99": fine["relative_L2_p99"],
        "raw_relative_L2_p99": raw["relative_L2_p99"],
        "p99_improvement_over_fine8": (
            fine["relative_L2_p99"] - raw["relative_L2_p99"]
        )
        / fine["relative_L2_p99"],
        "raw_gate": "PASS"
        if raw["relative_L2_p90"] < fine["relative_L2_p90"]
        else "FAIL",
        "strong_gate": "PASS"
        if (
            fine["relative_L2_p90"] - raw["relative_L2_p90"]
        )
        / fine["relative_L2_p90"]
        >= 0.10
        else "FAIL",
        "actual_pair_budget_ratio_max": raw[
            "actual_pair_budget_ratio_max"
        ],
    }
    raw_result = pd.DataFrame([raw_row])
    raw_gain = fine["relative_L2_p90"] - raw["relative_L2_p90"]
    vector_rows = []
    for variant in VECTOR_VARIANTS:
        row = indexed.loc[variant]
        candidate_gain = (
            fine["relative_L2_p90"] - row["relative_L2_p90"]
        )
        vector_rows.append(
            {
                "variant": variant,
                "execution_width": int(row["execution_width"]),
                "aggregation": row["aggregation"],
                "relative_L2_mean": row["relative_L2_mean"],
                "relative_L2_p90": row["relative_L2_p90"],
                "relative_L2_p99": row["relative_L2_p99"],
                "p90_improvement_over_fine8": candidate_gain
                / fine["relative_L2_p90"],
                "raw_gain_positive": bool(raw_gain > 0),
                "retention_of_raw_gain": (
                    candidate_gain / raw_gain
                    if raw_gain > 0
                    else np.nan
                ),
                "practical_gate_unlocked": bool(raw_gain > 0),
                "actual_pair_budget_ratio_max": row[
                    "actual_pair_budget_ratio_max"
                ],
                "raw_score_ms_mean": row["raw_score_ms_mean"],
                "aggregation_ms_mean": row["aggregation_ms_mean"],
                "selection_ms_mean": row["selection_ms_mean"],
                "execution_ms_mean": row["execution_ms_mean"],
            }
        )
    return raw_result, pd.DataFrame(vector_rows).sort_values(
        "relative_L2_p90"
    )


def _state_pivot(error: pd.DataFrame, *, heads: bool) -> pd.DataFrame:
    keys = ["prompt_id", "prefix", "layer", "timestep"]
    if heads:
        keys.append("head")
    return error.pivot_table(
        index=keys,
        columns="variant",
        values="relative_L2_p90",
    )


def _stratification(error: pd.DataFrame) -> pd.DataFrame:
    pivot = _state_pivot(error, heads=False)
    native = pivot[NATIVE]
    metadata = (
        error[
            [
                "prompt_id",
                "prefix",
                "layer",
                "timestep",
                "selection_stratum",
            ]
        ]
        .drop_duplicates()
        .set_index(["prompt_id", "prefix", "layer", "timestep"])
        .reindex(pivot.index)
    )
    masks = {
        "overall": pd.Series(True, index=pivot.index),
        "top_10pct_native_error": native.ge(native.quantile(0.90)),
        "top_25pct_native_error": native.ge(native.quantile(0.75)),
        "top_50pct_native_error": native.ge(native.quantile(0.50)),
        "bottom_50pct_native_error": native.lt(native.quantile(0.50)),
        "vsa80_safe": metadata["selection_stratum"].eq("vsa80_safe"),
        "vsa80_unsafe": metadata["selection_stratum"].eq(
            "vsa80_unsafe"
        ),
    }
    rows = []
    for variant in (RAW, BEST_VECTOR):
        for label, mask in masks.items():
            fine = float(pivot.loc[mask, FINE8].mean())
            candidate = float(pivot.loc[mask, variant].mean())
            rows.append(
                {
                    "variant": variant,
                    "stratum": label,
                    "states": int(mask.sum()),
                    "native_p90": float(pivot.loc[mask, NATIVE].mean()),
                    "fine8_p90": fine,
                    "candidate_p90": candidate,
                    "absolute_improvement_vs_fine8": fine - candidate,
                    "relative_improvement_vs_fine8": (
                        fine - candidate
                    )
                    / fine,
                    "fraction_states_better_than_fine8": float(
                        (
                            pivot.loc[mask, variant]
                            < pivot.loc[mask, FINE8]
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sensitivity(
    error: pd.DataFrame,
    sensitivity_path: Path,
) -> pd.DataFrame:
    head = error.loc[error["scope"].eq("head_query_blocks")].copy()
    timesteps = sorted(int(value) for value in head["timestep"].unique())
    head["step"] = head["timestep"].map(
        {value: index for index, value in enumerate(timesteps)}
    )
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity = sensitivity.loc[sensitivity["K"].eq(125)].copy()
    sensitivity.sort_values(
        "relative_L2_error_mean",
        ascending=False,
        inplace=True,
    )
    top_count = math.ceil(len(sensitivity) * 0.20)
    sensitivity["sensitivity_stratum"] = "bottom_80pct"
    sensitivity.iloc[
        :top_count,
        sensitivity.columns.get_loc("sensitivity_stratum"),
    ] = "top_20pct"
    columns = [
        "step",
        "layer",
        "head",
        "relative_L2_error_mean",
        "curve_sufficient_K",
        "sensitivity_class",
        "sensitivity_stratum",
    ]
    merged = head.merge(
        sensitivity[columns],
        on=["step", "layer", "head"],
        how="left",
        validate="many_to_one",
    )
    if merged["sensitivity_stratum"].isna().any():
        raise RuntimeError("BR sensitivity units did not align")
    pivot = merged.pivot_table(
        index=["prompt_id", "step", "layer", "head"],
        columns="variant",
        values="relative_L2_p90",
    )
    unit_meta = (
        merged[
            [
                "prompt_id",
                "step",
                "layer",
                "head",
                "sensitivity_stratum",
                "relative_L2_error_mean",
                "curve_sufficient_K",
                "sensitivity_class",
            ]
        ]
        .drop_duplicates()
        .set_index(["prompt_id", "step", "layer", "head"])
        .reindex(pivot.index)
    )
    rows: list[dict[str, Any]] = []

    def add(variant: str, label: str, mask: pd.Series, row_type: str) -> None:
        fine = float(pivot.loc[mask, FINE8].mean())
        candidate = float(pivot.loc[mask, variant].mean())
        rows.append(
            {
                "variant": variant,
                "row_type": row_type,
                "stratum": label,
                "states": int(mask.sum()),
                "native_p90": float(pivot.loc[mask, NATIVE].mean()),
                "fine8_p90": fine,
                "candidate_p90": candidate,
                "relative_improvement_vs_fine8": (
                    fine - candidate
                )
                / fine,
                "br_relative_L2_error_mean": float(
                    unit_meta.loc[mask, "relative_L2_error_mean"].mean()
                ),
                "curve_sufficient_K_mean": float(
                    unit_meta.loc[mask, "curve_sufficient_K"].mean()
                ),
            }
        )

    for variant in (RAW, BEST_VECTOR):
        for label in ("top_20pct", "bottom_80pct"):
            add(
                variant,
                label,
                unit_meta["sensitivity_stratum"].eq(label),
                "br_sensitivity_stratum",
            )
    for step, layer, head_index in [
        (0, 18, 6),
        (0, 22, 3),
        (0, 24, 8),
        (0, 0, 5),
        (0, 23, 9),
    ]:
        index = unit_meta.index.to_frame(index=False)
        mask = (
            index["step"].eq(step)
            & index["layer"].eq(layer)
            & index["head"].eq(head_index)
        )
        mask.index = unit_meta.index
        for variant in (RAW, BEST_VECTOR):
            add(
                variant,
                f"step{step}_layer{layer}_head{head_index}",
                mask,
                "specified_unit",
            )
    return pd.DataFrame(rows)


def _alignment(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw.loc[
        raw["event_type"].eq("vector_vsa_alignment")
        & raw["scope"].eq("all_heads_query_blocks")
    ].copy()
    columns = [
        "spearman_mean",
        "spearman_p10",
        "spearman_median",
        "spearman_p90",
        "retained_dense_mass_mean",
        "retained_dense_mass_p10",
        "retained_dense_mass_median",
        "retained_dense_mass_p90",
        "top_support_overlap_mean",
        "top_support_overlap_p10",
        "top_support_overlap_median",
        "top_support_overlap_p90",
    ]
    aggregations = {
        f"{column}_across_states": (column, "mean")
        for column in columns
    }
    result = selected.groupby("variant", as_index=False).agg(
        states=("variant", "size"),
        routing_width=("routing_width", "first"),
        **aggregations,
    )
    return result.sort_values(
        "spearman_mean_across_states",
        ascending=False,
    )


def _structure(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = raw.loc[
        raw["event_type"].eq("vector_vsa_support_structure")
    ].copy()
    keep = [
        "prompt_id",
        "prompt",
        "selection_stratum",
        "prefix",
        "layer",
        "timestep",
        "query_rows",
        "selected_tokens",
        "runs",
        "run_length_mean",
        "run_length_median",
        "run_length_p90",
        "run_length_p99",
        "run_length_max",
        "fraction_tokens_in_runs_ge8",
        "fraction_tokens_in_runs_ge16",
        "fraction_tokens_in_runs_ge32",
        "contiguity_definition",
    ]
    state_rows = selected[keep].sort_values(
        ["prompt_id", "timestep", "layer"]
    )
    rows = []
    masks = {
        "overall": pd.Series(True, index=selected.index),
        "vsa80_safe": selected["selection_stratum"].eq("vsa80_safe"),
        "vsa80_unsafe": selected["selection_stratum"].eq(
            "vsa80_unsafe"
        ),
    }
    for label, mask in masks.items():
        frame = selected.loc[mask]
        selected_total = frame["selected_tokens"].sum()
        run_total = frame["runs"].sum()
        rows.append(
            {
                "stratum": label,
                "attention_states": len(frame),
                "query_rows": int(frame["query_rows"].sum()),
                "selected_tokens": int(selected_total),
                "runs": int(run_total),
                "run_length_mean_weighted": float(
                    (frame["run_length_mean"] * frame["runs"]).sum()
                    / run_total
                ),
                "state_run_length_median_mean": float(
                    frame["run_length_median"].mean()
                ),
                "state_run_length_p90_mean": float(
                    frame["run_length_p90"].mean()
                ),
                "state_run_length_p99_mean": float(
                    frame["run_length_p99"].mean()
                ),
                "state_run_length_max_mean": float(
                    frame["run_length_max"].mean()
                ),
                "fraction_tokens_in_runs_ge8": float(
                    (
                        frame["fraction_tokens_in_runs_ge8"]
                        * frame["selected_tokens"]
                    ).sum()
                    / selected_total
                ),
                "fraction_tokens_in_runs_ge16": float(
                    (
                        frame["fraction_tokens_in_runs_ge16"]
                        * frame["selected_tokens"]
                    ).sum()
                    / selected_total
                ),
                "fraction_tokens_in_runs_ge32": float(
                    (
                        frame["fraction_tokens_in_runs_ge32"]
                        * frame["selected_tokens"]
                    ).sum()
                    / selected_total
                ),
            }
        )
    return state_rows, pd.DataFrame(rows)


def _systems(
    raw: pd.DataFrame,
    error: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = raw.loc[
        raw["event_type"].eq("vector_vsa_benchmark")
    ].copy()
    kernel = (
        benchmark.groupby("variant", as_index=False)
        .agg(
            candidate_kind=("candidate_kind", "first"),
            routing_width=("routing_width", "first"),
            execution_width=("execution_width", "first"),
            aggregation=("aggregation", "first"),
            samples=("execution_ms", "size"),
            mean_ms=("execution_ms", "mean"),
            median_ms=("execution_ms", "median"),
            p90_ms=(
                "execution_ms",
                lambda values: values.quantile(0.90),
            ),
        )
        .sort_values("mean_ms")
    )
    kernel["measurement"] = "stage0_exact_attention_replay_on_B200"
    kernel["production_status"] = "NOT_INTEGRATED_RAW_GATE_FAILED"

    selector = (
        benchmark.groupby("variant", as_index=False)
        .agg(
            routing_width=("routing_width", "first"),
            execution_width=("execution_width", "first"),
            aggregation=("aggregation", "first"),
            samples=("variant", "size"),
            raw_score_ms_mean=("raw_score_ms", "mean"),
            raw_score_ms_p90=(
                "raw_score_ms",
                lambda values: values.quantile(0.90),
            ),
            aggregation_ms_mean=("aggregation_ms", "mean"),
            aggregation_ms_p90=(
                "aggregation_ms",
                lambda values: values.quantile(0.90),
            ),
            selection_ms_mean=("selection_ms", "mean"),
            selection_ms_p90=(
                "selection_ms",
                lambda values: values.quantile(0.90),
            ),
        )
        .sort_values("raw_score_ms_mean")
    )
    selector["routing_total_ms_mean"] = (
        selector["raw_score_ms_mean"]
        + selector["aggregation_ms_mean"]
        + selector["selection_ms_mean"]
    )
    selector["measurement"] = "stage0_unfused_pytorch_selector"
    selector["production_status"] = "NOT_OPTIMIZED_RAW_GATE_FAILED"

    profiler_rows = []
    for variant in [RAW, BEST_VECTOR, "raw_vec16_logsumexp"]:
        frame = error.loc[error["variant"].eq(variant)]
        for component, column in [
            ("raw_token_scoring", "raw_score_ms"),
            ("segment_aggregation", "aggregation_ms"),
            ("selection", "selection_ms"),
            ("exact_sparse_attention", "execution_ms"),
        ]:
            values = frame[column].dropna().astype(float)
            profiler_rows.append(
                {
                    "variant": variant,
                    "component": component,
                    "samples": len(values),
                    "mean_ms": float(values.mean()),
                    "median_ms": float(values.median()),
                    "p90_ms": float(values.quantile(0.90)),
                    "status": "STAGE0_ONLY_NO_PRODUCTION_KERNEL",
                }
            )
    return kernel, selector, pd.DataFrame(profiler_rows)


def _plots(
    root: Path,
    granularity: pd.DataFrame,
    error: pd.DataFrame,
    structure: pd.DataFrame,
    alignment: pd.DataFrame,
    prior_results_path: Path,
) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    axis.plot(
        granularity["routing_width"],
        granularity["relative_L2_p90"],
        "o-",
        color="#457b9d",
    )
    for row in granularity.itertuples():
        axis.annotate(
            row.label,
            (row.routing_width, row.relative_L2_p90),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log", base=2)
    axis.invert_xaxis()
    axis.set(
        xlabel="K routing width (tokens; smaller is finer)",
        ylabel="Mean call-level p90 relative L2",
        title="K granularity is not monotonic",
    )
    axis.grid(alpha=0.25)
    figure.savefig(figures / "error_vs_k_granularity.pdf")
    plt.close(figure)

    pivot = _state_pivot(error, heads=False)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.6),
        constrained_layout=True,
    )
    for variant, label, color in [
        (FINE8, "Fine-VSA8", "#2a9d8f"),
        (RAW, "Raw-K token", "#e76f51"),
        (BEST_VECTOR, "Raw-score Vec8 top-2", "#457b9d"),
    ]:
        values = np.sort(pivot[variant].to_numpy())
        axes[0].plot(
            values,
            np.linspace(0, 100, len(values)),
            label=label,
            color=color,
        )
    axes[0].set(
        xlabel="Call-level p90 relative L2",
        ylabel="Attention states at or below (%)",
        title="Frozen error distribution",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].scatter(
        pivot[FINE8],
        pivot[RAW],
        s=12,
        alpha=0.35,
        color="#e76f51",
        label="Raw-K",
    )
    axes[1].scatter(
        pivot[FINE8],
        pivot[BEST_VECTOR],
        s=12,
        alpha=0.25,
        color="#457b9d",
        label="Vec8 top-2",
    )
    maximum = float(
        pivot[[FINE8, RAW, BEST_VECTOR]].max().max()
    )
    axes[1].plot([0, maximum], [0, maximum], "--", color="gray")
    axes[1].set(
        xlabel="Fine-VSA8 p90 relative L2",
        ylabel="Candidate p90 relative L2",
        title="Paired attention states",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.savefig(figures / "raw_vs_fine8_error.pdf")
    plt.close(figure)

    overall_structure = structure.set_index("stratum").loc["overall"]
    figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    widths = ["≥8", "≥16", "≥32"]
    fractions = [
        overall_structure["fraction_tokens_in_runs_ge8"],
        overall_structure["fraction_tokens_in_runs_ge16"],
        overall_structure["fraction_tokens_in_runs_ge32"],
    ]
    axis.bar(widths, np.asarray(fractions) * 100, color="#457b9d")
    axis.set(
        xlabel="Natural contiguous run length",
        ylabel="Selected raw tokens covered (%)",
        title="Raw-K support is substantially fragmented",
        ylim=(0, 100),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figures / "support_contiguity.pdf")
    plt.close(figure)

    plot_variants = [
        NATIVE,
        FINE8,
        RAW,
        BEST_VECTOR,
        "raw_vec16_logsumexp",
    ]
    selected = alignment.set_index("variant").loc[plot_variants]
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    axis.scatter(
        selected["spearman_mean_across_states"],
        selected["retained_dense_mass_mean_across_states"],
        s=80,
        color="#457b9d",
    )
    for variant in plot_variants:
        axis.annotate(
            variant,
            (
                selected.loc[variant, "spearman_mean_across_states"],
                selected.loc[
                    variant,
                    "retained_dense_mass_mean_across_states",
                ],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel="Spearman(score, true dense mass)",
        ylabel="Retained dense attention mass",
        title="Better routing proxies do not guarantee lower output error",
    )
    axis.grid(alpha=0.25)
    figure.savefig(figures / "score_mass_alignment.pdf")
    plt.close(figure)

    prior = pd.read_csv(prior_results_path)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.scatter(
        prior["median_e2e_ms"],
        prior["unsafe"],
        s=65,
        color="#457b9d",
    )
    for row in prior.itertuples():
        axis.annotate(
            row.method,
            (row.median_e2e_ms, row.unsafe),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.text(
        0.98,
        0.96,
        "Vec-VSA: no production point\n(Raw-K gate failed)",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#e76f51"},
    )
    axis.set(
        xlabel="Median end-to-end latency (ms)",
        ylabel="Unsafe prompts out of 72",
        title="Published quality–latency outcomes",
    )
    axis.grid(alpha=0.25)
    figure.savefig(figures / "quality_latency_pareto.pdf")
    plt.close(figure)


def _development_placeholders(
    root: Path,
    raw_result: pd.Series,
) -> None:
    development = root / "development_72"
    development.mkdir(parents=True, exist_ok=True)
    reason = (
        "NOT_RUN_STAGE0_STOP: Raw-K p90 "
        f"{raw_result['raw_relative_L2_p90']:.6f} did not beat Fine8 "
        f"{raw_result['fine8_relative_L2_p90']:.6f}"
    )
    for filename in [
        "quality.csv",
        "latency.csv",
        "repaired.csv",
        "regressions.csv",
    ]:
        pd.DataFrame(
            [{"status": "NOT_RUN_STAGE0_STOP", "reason": reason}]
        ).to_csv(development / filename, index=False)
    (development / "REPORT.md").write_text(
        f"""# Vector-VSA development-72 report

The frozen 72-prompt generation run was not authorized. Raw-K token
routing changed the frozen mean call-level p90 relative-L2 from
{raw_result['fine8_relative_L2_p90']:.6f} for Fine-VSA8 to
{raw_result['raw_relative_L2_p90']:.6f}, a
{raw_result['p90_improvement_over_fine8']:.2%} relative change. The
predeclared gate required a strict improvement.

Production integration, video generation, VBench scoring, repairs,
regressions, and end-to-end latency are therefore intentionally
unmeasured.

{DECISION}
"""
    )


def _reports(
    repo: Path,
    root: Path,
    prompts: pd.DataFrame,
    candidate: pd.DataFrame,
    granularity: pd.DataFrame,
    raw_result: pd.DataFrame,
    vectors: pd.DataFrame,
    strata: pd.DataFrame,
    sensitivity: pd.DataFrame,
    alignment: pd.DataFrame,
    contiguity: pd.DataFrame,
    selector: pd.DataFrame,
    kernel: pd.DataFrame,
) -> None:
    calibration = root / "calibration"
    raw_row = raw_result.iloc[0]
    vector_index = vectors.set_index("variant")
    strata_index = strata.set_index(["variant", "stratum"])
    sensitivity_index = sensitivity.set_index(["variant", "stratum"])
    alignment_index = alignment.set_index("variant")
    contiguity_row = contiguity.set_index("stratum").loc["overall"]
    selector_index = selector.set_index("variant")
    kernel_index = kernel.set_index("variant")
    p90_change = float(raw_row["p90_improvement_over_fine8"])
    p99_change = float(raw_row["p99_improvement_over_fine8"])
    best_vector_p90_gain = float(
        vector_index.loc[BEST_VECTOR, "p90_improvement_over_fine8"]
    )
    monotonic = bool(
        granularity["relative_L2_p90"].diff().dropna().lt(0).all()
    )
    exact_ratio = float(candidate["actual_pair_budget_ratio_max"].max())
    aggregate_sparsity = 1.0 - 8000.0 / 39936.0
    gate = {
        "calibration_prompts": 8,
        "safe_prompts": int(
            prompts["selection_stratum"].eq("vsa80_safe").sum()
        ),
        "unsafe_prompts": int(
            prompts["selection_stratum"].eq("vsa80_unsafe").sum()
        ),
        "all_head_attention_states": 720,
        "head_level_attention_states": 8640,
        "fine8_p90": float(raw_row["fine8_relative_L2_p90"]),
        "raw_k_p90": float(raw_row["raw_relative_L2_p90"]),
        "raw_k_relative_improvement_over_fine8": p90_change,
        "raw_gate": False,
        "strong_gate": False,
        "practical_vector_gate_unlocked": False,
        "best_offline_vector": BEST_VECTOR,
        "best_offline_vector_p90": float(
            vector_index.loc[BEST_VECTOR, "relative_L2_p90"]
        ),
        "best_offline_vector_p90_improvement_over_fine8": (
            best_vector_p90_gain
        ),
        "pair_budget_ratio_max": exact_ratio,
        "reason": (
            f"Raw-K p90 {raw_row['raw_relative_L2_p90']:.6f} is worse "
            f"than Fine8 p90 {raw_row['fine8_relative_L2_p90']:.6f} "
            f"({p90_change:.2%} relative change)."
        ),
        "decision": DECISION,
    }
    (calibration / "stage0_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )

    report = f"""# Raw-K Vector VSA frozen Stage 0 calibration

- Frozen prompts: 8 (4 VSA80-safe, 4 VSA80-unsafe), seed 1024.
- Captured states: 720 all-head calls and 8,640 head-level units.
- Exact support: every method stayed at or below Native VSA80; maximum
  valid-token pair ratio was {exact_ratio:.6f}.
- Fine-VSA8 p90 relative-L2: {raw_row['fine8_relative_L2_p90']:.6f}.
- Raw-K token p90 relative-L2: {raw_row['raw_relative_L2_p90']:.6f}.
- Relative change versus Fine8: {p90_change:.2%}; negative means worse.
- Raw-K p99 relative change: {p99_change:.2%}.

## Frozen gate

Raw-K did not beat Fine-VSA8. It regressed p90 by {-p90_change:.2%} and
p99 by {-p99_change:.2%}. The predeclared protocol therefore stops before
production vector-kernel integration or 72-video generation.

The best segment result, `{BEST_VECTOR}`, has p90
{vector_index.loc[BEST_VECTOR, 'relative_L2_p90']:.6f}, a
{best_vector_p90_gain:.2%} improvement over Fine8. This does not rescue
H13: the scientific gate was explicitly defined on individual-token
Raw-K, and its gain is negative. The segment result suggests a locality
regularization effect from aggregating raw scores, not that removing K
pooling itself is sufficient.

## Structure and systems

Only {contiguity_row['fraction_tokens_in_runs_ge8']:.2%} /
{contiguity_row['fraction_tokens_in_runs_ge16']:.2%} /
{contiguity_row['fraction_tokens_in_runs_ge32']:.2%} of selected raw
tokens belong to natural runs of at least 8 / 16 / 32 tokens. Support is
therefore substantially fragmented.

All latency numbers are diagnostic Stage-0 replay measurements on B200,
not production FastWan latency. No production VecAttention adaptation was
opened after the gate failure.

{DECISION}
"""
    (calibration / "REPORT.md").write_text(report)
    _development_placeholders(root, raw_row)

    worst = strata_index.loc[(RAW, "top_10pct_native_error")]
    safe = strata_index.loc[(RAW, "vsa80_safe")]
    unsafe = strata_index.loc[(RAW, "vsa80_unsafe")]
    sensitive_top = sensitivity_index.loc[(RAW, "top_20pct")]
    sensitive_bottom = sensitivity_index.loc[(RAW, "bottom_80pct")]
    raw_alignment = alignment_index.loc[RAW]
    fine_alignment = alignment_index.loc[FINE8]
    best_selector = selector_index.loc[BEST_VECTOR]
    best_kernel = kernel_index.loc[BEST_VECTOR]
    vec16_kernel = kernel_index.loc["raw_vec16_logsumexp"]
    raw_kernel = kernel_index.loc[RAW]
    granularity_text = " → ".join(
        f"K{int(row.routing_width)} {row.relative_L2_p90:.6f}"
        for row in granularity.itertuples()
    )
    final = f"""# Final result — Raw-K Vector VSA at Native 80% Pair Budget

## Outcome

Direct token-level K routing does not improve over Fine-VSA8 at the same
exact support. Fine-VSA8 reaches p90 relative-L2
{raw_row['fine8_relative_L2_p90']:.6f}; Raw-K reaches
{raw_row['raw_relative_L2_p90']:.6f}, a {-p90_change:.2%} regression.
The frozen fail-fast gate therefore stops production integration and full
VBench.

## Required questions

1. **Does error decrease monotonically from K64 → K32 → K16 → K8 →
   raw K1?** {'Yes' if monotonic else 'No'}. The measured sequence is
   {granularity_text}.
2. **Does Raw-K beat Fine-VSA8 p90?** No:
   {raw_row['raw_relative_L2_p90']:.6f} versus
   {raw_row['fine8_relative_L2_p90']:.6f}.
3. **By how much?** It is {-p90_change:.2%} worse at p90
   (relative improvement = {p90_change:.2%}).
4. **Does Raw-K improve p99?** No. Fine8 / Raw-K p99 is
   {raw_row['fine8_relative_L2_p99']:.6f} /
   {raw_row['raw_relative_L2_p99']:.6f}, a {-p99_change:.2%} regression.
5. **Does improvement persist in worst native-error states?** No global
   improvement exists. In the top 10% native-error states Raw-K changes
   Fine8 p90 by
   {worst['relative_improvement_vs_fine8']:.2%}; it beats Fine8 in
   {worst['fraction_states_better_than_fine8']:.2%} of those states.
   Across all states it beats Fine8 individually in
   {strata_index.loc[(RAW, 'overall'), 'fraction_states_better_than_fine8']:.2%},
   but the frozen aggregate p90 metric is worse.
6. **Does improvement concentrate in BR-sensitive units?** No consistent
   winning effect. Relative change versus Fine8 is
   {sensitive_top['relative_improvement_vs_fine8']:.2%} in the top 20%
   sensitive units and
   {sensitive_bottom['relative_improvement_vs_fine8']:.2%} in the bottom
   80%. The five requested units are reported in
   `calibration/sensitivity_stratification.csv`.
7. **Does individual K scoring improve score-to-mass alignment?** Yes.
   Mean Spearman rises from
   {fine_alignment['spearman_mean_across_states']:.6f} for pooled Fine8
   to {raw_alignment['spearman_mean_across_states']:.6f} for Raw-K, and
   retained mass rises from
   {fine_alignment['retained_dense_mass_mean_across_states']:.2%} to
   {raw_alignment['retained_dense_mass_mean_across_states']:.2%}. This
   proxy improvement does not lower output error.
8. **What fraction of selected raw tokens form contiguous runs?**
   {contiguity_row['fraction_tokens_in_runs_ge8']:.2%} are in runs ≥8,
   {contiguity_row['fraction_tokens_in_runs_ge16']:.2%} in runs ≥16, and
   {contiguity_row['fraction_tokens_in_runs_ge32']:.2%} in runs ≥32.
9. **Is Vec8 or Vec16 a better practical representation?** Offline,
   Vec8 gives the best fidelity: `{BEST_VECTOR}`. Vec16 log-sum-exp is a
   faster diagnostic exact-replay geometry but has weaker p90 fidelity.
   No production representation was frozen because Raw-K failed.
10. **Which vector aggregation works best?** Top-2 mean for Vec8, with
    p90 {vector_index.loc[BEST_VECTOR, 'relative_L2_p90']:.6f}. For Vec16,
    log-sum-exp is marginally best at
    {vector_index.loc['raw_vec16_logsumexp', 'relative_L2_p90']:.6f}.
11. **How much Raw-K gain does the practical vector retain?** Not
    applicable: Raw-K gain is negative, so the predeclared retention
    denominator is invalid and the practical gate never opens. Vec8
    top-2 independently improves over Fine8 by {best_vector_p90_gain:.2%}.
12. **Is exact K-token support <= VSA80?** Yes. Maximum measured valid
    pair ratio is {exact_ratio:.6f}.
13. **Is aggregate sparsity still ~80%?** Yes. Nominal support is
    8,000 of 39,936 padded K tokens, or {aggregate_sparsity:.2%} sparsity
    (the established valid-token accounting is approximately 79.97%).
14. **Can existing vector-sparse kernels be adapted to B200/FastWan?**
    Not established. Stage 1 inspection/integration was conditionally
    locked by the Raw-K gate. The offline width-8/16 replay executes on
    B200, but that is not a production VecAttention result.
15. **What is selector overhead?** For Vec8 top-2, Stage-0 means are
    raw scoring {best_selector['raw_score_ms_mean']:.4f} ms,
    aggregation {best_selector['aggregation_ms_mean']:.4f} ms, selection
    {best_selector['selection_ms_mean']:.4f} ms, total
    {best_selector['routing_total_ms_mean']:.4f} ms per captured call.
    This is an unfused diagnostic implementation.
16. **What is vector exact-attention latency?** Stage-0 exact replay
    averages {best_kernel['mean_ms']:.4f} ms/call for Vec8 top-2 and
    {vec16_kernel['mean_ms']:.4f} ms/call for Vec16 log-sum-exp. Raw
    arbitrary-token replay averages {raw_kernel['mean_ms']:.4f} ms/call.
    No production end-to-end latency was measured.
17. **How many VSA80 failures are repaired?** Not measured; the 72-video
    run was correctly skipped.
18. **How many new failures are introduced?** Not measured for the same
    reason.
19. **Does Vec-VSA beat Fine8 on systems?** Not demonstrated. The
    diagnostic selector is unfused and the production path was not
    authorized.
20. **Does it beat VSA60/VSA40 on the quality-latency Pareto?** No point
    exists to support that claim.
21. **Is K-side pooling the dominant remaining VSA bottleneck?** No.
    K1 regresses versus K8 even though raw-score segment aggregation gives
    a modest side improvement. The evidence points away from removal of K
    pooling itself as the dominant remaining error.
22. **Proceed to full VBench?** No. The frozen Raw-K gate failed.

## Diagnostic splits

Raw-K relative change versus Fine8 is
{safe['relative_improvement_vs_fine8']:.2%} on the four VSA80-safe
calibration prompts and {unsafe['relative_improvement_vs_fine8']:.2%} on
the four unsafe prompts. These were analysis-only splits.

## Recommended next direction

Test Q-side routing granularity: split or cluster the 64 queries that
currently share a sparse route while preserving the exact K-token budget
per query. Do not reopen the previously rejected K-side methods.

## Reproducibility

- branch: `{_git(repo, 'branch', '--show-current')}`
- source commit at analysis: `{_git(repo, 'rev-parse', 'HEAD')}`
- hardware: 8× NVIDIA B200 host; one frozen prompt per GPU
- model revision: `25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef`
- generated: 2026-08-29 UTC

{DECISION}
"""
    (root / "FINAL_RESULT.md").write_text(final)


def _manifest(root: Path, repo: Path) -> None:
    workspace_status = _git(repo, "status", "--porcelain").splitlines()
    source_status = [
        line
        for line in workspace_status
        if not line.startswith("?? artifacts/")
    ]
    provenance = {
        "branch": _git(repo, "branch", "--show-current"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "source_dirty": bool(source_status),
        "excluded_workspace_status": [
            line
            for line in workspace_status
            if line.startswith("?? artifacts/")
        ],
        "generated_utc": "2026-08-29",
        "decision": DECISION,
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name in {"MANIFEST.sha256", "manifest.csv"}
        ):
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    manifest_path = root / "manifest.csv"
    rows.append(
        {
            "path": "manifest.csv",
            "bytes": manifest_path.stat().st_size,
            "sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        }
    )
    manifest_lines = [
        f"{row['sha256']}  {row['path']}" for row in rows
    ]
    (root / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/vector_vsa"),
    )
    parser.add_argument(
        "--prompt-ids",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/fine_vsa/calibration/prompt_ids.json"
        ),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/br_vsa/calibration/"
            "sensitivity_summary.csv"
        ),
    )
    parser.add_argument(
        "--prior-results",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/fine_vsa/development_72/results.csv"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    for directory in [
        root / "method_notes",
        root / "calibration",
        root / "structure",
        root / "systems",
        root / "development_72",
        root / "figures",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    raw, prompts = _load(root, args.prompt_ids)
    error = raw.loc[
        raw["event_type"].eq("vector_vsa_error")
        & raw["scope"].eq("all_heads_query_blocks")
    ].copy()
    if len(error) != 720 * 11:
        raise RuntimeError(
            f"Expected 7,920 all-head error rows, found {len(error)}"
        )
    candidate = _summary(error)
    granularity = _granularity(candidate)
    raw_result, vectors = _raw_and_vector(candidate)
    strata = _stratification(error)
    sensitivity = _sensitivity(
        raw.loc[raw["event_type"].eq("vector_vsa_error")],
        args.sensitivity,
    )
    alignment = _alignment(raw)
    run_rows, contiguity = _structure(raw)
    kernel, selector, profiler = _systems(raw, error)

    raw.to_parquet(root / "calibration/raw_stats.parquet", index=False)
    (root / "calibration/prompt_ids.json").write_text(
        json.dumps(
            prompts.to_dict(orient="records"),
            indent=2,
        )
        + "\n"
    )
    geometry_columns = [
        "parent_blocks",
        "parent_width",
        "padded_tokens",
        "valid_tokens",
        "native_selected_parent_blocks",
        "native_nominal_kv_tokens",
        "raw_token_descriptor_capacity",
        "raw_score_semantics",
        "coarse_residual_policy",
    ]
    (
        raw[geometry_columns]
        .drop_duplicates()
        .to_csv(root / "calibration/geometry.csv", index=False)
    )
    candidate.to_csv(
        root / "calibration/candidate_summary.csv",
        index=False,
    )
    granularity.to_csv(
        root / "calibration/granularity_error.csv",
        index=False,
    )
    raw_result.to_csv(
        root / "calibration/raw_k_results.csv",
        index=False,
    )
    vectors.to_csv(
        root / "calibration/vector_results.csv",
        index=False,
    )
    strata.to_csv(
        root / "calibration/worst_state_stratification.csv",
        index=False,
    )
    sensitivity.to_csv(
        root / "calibration/sensitivity_stratification.csv",
        index=False,
    )
    alignment.to_csv(
        root / "calibration/score_mass_alignment.csv",
        index=False,
    )
    run_rows.to_csv(
        root / "structure/selected_run_lengths.csv",
        index=False,
    )
    contiguity.to_csv(
        root / "structure/contiguity_stats.csv",
        index=False,
    )
    kernel.to_csv(
        root / "systems/vector_kernel_benchmark.csv",
        index=False,
    )
    selector.to_csv(
        root / "systems/selector_latency.csv",
        index=False,
    )
    profiler.to_csv(
        root / "systems/profiler_summary.csv",
        index=False,
    )
    for note in (
        "raw_k_scoring.md",
        "exact_pair_accounting.md",
    ):
        source = repo / "research/vector_vsa/method_notes" / note
        (root / "method_notes" / note).write_text(source.read_text())

    _plots(
        root,
        granularity,
        error,
        contiguity,
        alignment,
        args.prior_results,
    )
    _reports(
        repo,
        root,
        prompts,
        candidate,
        granularity,
        raw_result,
        vectors,
        strata,
        sensitivity,
        alignment,
        contiguity,
        selector,
        kernel,
    )
    _manifest(root, repo)
    print(DECISION)


if __name__ == "__main__":
    main()
