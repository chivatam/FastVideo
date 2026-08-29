from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DECISION = "DECISION: STOP — ATTENTION-COHERENT KV64 BLOCK FORMATION DOES NOT RECOVER ENOUGH FINE8 FIDELITY"
BASELINES = ("native64_spatial", "fine8_spatial")
CLUSTERS = ("k_head_pca64", "k_shared_pca64")
EXPECTED_WINNER = "k_shared_pca64"


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def _summarize(
    frame: pd.DataFrame,
    *,
    group: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, selected in frame.groupby(group):
        row: dict[str, Any] = {
            group: variant,
            "samples": len(selected),
        }
        for column in columns:
            values = selected[column].dropna().astype(float)
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_p10"] = _quantile(values, 0.10)
            row[f"{column}_p90"] = _quantile(values, 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_calibration(
    root: Path,
    prompt_ids_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_dir = root / "calibration/run/phase0/stats"
    paths = sorted(stats_dir.glob("*.parquet"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 calibration traces, found {len(paths)}")
    raw = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    records_dir = root / "calibration/run/phase0/records"
    job_to_prompt = {path.stem: json.loads(path.read_text())["prompt_id"] for path in records_dir.glob("*.json")}
    prompt_meta = pd.DataFrame(json.loads(prompt_ids_path.read_text()))
    raw["prompt_id"] = raw["job_id"].map(job_to_prompt)
    if raw["prompt_id"].isna().any():
        raise RuntimeError("Calibration trace has an unmapped job ID")
    raw = raw.merge(
        prompt_meta[
            [
                "prompt_id",
                "prompt",
                "selection_stratum",
                "vsa80_quality_safe",
            ]
        ],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    return raw, prompt_meta


def _candidate_error(error: pd.DataFrame) -> pd.DataFrame:
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
    metadata = [
        "candidate_kind",
        "grouping",
        "shared_across_heads",
        "fixed_slot_width",
        "selected_blocks",
        "nominal_kv_tokens",
        "nominal_pair_budget_ratio",
    ]
    aggregations: dict[str, tuple[str, str]] = {column: (column, "mean") for column in metrics}
    aggregations.update({column: (column, "first") for column in metadata})
    aggregations.update(
        {
            "attention_calls": ("variant", "size"),
            "actual_pair_budget_ratio_mean": (
                "actual_pair_budget_ratio",
                "mean",
            ),
            "actual_pair_budget_ratio_max": (
                "actual_pair_budget_ratio",
                "max",
            ),
            "actual_kv_tokens_mean": ("actual_kv_tokens_mean", "mean"),
            "execution_ms_mean": ("execution_ms", "mean"),
            "execution_ms_median": ("execution_ms", "median"),
            "execution_ms_p90": (
                "execution_ms",
                lambda values: values.quantile(0.90),
            ),
        }
    )
    return error.groupby("variant", as_index=False).agg(**aggregations).sort_values("relative_L2_p90")


def _state_pivot(error: pd.DataFrame) -> pd.DataFrame:
    return error.pivot_table(
        index=["prompt_id", "prefix", "layer", "timestep"],
        columns="variant",
        values="relative_L2_p90",
    )


def _benefit_rows(
    pivot: pd.DataFrame,
    *,
    masks: dict[str, pd.Series],
    row_type: str,
) -> list[dict[str, Any]]:
    native = pivot["native64_spatial"]
    fine = pivot["fine8_spatial"]
    rows: list[dict[str, Any]] = []
    for variant in CLUSTERS:
        for stratum, mask in masks.items():
            native_error = float(native.loc[mask].mean())
            fine_error = float(fine.loc[mask].mean())
            candidate_error = float(pivot.loc[mask, variant].mean())
            denominator = native_error - fine_error
            rows.append(
                {
                    "variant": variant,
                    "row_type": row_type,
                    "stratum": stratum,
                    "states": int(mask.sum()),
                    "native_p90": native_error,
                    "fine8_p90": fine_error,
                    "candidate_p90": candidate_error,
                    "benefit_retained": (
                        (native_error - candidate_error) / denominator if abs(denominator) > 1e-12 else float("nan")
                    ),
                    "absolute_improvement_vs_native": (native_error - candidate_error),
                    "relative_improvement_vs_native": (native_error - candidate_error) / native_error,
                }
            )
    return rows


def _benefit_tables(
    error: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = _state_pivot(error)
    native = pivot["native64_spatial"]
    masks = {
        "overall": pd.Series(True, index=pivot.index),
        "top_10pct_native_error": native.ge(native.quantile(0.90)),
        "top_25pct_native_error": native.ge(native.quantile(0.75)),
        "top_50pct_native_error": native.ge(native.quantile(0.50)),
        "bottom_50pct_native_error": native.lt(native.quantile(0.50)),
    }
    worst = pd.DataFrame(_benefit_rows(pivot, masks=masks, row_type="native_error_stratum"))
    overall = worst.loc[worst["stratum"].eq("overall")].copy()

    prompt_stratum = (
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
        .reindex(pivot.index)["selection_stratum"]
    )
    diagnostic_masks = {
        "vsa80_safe": prompt_stratum.eq("vsa80_safe"),
        "vsa80_unsafe": prompt_stratum.eq("vsa80_unsafe"),
    }
    safe_unsafe = pd.DataFrame(
        _benefit_rows(
            pivot,
            masks=diagnostic_masks,
            row_type="calibration_prompt_stratum",
        )
    )
    return overall, worst, safe_unsafe


def _event_summary(
    raw: pd.DataFrame,
    *,
    event_type: str,
    scope: str,
    columns: list[str],
) -> pd.DataFrame:
    selected = raw.loc[raw["event_type"].eq(event_type) & raw["scope"].eq(scope)]
    return _summarize(selected, group="variant", columns=columns)


def _adjusted_rand(left: Any, right: Any) -> float:
    labels_a = np.asarray(left, dtype=np.int64)
    labels_b = np.asarray(right, dtype=np.int64)
    if labels_a.shape != labels_b.shape:
        raise ValueError("ARI label vectors must have equal shapes")
    _, inverse_a = np.unique(labels_a, return_inverse=True)
    _, inverse_b = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros(
        (inverse_a.max() + 1, inverse_b.max() + 1),
        dtype=np.int64,
    )
    np.add.at(contingency, (inverse_a, inverse_b), 1)

    def choose2(values: np.ndarray) -> float:
        values = values.astype(np.float64)
        return float((values * (values - 1.0) / 2.0).sum())

    total = labels_a.size
    if total < 2:
        return 1.0
    pair_total = total * (total - 1.0) / 2.0
    agreement = choose2(contingency)
    row_pairs = choose2(contingency.sum(axis=1))
    column_pairs = choose2(contingency.sum(axis=0))
    expected = row_pairs * column_pairs / pair_total
    maximum = 0.5 * (row_pairs + column_pairs)
    if abs(maximum - expected) < 1e-12:
        return 1.0
    return (agreement - expected) / (maximum - expected)


def _stability_rows(trace: pd.DataFrame) -> pd.DataFrame:
    comparisons: list[dict[str, Any]] = []

    def append(
        *,
        variant: str,
        comparison: str,
        values: list[float],
        interpretation: str,
    ) -> None:
        array = np.asarray(values, dtype=np.float64)
        comparisons.append(
            {
                "variant": variant,
                "comparison": comparison,
                "pairs": int(array.size),
                "ari_mean": float(array.mean()) if array.size else np.nan,
                "ari_median": (float(np.median(array)) if array.size else np.nan),
                "ari_p10": (float(np.quantile(array, 0.10)) if array.size else np.nan),
                "ari_p90": (float(np.quantile(array, 0.90)) if array.size else np.nan),
                "interpretation": interpretation,
            }
        )

    for variant in CLUSTERS:
        selected = trace.loc[trace["variant"].eq(variant)]

        step_values: list[float] = []
        for _, group in selected.groupby(
            ["prompt_id", "layer", "head"],
            dropna=False,
        ):
            ordered = group.sort_values("timestep")
            labels = ordered["sample_cluster_labels"].tolist()
            step_values.extend(_adjusted_rand(left, right) for left, right in zip(labels, labels[1:], strict=False))
        append(
            variant=variant,
            comparison="adjacent_denoising_steps",
            values=step_values,
            interpretation=(
                "ARI of sampled original-token cluster labels for adjacent steps at fixed prompt/layer/head"
            ),
        )

        prompt_values: list[float] = []
        for _, group in selected.groupby(
            ["layer", "timestep", "head"],
            dropna=False,
        ):
            ordered = group.sort_values("prompt_id")
            labels = ordered["sample_cluster_labels"].tolist()
            if labels:
                prompt_values.extend(_adjusted_rand(labels[0], other) for other in labels[1:])
        append(
            variant=variant,
            comparison="cross_prompt_static_layer",
            values=prompt_values,
            interpretation=("ARI against the first frozen prompt at fixed layer/step/head"),
        )

    head = trace.loc[trace["variant"].eq("k_head_pca64")]
    head_values: list[float] = []
    for _, group in head.groupby(
        ["prompt_id", "layer", "timestep"],
        dropna=False,
    ):
        labels = group.sort_values("head")["sample_cluster_labels"].tolist()
        if labels:
            head_values.extend(_adjusted_rand(labels[0], other) for other in labels[1:])
    append(
        variant="k_head_pca64",
        comparison="cross_head_same_state",
        values=head_values,
        interpretation=(
            "ARI against head 0 at fixed prompt/layer/step; measures whether head-specific assignments are reusable"
        ),
    )
    append(
        variant="k_shared_pca64",
        comparison="cross_head_same_state",
        values=[1.0],
        interpretation=(
            "Exact by construction: one assignment is derived from mean-normalized K and reused across all heads"
        ),
    )
    return pd.DataFrame(comparisons)


def _latency_tables(
    raw: pd.DataFrame,
    error: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = raw.loc[raw["event_type"].eq("cluster_vsa_benchmark")].copy()
    clustering = _summarize(
        benchmark,
        group="variant",
        columns=["grouping_ms", "centroid_ms", "scoring_ms", "selection_ms"],
    )
    clustering["measurement"] = "stage0_cuda_event_replay"

    permutation = _summarize(
        benchmark,
        group="variant",
        columns=["permutation_build_ms", "permutation_ms"],
    )
    permutation["measurement"] = "stage0_cuda_event_replay"
    permutation["combined_mean_ms"] = permutation["permutation_build_ms_mean"] + permutation["permutation_ms_mean"]

    sparse = _summarize(
        error,
        group="variant",
        columns=["execution_ms"],
    )
    sparse["measurement"] = "stage0_exact_attention_replay"
    sparse["production_kernel_status"] = "NOT_INTEGRATED_STAGE0_STOP"

    profiler_rows: list[dict[str, Any]] = []
    component_columns = [
        ("cluster_assignment", "grouping_ms"),
        ("permutation_construction", "permutation_build_ms"),
        ("kv_and_analysis_permutation", "permutation_ms"),
        ("centroid_calculation", "centroid_ms"),
        ("coarse_scoring", "scoring_ms"),
        ("topk_selection", "selection_ms"),
        ("exact_sparse_attention", "execution_ms"),
    ]
    for variant, selected in benchmark.groupby("variant"):
        for component, column in component_columns:
            values = selected[column].dropna().astype(float)
            profiler_rows.append(
                {
                    "variant": variant,
                    "component": component,
                    "samples": len(values),
                    "mean_ms": float(values.mean()),
                    "median_ms": float(values.median()),
                    "p90_ms": _quantile(values, 0.90),
                    "status": "STAGE0_ONLY_NO_PRODUCTION_INTEGRATION",
                }
            )
        routing_columns = [
            "grouping_ms",
            "permutation_build_ms",
            "permutation_ms",
            "centroid_ms",
            "scoring_ms",
            "selection_ms",
        ]
        total = selected[routing_columns].sum(axis=1)
        profiler_rows.append(
            {
                "variant": variant,
                "component": "total_new_grouping_and_routing",
                "samples": len(total),
                "mean_ms": float(total.mean()),
                "median_ms": float(total.median()),
                "p90_ms": _quantile(total, 0.90),
                "status": "STAGE0_ONLY_NO_PRODUCTION_INTEGRATION",
            }
        )
    return (
        clustering,
        permutation,
        sparse,
        pd.DataFrame(profiler_rows),
    )


def _plot_block_coherence(summary: pd.DataFrame, output: Path) -> None:
    order = ["native64_spatial", "k_head_pca64", "k_shared_pca64"]
    selected = summary.set_index("variant").loc[order]
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    positions = np.arange(len(order))
    axis.bar(
        positions,
        selected["mean_mean"],
        color=["#e76f51", "#2a9d8f", "#457b9d"],
    )
    axis.errorbar(
        positions,
        selected["mean_mean"],
        yerr=[
            selected["mean_mean"] - selected["p10_mean"],
            selected["p90_mean"] - selected["mean_mean"],
        ],
        fmt="none",
        color="black",
        capsize=4,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(["Native spatial", "Head-specific", "Shared-head"])
    axis.set(
        ylabel="Mean K coherence",
        title="Fixed-size KV64 block coherence",
        ylim=(0.55, 0.92),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output)
    plt.close(figure)


def _plot_alignment(summary: pd.DataFrame, output: Path) -> None:
    order = ["native64_spatial", "k_head_pca64", "k_shared_pca64"]
    selected = summary.set_index("variant").loc[order]
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    axis.scatter(
        selected["spearman_mean_mean"],
        selected["top125_mass_recall_mean_mean"],
        s=100,
        color=["#e76f51", "#2a9d8f", "#457b9d"],
    )
    for variant, label in zip(
        order,
        ["Native spatial", "Head-specific", "Shared-head"],
        strict=True,
    ):
        axis.annotate(
            label,
            (
                selected.loc[variant, "spearman_mean_mean"],
                selected.loc[variant, "top125_mass_recall_mean_mean"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set(
        xlabel="Spearman(coarse score, true dense block mass)",
        ylabel="Top-125 retained dense mass",
        title="Routing-score fidelity",
    )
    axis.grid(alpha=0.25)
    figure.savefig(output)
    plt.close(figure)


def _plot_benefit(benefit: pd.DataFrame, output: Path) -> None:
    selected = benefit.sort_values("benefit_retained")
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    values = selected["benefit_retained"] * 100.0
    labels = selected["variant"].replace(
        {
            "k_head_pca64": "Head-specific K clusters",
            "k_shared_pca64": "Shared-head K clusters",
        }
    )
    axis.barh(labels, values, color=["#457b9d", "#2a9d8f"])
    axis.axvline(70, color="#e76f51", linestyle="--", label="Frozen GO gate")
    axis.axvline(
        66.75,
        color="#f4a261",
        linestyle=":",
        label="Previous hierarchical result",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Fine8 p90 benefit retained (%)",
        title="Cluster-VSA frozen Stage 0 gate",
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output)
    plt.close(figure)


def _plot_worst_states(
    worst: pd.DataFrame,
    winner: str,
    output: Path,
) -> None:
    order = [
        "top_10pct_native_error",
        "top_25pct_native_error",
        "top_50pct_native_error",
        "bottom_50pct_native_error",
        "overall",
    ]
    selected = worst.loc[worst["variant"].eq(winner)].set_index("stratum").loc[order]
    labels = ["Top 10%", "Top 25%", "Top 50%", "Bottom 50%", "Overall"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    values = selected["benefit_retained"] * 100.0
    axis.bar(labels, values, color=np.where(values >= 0, "#2a9d8f", "#e76f51"))
    axis.axhline(70, color="#f4a261", linestyle="--", label="GO gate")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        ylabel="Fine8 p90 benefit retained (%)",
        title="Best Cluster-VSA candidate by native-error stratum",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output)
    plt.close(figure)


def _plot_cluster_visualization(
    assignments: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )
    for axis, variant, title in [
        (axes[0], "k_head_pca64", "Head-specific clustering"),
        (axes[1], "k_shared_pca64", "Shared-head clustering"),
    ]:
        row = assignments.loc[assignments["variant"].eq(variant)].iloc[0]
        permutation = np.asarray(
            row["valid_token_permutation"],
            dtype=np.int64,
        )
        clustered_position = np.arange(permutation.size)
        axis.scatter(
            clustered_position[::4],
            permutation[::4],
            s=1,
            alpha=0.35,
            color="#457b9d",
            rasterized=True,
        )
        axis.set(
            xlabel="Position after similarity permutation",
            ylabel="Original padded token index",
            title=title,
        )
        axis.grid(alpha=0.15)
    figure.savefig(output)
    plt.close(figure)


def _plot_quality_latency(
    prior_results_path: Path,
    output: Path,
) -> pd.DataFrame:
    systems = pd.read_csv(prior_results_path)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.scatter(
        systems["median_e2e_ms"],
        systems["unsafe"],
        s=65,
        color="#457b9d",
    )
    for row in systems.itertuples():
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
        "Cluster-VSA: no production point\n(Stage 0 gate failed)",
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
    figure.savefig(output)
    plt.close(figure)
    return systems


def _write_placeholders(
    development: Path,
    *,
    winner_benefit: float,
) -> None:
    reason = f"NOT_RUN_STAGE0_STOP: best Fine8 p90 benefit retention {winner_benefit:.2%} < 70% gate"
    contents = {
        "results.csv": {
            "method": "Cluster-VSA",
            "status": "NOT_RUN_STAGE0_STOP",
            "reason": reason,
            "unsafe": np.nan,
            "repaired": np.nan,
            "new_failures": np.nan,
            "median_e2e_ms": np.nan,
        },
        "quality.csv": {
            "status": "NOT_RUN_STAGE0_STOP",
            "reason": reason,
        },
        "latency.csv": {
            "status": "NOT_RUN_STAGE0_STOP",
            "reason": reason,
        },
        "repaired.csv": {
            "status": "NOT_RUN_STAGE0_STOP",
            "reason": reason,
        },
        "regressions.csv": {
            "status": "NOT_RUN_STAGE0_STOP",
            "reason": reason,
        },
    }
    for name, row in contents.items():
        pd.DataFrame([row]).to_csv(development / name, index=False)
    (development / "REPORT.md").write_text(
        f"""# Cluster-VSA development-72 report

The frozen 72-prompt run was not authorized. The best K-only fixed-size
candidate retained {winner_benefit:.2%} of Pure Fine8's overall p90
fidelity gain, below the predeclared 70% gate.

Q/K independent clustering was also not opened because K-only clustering
did not improve the frozen overall p90 fidelity metric. Bidirectional
co-clustering, production integration, VBench generation, and production
latency measurement were therefore correctly skipped.

{DECISION}
"""
    )


def _write_reports(
    *,
    repo: Path,
    root: Path,
    prompt_meta: pd.DataFrame,
    candidates: pd.DataFrame,
    benefit: pd.DataFrame,
    worst: pd.DataFrame,
    safe_unsafe: pd.DataFrame,
    coherence: pd.DataFrame,
    alignment: pd.DataFrame,
    internal_mass: pd.DataFrame,
    stability: pd.DataFrame,
    profiler: pd.DataFrame,
    winner: str,
) -> dict[str, Any]:
    calibration = root / "calibration"
    development = root / "development_72"
    candidate = candidates.set_index("variant")
    overall = benefit.set_index("variant")
    winner_row = overall.loc[winner]
    winner_error = float(winner_row["candidate_p90"])
    native_error = float(winner_row["native_p90"])
    fine_error = float(winner_row["fine8_p90"])
    winner_benefit = float(winner_row["benefit_retained"])
    winner_candidate = candidate.loc[winner]
    p99_native = float(candidate.loc["native64_spatial", "relative_L2_p99"])
    p99_winner = float(candidate.loc[winner, "relative_L2_p99"])
    p99_relative = (p99_native - p99_winner) / p99_native
    p90_relative = (native_error - winner_error) / native_error
    strata = worst.loc[worst["variant"].eq(winner)].set_index("stratum")
    diagnostics = safe_unsafe.loc[safe_unsafe["variant"].eq(winner)].set_index("stratum")
    coherence_index = coherence.set_index("variant")
    alignment_index = alignment.set_index("variant")
    mass_index = internal_mass.set_index("variant")
    profiler_index = profiler.set_index(["variant", "component"])
    routing = profiler_index.loc[(winner, "total_new_grouping_and_routing")]
    exact = profiler_index.loc[(winner, "exact_sparse_attention")]

    gate = {
        "calibration_prompts": 8,
        "safe_prompts": int(prompt_meta["selection_stratum"].eq("vsa80_safe").sum()),
        "unsafe_prompts": int(prompt_meta["selection_stratum"].eq("vsa80_unsafe").sum()),
        "attention_states": 720,
        "cluster_candidates": len(CLUSTERS),
        "winner": winner,
        "native_p90": native_error,
        "fine8_p90": fine_error,
        "winner_p90": winner_error,
        "winner_benefit_retained": winner_benefit,
        "go_threshold": 0.70,
        "strong_threshold": 0.80,
        "pair_budget_ratio_max": float(winner_candidate["actual_pair_budget_ratio_max"]),
        "fixed_slot_width": int(winner_candidate["fixed_slot_width"]),
        "k_only_improves_frozen_metric": winner_error < native_error,
        "qk_independent_unlocked": False,
        "qk_coclustering_unlocked": False,
        "go": False,
        "reason": (
            f"The best K-only fixed-size candidate retained "
            f"{winner_benefit:.2%} of Fine8's p90 fidelity benefit and "
            f"{'improved' if winner_error < native_error else 'regressed'} "
            "the frozen overall p90 metric; the predeclared gate requires "
            "at least 70% retention and an improvement over native VSA64."
        ),
        "decision": DECISION,
    }
    (calibration / "stage0_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    report = f"""# Cluster-VSA Stage 0 calibration

- Frozen prompts: 8 (4 VSA80-safe, 4 VSA80-unsafe).
- Attention states: 720 all-head call states and 8,640 head-level states.
- Native geometry: 624 KV64 slots, 39,936 padded tokens, 32,760 valid
  tokens, K=125, nominal 8,000 selected tokens.
- Full cluster slots: 455 × 64 tokens; native-equivalent ragged boundary
  capacities: 65 × 32, 91 × 16, and 13 × 8.
- Candidate representations: head-specific K clustering and one shared K
  assignment reused across heads.
- Pair budget: selected capacity histograms exactly match native VSA64
  per query row; maximum valid-token pair ratio is
  {float(winner_candidate["actual_pair_budget_ratio_max"]):.6f}.

## Frozen fidelity gate

| Method | Mean call-level p90 relative-L2 | Fine8 benefit retained |
|---|---:|---:|
| Native VSA64 | {native_error:.6f} | 0.00% |
| Head-specific K clusters | {overall.loc["k_head_pca64", "candidate_p90"]:.6f} | {overall.loc["k_head_pca64", "benefit_retained"]:.2%} |
| Shared-head K clusters | {winner_error:.6f} | {winner_benefit:.2%} |
| Pure Fine8 | {fine_error:.6f} | 100.00% |

The best candidate is `{winner}`, but it regresses the frozen overall p90
metric by {-p90_relative:.2%}. It therefore fails both the improvement
condition and the 70% benefit-retention threshold.

## Why the hypothesis looked promising locally

Head-specific clustering raises mean K coherence from
{coherence_index.loc["native64_spatial", "mean_mean"]:.6f} to
{coherence_index.loc["k_head_pca64", "mean_mean"]:.6f}. Its coarse-score
Spearman correlation rises from
{alignment_index.loc["native64_spatial", "spearman_mean_mean"]:.6f} to
{alignment_index.loc["k_head_pca64", "spearman_mean_mean"]:.6f}, and
Top-125 dense-mass recall rises from
{alignment_index.loc["native64_spatial", "top125_mass_recall_mean_mean"]:.6f}
to
{alignment_index.loc["k_head_pca64", "top125_mass_recall_mean_mean"]:.6f}.

Those proxy improvements do not translate to global output fidelity.
Shared-head clustering helps the top 10/25/50% native-error states, but
strongly regresses the bottom half:

| Stratum | Fine8 benefit retained | Relative improvement vs native |
|---|---:|---:|
| Top 10% | {strata.loc["top_10pct_native_error", "benefit_retained"]:.2%} | {strata.loc["top_10pct_native_error", "relative_improvement_vs_native"]:.2%} |
| Top 25% | {strata.loc["top_25pct_native_error", "benefit_retained"]:.2%} | {strata.loc["top_25pct_native_error", "relative_improvement_vs_native"]:.2%} |
| Top 50% | {strata.loc["top_50pct_native_error", "benefit_retained"]:.2%} | {strata.loc["top_50pct_native_error", "relative_improvement_vs_native"]:.2%} |
| Bottom 50% | {strata.loc["bottom_50pct_native_error", "benefit_retained"]:.2%} | {strata.loc["bottom_50pct_native_error", "relative_improvement_vs_native"]:.2%} |
| Overall | {winner_benefit:.2%} | {p90_relative:.2%} |

Safe/unsafe calibration retention is
{diagnostics.loc["vsa80_safe", "benefit_retained"]:.2%} /
{diagnostics.loc["vsa80_unsafe", "benefit_retained"]:.2%}; this split is
diagnostic only and was not used for tuning.

## Fail-fast decision

Variant C was allowed only if K-only clustering improved frozen offline
fidelity. It did not. Variant D required a clear Variant A/B improvement.
Neither condition was met, so Q/K independent clustering, co-clustering,
production integration, and the 72-prompt generation run were not started.

{DECISION}
"""
    (calibration / "REPORT.md").write_text(report)
    _write_placeholders(development, winner_benefit=winner_benefit)

    shared_step = stability.loc[
        stability["variant"].eq("k_shared_pca64") & stability["comparison"].eq("adjacent_denoising_steps")
    ].iloc[0]
    shared_prompt = stability.loc[
        stability["variant"].eq("k_shared_pca64") & stability["comparison"].eq("cross_prompt_static_layer")
    ].iloc[0]
    final = f"""# Final result — Fixed-Size Cluster-VSA at Native 80% Pair Budget

## Outcome

Training-free fixed-size K clustering improves some local routing proxies
and helps high-error attention states, but it does not recover Fine8-like
fidelity overall. The best candidate, one K-derived permutation shared
across heads, changes mean call-level p90 relative-L2 from
{native_error:.6f} to {winner_error:.6f}; that is a {p90_relative:.2%}
relative change and {winner_benefit:.2%} of Fine8's p90 gain. The frozen
progression requirement was at least 70% retention plus improvement over
native VSA64.

## Required questions

1. **How coherent are native spatial KV64 blocks in K space?** Mean
   coherence is {coherence_index.loc["native64_spatial", "mean_mean"]:.6f}
   (p10/median/p90
   {coherence_index.loc["native64_spatial", "p10_mean"]:.6f} /
   {coherence_index.loc["native64_spatial", "median_mean"]:.6f} /
   {coherence_index.loc["native64_spatial", "p90_mean"]:.6f}).
2. **How much does fixed-size K clustering improve coherence?**
   Head-specific clustering reaches
   {coherence_index.loc["k_head_pca64", "mean_mean"]:.6f}, an absolute
   gain of
   {coherence_index.loc["k_head_pca64", "mean_mean"] - coherence_index.loc["native64_spatial", "mean_mean"]:.6f}.
   Shared-head clustering instead falls to
   {coherence_index.loc["k_shared_pca64", "mean_mean"]:.6f}.
3. **Does better coherence improve coarse-score correlation with true
   dense mass?** Yes locally. Head-specific Spearman rises from
   {alignment_index.loc["native64_spatial", "spearman_mean_mean"]:.6f} to
   {alignment_index.loc["k_head_pca64", "spearman_mean_mean"]:.6f}, and
   Top-125 mass recall rises from
   {alignment_index.loc["native64_spatial", "top125_mass_recall_mean_mean"]:.6f}
   to
   {alignment_index.loc["k_head_pca64", "top125_mass_recall_mean_mean"]:.6f}.
   This proxy gain does not preserve output fidelity.
4. **Does K-only clustering reduce dense-relative sparse output error?**
   No on the frozen overall p90 metric. The best candidate is worse:
   {native_error:.6f} → {winner_error:.6f}.
5. **What is its p90/p99 improvement?** Relative p90 improvement is
   {p90_relative:.2%}; relative p99 improvement is {p99_relative:.2%}
   ({p99_native:.6f} → {p99_winner:.6f}). Negative means regression.
6. **How much of Fine8's p90 gain is retained?** {winner_benefit:.2%}.
7. **Does it exceed the previous 66.75% hierarchical result?** No.
8. **Does it pass the frozen 70% offline gate?** No. It also fails the
   required native-error improvement condition.
9. **Does Q/K independent clustering improve over K-only?** Not run.
   Variant C was conditionally unlocked only by a K-only fidelity
   improvement, which did not occur.
10. **Does bidirectional Q-K co-clustering improve further?** Not run.
    Variant D required a clear A/B improvement.
11. **Do selected clustered blocks have less internal mass
    concentration?** No. Native top-8/16/32 mean fractions are
    {mass_index.loc["native64_spatial", "top8_mean_mean"]:.2%} /
    {mass_index.loc["native64_spatial", "top16_mean_mean"]:.2%} /
    {mass_index.loc["native64_spatial", "top32_mean_mean"]:.2%}; the best
    candidate has
    {mass_index.loc[winner, "top8_mean_mean"]:.2%} /
    {mass_index.loc[winner, "top16_mean_mean"]:.2%} /
    {mass_index.loc[winner, "top32_mean_mean"]:.2%}, which is more
    concentrated.
12. **Does clustering help the worst native-error states?** Yes, but
    unevenly: top-10/25/50% benefit retention is
    {strata.loc["top_10pct_native_error", "benefit_retained"]:.2%} /
    {strata.loc["top_25pct_native_error", "benefit_retained"]:.2%} /
    {strata.loc["top_50pct_native_error", "benefit_retained"]:.2%};
    bottom-50% retention is
    {strata.loc["bottom_50pct_native_error", "benefit_retained"]:.2%}.
13. **Is exact pair budget still <= VSA80?** Yes. The selected capacity
    histogram exactly matches native per query; maximum measured ratio is
    {float(winner_candidate["actual_pair_budget_ratio_max"]):.6f}.
14. **Is aggregate sparsity still ~80%?** Yes: nominal support remains
    8,000 of 39,936 padded tokens, or 79.97% sparsity.
15. **Can clusters be contiguous KV64 blocks through permutation?** Yes.
    Recursive-cluster traversal becomes one contiguous permutation; exact
    attention uses original K/V in fixed 64-wide slots.
16. **What is clustering/permutation overhead?** For the best candidate,
    total new Stage-0 grouping and routing averages
    {routing["mean_ms"]:.4f} ms per captured call (p90
    {routing["p90_ms"]:.4f} ms). This is an offline replay measurement,
    not production E2E.
17. **Can grouping be reused across heads or steps?** Across heads, yes
    structurally for the shared candidate because one assignment is reused.
    Adjacent-step sampled-label ARI is {shared_step["ari_mean"]:.4f};
    cross-prompt static-layer ARI is {shared_prompt["ari_mean"]:.4f}.
    No production reuse policy was frozen after the fidelity failure.
18. **What is production exact-attention latency?** Not measured.
    Offline exact-attention replay for the best candidate averages
    {exact["mean_ms"]:.4f} ms per captured call (p90
    {exact["p90_ms"]:.4f} ms).
19. **How many VSA80 failures are repaired?** Not measured; the 72-prompt
    run was correctly skipped.
20. **How many new failures are introduced?** Not measured for the same
    reason.
21. **Does the method approach Fine8 quality?** No globally. It improves
    difficult states but retains {winner_benefit:.2%} overall.
22. **Does it approach VSA80 systems efficiency?** Contiguous KV64
    execution is structurally compatible, but clustering adds
    {routing["mean_ms"]:.4f} ms/call offline and production E2E was not
    authorized.
23. **Does it beat VSA60/VSA40 on quality-latency Pareto?** Not
    demonstrated; there is no generated quality/latency point.
24. **Should it proceed to full VBench?** No. The frozen Stage-0 gate
    failed, so the experiment stops here.

## Reproducibility

- branch: `{_git_value(repo, "branch", "--show-current")}`
- source commit at package generation:
  `{_git_value(repo, "rev-parse", "HEAD")}`
- model/seed: frozen FastVideo setup, seed 1024
- calibration: 8 frozen prompts, 720 attention states
- host: 8× NVIDIA B200
- generated: 2026-08-29 UTC

{DECISION}
"""
    (root / "FINAL_RESULT.md").write_text(final)
    return gate


def _write_manifest(root: Path, gate: dict[str, Any]) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path == root / "manifest.json"
            or str(relative).startswith("calibration/run/")
            or str(relative).startswith("smoke")
        ):
            continue
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "experiment": "cluster_vsa",
        "date_utc": "2026-08-29",
        "stage0_go": gate["go"],
        "decision": DECISION,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def analyze(
    *,
    repo: Path,
    root: Path,
    prompt_ids_path: Path,
    prior_results_path: Path,
) -> dict[str, Any]:
    calibration = root / "calibration"
    grouping = root / "grouping"
    systems = root / "systems"
    development = root / "development_72"
    figures = root / "figures"
    method_notes = root / "method_notes"
    for directory in (
        calibration,
        grouping,
        systems,
        development,
        figures,
        method_notes,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    source_notes = repo / "research/cluster_vsa/method_notes"
    for source in source_notes.glob("*.md"):
        (method_notes / source.name).write_text(source.read_text())

    raw, prompt_meta = _load_calibration(root, prompt_ids_path)
    (calibration / "prompt_ids.json").write_text(json.dumps(prompt_meta.to_dict(orient="records"), indent=2) + "\n")
    raw.to_parquet(calibration / "raw_stats.parquet", index=False)

    geometry_columns = [
        "parent_blocks",
        "parent_width",
        "padded_tokens",
        "valid_tokens",
        "native_selected_parent_blocks",
        "native_nominal_kv_tokens",
        "cluster_slot_size_policy",
        "centroid_policy",
    ]
    geometry = raw[geometry_columns].dropna(how="all").drop_duplicates().iloc[0].to_dict()
    (calibration / "geometry.json").write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n")

    error = raw.loc[raw["event_type"].eq("cluster_vsa_error") & raw["scope"].eq("all_heads_query_blocks")].copy()
    if set(error["variant"].unique()) != set(BASELINES + CLUSTERS):
        raise RuntimeError("Unexpected Cluster-VSA variant set")
    if error["job_id"].nunique() != 8 or len(error) != 2880:
        raise RuntimeError("Expected 8 jobs and 2,880 all-head error rows")
    if float(error["actual_pair_budget_ratio"].max()) > 1.0 + 1e-9:
        raise RuntimeError("Pair budget violation")

    candidates = _candidate_error(error)
    candidates.to_csv(calibration / "candidate_error.csv", index=False)
    benefit, worst, safe_unsafe = _benefit_tables(error)
    benefit.to_csv(
        calibration / "fine8_benefit_retained.csv",
        index=False,
    )
    worst.to_csv(
        calibration / "worst_state_stratification.csv",
        index=False,
    )
    safe_unsafe.to_csv(
        calibration / "safe_unsafe_stratification.csv",
        index=False,
    )
    winner = benefit.sort_values(
        "benefit_retained",
        ascending=False,
    ).iloc[0]["variant"]
    if winner != EXPECTED_WINNER:
        raise RuntimeError(f"Unexpected winner: {winner}")

    coherence = _event_summary(
        raw,
        event_type="cluster_block_coherence",
        scope="all_heads_blocks",
        columns=["mean", "p10", "median", "p90"],
    )
    coherence.to_csv(calibration / "block_coherence.csv", index=False)
    alignment = _event_summary(
        raw,
        event_type="cluster_coarse_true_mass_alignment",
        scope="all_heads_query_blocks",
        columns=[
            "spearman_mean",
            "spearman_p10",
            "spearman_median",
            "spearman_p90",
            "top125_mass_recall_mean",
            "top125_mass_recall_p10",
            "top125_mass_recall_median",
            "top125_mass_recall_p90",
        ],
    )
    alignment.to_csv(
        calibration / "coarse_true_mass_alignment.csv",
        index=False,
    )
    internal_mass = _event_summary(
        raw,
        event_type="cluster_internal_mass",
        scope="all_heads_selected_blocks",
        columns=[
            "top8_mean",
            "top8_p10",
            "top8_median",
            "top8_p90",
            "top16_mean",
            "top16_p10",
            "top16_median",
            "top16_p90",
            "top32_mean",
            "top32_p10",
            "top32_median",
            "top32_p90",
        ],
    )
    internal_mass.to_csv(
        calibration / "selected_block_internal_mass.csv",
        index=False,
    )

    assignments = raw.loc[raw["event_type"].eq("cluster_assignment")].copy()
    assignment_columns = [
        "prompt_id",
        "prompt",
        "selection_stratum",
        "prefix",
        "layer",
        "timestep",
        "variant",
        "head",
        "shared_across_heads",
        "valid_token_permutation",
        "cluster_sizes",
    ]
    assignments[assignment_columns].to_parquet(
        grouping / "cluster_assignments.parquet",
        index=False,
    )

    trace = raw.loc[raw["event_type"].eq("cluster_grouping_trace")].copy()
    permutation_stats = _summarize(
        trace,
        group="variant",
        columns=[
            "mean_adjacent_original_index_jump",
            "median_adjacent_original_index_jump",
        ],
    )
    checksum_counts = (
        trace.groupby("variant")["permutation_checksum"].nunique().rename("unique_permutation_checksums").reset_index()
    )
    permutation_stats = permutation_stats.merge(
        checksum_counts,
        on="variant",
        validate="one_to_one",
    )
    permutation_stats.to_csv(
        grouping / "permutation_stats.csv",
        index=False,
    )
    stability = _stability_rows(trace)
    stability.to_csv(grouping / "stability.csv", index=False)

    clustering_latency, permutation_latency, sparse_latency, profiler = _latency_tables(raw, error)
    clustering_latency.to_csv(
        systems / "clustering_latency.csv",
        index=False,
    )
    permutation_latency.to_csv(
        systems / "permutation_latency.csv",
        index=False,
    )
    sparse_latency.to_csv(
        systems / "sparse_kernel_latency.csv",
        index=False,
    )
    profiler.to_csv(systems / "profiler_summary.csv", index=False)

    _plot_block_coherence(
        coherence,
        figures / "block_coherence.pdf",
    )
    _plot_alignment(
        alignment,
        figures / "coarse_vs_true_mass.pdf",
    )
    _plot_benefit(
        benefit,
        figures / "fine8_benefit_retained.pdf",
    )
    _plot_worst_states(
        worst,
        winner,
        figures / "worst_state_error.pdf",
    )
    _plot_cluster_visualization(
        assignments,
        figures / "cluster_visualization.pdf",
    )
    published = _plot_quality_latency(
        prior_results_path,
        figures / "quality_latency_pareto.pdf",
    )
    published.to_csv(root / "published_systems_reference.csv", index=False)

    gate = _write_reports(
        repo=repo,
        root=root,
        prompt_meta=prompt_meta,
        candidates=candidates,
        benefit=benefit,
        worst=worst,
        safe_unsafe=safe_unsafe,
        coherence=coherence,
        alignment=alignment,
        internal_mass=internal_mass,
        stability=stability,
        profiler=profiler,
        winner=winner,
    )
    _write_manifest(root, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/cluster_vsa"),
    )
    parser.add_argument(
        "--prompt-ids",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/fine_vsa/calibration/prompt_ids.json"),
    )
    parser.add_argument(
        "--prior-results",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/fine_vsa/development_72/results.csv"),
    )
    args = parser.parse_args()
    gate = analyze(
        repo=args.repo.resolve(),
        root=args.root.resolve(),
        prompt_ids_path=args.prompt_ids.resolve(),
        prior_results_path=args.prior_results.resolve(),
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
