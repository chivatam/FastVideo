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

DECISION = (
    "DECISION: STOP — FINE SCORING CANNOT PRESERVE FINE-VSA QUALITY "
    "UNDER COARSE EXECUTION"
)
WINNER_FALLBACK = "s8_e64_max_all_l0.00"
BASELINES = ("native64_global", "kv8_global")


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _quantile(values: pd.Series, value: float) -> float:
    return float(values.quantile(value))


def _load_calibration(
    root: Path,
    prompt_ids_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_dir = root / "calibration/run/phase0/stats"
    frames = [
        pd.read_parquet(path)
        for path in sorted(stats_dir.glob("*.parquet"))
    ]
    if len(frames) != 8:
        raise RuntimeError(
            f"Expected 8 calibration traces, found {len(frames)}"
        )
    raw = pd.concat(frames, ignore_index=True)
    records = root / "calibration/run/phase0/records"
    job_to_prompt = {
        path.stem: json.loads(path.read_text())["prompt_id"]
        for path in records.glob("*.json")
    }
    prompts = pd.DataFrame(json.loads(prompt_ids_path.read_text()))
    prompt_meta = prompts[
        ["prompt_id", "prompt", "selection_stratum"]
    ].copy()
    raw["prompt_id"] = raw["job_id"].map(job_to_prompt)
    if raw["prompt_id"].isna().any():
        raise RuntimeError("Calibration trace has an unmapped job ID")
    raw = raw.merge(
        prompt_meta,
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
        "score_width",
        "execution_width",
        "aggregation",
        "soft_native_prior",
        "parent_pool",
        "nominal_kv_tokens",
        "nominal_pair_budget_ratio",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "mean") for column in metrics
    }
    aggregations.update(
        {
            column: (column, "first")
            for column in metadata
        }
    )
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
            "fine_score_ms_mean": ("fine_score_ms", "mean"),
            "aggregation_ms_mean": ("aggregation_ms", "mean"),
            "selection_ms_mean": ("selection_ms", "mean"),
            "execution_ms_mean": ("execution_ms", "mean"),
        }
    )
    summary = (
        error.groupby("variant", as_index=False)
        .agg(**aggregations)
        .sort_values("relative_L2_p90")
    )
    return summary


def _state_pivot(
    error: pd.DataFrame,
    *,
    head_scope: bool,
) -> pd.DataFrame:
    keys = ["prompt_id", "prefix", "layer", "timestep"]
    if head_scope:
        keys.append("head")
    return error.pivot_table(
        index=keys,
        columns="variant",
        values="relative_L2_p90",
    )


def _benefit_table(
    error: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    pivot = _state_pivot(error, head_scope=False)
    native = pivot["native64_global"]
    fine = pivot["kv8_global"]
    masks = {
        "overall": pd.Series(True, index=native.index),
        "top_10pct_native_error": native.ge(native.quantile(0.90)),
        "top_25pct_native_error": native.ge(native.quantile(0.75)),
        "top_50pct_native_error": native.ge(native.quantile(0.50)),
        "bottom_50pct_native_error": native.lt(native.quantile(0.50)),
    }
    hierarchical = candidates.loc[
        candidates["candidate_kind"].eq("hierarchical")
    ]
    rows: list[dict[str, Any]] = []
    for candidate in hierarchical.to_dict(orient="records"):
        variant = candidate["variant"]
        for stratum, mask in masks.items():
            native_error = float(native.loc[mask].mean())
            fine_error = float(fine.loc[mask].mean())
            candidate_error = float(pivot.loc[mask, variant].mean())
            denominator = native_error - fine_error
            retained = (
                (native_error - candidate_error) / denominator
                if abs(denominator) > 1e-12
                else float("nan")
            )
            rows.append(
                {
                    "variant": variant,
                    "stratum": stratum,
                    "states": int(mask.sum()),
                    "native_p90": native_error,
                    "fine8_p90": fine_error,
                    "candidate_p90": candidate_error,
                    "benefit_retained": retained,
                    "relative_improvement_vs_native": (
                        native_error - candidate_error
                    )
                    / native_error,
                    "score_width": candidate["score_width"],
                    "execution_width": candidate["execution_width"],
                    "aggregation": candidate["aggregation"],
                    "soft_native_prior": candidate[
                        "soft_native_prior"
                    ],
                    "parent_pool": candidate["parent_pool"],
                    "actual_pair_budget_ratio_max": candidate[
                        "actual_pair_budget_ratio_max"
                    ],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stratum", "benefit_retained"],
        ascending=[True, False],
    )


def _sensitivity_table(
    error: pd.DataFrame,
    benefit: pd.DataFrame,
    sensitivity_path: Path,
) -> pd.DataFrame:
    head = error.loc[
        error["scope"].eq("head_query_blocks")
    ].copy()
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
    merged = head.merge(
        sensitivity[
            [
                "step",
                "layer",
                "head",
                "relative_L2_error_mean",
                "curve_sufficient_K",
                "sensitivity_class",
                "sensitivity_stratum",
            ]
        ],
        on=["step", "layer", "head"],
        how="left",
        validate="many_to_one",
    )
    if merged["sensitivity_stratum"].isna().any():
        raise RuntimeError("Could not align hierarchical and BR-VSA units")
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
    )
    hierarchical = benefit.loc[
        benefit["stratum"].eq("overall"), "variant"
    ].tolist()
    rows: list[dict[str, Any]] = []

    def append_row(
        variant: str,
        label: str,
        mask: pd.Series,
        *,
        row_type: str,
    ) -> None:
        native_error = float(
            pivot.loc[mask, "native64_global"].mean()
        )
        fine_error = float(pivot.loc[mask, "kv8_global"].mean())
        candidate_error = float(pivot.loc[mask, variant].mean())
        denominator = native_error - fine_error
        rows.append(
            {
                "variant": variant,
                "row_type": row_type,
                "stratum": label,
                "states": int(mask.sum()),
                "native_p90": native_error,
                "fine8_p90": fine_error,
                "candidate_p90": candidate_error,
                "benefit_retained": (
                    native_error - candidate_error
                )
                / denominator
                if abs(denominator) > 1e-12
                else float("nan"),
                "relative_improvement_vs_native": (
                    native_error - candidate_error
                )
                / native_error,
            }
        )

    for variant in hierarchical:
        for stratum in ("top_20pct", "bottom_80pct"):
            append_row(
                variant,
                stratum,
                unit_meta["sensitivity_stratum"].eq(stratum),
                row_type="br_sensitivity_stratum",
            )

    winner = (
        benefit.loc[benefit["stratum"].eq("overall")]
        .sort_values("benefit_retained", ascending=False)
        .iloc[0]["variant"]
    )
    specified_units = [
        (0, 18, 6),
        (0, 22, 3),
        (0, 24, 8),
        (0, 0, 5),
        (0, 23, 9),
    ]
    index_frame = unit_meta.index.to_frame(index=False)
    for step, layer, head_index in specified_units:
        unit_mask = (
            index_frame["step"].eq(step)
            & index_frame["layer"].eq(layer)
            & index_frame["head"].eq(head_index)
        )
        unit_mask.index = unit_meta.index
        append_row(
            winner,
            f"step{step}_layer{layer}_head{head_index}",
            unit_mask,
            row_type="specified_unit",
        )
    return pd.DataFrame(rows)


def _benchmark_tables(
    raw: pd.DataFrame,
    error: pd.DataFrame,
    winner: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = raw.loc[
        raw["event_type"].eq("hierarchical_scoring_benchmark")
    ]
    scoring_rows = []
    for label, column in (
        ("native64_score", "native_score_ms"),
        ("score8", "score8_ms"),
        ("score16", "score16_ms"),
    ):
        values = benchmark[column].dropna()
        scoring_rows.append(
            {
                "component": label,
                "samples": len(values),
                "mean_ms": float(values.mean()),
                "median_ms": float(values.median()),
                "p90_ms": _quantile(values, 0.90),
                "measurement": "stage0_cuda_event_replay",
            }
        )
    scoring = pd.DataFrame(scoring_rows)

    execution = (
        error.groupby("variant", as_index=False)
        .agg(
            score_width=("score_width", "first"),
            execution_width=("execution_width", "first"),
            nominal_kv_tokens=("nominal_kv_tokens", "first"),
            actual_pair_budget_ratio_max=(
                "actual_pair_budget_ratio",
                "max",
            ),
            samples=("execution_ms", "size"),
            mean_ms=("execution_ms", "mean"),
            median_ms=("execution_ms", "median"),
            p90_ms=("execution_ms", lambda values: values.quantile(0.90)),
        )
        .sort_values("mean_ms")
    )
    execution["measurement"] = "stage0_exact_attention_replay"

    winner_rows = error.loc[error["variant"].eq(winner)]
    parts = {
        "fine_score": winner_rows["fine_score_ms"],
        "aggregation": winner_rows["aggregation_ms"],
        "selection": winner_rows["selection_ms"],
        "exact_attention_replay": winner_rows["execution_ms"],
    }
    profiler_rows = []
    for component, values in parts.items():
        values = values.dropna()
        profiler_rows.append(
            {
                "variant": winner,
                "component": component,
                "samples": len(values),
                "mean_ms": float(values.mean()),
                "median_ms": float(values.median()),
                "p90_ms": _quantile(values, 0.90),
                "status": "STAGE0_ONLY_NO_PRODUCTION_KERNEL",
            }
        )
    profiler = pd.DataFrame(profiler_rows)
    return scoring, execution, profiler


def _plot_benefit(benefit: pd.DataFrame, output: Path) -> None:
    overall = (
        benefit.loc[benefit["stratum"].eq("overall")]
        .sort_values("benefit_retained", ascending=False)
        .head(20)
        .sort_values("benefit_retained")
    )
    figure, axis = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True,
    )
    values = overall["benefit_retained"] * 100
    colors = np.where(values >= 70, "#2a9d8f", "#457b9d")
    axis.barh(overall["variant"], values, color=colors)
    axis.axvline(70, color="#e76f51", linestyle="--", label="GO gate")
    axis.axvline(
        85,
        color="#f4a261",
        linestyle=":",
        label="Strong target",
    )
    axis.set(
        xlabel="Fine8 p90 fidelity benefit retained (%)",
        title="Hierarchical VSA Stage 0 candidates",
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output)
    plt.close(figure)


def _plot_error_distribution(
    error: pd.DataFrame,
    winner: str,
    output: Path,
) -> None:
    pivot = _state_pivot(error, head_scope=False)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )
    for variant, label, color in (
        ("native64_global", "Native VSA64", "#e76f51"),
        (winner, "Best hierarchical", "#457b9d"),
        ("kv8_global", "Pure Fine8", "#2a9d8f"),
    ):
        values = np.sort(pivot[variant].to_numpy())
        percentile = np.linspace(0, 100, len(values))
        axes[0].plot(values, percentile, label=label, color=color)
    axes[0].set(
        xlabel="Call-level p90 relative L2",
        ylabel="Attention states at or below (%)",
        title="Error distribution",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    native = pivot["native64_global"]
    axes[1].scatter(
        native,
        pivot[winner],
        alpha=0.35,
        s=14,
        label="Best hierarchical",
    )
    axes[1].scatter(
        native,
        pivot["kv8_global"],
        alpha=0.25,
        s=14,
        label="Pure Fine8",
    )
    maximum = float(
        max(native.max(), pivot[winner].max(), pivot["kv8_global"].max())
    )
    axes[1].plot([0, maximum], [0, maximum], "--", color="gray")
    axes[1].set(
        xlabel="Native VSA64 p90 relative L2",
        ylabel="Candidate p90 relative L2",
        title="Paired attention states",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output)
    plt.close(figure)


def _plot_sensitive(
    sensitivity: pd.DataFrame,
    winner: str,
    output: Path,
) -> None:
    selected = sensitivity.loc[
        sensitivity["variant"].eq(winner)
    ].copy()
    selected["benefit_percent"] = selected["benefit_retained"] * 100
    figure, axis = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
    )
    axis.bar(
        selected["stratum"],
        selected["benefit_percent"],
        color=np.where(
            selected["row_type"].eq("specified_unit"),
            "#8d99ae",
            "#457b9d",
        ),
    )
    axis.axhline(70, color="#e76f51", linestyle="--", label="GO gate")
    axis.set(
        ylabel="Fine8 benefit retained (%)",
        title="Best candidate on BR-VSA sensitivity strata",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output)
    plt.close(figure)


def _plot_granularity(benefit: pd.DataFrame, output: Path) -> None:
    overall = benefit.loc[benefit["stratum"].eq("overall")]
    matrix = overall.pivot_table(
        index="score_width",
        columns="execution_width",
        values="benefit_retained",
        aggfunc="max",
    ).sort_index()
    figure, axis = plt.subplots(
        figsize=(6.5, 4.8),
        constrained_layout=True,
    )
    image = axis.imshow(
        matrix.to_numpy() * 100,
        cmap="viridis",
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column] * 100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(
        [f"KV{int(value)}" for value in matrix.columns]
    )
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels(
        [f"Score{int(value)}" for value in matrix.index]
    )
    axis.set(
        xlabel="Execution granularity",
        ylabel="Scoring granularity",
        title="Best benefit retained by geometry",
    )
    figure.colorbar(image, ax=axis, label="Benefit retained (%)")
    figure.savefig(output)
    plt.close(figure)


def _plot_quality_latency(
    prior_results: Path,
    output: Path,
) -> pd.DataFrame:
    systems = pd.read_csv(prior_results)
    systems = systems.loc[
        systems["method"].isin(
            [
                "Dense BF16",
                "VSA80",
                "VSA60",
                "VSA40",
                "Adaptive-K",
                "CH-VSA",
                "BR-VSA",
                "Fine-VSA8",
            ]
        )
    ].copy()
    figure, axis = plt.subplots(
        figsize=(8, 5.5),
        constrained_layout=True,
    )
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
        "Hier-VSA: not generated\\n(Stage 0 gate failed)",
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


def _write_reports(
    *,
    repo: Path,
    root: Path,
    prompt_meta: pd.DataFrame,
    candidates: pd.DataFrame,
    benefit: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scoring: pd.DataFrame,
    execution: pd.DataFrame,
    winner: str,
) -> dict[str, Any]:
    calibration = root / "calibration"
    development = root / "development_72"
    winner_candidate = candidates.set_index("variant").loc[winner]
    overall = benefit.loc[
        benefit["stratum"].eq("overall")
    ].set_index("variant")
    winner_benefit = float(overall.loc[winner, "benefit_retained"])
    winner_error = float(overall.loc[winner, "candidate_p90"])
    native_error = float(overall.loc[winner, "native_p90"])
    fine_error = float(overall.loc[winner, "fine8_p90"])
    gate = {
        "calibration_prompts": 8,
        "safe_prompts": int(
            prompt_meta["selection_stratum"].eq("vsa80_safe").sum()
        ),
        "unsafe_prompts": int(
            prompt_meta["selection_stratum"].eq("vsa80_unsafe").sum()
        ),
        "candidate_count": int(
            candidates["candidate_kind"].eq("hierarchical").sum()
        ),
        "winner": winner,
        "winner_score_width": int(winner_candidate["score_width"]),
        "winner_execution_width": int(
            winner_candidate["execution_width"]
        ),
        "winner_aggregation": winner_candidate["aggregation"],
        "winner_soft_native_prior": float(
            winner_candidate["soft_native_prior"]
        ),
        "winner_parent_pool": winner_candidate["parent_pool"],
        "native_p90": native_error,
        "fine8_p90": fine_error,
        "winner_p90": winner_error,
        "winner_benefit_retained": winner_benefit,
        "go_threshold": 0.70,
        "strong_threshold": 0.85,
        "pair_budget_ratio_max": float(
            winner_candidate["actual_pair_budget_ratio_max"]
        ),
        "go": bool(
            winner_benefit >= 0.70
            and winner_candidate["actual_pair_budget_ratio_max"] <= 1.0
        ),
        "reason": (
            f"The best GPU-friendly candidate retained "
            f"{winner_benefit:.2%} of Fine8's p90 fidelity benefit, below "
            "the predeclared 70% generation gate."
        ),
        "decision": DECISION,
    }
    (calibration / "stage0_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )

    strata = benefit.loc[
        benefit["variant"].eq(winner)
    ].set_index("stratum")
    sensitive = sensitivity.loc[
        sensitivity["variant"].eq(winner)
    ].set_index("stratum")
    report = f"""# Hierarchical VSA Stage 0 calibration

- Frozen prompts: 8 (4 VSA80-safe, 4 VSA80-unsafe).
- Attention states: 720 all-head call states and 8,640 head-level states.
- Candidates: 72 hierarchical variants plus Native VSA64 and Pure Fine8.
- Pair budget: every candidate stayed at or below Native VSA80; the winner's
  maximum valid-token ratio was
  {float(winner_candidate['actual_pair_budget_ratio_max']):.6f}.
- Best candidate: `{winner}` — Score8, KV64 execution, max aggregation,
  no native prior, all parents.
- Native / winner / Fine8 mean call-level p90 relative-L2:
  {native_error:.6f} / {winner_error:.6f} / {fine_error:.6f}.
- Fine8 benefit retained: {winner_benefit:.2%}.
- Predeclared GO threshold: 70%; strong target: 85%.

## Native-error strata

| Stratum | Benefit retained | Relative improvement vs native |
|---|---:|---:|
| Top 10% | {strata.loc['top_10pct_native_error', 'benefit_retained']:.2%} | {strata.loc['top_10pct_native_error', 'relative_improvement_vs_native']:.2%} |
| Top 25% | {strata.loc['top_25pct_native_error', 'benefit_retained']:.2%} | {strata.loc['top_25pct_native_error', 'relative_improvement_vs_native']:.2%} |
| Top 50% | {strata.loc['top_50pct_native_error', 'benefit_retained']:.2%} | {strata.loc['top_50pct_native_error', 'relative_improvement_vs_native']:.2%} |
| Bottom 50% | {strata.loc['bottom_50pct_native_error', 'benefit_retained']:.2%} | {strata.loc['bottom_50pct_native_error', 'relative_improvement_vs_native']:.2%} |

## BR-VSA sensitivity strata

The winner retained
{sensitive.loc['top_20pct', 'benefit_retained']:.2%} of Fine8's benefit in
the top 20% BR-sensitive units and
{sensitive.loc['bottom_80pct', 'benefit_retained']:.2%} in the bottom 80%.
It helps difficult units, but not enough to cross the global gate.

## Decision

Stage 1 kernel work and the 72-video generation/VBench run were not started.
This is the required fail-fast outcome, not missing work.

{DECISION}
"""
    (calibration / "REPORT.md").write_text(report)

    reason = (
        "NOT_RUN_STAGE0_STOP: best benefit retention "
        f"{winner_benefit:.2%} < 70% gate"
    )
    placeholders = {
        "results.csv": [
            {
                "method": "Hier-VSA",
                "status": "NOT_RUN_STAGE0_STOP",
                "reason": reason,
                "unsafe": np.nan,
                "repaired": np.nan,
                "new_failures": np.nan,
                "median_e2e_ms": np.nan,
                "median_attention_ms": np.nan,
            }
        ],
        "quality.csv": [
            {
                "status": "NOT_RUN_STAGE0_STOP",
                "reason": reason,
            }
        ],
        "latency.csv": [
            {
                "status": "NOT_RUN_STAGE0_STOP",
                "reason": reason,
            }
        ],
        "repaired.csv": [
            {
                "status": "NOT_RUN_STAGE0_STOP",
                "reason": reason,
            }
        ],
        "regressions.csv": [
            {
                "status": "NOT_RUN_STAGE0_STOP",
                "reason": reason,
            }
        ],
    }
    for name, rows in placeholders.items():
        pd.DataFrame(rows).to_csv(development / name, index=False)
    development_report = f"""# Hierarchical VSA development-72 report

The 72-video run was not authorized by the predeclared Stage 0 gate.
The best candidate retained {winner_benefit:.2%} of Pure Fine8's p90
fidelity benefit, below the required 70%.

Therefore unsafe count, repairs, regressions, VBench metrics, and production
latency are intentionally unmeasured.

{DECISION}
"""
    (development / "REPORT.md").write_text(development_report)

    score8 = scoring.set_index("component").loc["score8"]
    score16 = scoring.set_index("component").loc["score16"]
    winner_exec = execution.set_index("variant").loc[winner]
    final = f"""# Final result — Hierarchical Fine Scoring with Coarse Execution

## Outcome

The hypothesis is partially supported but fails the predeclared progression
gate. Fine scoring improves which KV64 blocks are selected, especially in
the worst native-error states, but it preserves only {winner_benefit:.2%} of
Pure Fine8's overall p90 fidelity gain. The required threshold was 70%.

## Frozen winner

`{winner}`:

- scoring width: {int(winner_candidate['score_width'])}
- execution width: {int(winner_candidate['execution_width'])}
- aggregation: {winner_candidate['aggregation']}
- native-prior weight: {float(winner_candidate['soft_native_prior']):.2f}
- candidate pool: {winner_candidate['parent_pool']}
- exact-pair ratio versus VSA80: <= {float(winner_candidate['actual_pair_budget_ratio_max']):.6f}

## Required questions

1. **Does fine sub-block scoring improve KV64 ranking?** Yes. The winner
   lowers p90 relative-L2 from {native_error:.6f} to {winner_error:.6f}
   ({(native_error - winner_error) / native_error:.2%}).
2. **Score8 or Score16?** Score8. Best Score8→KV64 retention is
   {winner_benefit:.2%}; the best Score16→KV64 result is
   {benefit.loc[(benefit['stratum'].eq('overall')) & (benefit['score_width'].eq(16)) & (benefit['execution_width'].eq(64)), 'benefit_retained'].max():.2%}.
3. **Best aggregation?** Max. Hierarchical mass, plain log-sum-exp, and
   top-2 mean are weaker.
4. **Fine8 gain retained?** {winner_benefit:.2%} overall.
5. **Does the gain persist in worst states?** Yes: top-10/25/50% retention
   is {strata.loc['top_10pct_native_error', 'benefit_retained']:.2%} /
   {strata.loc['top_25pct_native_error', 'benefit_retained']:.2%} /
   {strata.loc['top_50pct_native_error', 'benefit_retained']:.2%}.
6. **Does it disproportionately help BR-sensitive units?** It helps them,
   but retention is {sensitive.loc['top_20pct', 'benefit_retained']:.2%}
   in the top 20%, versus
   {sensitive.loc['bottom_80pct', 'benefit_retained']:.2%} in the bottom
   80%; this does not explain enough of Fine8's total advantage.
7. **Does a soft native prior help?** No. The winner uses lambda=0; 0.25
   and 0.50 reduce benefit.
8. **Does top-300 pruning retain most benefit?** Yes in the literal sense:
   for the winner's Score8/KV64/max geometry, top-300 retains
   {overall.loc['s8_e64_max_top300_l0.00', 'benefit_retained'] / winner_benefit:.2%}
   of the all-parent hierarchical benefit. Its absolute Fine8 benefit
   retention is only
   {overall.loc['s8_e64_max_top300_l0.00', 'benefit_retained']:.2%}, so it
   remains well below the generation gate.
9. **Exact pair count <= VSA80?** Yes; all measured ratios are <=1.0.
10. **Aggregate sparsity still ~80%?** Yes by the exact-pair definition:
    KV64 uses 125×64 nominal tokens and KV128 uses 62×128.
11. **Can the native KV64 kernel be reused?** Structurally yes for the
    winner, because execution remains KV64. Production integration was not
    performed after the gate failed.
12. **Fine-scoring overhead?** Stage 0 Score8 matrix scoring averaged
    {score8['mean_ms']:.4f} ms per captured call; Score16 averaged
    {score16['mean_ms']:.4f} ms. For the winner, scoring plus aggregation
    plus selection averaged
    {float(winner_candidate['fine_score_ms_mean'] + winner_candidate['aggregation_ms_mean'] + winner_candidate['selection_ms_mean']):.4f}
    ms. These exclude production integration.
13. **Exact-attention latency?** Offline exact-attention replay for the
    winner averaged {winner_exec['mean_ms']:.4f} ms per captured call.
    No production end-to-end latency was measured.
14. **VSA80 failures repaired?** Not measured; the 72-video run was
    correctly skipped.
15. **New failures?** Not measured for the same reason.
16. **Does it approach Fine8 quality?** It moves substantially toward it,
    but misses the minimum 70% retention criterion.
17. **Does it approach VSA80 systems performance?** The KV64 geometry is
    compatible, but this was not established in production.
18. **Does it beat VSA60/VSA40 on Pareto?** Not demonstrated; no generation
    quality/latency point exists.
19. **Is Hilbert3D ordering necessary?** Unknown and not justified by this
    failed primary gate; it was intentionally not opened.
20. **Proceed to full VBench?** No.

## Reproducibility

- branch: `{_git_value(repo, 'branch', '--show-current')}`
- source commit at analysis: `{_git_value(repo, 'rev-parse', 'HEAD')}`
- calibration prompts: 8 frozen prompts from Fine-VSA Stage 0
- generated on: 2026-08-29 UTC

{DECISION}
"""
    (root / "FINAL_RESULT.md").write_text(final)
    return gate


def _write_manifest(root: Path, gate: dict[str, Any]) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "run/phase0" in path.as_posix():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    manifest = {
        "experiment": "hierarchical_vsa",
        "date_utc": "2026-08-29",
        "stage0_go": gate["go"],
        "decision": DECISION,
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def analyze(
    *,
    repo: Path,
    root: Path,
    prompt_ids_path: Path,
    sensitivity_path: Path,
    prior_results_path: Path,
) -> dict[str, Any]:
    calibration = root / "calibration"
    kernel = root / "kernel"
    development = root / "development_72"
    figures = root / "figures"
    method_notes = root / "method_notes"
    for directory in (
        calibration,
        kernel,
        development,
        figures,
        method_notes,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    source_note = (
        repo
        / "research/hierarchical_vsa/method_notes/"
        "hierarchical_scoring.md"
    )
    (method_notes / "hierarchical_scoring.md").write_text(
        source_note.read_text()
    )
    prompt_meta = pd.DataFrame(json.loads(prompt_ids_path.read_text()))
    (calibration / "prompt_ids.json").write_text(
        json.dumps(
            prompt_meta.to_dict(orient="records"),
            indent=2,
        )
        + "\n"
    )

    raw, prompt_meta = _load_calibration(root, prompt_ids_path)
    raw.to_parquet(calibration / "raw_stats.parquet", index=False)
    error = raw.loc[
        raw["event_type"].eq("hierarchical_vsa_error")
        & raw["scope"].eq("all_heads_query_blocks")
    ].copy()
    if error["variant"].nunique() != 74:
        raise RuntimeError("Expected 74 total variants")
    if error["job_id"].nunique() != 8:
        raise RuntimeError("Expected 8 calibration jobs")
    if float(error["actual_pair_budget_ratio"].max()) > 1.0 + 1e-9:
        raise RuntimeError("Pair budget violation")

    candidates = _candidate_error(error)
    candidates.to_csv(calibration / "candidate_error.csv", index=False)
    benefit = _benefit_table(error, candidates)
    benefit.to_csv(calibration / "benefit_retained.csv", index=False)
    ablation = benefit.loc[
        benefit["stratum"].eq("overall")
    ].sort_values("benefit_retained", ascending=False)
    ablation.to_csv(
        calibration / "aggregation_ablation.csv",
        index=False,
    )
    sensitivity = _sensitivity_table(
        raw.loc[
            raw["event_type"].eq("hierarchical_vsa_error")
        ],
        benefit,
        sensitivity_path,
    )
    sensitivity.to_csv(
        calibration / "sensitivity_stratified.csv",
        index=False,
    )

    winner = (
        benefit.loc[benefit["stratum"].eq("overall")]
        .sort_values("benefit_retained", ascending=False)
        .iloc[0]["variant"]
    )
    if winner != WINNER_FALLBACK:
        raise RuntimeError(f"Unexpected winner: {winner}")
    scoring, execution, profiler = _benchmark_tables(
        raw,
        error,
        winner,
    )
    scoring.to_csv(kernel / "scoring_benchmark.csv", index=False)
    execution.to_csv(kernel / "execution_benchmark.csv", index=False)
    profiler.to_csv(kernel / "profiler_summary.csv", index=False)

    _plot_benefit(benefit, figures / "benefit_retained.pdf")
    _plot_error_distribution(
        error,
        winner,
        figures / "error_distribution.pdf",
    )
    _plot_sensitive(
        sensitivity,
        winner,
        figures / "sensitive_units.pdf",
    )
    _plot_granularity(
        benefit,
        figures / "scoring_vs_execution_granularity.pdf",
    )
    systems = _plot_quality_latency(
        prior_results_path,
        figures / "quality_latency_pareto.pdf",
    )
    systems.to_csv(root / "published_systems_reference.csv", index=False)

    gate = _write_reports(
        repo=repo,
        root=root,
        prompt_meta=prompt_meta,
        candidates=candidates,
        benefit=benefit,
        sensitivity=sensitivity,
        scoring=scoring,
        execution=execution,
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
        default=Path("artifacts/hierarchical_vsa"),
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
    gate = analyze(
        repo=args.repo.resolve(),
        root=args.root.resolve(),
        prompt_ids_path=args.prompt_ids.resolve(),
        sensitivity_path=args.sensitivity.resolve(),
        prior_results_path=args.prior_results.resolve(),
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
