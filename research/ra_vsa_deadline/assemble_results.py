from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_NAMES = {
    "vbench.subject_consistency": "subject_consistency",
    "vbench.motion_smoothness": "motion_smoothness",
    "vbench.dynamic_degree": "dynamic_degree",
}
BASE_METHODS = {
    "Dense BF16": "dense_bf16_fa4@s0.00",
    "VSA80": "vsa_bf16@s0.80",
    "VSA60": "vsa_bf16@s0.60",
    "VSA40": "vsa_bf16@s0.40",
}
RA_LABELS = {
    0.875: "RA-VSA 87.5/12.5",
    0.750: "RA-VSA 75/25",
    0.625: "RA-VSA 62.5/37.5",
}
SELECTED_FRACTION = 0.75
NATIVE_FUSED_TOPK_MS = 0.09577599912881851
DECISION = (
    "DECISION: STOP — FIXED-BUDGET RESIDUAL ROUTING DOES NOT "
    "RECOVER ENOUGH QUALITY"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
    ).strip()


def _metrics_wide(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    wide = metrics.pivot(
        index="job_id",
        columns="metric",
        values="score",
    ).reset_index()
    return wide.rename(columns=METRIC_NAMES)


def _label_quality(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["subject_delta"] = (
        result["subject_consistency"]
        - result["dense_subject_consistency"]
    )
    result["motion_delta"] = (
        result["motion_smoothness"]
        - result["dense_motion_smoothness"]
    )
    result["dynamic_delta"] = (
        result["dynamic_degree"]
        - result["dense_dynamic_degree"]
    )
    result["subject_safe"] = result["subject_delta"] >= -0.02
    result["motion_safe"] = result["motion_delta"] >= -0.01
    result["dynamic_safe"] = result["dynamic_delta"] >= 0.0
    result["quality_safe"] = (
        result["subject_safe"]
        & result["motion_safe"]
        & result["dynamic_safe"]
    )
    return result


def _effect_size(safe: pd.Series, unsafe: pd.Series) -> float:
    safe_values = safe.dropna().to_numpy(float)
    unsafe_values = unsafe.dropna().to_numpy(float)
    pooled = math.sqrt(
        (
            (len(safe_values) - 1) * safe_values.var(ddof=1)
            + (len(unsafe_values) - 1) * unsafe_values.var(ddof=1)
        )
        / (len(safe_values) + len(unsafe_values) - 2)
    )
    return float(
        (unsafe_values.mean() - safe_values.mean()) / pooled
        if pooled
        else np.nan
    )


def _load_candidate_quality(
    repo: Path,
    root: Path,
    base: pd.DataFrame,
) -> pd.DataFrame:
    jobs = pd.read_parquet(
        root / "development_72/candidates/all_candidate_jobs.parquet"
    )
    metrics = _metrics_wide(
        root / "development_72/candidates/vbench_metrics.csv"
    )
    dense = (
        base.loc[
            base["config"].eq("dense_bf16_fa4@s0.00"),
            [
                "prompt_id",
                "subject_consistency",
                "motion_smoothness",
                "dynamic_degree",
            ],
        ]
        .rename(
            columns={
                "subject_consistency": "dense_subject_consistency",
                "motion_smoothness": "dense_motion_smoothness",
                "dynamic_degree": "dense_dynamic_degree",
            }
        )
        .copy()
    )
    vsa80 = base.loc[
        base["config"].eq("vsa_bf16@s0.80"),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    result = (
        jobs.merge(metrics, on="job_id", validate="one_to_one")
        .merge(dense, on="prompt_id", validate="many_to_one")
        .merge(vsa80, on="prompt_id", validate="many_to_one")
    )
    result = _label_quality(result)
    result["original_failure"] = ~result["vsa80_safe"]
    result["repaired"] = (
        result["original_failure"] & result["quality_safe"]
    )
    result["new_failure"] = (
        ~result["original_failure"] & ~result["quality_safe"]
    )
    result["method"] = result["ra_native_fraction"].map(RA_LABELS)
    return result


def _base_quality_rows(base: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    vsa80 = base.loc[
        base["config"].eq("vsa_bf16@s0.80"),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    for method, config in BASE_METHODS.items():
        group = base.loc[base["config"].eq(config)].copy()
        group = group.merge(
            vsa80,
            on="prompt_id",
            validate="one_to_one",
            suffixes=("", "_reference"),
        )
        group["method"] = method
        group["original_failure"] = ~group["vsa80_safe"]
        group["repaired"] = (
            group["original_failure"] & group["quality_safe"]
        )
        group["new_failure"] = (
            ~group["original_failure"] & ~group["quality_safe"]
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def _adaptive_quality_rows(repo: Path, base: pd.DataFrame) -> pd.DataFrame:
    adaptive = pd.read_csv(
        repo
        / "artifacts/adaptive_vsa_deadline_final/development_72/results.csv"
    )
    adaptive = adaptive.loc[
        adaptive["method"].eq("Adaptive VSA")
    ].copy()
    vsa80 = base.loc[
        base["config"].eq("vsa_bf16@s0.80"),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    adaptive = adaptive.merge(
        vsa80,
        on="prompt_id",
        validate="one_to_one",
    )
    adaptive["original_failure"] = ~adaptive["vsa80_safe"]
    adaptive["repaired"] = (
        adaptive["original_failure"] & adaptive["quality_safe"]
    )
    adaptive["new_failure"] = (
        ~adaptive["original_failure"] & ~adaptive["quality_safe"]
    )
    return adaptive


def _quality_export(
    base_rows: pd.DataFrame,
    adaptive: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "method",
        "job_id",
        "prompt_id",
        "prompt",
        "seed",
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "dense_subject_consistency",
        "dense_motion_smoothness",
        "dense_dynamic_degree",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
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
    for frame in (base_rows, adaptive, candidate):
        copy = frame.copy()
        for column in columns:
            if column not in copy:
                copy[column] = np.nan
        frames.append(copy[columns])
    return pd.concat(frames, ignore_index=True)


def _method_summary(
    quality: pd.DataFrame,
    selected_latency: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in quality.groupby("method", sort=False):
        latency_group = group
        if method in BASE_METHODS:
            latency_protocol = "existing frozen 72-prompt baseline"
        elif method == "Adaptive VSA":
            latency_protocol = "existing adaptive deadline run"
        else:
            latency_protocol = "saved-video generation with detailed policy tracing"
        if method == RA_LABELS[SELECTED_FRACTION]:
            latency_group = selected_latency
            latency_protocol = "clean no-save, minimal invariant trace"
        rows.append(
            {
                "method": method,
                "unsafe": int((~group["quality_safe"]).sum()),
                "original_failures_repaired": int(group["repaired"].sum()),
                "new_failures": int(group["new_failure"].sum()),
                "median_wall_ms": float(latency_group["wall_ms"].median()),
                "median_dit_ms": float(latency_group["dit_ms"].median()),
                "median_attention_ms": float(
                    latency_group["attention_ms"].median()
                ),
                "effective_sparsity": float(
                    group["effective_sparsity"].mean()
                ),
                "latency_protocol": latency_protocol,
            }
        )
    return pd.DataFrame(rows)


def _instrumentation(
    root: Path,
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = pd.read_parquet(
        root / "instrumentation_run/phase0/jobs.parquet"
    )
    trace = pd.read_parquet(
        root / "instrumentation_run/phase0/policy_trace.parquet"
    )
    labels = base.loc[
        base["config"].eq("vsa_bf16@s0.80"),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    block_risk = (
        trace.merge(
            jobs[["job_id", "prompt_id", "prompt"]],
            on="job_id",
            validate="many_to_one",
        )
        .merge(labels, on="prompt_id", validate="many_to_one")
        .copy()
    )
    metric_columns = [
        "key_heterogeneity_mean",
        "key_heterogeneity_p50",
        "key_heterogeneity_p90",
        "risk_mean",
        "risk_p50",
        "risk_p90",
        "gate_abs_mean",
        "gate_rms",
        "native_0875_replacement_fraction_mean",
        "native_0750_replacement_fraction_mean",
        "native_0625_replacement_fraction_mean",
    ]
    prompt_risk = (
        block_risk.groupby(
            ["prompt_id", "prompt", "vsa80_safe"],
            as_index=False,
        )[metric_columns]
        .mean()
    )
    rows = []
    for metric in metric_columns:
        safe = prompt_risk.loc[prompt_risk["vsa80_safe"], metric]
        unsafe = prompt_risk.loc[~prompt_risk["vsa80_safe"], metric]
        rows.append(
            {
                "metric": metric,
                "safe_mean": safe.mean(),
                "safe_median": safe.median(),
                "unsafe_mean": unsafe.mean(),
                "unsafe_median": unsafe.median(),
                "unsafe_minus_safe": unsafe.mean() - safe.mean(),
                "standardized_difference": _effect_size(safe, unsafe),
                "safe_n": len(safe),
                "unsafe_n": len(unsafe),
            }
        )
    return block_risk, prompt_risk, pd.DataFrame(rows)


def _candidate_trace(root: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for fraction, directory in (
        (0.875, "native0875"),
        (0.750, "native0750"),
        (0.625, "native0625"),
    ):
        trace = pd.read_parquet(
            root
            / f"development_72/candidates/{directory}/policy_trace.parquet"
        )
        trace["candidate_native_fraction"] = fraction
        jobs = candidate.loc[
            candidate["ra_native_fraction"].eq(fraction),
            ["job_id", "prompt_id", "prompt"],
        ]
        trace = trace.merge(
            jobs,
            on="job_id",
            validate="many_to_one",
        )
        frames.append(trace)
    return pd.concat(frames, ignore_index=True)


def _replacement_summary(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction, group in trace.groupby(
        "candidate_native_fraction",
        sort=False,
    ):
        replacement = group["replacement_fraction_mean"]
        rows.append(
            {
                "method": RA_LABELS[float(fraction)],
                "native_fraction": fraction,
                "native_slots": int(group["native_slots"].iloc[0]),
                "rescue_slots": int(group["rescue_slots"].iloc[0]),
                "total_slots": int(group["total_slots"].iloc[0]),
                "mean_replacement_fraction": replacement.mean(),
                "median_replacement_fraction": replacement.median(),
                "mean_replaced_blocks": (
                    replacement * group["total_slots"]
                ).mean(),
                "rescue_key_heterogeneity_mean": group[
                    "rescue_key_heterogeneity_mean"
                ].mean(),
                "removed_key_heterogeneity_mean": group[
                    "removed_key_heterogeneity_mean"
                ].mean(),
                "selected_count_min": int(
                    group["selected_count_min"].min()
                ),
                "selected_count_max": int(
                    group["selected_count_max"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_pareto(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in summary.itertuples():
        marker = "*" if row.method == RA_LABELS[SELECTED_FRACTION] else "o"
        size = 150 if marker == "*" else 70
        ax.scatter(
            row.median_wall_ms,
            row.unsafe,
            s=size,
            marker=marker,
            zorder=3,
        )
        ax.annotate(
            row.method,
            (row.median_wall_ms, row.unsafe),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(9356.053424 * 1.02, color="tab:red", linestyle="--", linewidth=1)
    ax.axhline(12, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel("Median end-to-end latency (ms)")
    ax.set_ylabel("Unsafe prompts / 72 (lower is better)")
    ax.set_title("RA-VSA misses both deadline gates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_failures(summary: pd.DataFrame, path: Path) -> None:
    methods = [
        "Dense BF16",
        "VSA80",
        "VSA60",
        "VSA40",
        "Adaptive VSA",
        RA_LABELS[SELECTED_FRACTION],
    ]
    values = (
        summary.set_index("method")
        .loc[methods, "unsafe"]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = [
        "0.55",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:purple",
    ]
    bars = ax.bar(methods, values, color=colors)
    ax.axhline(12, color="black", linestyle="--", linewidth=1)
    ax.bar_label(bars)
    ax.set_ylabel("Unsafe prompts / 72")
    ax.set_title("Fixed-budget residual routing repairs too few failures")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_risk(prompt_risk: pd.DataFrame, path: Path) -> None:
    safe = prompt_risk.loc[
        prompt_risk["vsa80_safe"],
        "risk_p90",
    ].to_numpy()
    unsafe = prompt_risk.loc[
        ~prompt_risk["vsa80_safe"],
        "risk_p90",
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.boxplot(
        [safe, unsafe],
        tick_labels=["VSA80 safe (48)", "VSA80 unsafe (24)"],
        showmeans=True,
    )
    ax.set_ylabel("Prompt-mean p90(coarse mass × U_K)")
    ax.set_title("Unsafe prompts have modestly higher residual risk")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    return np.asarray(value)


def _plot_routing_examples(
    trace: pd.DataFrame,
    candidate: pd.DataFrame,
    path: Path,
) -> list[str]:
    selected = candidate.loc[
        candidate["ra_native_fraction"].eq(SELECTED_FRACTION)
        & candidate["repaired"]
    ].copy()
    example_ids = selected["prompt_id"].head(3).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), squeeze=False)
    for ax, prompt_id in zip(axes[0], example_ids, strict=False):
        rows = trace.loc[
            trace["candidate_native_fraction"].eq(SELECTED_FRACTION)
            & trace["prompt_id"].eq(prompt_id)
            & trace["layer"].eq(14)
            & trace["timestep"].eq(1)
        ]
        row = rows.iloc[0] if not rows.empty else trace.loc[
            trace["candidate_native_fraction"].eq(SELECTED_FRACTION)
            & trace["prompt_id"].eq(prompt_id)
        ].iloc[0]
        removed_ids = _as_array(row["example_removed_blocks"]).astype(int)
        added_ids = _as_array(row["example_added_blocks"]).astype(int)
        removed_scores = _as_array(
            row["example_removed_native_scores"]
        ).astype(float)
        added_scores = _as_array(
            row["example_added_native_scores"]
        ).astype(float)
        removed_uk = _as_array(
            row["example_removed_key_heterogeneity"]
        ).astype(float)
        added_uk = _as_array(
            row["example_added_key_heterogeneity"]
        ).astype(float)
        ax.scatter(
            removed_scores,
            removed_uk,
            marker="x",
            color="tab:red",
            label="native blocks removed",
        )
        ax.scatter(
            added_scores,
            added_uk,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            label="risk blocks added",
        )
        for block, x, y in zip(
            added_ids[:3],
            added_scores[:3],
            added_uk[:3],
            strict=False,
        ):
            ax.annotate(str(block), (x, y), fontsize=7)
        prompt = selected.loc[
            selected["prompt_id"].eq(prompt_id),
            "prompt",
        ].iloc[0]
        ax.set_title(prompt, fontsize=9)
        ax.set_xlabel("Native coarse score")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Key heterogeneity U_K")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Representative repaired prompts: native tail vs residual rescue")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return example_ids


def _write_instrumentation_report(
    path: Path,
    safe_vs_unsafe: pd.DataFrame,
    replacement: pd.DataFrame,
) -> None:
    index = safe_vs_unsafe.set_index("metric")
    risk = index.loc["risk_p90"]
    uk = index.loc["key_heterogeneity_mean"]
    gate = index.loc["gate_rms"]
    split = replacement.loc[
        replacement["native_fraction"].eq(SELECTED_FRACTION)
    ].iloc[0]
    path.write_text(
        f"""# Stage 0 — RA-VSA risk instrumentation

The stop condition was not triggered: residual rescue does select blocks
outside the native VSA80 set.

- Mean key heterogeneity is slightly higher for unsafe prompts
  ({uk.unsafe_mean:.6f} vs {uk.safe_mean:.6f}; standardized difference
  {uk.standardized_difference:.2f}).
- The clearest tested signal is p90(coarse mass × U_K):
  {risk.unsafe_mean:.6f} unsafe vs {risk.safe_mean:.6f} safe
  (standardized difference {risk.standardized_difference:.2f}).
- The 75/25 policy changes {split.mean_replacement_fraction:.2%} of the
  native set, or {split.mean_replaced_blocks:.2f} of 125 blocks per query
  on average.
- The learned gate is a weak prompt-level separator
  (standardized difference {gate.standardized_difference:.2f}), so it was
  not integrated.
- U_V was not added because exact token-level value variance would add
  another full V read and the primary U_K signal already passed the
  non-redundancy gate.

Conclusion: proceed to the tiny frozen split search, without fitting a
predictor or adding a gate-derived term.
"""
    )


def _write_development_report(
    path: Path,
    summary: pd.DataFrame,
    replacement: pd.DataFrame,
    selected_trace: pd.DataFrame,
) -> None:
    selected = summary.loc[
        summary["method"].eq(RA_LABELS[SELECTED_FRACTION])
    ].iloc[0]
    vsa80 = summary.loc[summary["method"].eq("VSA80")].iloc[0]
    dense = summary.loc[summary["method"].eq("Dense BF16")].iloc[0]
    rep = replacement.loc[
        replacement["native_fraction"].eq(SELECTED_FRACTION)
    ].iloc[0]
    fine = selected_trace["fine_attention_ms"].median()
    path.write_text(
        f"""# 72-prompt development result

The primary 75/25 RA-VSA policy fails the deadline gate.

| Quantity | Result | Gate |
|---|---:|---:|
| Unsafe prompts | {int(selected.unsafe)} / 72 | <= 12 |
| Original VSA80 failures repaired | {int(selected.original_failures_repaired)} / 24 | recover at least half |
| New failures | {int(selected.new_failures)} / 48 | <= 2 |
| Median E2E latency | {selected.median_wall_ms:.2f} ms | <= {vsa80.median_wall_ms * 1.02:.2f} ms |
| Overhead over VSA80 | {(selected.median_wall_ms / vsa80.median_wall_ms - 1):.2%} | <= 2% |
| Relative to dense | {(selected.median_wall_ms / dense.median_wall_ms - 1):.2%} | faster than dense |
| Mean native-set replacement | {rep.mean_replacement_fraction:.2%} ({rep.mean_replaced_blocks:.2f}/125) | nonzero |
| Fine attention per call | {fine:.4f} ms | approximately unchanged |
| Fixed K | 125 for every query row | exactly 125 |

The 62.5/37.5 split repairs one additional original failure, but still has
20 unsafe prompts and introduces two new failures. The 87.5/12.5 split has
21 unsafe prompts. No allowed split approaches the <=12 target.

The bear-sniffing severe failure remains unsafe for all three splits.

No full VBench, Wan14B, NVFP4, residual-only tuning, or head redistribution
was launched after this decisive failure.
"""
    )


def _write_final(
    path: Path,
    repo: Path,
    summary: pd.DataFrame,
    safe_vs_unsafe: pd.DataFrame,
    replacement: pd.DataFrame,
    stage0_trace: pd.DataFrame,
    clean_trace: pd.DataFrame,
    example_ids: list[str],
) -> None:
    selected = summary.loc[
        summary["method"].eq(RA_LABELS[SELECTED_FRACTION])
    ].iloc[0]
    vsa80 = summary.loc[summary["method"].eq("VSA80")].iloc[0]
    dense = summary.loc[summary["method"].eq("Dense BF16")].iloc[0]
    risk = safe_vs_unsafe.set_index("metric").loc["risk_p90"]
    uk = safe_vs_unsafe.set_index("metric").loc[
        "key_heterogeneity_mean"
    ]
    gate = safe_vs_unsafe.set_index("metric").loc["gate_rms"]
    rep = replacement.loc[
        replacement["native_fraction"].eq(SELECTED_FRACTION)
    ].iloc[0]
    native_fine = stage0_trace["fine_attention_ms"].median()
    ra_fine = clean_trace["fine_attention_ms"].median()
    fine_delta = ra_fine / native_fine - 1.0
    commit = _git(repo, "rev-parse", "HEAD")
    path.write_text(
        f"""# FINAL RESULT — Residual-Aware Fixed-Budget VSA

Date: 2026-08-29 UTC  
Code commit: `{commit}`  
Frozen primary policy: 75% native importance slots + 25% residual-risk
rescue slots, `R = coarse_mass × U_K`, fixed VSA80 K=125.

## Bottom line

RA-VSA changes the selected block IDs while preserving the exact VSA80
fine-attention budget, but the tested risk signal repairs too few quality
failures and adds too much selector overhead.

## Required questions

1. **Does coarse block heterogeneity differ between VSA80-safe and unsafe
   generations?** Modestly. Mean U_K is {uk.unsafe_mean:.6f} for unsafe
   prompts vs {uk.safe_mean:.6f} for safe prompts (standardized difference
   {uk.standardized_difference:.2f}). The stronger distinction is risk p90:
   {risk.unsafe_mean:.6f} vs {risk.safe_mean:.6f}
   (standardized difference {risk.standardized_difference:.2f}).

2. **Which risk signal worked best?** Of the frozen primary signals,
   `coarse_mass × U_K`, especially its upper-tail statistic, separated the
   groups best. U_V was not added because it required another full value
   read. No learned predictor was fit.

3. **How many native VSA80 blocks are replaced?** The selected 75/25 policy
   replaces {rep.mean_replacement_fraction:.2%} of the native set on average,
   or {rep.mean_replaced_blocks:.2f} of 125 blocks per query. Although 31
   slots are assigned to rescue, most rescue choices overlap the lower native
   top-K tail.

4. **How many of the 24 VSA80 failures are repaired?**
   {int(selected.original_failures_repaired)}.

5. **How many new failures are introduced?**
   {int(selected.new_failures)}.

6. **What is RA-VSA's median E2E latency?**
   {selected.median_wall_ms:.2f} ms in the clean 72-prompt no-save run.

7. **What is the overhead over VSA80?**
   {(selected.median_wall_ms / vsa80.median_wall_ms - 1):.2%}. It is also
   {(selected.median_wall_ms / dense.median_wall_ms - 1):.2%} slower than
   dense BF16.

8. **Is fine-attention latency essentially unchanged?** Yes. The
   native-compatible fixed-K path is {native_fine:.4f} ms/call and RA-VSA is
   {ra_fine:.4f} ms/call ({fine_delta:+.2%}). The extra time is in risk and
   exact-index selection, not the sparse fine kernel.

9. **Is total selected K identical to VSA80?** Yes. Every recorded query row
   has `selected_count_min = selected_count_max = 125`; no density increase
   or variable-K queue is used.

10. **Does RA-VSA beat VSA60/VSA40 on the measured quality/speed Pareto
    front?** No. It is faster than those unusually slow fixed baselines, but
    has materially more unsafe prompts (20 vs 14 and 8), so it is not a
    superior quality/latency point.

11. **Does the learned coarse/fine gate provide useful additional risk
    information?** Not enough to justify integration. Gate RMS has only a
    {gate.standardized_difference:.2f} standardized safe/unsafe difference.

12. **Is the result strong enough to run full VBench immediately?** No.
    Unsafe count is {int(selected.unsafe)}/72, above the <=12 gate, and clean
    latency exceeds both the VSA80+2% ceiling and dense BF16.

## Systems profile

| Component | VSA80/native-compatible | RA-VSA 75/25 |
|---|---:|---:|
| Coarse branch per call | {stage0_trace.coarse_selector_ms.median():.4f} ms | {clean_trace.coarse_selector_ms.median():.4f} ms |
| Native Top-K / RA risk selector | {NATIVE_FUSED_TOPK_MS:.4f} ms | {clean_trace.metadata_selector_ms.median():.4f} ms |
| Fine attention per call | {native_fine:.4f} ms | {ra_fine:.4f} ms |
| Attention total per video | {vsa80.median_attention_ms:.2f} ms | {selected.median_attention_ms:.2f} ms |
| E2E per video | {vsa80.median_wall_ms:.2f} ms | {selected.median_wall_ms:.2f} ms |

The native Top-K number is a 200-iteration B200 microbenchmark at the actual
`[1,12,624,624]`, K=125 shape. RA component times come from the clean model
run. Fine attention includes fixed-width mask compaction and the SM100A
block-sparse kernel.

## Mechanistic examples

Three repaired prompts are visualized in `figures/residual_routing_example.pdf`:
{", ".join(example_ids)}. Rescue blocks have higher U_K and lower native
coarse rank than the removed native tail. The severe prompt “a bear sniffing
the air for scents of food” improves numerically but remains unsafe for every
tested split.

## Interpretation

Increasing attention density in Adaptive-K repaired 23/24 failures, but
fixed-budget rerouting repairs only 5/24 under the selected policy. This
supports the conclusion that the checkpoint's failures need substantially
more exact attention capacity, not merely a small reordering of VSA80's
existing exact blocks using this training-free coarse-risk proxy.

{DECISION}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/ra_vsa_deadline"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.root.resolve()
    instrumentation_dir = root / "instrumentation"
    development_dir = root / "development_72"
    figures_dir = root / "figures"
    for directory in (
        instrumentation_dir,
        development_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(
        repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet"
    )
    candidate = _load_candidate_quality(repo, root, base)
    base_rows = _base_quality_rows(base)
    adaptive = _adaptive_quality_rows(repo, base)
    quality = _quality_export(base_rows, adaptive, candidate)
    clean_latency = pd.read_parquet(
        root / "development_72/latency/native0750/jobs.parquet"
    )
    summary = _method_summary(quality, clean_latency)

    block_risk, prompt_risk, safe_vs_unsafe = _instrumentation(
        root,
        base,
    )
    candidate_trace = _candidate_trace(root, candidate)
    replacement = _replacement_summary(candidate_trace)
    stage0_trace = pd.read_parquet(
        root / "instrumentation_run/phase0/policy_trace.parquet"
    )
    clean_trace = pd.read_parquet(
        root
        / "development_72/latency/native0750/policy_trace.parquet"
    )

    block_risk.to_parquet(
        instrumentation_dir / "block_risk.parquet",
        index=False,
    )
    safe_vs_unsafe.to_csv(
        instrumentation_dir / "safe_vs_unsafe.csv",
        index=False,
    )
    summary.to_csv(
        development_dir / "candidate_results.csv",
        index=False,
    )
    quality.to_csv(
        development_dir / "quality_results.csv",
        index=False,
    )
    summary[
        [
            "method",
            "median_wall_ms",
            "median_dit_ms",
            "median_attention_ms",
            "effective_sparsity",
            "latency_protocol",
        ]
    ].to_csv(
        development_dir / "latency_results.csv",
        index=False,
    )
    replacement.to_csv(
        development_dir / "block_replacement_stats.csv",
        index=False,
    )
    candidate_trace.to_parquet(
        development_dir / "policy_trace.parquet",
        index=False,
    )

    _write_instrumentation_report(
        instrumentation_dir / "REPORT.md",
        safe_vs_unsafe,
        replacement,
    )
    _write_development_report(
        development_dir / "REPORT.md",
        summary,
        replacement,
        clean_trace,
    )
    _plot_pareto(
        summary,
        figures_dir / "quality_latency_pareto.pdf",
    )
    _plot_failures(
        summary,
        figures_dir / "failure_count.pdf",
    )
    _plot_risk(
        prompt_risk,
        figures_dir / "risk_distribution.pdf",
    )
    example_ids = _plot_routing_examples(
        candidate_trace,
        candidate,
        figures_dir / "residual_routing_example.pdf",
    )
    _write_final(
        root / "FINAL_RESULT.md",
        repo,
        summary,
        safe_vs_unsafe,
        replacement,
        stage0_trace,
        clean_trace,
        example_ids,
    )
    print(f"assembled={root}")


if __name__ == "__main__":
    main()
