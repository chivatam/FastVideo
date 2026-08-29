from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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
VSA80_WALL_MS = 9356.053424499805
DENSE_WALL_MS = 9586.63877949948
FINAL_DECISION = (
    "DECISION: STOP — COMPRESSED OMITTED-SUPPORT RECONSTRUCTION "
    "DOES NOT RECOVER ENOUGH QUALITY"
)
EXAMPLE_PROMPTS = [
    "a bear sniffing the air for scents of food",
    "a person giving a presentation to a room full of colleagues",
    "a car slowing down to stop",
    "a bear hunting for prey",
    "a person swimming in ocean",
    "a cat playing in park",
]


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


def _load_baselines(
    repo: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(
        repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet"
    )
    vsa80 = labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        [
            "prompt_id",
            "quality_safe",
            "subject_delta",
            "motion_delta",
            "dynamic_delta",
        ],
    ].rename(
        columns={
            "quality_safe": "vsa80_safe",
            "subject_delta": "vsa80_subject_delta",
            "motion_delta": "vsa80_motion_delta",
            "dynamic_delta": "vsa80_dynamic_delta",
        }
    )
    dense = labels.loc[
        labels["config"].eq(BASE_CONFIGS["Dense BF16"]),
        [
            "prompt_id",
            "subject_consistency",
            "motion_smoothness",
            "dynamic_degree",
        ],
    ].rename(
        columns={
            "subject_consistency": "dense_subject_consistency",
            "motion_smoothness": "dense_motion_smoothness",
            "dynamic_degree": "dense_dynamic_degree",
        }
    )
    return labels, dense, vsa80


def _aggregate_trace(path: Path) -> pd.DataFrame:
    trace = pd.read_parquet(path)
    trace = trace.loc[
        trace["event_type"].eq("compressed_support_policy")
    ].copy()
    aggregations: dict[str, tuple[str, str]] = {
        "trace_rows": ("event_type", "size"),
        "selected_count_min": ("selected_count_min", "min"),
        "selected_count_max": ("selected_count_max", "max"),
        "retained_mass_mean": ("retained_mass_mean", "mean"),
        "omitted_mass_mean": ("omitted_mass_mean", "mean"),
        "correction_abs_mean": ("correction_abs_mean", "mean"),
        "correction_rms": ("correction_rms", "mean"),
        "correction_relative_l2": (
            "correction_relative_l2",
            "mean",
        ),
        "halo_abs_mean": ("halo_abs_mean", "mean"),
        "halo_rms": ("halo_rms", "mean"),
        "halo_weight_mean": ("halo_weight_mean", "mean"),
        "halo_weight_p50": ("halo_weight_p50", "mean"),
        "halo_weight_p90": ("halo_weight_p90", "mean"),
        "native_coarse_ms": ("native_coarse_ms", "mean"),
        "native_topk_ms": ("native_topk_ms", "mean"),
        "native_exact_fine_ms": ("native_exact_fine_ms", "mean"),
    }
    for column in (
        "rectification_ms",
        "compressed_halo_fused_ms",
        "merge_ms",
    ):
        if column in trace.columns:
            aggregations[column] = (column, "mean")
    return trace.groupby("job_id", as_index=False).agg(**aggregations)


def _load_candidate(
    *,
    quality_root: Path,
    latency_root: Path,
    method: str,
    dense: pd.DataFrame,
    vsa80: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_jobs = pd.read_parquet(quality_root / "jobs.parquet")
    quality_metrics = _metric_wide(
        quality_root / "vbench_metrics.csv"
    )
    quality_trace = _aggregate_trace(
        quality_root / "policy_trace.parquet"
    )
    quality = (
        quality_jobs.merge(
            quality_metrics,
            on="job_id",
            validate="one_to_one",
        )
        .merge(dense, on="prompt_id", validate="one_to_one")
        .merge(vsa80, on="prompt_id", validate="one_to_one")
        .merge(quality_trace, on="job_id", validate="one_to_one")
    )
    quality = _add_quality_labels(quality)
    quality["method"] = method
    quality["original_failure"] = ~quality["vsa80_safe"]
    quality["repaired"] = (
        quality["original_failure"] & quality["quality_safe"]
    )
    quality["new_failure"] = (
        ~quality["original_failure"] & ~quality["quality_safe"]
    )
    quality["subject_recovery"] = (
        quality["subject_delta"] - quality["vsa80_subject_delta"]
    )
    quality["motion_recovery"] = (
        quality["motion_delta"] - quality["vsa80_motion_delta"]
    )

    latency_jobs = pd.read_parquet(latency_root / "jobs.parquet")
    latency_trace = _aggregate_trace(
        latency_root / "policy_trace.parquet"
    )
    latency = latency_jobs.merge(
        latency_trace,
        on="job_id",
        validate="one_to_one",
    )
    latency["method"] = method
    clean_columns = [
        "prompt_id",
        "wall_ms",
        "dit_ms",
        "attention_ms",
        "peak_hbm_bytes",
    ]
    quality = quality.merge(
        latency[clean_columns].rename(
            columns={
                "wall_ms": "clean_wall_ms",
                "dit_ms": "clean_dit_ms",
                "attention_ms": "clean_attention_ms",
                "peak_hbm_bytes": "clean_peak_hbm_bytes",
            }
        ),
        on="prompt_id",
        validate="one_to_one",
    )
    return quality, latency, quality_trace


def _baseline_summary(labels: pd.DataFrame) -> pd.DataFrame:
    vsa80 = labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    rows = []
    for method, config in BASE_CONFIGS.items():
        group = labels.loc[labels["config"].eq(config)].merge(
            vsa80,
            on="prompt_id",
            validate="one_to_one",
        )
        original_failure = ~group["vsa80_safe"]
        rows.append(
            {
                "method": method,
                "unsafe": int((~group["quality_safe"]).sum()),
                "repaired": int(
                    (original_failure & group["quality_safe"]).sum()
                ),
                "new_failures": int(
                    (
                        ~original_failure
                        & ~group["quality_safe"]
                    ).sum()
                ),
                "median_e2e_ms": float(group["wall_ms"].median()),
                "median_dit_ms": float(group["dit_ms"].median()),
                "median_attention_ms": float(
                    group["attention_ms"].median()
                ),
                "native_exact_k": (
                    "dense"
                    if method == "Dense BF16"
                    else {
                        "VSA80": 125,
                        "VSA60": 250,
                        "VSA40": 375,
                    }[method]
                ),
            }
        )
    return pd.DataFrame(rows)


def _adaptive_summary(repo: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        repo
        / "artifacts/adaptive_vsa_deadline_final/development_72/results.csv"
    )
    vsa80 = frame.loc[
        frame["method"].eq("Fixed VSA80"),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    adaptive = frame.loc[
        frame["method"].eq("Adaptive VSA")
    ].merge(vsa80, on="prompt_id", validate="one_to_one")
    original_failure = ~adaptive["vsa80_safe"]
    return {
        "method": "Adaptive-K",
        "unsafe": int((~adaptive["quality_safe"]).sum()),
        "repaired": int(
            (original_failure & adaptive["quality_safe"]).sum()
        ),
        "new_failures": int(
            (~original_failure & ~adaptive["quality_safe"]).sum()
        ),
        "median_e2e_ms": float(adaptive["wall_ms"].median()),
        "median_dit_ms": float(adaptive["dit_ms"].median()),
        "median_attention_ms": float(
            adaptive["attention_ms"].median()
        ),
        "native_exact_k": "variable",
        "effective_sparsity": float(
            adaptive["effective_sparsity"].mean()
        ),
    }


def _candidate_summary(
    quality: pd.DataFrame,
    latency: pd.DataFrame,
    *,
    method: str,
    compressed: bool,
) -> dict[str, Any]:
    unsafe = int((~quality["quality_safe"]).sum())
    repaired = int(quality["repaired"].sum())
    new_failures = int(quality["new_failure"].sum())
    median_e2e = float(latency["wall_ms"].median())
    median_dit = float(latency["dit_ms"].median())
    median_attention = float(latency["attention_ms"].median())
    exact_fine = float(latency["native_exact_fine_ms"].mean())
    row: dict[str, Any] = {
        "method": method,
        "unsafe": unsafe,
        "repaired": repaired,
        "new_failures": new_failures,
        "subject_failures": int((~quality["subject_safe"]).sum()),
        "motion_failures": int((~quality["motion_safe"]).sum()),
        "dynamic_regressions": int(
            (~quality["dynamic_safe"]).sum()
        ),
        "median_e2e_ms": median_e2e,
        "median_dit_ms": median_dit,
        "median_attention_ms": median_attention,
        "delta_e2e_vs_vsa80_pct": (
            100.0 * (median_e2e / VSA80_WALL_MS - 1.0)
        ),
        "delta_attention_vs_vsa80_pct": (
            100.0
            * (
                median_attention
                / 274.9595195055008
                - 1.0
            )
        ),
        "faster_than_dense": median_e2e < DENSE_WALL_MS,
        "within_vsa80_2pct": (
            median_e2e <= VSA80_WALL_MS * 1.02
        ),
        "native_exact_k": 125,
        "selected_count_min": int(
            latency["selected_count_min"].min()
        ),
        "selected_count_max": int(
            latency["selected_count_max"].max()
        ),
        "native_exact_fine_ms_per_call": exact_fine,
        "native_coarse_ms_per_call": float(
            latency["native_coarse_ms"].mean()
        ),
        "native_topk_ms_per_call": float(
            latency["native_topk_ms"].mean()
        ),
        "quality_gate": unsafe <= 12,
        "repair_gate": repaired >= 12,
        "new_failure_gate": new_failures <= 2,
        "latency_gate": (
            median_e2e <= VSA80_WALL_MS * 1.02
            and median_e2e < DENSE_WALL_MS
        ),
        "passes_primary_gate": (
            unsafe <= 12
            and repaired >= 12
            and new_failures <= 2
            and median_e2e <= VSA80_WALL_MS * 1.02
            and median_e2e < DENSE_WALL_MS
        ),
    }
    if compressed:
        row.update(
            {
                "exact_fine_tokens": 125 * 64,
                "compressed_support_tokens": 499,
                "dense_tokens": 624 * 64,
                "dense_equivalent_support_ratio": (
                    (125 * 64 + 499) / (624 * 64)
                ),
                "compressed_halo_ms_per_call": float(
                    latency["compressed_halo_fused_ms"].mean()
                ),
                "merge_ms_per_call": float(
                    latency["merge_ms"].mean()
                ),
                "halo_weight_mean": float(
                    quality["halo_weight_mean"].mean()
                ),
                "halo_weight_p50": float(
                    quality["halo_weight_p50"].mean()
                ),
                "halo_weight_p90": float(
                    quality["halo_weight_p90"].mean()
                ),
            }
        )
    else:
        row["rectification_ms_per_call"] = float(
            latency["rectification_ms"].mean()
        )
    return row


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in (
        "omitted_mass_mean",
        "correction_relative_l2",
        "correction_rms",
        "halo_weight_mean",
        "halo_rms",
    ):
        if feature not in frame.columns:
            continue
        for outcome in ("subject_recovery", "motion_recovery"):
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "pearson": frame[feature].corr(
                        frame[outcome]
                    ),
                    "spearman": frame[feature].corr(
                        frame[outcome],
                        method="spearman",
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_quality_latency(
    headline: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = {
        "Dense BF16": "#4c78a8",
        "VSA80": "#9d9d9d",
        "VSA60": "#79706e",
        "VSA40": "#bab0ac",
        "Adaptive-K": "#f58518",
        "Rectified VSA": "#e45756",
        "CH-VSA": "#54a24b",
    }
    for row in headline.itertuples():
        ax.scatter(
            row.median_e2e_ms,
            row.unsafe,
            s=90,
            color=colors[row.method],
            zorder=3,
        )
        ax.annotate(
            row.method,
            (row.median_e2e_ms, row.unsafe),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(
        12,
        color="#b279a2",
        linestyle="--",
        linewidth=1.2,
        label="unsafe gate",
    )
    ax.axvline(
        VSA80_WALL_MS * 1.02,
        color="#ff9da6",
        linestyle="--",
        linewidth=1.2,
        label="VSA80 +2%",
    )
    ax.set_xlabel("Median end-to-end latency (ms)")
    ax.set_ylabel("Unsafe prompts /72 (lower is better)")
    ax.set_title("Quality–latency gate: neither correction passes")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_failure_count(
    headline: pd.DataFrame,
    output: Path,
) -> None:
    frame = headline.loc[
        headline["method"].isin(
            [
                "VSA80",
                "VSA60",
                "VSA40",
                "Adaptive-K",
                "Rectified VSA",
                "CH-VSA",
            ]
        )
    ].copy()
    x = np.arange(len(frame))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(
        x - width,
        frame["unsafe"],
        width,
        label="unsafe",
        color="#e45756",
    )
    ax.bar(
        x,
        frame["repaired"],
        width,
        label="repaired",
        color="#54a24b",
    )
    ax.bar(
        x + width,
        frame["new_failures"],
        width,
        label="new failures",
        color="#f2cf5b",
    )
    ax.axhline(12, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, frame["method"], rotation=20, ha="right")
    ax.set_ylabel("Prompt count")
    ax.set_title("Failure accounting relative to native VSA80")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_contributions(
    rectified: pd.DataFrame,
    compressed: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, frame, title in (
        (axes[0], rectified, "Rectified VSA"),
        (axes[1], compressed, "CH-VSA"),
    ):
        colors = np.where(
            frame["repaired"],
            "#54a24b",
            np.where(frame["new_failure"], "#e45756", "#4c78a8"),
        )
        ax.scatter(
            frame["correction_relative_l2"],
            frame["subject_recovery"],
            c=colors,
            alpha=0.8,
            s=34,
        )
        pearson = frame["correction_relative_l2"].corr(
            frame["subject_recovery"]
        )
        spearman = frame["correction_relative_l2"].corr(
            frame["subject_recovery"],
            method="spearman",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{title}\nPearson {pearson:.2f}, Spearman {spearman:.2f}"
        )
        ax.set_xlabel("Mean correction relative L2")
        ax.set_ylabel("SC recovery vs VSA80")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Larger omitted-support corrections do not predict recovery"
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _example_table(
    *,
    labels: pd.DataFrame,
    rectified: pd.DataFrame,
    compressed: pd.DataFrame,
) -> pd.DataFrame:
    baseline = labels.loc[
        labels["config"].isin(
            [
                BASE_CONFIGS["Dense BF16"],
                BASE_CONFIGS["VSA80"],
            ]
        ),
        [
            "config",
            "prompt",
            "subject_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "quality_safe",
            "wall_ms",
        ],
    ].copy()
    baseline["method"] = baseline["config"].map(
        {
            BASE_CONFIGS["Dense BF16"]: "Dense BF16",
            BASE_CONFIGS["VSA80"]: "VSA80",
        }
    )
    candidate_columns = [
        "prompt",
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "quality_safe",
        "clean_wall_ms",
        "retained_mass_mean",
        "correction_relative_l2",
        "halo_weight_mean",
        "halo_rms",
    ]
    rect = rectified[candidate_columns].copy()
    rect["method"] = "Rectified VSA"
    rect = rect.rename(columns={"clean_wall_ms": "wall_ms"})
    ch = compressed[candidate_columns].copy()
    ch["method"] = "CH-VSA"
    ch = ch.rename(columns={"clean_wall_ms": "wall_ms"})
    baseline["retained_mass_mean"] = np.nan
    baseline["correction_relative_l2"] = np.nan
    baseline["halo_weight_mean"] = np.nan
    baseline["halo_rms"] = np.nan
    combined = pd.concat(
        [
            baseline[
                [
                    "prompt",
                    "method",
                    "subject_consistency",
                    "motion_smoothness",
                    "dynamic_degree",
                    "quality_safe",
                    "wall_ms",
                    "retained_mass_mean",
                    "correction_relative_l2",
                    "halo_weight_mean",
                    "halo_rms",
                ]
            ],
            rect,
            ch,
        ],
        ignore_index=True,
    )
    return combined.loc[
        combined["prompt"].isin(EXAMPLE_PROMPTS)
    ].copy()


def _plot_examples(
    examples: pd.DataFrame,
    output: Path,
) -> None:
    order = ["Dense BF16", "VSA80", "Rectified VSA", "CH-VSA"]
    with PdfPages(output) as pdf:
        for prompt in EXAMPLE_PROMPTS:
            frame = examples.loc[
                examples["prompt"].eq(prompt)
            ].copy()
            frame["method"] = pd.Categorical(
                frame["method"],
                order,
                ordered=True,
            )
            frame = frame.sort_values("method")
            values = []
            for row in frame.itertuples():
                values.append(
                    [
                        str(row.method),
                        f"{row.subject_consistency:.4f}",
                        f"{row.motion_smoothness:.4f}",
                        f"{row.dynamic_degree:.0f}",
                        "safe" if row.quality_safe else "unsafe",
                        f"{row.wall_ms:.1f}",
                        (
                            "—"
                            if pd.isna(row.retained_mass_mean)
                            else f"{row.retained_mass_mean:.3f}"
                        ),
                        (
                            "—"
                            if pd.isna(row.correction_relative_l2)
                            else f"{row.correction_relative_l2:.3f}"
                        ),
                        (
                            "—"
                            if pd.isna(row.halo_weight_mean)
                            else f"{row.halo_weight_mean:.3f}"
                        ),
                    ]
                )
            fig, ax = plt.subplots(figsize=(11.5, 4.2))
            ax.axis("off")
            ax.set_title(prompt, fontsize=13, pad=18)
            table = ax.table(
                cellText=values,
                colLabels=[
                    "Method",
                    "SC",
                    "MS",
                    "DD",
                    "Rule",
                    "E2E ms",
                    "Retained",
                    "Correction L2",
                    "Halo weight",
                ],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1, 1.7)
            ax.text(
                0.01,
                0.05,
                (
                    "Frozen rule: SC delta ≥ -0.02, MS delta ≥ -0.01, "
                    "and no DD regression. Candidate E2E uses clean timing."
                ),
                transform=ax.transAxes,
                fontsize=8.5,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def _write_rectified_report(
    path: Path,
    summary: dict[str, Any],
    correlations: pd.DataFrame,
) -> None:
    corr = correlations.loc[
        correlations["feature"].eq("correction_relative_l2")
        & correlations["outcome"].eq("subject_recovery")
    ].iloc[0]
    path.write_text(
        f"""# Rectified VSA report

Date: 2026-08-29 UTC

## Result

Coarse-mass rectification fails the frozen 72-prompt gate.

| Measure | Result | Gate |
|---|---:|---:|
| Unsafe | {summary['unsafe']} / 72 | <= 12 |
| Original failures repaired | {summary['repaired']} / 24 | >= 12 |
| New failures | {summary['new_failures']} / 48 | <= 2 |
| Median clean E2E | {summary['median_e2e_ms']:.2f} ms | <= {VSA80_WALL_MS * 1.02:.2f} ms |
| Delta vs VSA80 | {summary['delta_e2e_vs_vsa80_pct']:+.2f}% | <= +2% |
| Native exact K | {summary['selected_count_min']}–{summary['selected_count_max']} | exactly 125 |

## Formulation

The implementation first verified the native output as
`O_fine + G_compress ⊙ O_coarse`. Rectification then uses
`m O_fine + G_compress ⊙ O_omitted_unconditional`, excluding selected
coarse support to avoid obvious double counting. No learned coefficient or
mask search was introduced.

## Quality

The method is harmful: it leaves {summary['unsafe']} prompts unsafe,
repairs only {summary['repaired']} of the 24 original VSA80 failures, and
creates {summary['new_failures']} new failures. Mean correction relative L2
is large, and correction magnitude is negatively associated with subject
recovery (Pearson {corr.pearson:.3f}, Spearman {corr.spearman:.3f}).

## Systems

Clean median E2E is {summary['median_e2e_ms']:.2f} ms, which is
{summary['delta_e2e_vs_vsa80_pct']:+.2f}% versus VSA80 and remains faster
than dense BF16. Rectification itself costs
{summary['rectification_ms_per_call']:.3f} ms/call. Native exact-fine
attention remains {summary['native_exact_fine_ms_per_call']:.3f} ms/call.

## Interpretation

Coarse retained mass is not a sufficient fine-query normalization proxy.
The correction changes the attention output too strongly and broadly,
destroying many previously safe generations. Rectification is rejected.
"""
    )


def _write_ch_report(
    path: Path,
    summary: dict[str, Any],
    correlations: pd.DataFrame,
) -> None:
    corr = correlations.loc[
        correlations["feature"].eq("correction_relative_l2")
        & correlations["outcome"].eq("subject_recovery")
    ].iloc[0]
    path.write_text(
        f"""# Compressed-Halo VSA report

Date: 2026-08-29 UTC

## Result

CH-VSA improves some native VSA80 failures but does not pass the frozen
quality gate.

| Measure | Result | Gate |
|---|---:|---:|
| Unsafe | {summary['unsafe']} / 72 | <= 12 |
| Original failures repaired | {summary['repaired']} / 24 | >= 12 |
| New failures | {summary['new_failures']} / 48 | <= 2 |
| Median clean E2E | {summary['median_e2e_ms']:.2f} ms | <= {VSA80_WALL_MS * 1.02:.2f} ms |
| Delta vs VSA80 | {summary['delta_e2e_vs_vsa80_pct']:+.2f}% | <= +2% |
| Native exact K | {summary['selected_count_min']}–{summary['selected_count_max']} | exactly 125 |

## Formulation

CH-VSA leaves the native 125 exact blocks and sparse fine kernel unchanged.
For each fine query it attends to the 499 omitted pooled K/V centroids,
adds the actual block-size log multiplicity, and merges exact and compressed
support with a common log-sum-exp normalization. The checkpoint's learned
coarse residual is then added in the same fused B200 kernel.

## Support and systems

- Exact fine support: {summary['exact_fine_tokens']} tokens.
- Compressed omitted support: {summary['compressed_support_tokens']} centroid tokens.
- Dense support: {summary['dense_tokens']} tokens.
- Dense-equivalent nominal support ratio: {100 * summary['dense_equivalent_support_ratio']:.2f}%.
- Halo kernel: {summary['compressed_halo_ms_per_call']:.3f} ms/call.
- Fused merge/coarse residual: {summary['merge_ms_per_call']:.3f} ms/call.
- Native exact-fine kernel: {summary['native_exact_fine_ms_per_call']:.3f} ms/call.

Clean median E2E is {summary['median_e2e_ms']:.2f} ms
({summary['delta_e2e_vs_vsa80_pct']:+.2f}% versus VSA80), so the systems
budget is met and the method remains faster than dense BF16.

## Quality and mechanism

CH-VSA repairs {summary['repaired']} original failures, but introduces
{summary['new_failures']} new failures and ends at {summary['unsafe']}/72
unsafe. Mean halo weight is {summary['halo_weight_mean']:.3f}; the average
90th-percentile halo weight is {summary['halo_weight_p90']:.3f}. Correction
magnitude does not positively predict subject recovery (Pearson
{corr.pearson:.3f}, Spearman {corr.spearman:.3f}).

## Decision

The method fails both the unsafe-count and new-failure gates and is worse
than fixed VSA60/VSA40 on quality. The optional one-probe method is not run:
its prerequisite required both rectification and CH to be promising, while
rectification was decisively harmful. No full VBench or Wan14B transfer is
authorized.

The broad concept of sparse residual or centroid compensation is not claimed
as novel. This experiment only tests whether VSA's already-computed pooled
representations can recover omitted fine support at fixed exact K.
"""
    )


def _write_final(
    path: Path,
    *,
    headline: pd.DataFrame,
    rectified_summary: dict[str, Any],
    ch_summary: dict[str, Any],
    rect_corr: pd.DataFrame,
    ch_corr: pd.DataFrame,
    examples: pd.DataFrame,
    commit: str,
) -> None:
    table = headline.copy()
    table["Median E2E"] = table["median_e2e_ms"].map(
        lambda value: f"{value:.2f} ms"
    )
    table["Δ vs VSA80"] = table["median_e2e_ms"].map(
        lambda value: f"{100 * (value / VSA80_WALL_MS - 1):+.2f}%"
    )
    table = table.rename(
        columns={
            "method": "Method",
            "unsafe": "Unsafe /72",
            "repaired": "Repaired /24",
            "new_failures": "New failures /48",
            "native_exact_k": "Native exact K",
        }
    )[
        [
            "Method",
            "Unsafe /72",
            "Repaired /24",
            "New failures /48",
            "Median E2E",
            "Δ vs VSA80",
            "Native exact K",
        ]
    ]
    rect_subject = rect_corr.loc[
        rect_corr["feature"].eq("correction_relative_l2")
        & rect_corr["outcome"].eq("subject_recovery")
    ].iloc[0]
    ch_subject = ch_corr.loc[
        ch_corr["feature"].eq("correction_relative_l2")
        & ch_corr["outcome"].eq("subject_recovery")
    ].iloc[0]
    example_lines = []
    for prompt in EXAMPLE_PROMPTS:
        frame = examples.loc[examples["prompt"].eq(prompt)]
        rect = frame.loc[frame["method"].eq("Rectified VSA")].iloc[0]
        ch = frame.loc[frame["method"].eq("CH-VSA")].iloc[0]
        example_lines.append(
            f"- **{prompt}:** Rectified "
            f"{'safe' if rect.quality_safe else 'unsafe'}, CH "
            f"{'safe' if ch.quality_safe else 'unsafe'}; "
            f"retained mass {ch.retained_mass_mean:.3f}, "
            f"halo weight {ch.halo_weight_mean:.3f}."
        )
    path.write_text(
        f"""# Final result — Compressed omitted-support VSA

Date: 2026-08-29 UTC  
Branch: `research`  
Commit: `{commit}`

## Headline

{_markdown_table(table)}

Both correction methods meet the clean latency budget and keep native exact
K at 125, but both fail the frozen quality gate. Rectification is severely
harmful. CH-VSA recovers 14 original failures yet creates 12 new failures,
ending with 22 unsafe prompts.

## Required answers

1. **Was native VSA's actual coarse/fine output formula verified from code?**
   Yes. Native output is independently normalized sparse
   `O_fine + G_compress ⊙ O_coarse`; the sparse kernel also exposes log2 LSE.
   The fused Top-K threshold required a rank-preserving numerical
   normalization to enforce its intended exact K=125 cardinality on
   high-range checkpoint rows.

2. **Does coarse-mass rectification improve quality?** No. Unsafe prompts
   increase from 24 to {rectified_summary['unsafe']}.

3. **How many VSA80 failures does it repair?**
   {rectified_summary['repaired']} / 24.

4. **Does it introduce new failures?**
   Yes, {rectified_summary['new_failures']} / 48.

5. **What is its latency overhead?** Median clean E2E is
   {rectified_summary['median_e2e_ms']:.2f} ms,
   {rectified_summary['delta_e2e_vs_vsa80_pct']:+.2f}% versus VSA80.
   Rectification costs {rectified_summary['rectification_ms_per_call']:.3f}
   ms per attention call.

6. **Does fine-query compressed halo improve quality further?** It improves
   substantially over rectification and modestly over native VSA80 in
   original-failure repair, but not enough to pass: unsafe is
   {ch_summary['unsafe']} / 72.

7. **How many failures does CH-VSA repair?**
   {ch_summary['repaired']} / 24; it also creates
   {ch_summary['new_failures']} new failures.

8. **Does it remain faster than dense?** Yes. CH median clean E2E is
   {ch_summary['median_e2e_ms']:.2f} ms versus {DENSE_WALL_MS:.2f} ms for
   dense BF16.

9. **Does native exact K remain exactly 125?** Yes. Both methods record
   selected-count minimum = maximum = 125 across all 6,480 attention calls.

10. **How much compressed omitted support is added?** 499 centroid tokens
    per query row in addition to 8,000 exact fine tokens. The nominal
    dense-equivalent support is
    {100 * ch_summary['dense_equivalent_support_ratio']:.2f}% of 39,936
    dense tokens.

11. **Is fine-attention kernel latency unchanged?** Yes in the tested paths.
    Rectified exact-fine latency is
    {rectified_summary['native_exact_fine_ms_per_call']:.3f} ms/call and CH
    is {ch_summary['native_exact_fine_ms_per_call']:.3f} ms/call, a
    {100 * (ch_summary['native_exact_fine_ms_per_call'] / rectified_summary['native_exact_fine_ms_per_call'] - 1):+.2f}%
    difference. Both call the same native sparse kernel.

12. **Is quality recovery associated with omitted-support correction
    magnitude?** No positive relationship is observed. Rectified
    correction-L2 versus SC recovery is Pearson {rect_subject.pearson:.3f}
    / Spearman {rect_subject.spearman:.3f}; CH is Pearson
    {ch_subject.pearson:.3f} / Spearman {ch_subject.spearman:.3f}.

13. **Does either method beat fixed VSA60/VSA40 on quality-vs-speed?** No.
    Rectified has {rectified_summary['unsafe']} unsafe and CH has
    {ch_summary['unsafe']}, versus 14 for VSA60 and 8 for VSA40. Adaptive-K
    also has much better quality at 2 unsafe, although it is slower.

14. **Is probe-based residual reconstruction necessary?** It is not run.
    The protocol allows it only if both B and C are promising but
    insufficient. Rectification is decisively harmful, so that prerequisite
    is false.

15. **Should we freeze the method and run full VBench?** No. Neither method
    passes the 72-prompt gate; therefore no full VBench or Wan14B transfer is
    run.

## Audited examples

{chr(10).join(example_lines)}

The multi-page `figures/repaired_examples.pdf` reports Dense, VSA80,
Rectified, and CH metrics plus retained mass, correction magnitude, halo
weight, and clean latency for all six examples. Failures are retained.

## Bounded claim

No claim is made for the first sparse residual reconstruction, compressed
attention, or centroid compensation method. The tested VSA-specific question
is whether pooled representations already learned and computed by VSA can
recover omitted fine-attention support without increasing exact fine K or
training another model. Under this checkpoint and gate, the answer is no.

{FINAL_DECISION}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/home/ec2-user/FastVideo"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/compressed_halo_vsa/final_package"
        ),
    )
    parser.add_argument(
        "--rectified-quality",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/compressed_halo_vsa/"
            "rectified_quality_v3"
        ),
    )
    parser.add_argument(
        "--ch-quality",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/compressed_halo_vsa/ch_quality"
        ),
    )
    parser.add_argument(
        "--rectified-latency",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/compressed_halo_vsa/clean_latency"
        ),
    )
    parser.add_argument(
        "--ch-latency",
        type=Path,
        default=Path(
            "/mnt/fastvideo-gpu0/compressed_halo_vsa/ch_clean_latency"
        ),
    )
    args = parser.parse_args()

    output = args.output
    rectified_dir = output / "rectified"
    ch_dir = output / "compressed_halo"
    native_dir = output / "native_analysis"
    figures_dir = output / "figures"
    for directory in (
        output,
        rectified_dir,
        ch_dir,
        native_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    labels, dense, vsa80 = _load_baselines(args.repo)
    rectified, rect_latency, _ = _load_candidate(
        quality_root=args.rectified_quality,
        latency_root=args.rectified_latency,
        method="Rectified VSA",
        dense=dense,
        vsa80=vsa80,
    )
    compressed, ch_latency, _ = _load_candidate(
        quality_root=args.ch_quality,
        latency_root=args.ch_latency,
        method="CH-VSA",
        dense=dense,
        vsa80=vsa80,
    )
    rectified_summary = _candidate_summary(
        rectified,
        rect_latency,
        method="Rectified VSA",
        compressed=False,
    )
    ch_summary = _candidate_summary(
        compressed,
        ch_latency,
        method="CH-VSA",
        compressed=True,
    )
    rect_corr = _correlations(rectified)
    ch_corr = _correlations(compressed)

    baseline = _baseline_summary(labels)
    adaptive = _adaptive_summary(args.repo)
    headline = pd.concat(
        [
            baseline,
            pd.DataFrame([adaptive]),
            pd.DataFrame(
                [
                    {
                        key: rectified_summary[key]
                        for key in (
                            "method",
                            "unsafe",
                            "repaired",
                            "new_failures",
                            "median_e2e_ms",
                            "median_dit_ms",
                            "median_attention_ms",
                            "native_exact_k",
                        )
                    },
                    {
                        key: ch_summary[key]
                        for key in (
                            "method",
                            "unsafe",
                            "repaired",
                            "new_failures",
                            "median_e2e_ms",
                            "median_dit_ms",
                            "median_attention_ms",
                            "native_exact_k",
                        )
                    },
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    vsa60 = headline.loc[headline["method"].eq("VSA60")].iloc[0]
    vsa40 = headline.loc[headline["method"].eq("VSA40")].iloc[0]
    adaptive_row = headline.loc[
        headline["method"].eq("Adaptive-K")
    ].iloc[0]
    for summary in (rectified_summary, ch_summary):
        summary["pareto_beats_vsa60_vsa40_adaptive"] = all(
            (
                summary["unsafe"] <= comparator.unsafe
                and summary["median_e2e_ms"]
                < comparator.median_e2e_ms
            )
            or (
                summary["unsafe"] < comparator.unsafe
                and summary["median_e2e_ms"]
                <= comparator.median_e2e_ms
            )
            for comparator in (vsa60, vsa40, adaptive_row)
        )

    pd.DataFrame([rectified_summary]).to_csv(
        rectified_dir / "results.csv",
        index=False,
    )
    pd.DataFrame([ch_summary]).to_csv(
        ch_dir / "results.csv",
        index=False,
    )
    rectified.to_csv(rectified_dir / "quality.csv", index=False)
    compressed.to_csv(ch_dir / "quality.csv", index=False)
    rect_latency.to_csv(rectified_dir / "latency.csv", index=False)
    ch_latency.to_csv(ch_dir / "latency.csv", index=False)
    compressed.to_parquet(
        ch_dir / "contribution_stats.parquet",
        index=False,
    )
    rect_corr.to_csv(
        rectified_dir / "correlations.csv",
        index=False,
    )
    ch_corr.to_csv(ch_dir / "correlations.csv", index=False)
    headline.to_csv(output / "headline.csv", index=False)

    shutil.copyfile(
        args.repo
        / "research/compressed_halo_vsa/native_vsa_equations.md",
        native_dir / "native_vsa_equations.md",
    )
    _write_rectified_report(
        rectified_dir / "REPORT.md",
        rectified_summary,
        rect_corr,
    )
    _write_ch_report(
        ch_dir / "REPORT.md",
        ch_summary,
        ch_corr,
    )

    examples = _example_table(
        labels=labels,
        rectified=rectified,
        compressed=compressed,
    )
    examples.to_csv(output / "audited_examples.csv", index=False)
    _plot_quality_latency(
        headline,
        figures_dir / "quality_latency.pdf",
    )
    _plot_failure_count(
        headline,
        figures_dir / "failure_count.pdf",
    )
    _plot_contributions(
        rectified,
        compressed,
        figures_dir / "omitted_support_contribution.pdf",
    )
    _plot_examples(
        examples,
        figures_dir / "repaired_examples.pdf",
    )

    commit = _git(args.repo, "rev-parse", "HEAD")
    _write_final(
        output / "FINAL_RESULT.md",
        headline=headline,
        rectified_summary=rectified_summary,
        ch_summary=ch_summary,
        rect_corr=rect_corr,
        ch_corr=ch_corr,
        examples=examples,
        commit=commit,
    )
    manifest = {
        "date_utc": "2026-08-29",
        "branch": _git(args.repo, "branch", "--show-current"),
        "commit": commit,
        "decision": FINAL_DECISION,
        "rectified_quality_root": str(args.rectified_quality),
        "compressed_halo_quality_root": str(args.ch_quality),
        "rectified_latency_root": str(args.rectified_latency),
        "compressed_halo_latency_root": str(args.ch_latency),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"output={output}")
    print(FINAL_DECISION)


if __name__ == "__main__":
    main()
