from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

VARIANT_NAMES = {
    "dense_bf16": "Dense BF16",
    "native64": "VSA80",
    "fine8": "Fine8",
    "native_anchor25": "NativeAnchor25",
    "native_anchor50": "NativeAnchor50",
    "calib_core25_tail": "CalibCore25 + Fine8Tail",
    "calib_core50_tail": "CalibCore50 + Fine8Tail",
    "calib_core25_only": "CalibCore25 static only",
    "calib_core50_only": "CalibCore50 static only",
}
CANDIDATES = ("calib_core25_tail", "calib_core50_tail")
NOMINAL_SPARSITY = 0.799679
STOP_DECISION = ("DECISION: STOP — CALIBRATED STATIC SUPPORT DOES NOT IMPROVE "
                 "FINE8 SAFETY AT FIXED 80% PAIR BUDGET")


def _load_run(
    run_root: Path,
    prompts_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted((run_root / "phase0/stats").glob("*.parquet"))
    prompts = pd.DataFrame(json.loads(prompts_path.read_text()))
    if len(paths) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} held-out traces, found {len(paths)}")
    stats = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    records = []
    for path in sorted((run_root / "phase0/records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            records.append({
                "job_id": record["job_id"],
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
            })
    jobs = pd.DataFrame(records)
    if len(jobs) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} successful jobs, found {len(jobs)}")
    if set(jobs["prompt_id"]) != set(prompts["prompt_id"]):
        raise RuntimeError("Held-out jobs do not match the frozen prompt set")
    return stats.merge(
        jobs,
        on="job_id",
        how="left",
        validate="many_to_one",
    ), jobs


def _metric_summary(error: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "relative_L2_mean",
        "relative_L2_median",
        "relative_L2_p90",
        "relative_L2_p99",
        "cosine_error_mean",
        "cosine_error_median",
        "cosine_error_p90",
        "cosine_error_p99",
        "actual_pair_budget_ratio",
        "unused_pair_capacity",
    ]
    summary = error.groupby("variant", as_index=False)[metrics].mean()
    summary["max_actual_pair_budget_ratio"] = error.groupby("variant")["actual_pair_budget_ratio"].max().values
    summary["attention_states"] = error.groupby("variant").size().values
    summary["method"] = summary["variant"].map(VARIANT_NAMES)
    return summary


def _catastrophic_table(error: pd.DataFrame, ) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["job_id", "prompt_id", "prompt", "timestep", "layer"]
    wide = error.pivot(
        index=keys,
        columns="variant",
        values="relative_L2_p90",
    ).reset_index()
    if len(wide) != 8 * 90:
        raise RuntimeError(f"Expected 720 held-out attention states, found {len(wide)}")
    baseline = wide["fine8"].gt(wide["native64"] * 1.05)
    rows = []
    summaries = []
    for candidate in CANDIDATES:
        threshold = np.maximum(
            wide["fine8"] * 1.20,
            wide["native64"] * 1.05,
        )
        event = wide[candidate].gt(threshold)
        candidate_rows = wide[keys].copy()
        candidate_rows["candidate"] = candidate
        candidate_rows["native_error"] = wide["native64"]
        candidate_rows["fine8_error"] = wide["fine8"]
        candidate_rows["candidate_error"] = wide[candidate]
        candidate_rows["candidate_minus_fine8"] = (wide[candidate] - wide["fine8"])
        candidate_rows["candidate_minus_vsa80"] = (wide[candidate] - wide["native64"])
        candidate_rows["frozen_threshold"] = threshold
        candidate_rows["fine8_baseline_catastrophic"] = baseline
        candidate_rows["candidate_catastrophic"] = event
        rows.append(candidate_rows)
        baseline_rate = float(baseline.mean())
        candidate_rate = float(event.mean())
        summaries.append({
            "variant":
            candidate,
            "method":
            VARIANT_NAMES[candidate],
            "attention_states":
            len(wide),
            "fine8_baseline_events":
            int(baseline.sum()),
            "fine8_baseline_rate":
            baseline_rate,
            "candidate_events":
            int(event.sum()),
            "candidate_rate":
            candidate_rate,
            "candidate_to_baseline_rate_ratio":
            candidate_rate / baseline_rate if baseline_rate > 0 else (0.0 if candidate_rate == 0 else float("inf")),
            "passes_75pct_rate_gate":
            candidate_rate <= 0.75 * baseline_rate,
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def _gate(
    summary: pd.DataFrame,
    catastrophic: pd.DataFrame,
) -> dict[str, Any]:
    indexed = summary.set_index("variant")
    fine_p90 = float(indexed.loc["fine8", "relative_L2_p90"])
    fine_p99 = float(indexed.loc["fine8", "relative_L2_p99"])
    catastrophic_index = catastrophic.set_index("variant")
    candidates: dict[str, dict[str, Any]] = {}
    passing = []
    for candidate in CANDIDATES:
        row = indexed.loc[candidate]
        tail = catastrophic_index.loc[candidate]
        conditions = {
            "p90_le_1p02_fine8": bool(float(row["relative_L2_p90"]) <= 1.02 * fine_p90),
            "p99_lt_fine8": bool(float(row["relative_L2_p99"]) < fine_p99),
            "catastrophic_rate_le_0p75_fine8": bool(tail["passes_75pct_rate_gate"]),
            "max_pair_ratio_le_1": bool(float(row["max_actual_pair_budget_ratio"]) <= 1.0 + 1e-9),
            "nominal_sparsity_matches": True,
        }
        passed = all(conditions.values())
        candidates[candidate] = {
            "method": VARIANT_NAMES[candidate],
            "relative_L2_p90": float(row["relative_L2_p90"]),
            "relative_L2_p99": float(row["relative_L2_p99"]),
            "fine8_relative_L2_p90": fine_p90,
            "fine8_relative_L2_p99": fine_p99,
            "catastrophic_rate": float(tail["candidate_rate"]),
            "fine8_catastrophic_rate": float(tail["fine8_baseline_rate"]),
            "max_exact_pair_ratio": float(row["max_actual_pair_budget_ratio"]),
            "nominal_sparsity": NOMINAL_SPARSITY,
            "conditions": conditions,
            "passes": passed,
        }
        if passed:
            passing.append(candidate)
    winner = None
    if passing:
        winner = min(
            passing,
            key=lambda candidate: (
                candidates[candidate]["catastrophic_rate"],
                candidates[candidate]["relative_L2_p99"],
                candidates[candidate]["relative_L2_p90"],
                -int("50" in candidate),
            ),
        )
    return {
        "metric_scope": "720 held-out prompt×step×layer states; each state uses "
        "all-head/query-block p90 relative-L2",
        "fine8_baseline_catastrophic_definition": "E_fine8 > 1.05 * E_native",
        "candidate_catastrophic_definition": "E_candidate > max(1.20 * E_fine8, 1.05 * E_native)",
        "nominal_sparsity": NOMINAL_SPARSITY,
        "candidates": candidates,
        "passing_candidates": passing,
        "winner": winner,
        "proceed_to_systems": winner is not None,
    }


def _offline_figures(
    error: pd.DataFrame,
    summary: pd.DataFrame,
    catastrophic: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    order = [
        "native64",
        "fine8",
        "native_anchor25",
        "calib_core25_tail",
        "native_anchor50",
        "calib_core50_tail",
        "calib_core25_only",
        "calib_core50_only",
    ]
    with PdfPages(figures / "offline_error_distribution.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(11, 5))
        data = [error.loc[error["variant"].eq(variant), "relative_L2_p90"] for variant in order]
        axis.boxplot(data, tick_labels=[VARIANT_NAMES[v] for v in order], showfliers=False)
        axis.set_ylabel("Per-state p90 relative-L2")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

    with PdfPages(figures / "catastrophic_tail.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = catastrophic["method"].tolist()
        x = np.arange(len(labels))
        axis.bar(
            x - 0.18,
            catastrophic["fine8_baseline_rate"],
            width=0.36,
            label="Fine8 baseline",
        )
        axis.bar(
            x + 0.18,
            catastrophic["candidate_rate"],
            width=0.36,
            label="Candidate",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel("Catastrophic-event rate")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

    comparison = summary.set_index("variant").loc[[
        "native_anchor25",
        "calib_core25_tail",
        "native_anchor50",
        "calib_core50_tail",
    ]]
    with PdfPages(figures / "native_vs_calibrated_anchor.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(8, 4))
        x = np.arange(len(comparison))
        axis.bar(
            x - 0.18,
            comparison["relative_L2_p90"],
            width=0.36,
            label="p90",
        )
        axis.bar(
            x + 0.18,
            comparison["relative_L2_p99"],
            width=0.36,
            label="p99",
        )
        axis.set_xticks(
            x,
            [VARIANT_NAMES[value] for value in comparison.index],
            rotation=20,
        )
        axis.set_ylabel("Mean call-level relative-L2 quantile")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def analyze_offline(root: Path) -> dict[str, Any]:
    heldout = root / "heldout_offline"
    stats, _ = _load_run(
        heldout / "run",
        heldout / "prompts.json",
    )
    error = stats.loc[stats["event_type"].eq("coretail_vsa_error") & stats["scope"].eq("all_heads_query_blocks")].copy()
    expected = 8 * 90 * len(VARIANT_NAMES)
    if len(error) != expected:
        raise RuntimeError(f"Expected {expected} held-out error rows, found {len(error)}")
    error.to_csv(heldout / "errors.csv", index=False)
    summary = _metric_summary(error)
    summary.to_csv(heldout / "p90_p99.csv", index=False)
    tail_rows, catastrophic = _catastrophic_table(error)
    tail_rows.to_csv(heldout / "catastrophic_tail.csv", index=False)
    catastrophic.to_csv(
        heldout / "catastrophic_tail_summary.csv",
        index=False,
    )

    accounting = stats.loc[stats["event_type"].eq("coretail_pair_accounting")].copy()
    accounting.to_csv(heldout / "pair_accounting.csv", index=False)
    ablation = summary.loc[summary["variant"].isin({
        "native_anchor25",
        "native_anchor50",
        "calib_core25_tail",
        "calib_core50_tail",
        "calib_core25_only",
        "calib_core50_only",
    })].copy()
    ablation.to_csv(
        heldout / "native_anchor_ablation.csv",
        index=False,
    )
    gate = _gate(summary, catastrophic)
    (heldout / "offline_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    _offline_figures(
        error,
        summary,
        catastrophic,
        root / "figures",
    )
    report_rows = summary.set_index("variant")
    candidate_lines = []
    for candidate in CANDIDATES:
        result = gate["candidates"][candidate]
        candidate_lines.append(f"- {result['method']}: p90 "
                               f"{result['relative_L2_p90']:.6f}, p99 "
                               f"{result['relative_L2_p99']:.6f}, catastrophic "
                               f"{result['catastrophic_rate']:.3%}, max pair ratio "
                               f"{result['max_exact_pair_ratio']:.6f}, "
                               f"**{'PASS' if result['passes'] else 'FAIL'}**.")
    decision = (f"Offline gate passed; systems winner: "
                f"{VARIANT_NAMES[gate['winner']]}." if gate["winner"] is not None else STOP_DECISION)
    report = f"""# Held-Out Offline Report

The frozen eight-prompt set produced 720 prompt × step × layer attention
states. Each state uses the p90 query-block relative-L2 aggregated over all
heads. Fine8's predeclared baseline catastrophic event is
`E_fine8 > 1.05 E_native`; candidate events use
`E_candidate > max(1.20 E_fine8, 1.05 E_native)`.

Fine8 mean call-level p90/p99 were
{report_rows.loc['fine8', 'relative_L2_p90']:.6f} /
{report_rows.loc['fine8', 'relative_L2_p99']:.6f}. Its baseline catastrophic
rate was {catastrophic['fine8_baseline_rate'].iloc[0]:.3%}.

{chr(10).join(candidate_lines)}

Static-only Core25/Core50 p90 were
{report_rows.loc['calib_core25_only', 'relative_L2_p90']:.6f} /
{report_rows.loc['calib_core50_only', 'relative_L2_p90']:.6f}.

{decision}
"""
    (heldout / "REPORT.md").write_text(report)
    return gate


def _calibration_figures(root: Path) -> None:
    external = root / "external_calibration"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stability = pd.read_csv(external / "core_stability.csv")
    coverage = pd.read_csv(external / "core_mass_coverage.csv")

    with PdfPages(figures / "stable_core_heatmap.pdf") as pdf:
        for step in sorted(stability["step"].unique()):
            for metric, title in [
                ("top31_jaccard_mean", "Core25 mean Jaccard"),
                ("top62_jaccard_mean", "Core50 mean Jaccard"),
                ("rank_correlation_mean", "Stable-rank correlation"),
            ]:
                table = stability.loc[stability["step"].eq(step)].pivot(
                    index="layer",
                    columns="head",
                    values=metric,
                )
                figure, axis = plt.subplots(figsize=(9, 7))
                image = axis.imshow(
                    table.values,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                )
                axis.set_title(f"Step {step}: {title}")
                axis.set_xlabel("Head")
                axis.set_ylabel("Layer")
                figure.colorbar(image, ax=axis)
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)

    with PdfPages(figures / "core_overlap.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(8, 4))
        grouped = stability.groupby("step", as_index=False)[[
            "top31_jaccard_mean",
            "top62_jaccard_mean",
        ]].mean()
        axis.plot(
            grouped["step"],
            grouped["top31_jaccard_mean"],
            marker="o",
            label="Core25",
        )
        axis.plot(
            grouped["step"],
            grouped["top62_jaccard_mean"],
            marker="o",
            label="Core50",
        )
        axis.set_xticks(grouped["step"])
        axis.set_xlabel("Denoising step")
        axis.set_ylabel("Mean prompt-to-stable Jaccard")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

    with PdfPages(figures / "core_mass_distribution.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(8, 4))
        for method, group in coverage.groupby("method"):
            by_step = group.groupby("step")["dense_mass_mean"].mean()
            axis.plot(
                by_step.index,
                by_step.values,
                marker="o",
                label=method,
            )
        axis.set_xlabel("Denoising step")
        axis.set_ylabel("Mean true dense mass coverage")
        axis.set_xticks(sorted(coverage["step"].unique()))
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def report_calibration(root: Path) -> None:
    external = root / "external_calibration"
    summary = json.loads((external / "calibration_summary.json").read_text())
    stability = pd.read_csv(external / "core_stability.csv")
    coverage = pd.read_csv(external / "core_mass_coverage.csv")
    by_step = stability.groupby("step", as_index=False).mean(numeric_only=True)
    coverage_overall = coverage.groupby(
        "method",
        as_index=False,
    ).mean(numeric_only=True)
    report = f"""# External Calibration Report

The static masks were frozen from 32 external VBench-2.0 prompts with zero
overlap with the eight held-out prompts or 72 development prompts. Dense BF16
capture covered 3 steps × 30 layers × 12 heads × 624 query regions.

- Core25 prompt-to-stable overlap/Jaccard:
  {summary['core25_overlap_mean']:.4f} / {summary['core25_jaccard_mean']:.4f}.
- Core50 prompt-to-stable overlap/Jaccard:
  {summary['core50_overlap_mean']:.4f} / {summary['core50_jaccard_mean']:.4f}.
- Mean prompt/stable full-rank correlation:
  {summary['rank_correlation_mean']:.4f}.
- Mean Core25 fraction often missed by Fine8:
  {stability['core25_often_missed_fraction'].mean():.3%}.
- Mean Core50 fraction often missed by Fine8:
  {stability['core50_often_missed_fraction'].mean():.3%}.

## Stability by denoising step

{by_step[['step', 'top31_jaccard_mean', 'top62_jaccard_mean', 'rank_correlation_mean']].to_markdown(index=False)}

## True dense mass coverage

{coverage_overall[['method', 'dense_mass_mean', 'dense_mass_p10', 'dense_mass_median', 'dense_mass_p90', 'mass_per_valid_token_mean']].to_markdown(index=False)}

The primary score remained the frozen linear p10 statistic; no quantile or core
ratio was selected from held-out results.
"""
    (external / "REPORT.md").write_text(report)
    _calibration_figures(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/coretail_vsa"),
    )
    parser.add_argument(
        "--stage",
        choices=("calibration", "offline", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage in {"calibration", "all"}:
        report_calibration(args.root)
    if args.stage in {"offline", "all"}:
        gate = analyze_offline(args.root)
        print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
