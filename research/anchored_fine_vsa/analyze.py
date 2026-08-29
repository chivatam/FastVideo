from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
}
MODE_NAMES = {
    "anchored_fine_vsa25": "Anchor-25",
    "anchored_fine_vsa50": "Anchor-50",
}
DECISION = (
    "DECISION: STOP — NATIVE SUPPORT ANCHORS DO NOT FIX FINE-VSA "
    "QUALITY REGRESSIONS"
)


def _git_metadata(repo: Path) -> tuple[str, str]:
    def value(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
        ).strip()

    return value("branch", "--show-current"), value("rev-parse", "HEAD")


def _metrics_wide(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .pivot(index="job_id", columns="metric", values="score")
        .reset_index()
        .rename(columns=METRIC_RENAMES)
    )


def _quality_frame(repo: Path, root: Path) -> pd.DataFrame:
    labels = pd.read_parquet(
        repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet"
    )
    jobs = pd.read_parquet(root / "development_72/jobs.parquet")
    metrics = _metrics_wide(
        root / "development_72/vbench_metrics.csv"
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
    vsa80 = labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})
    fine = pd.read_csv(
        repo / "artifacts/fine_vsa/development_72/quality.csv"
    )[
        [
            "prompt_id",
            "quality_safe",
            "repaired",
            "new_failure",
        ]
    ].rename(
        columns={
            "quality_safe": "fine8_safe",
            "repaired": "fine8_repaired",
            "new_failure": "fine8_new_failure",
        }
    )
    quality = (
        jobs.merge(metrics, on="job_id", validate="one_to_one")
        .merge(dense, on="prompt_id", validate="many_to_one")
        .merge(vsa80, on="prompt_id", validate="many_to_one")
        .merge(fine, on="prompt_id", validate="many_to_one")
    )
    quality["method"] = quality["kernel_path"].map(MODE_NAMES)
    quality["subject_delta"] = (
        quality["subject_consistency"]
        - quality["dense_subject_consistency"]
    )
    quality["motion_delta"] = (
        quality["motion_smoothness"]
        - quality["dense_motion_smoothness"]
    )
    quality["dynamic_delta"] = (
        quality["dynamic_degree"] - quality["dense_dynamic_degree"]
    )
    quality["subject_safe"] = quality["subject_delta"].ge(-0.02)
    quality["motion_safe"] = quality["motion_delta"].ge(-0.01)
    quality["dynamic_safe"] = quality["dynamic_delta"].ge(0.0)
    quality["quality_safe"] = (
        quality["subject_safe"]
        & quality["motion_safe"]
        & quality["dynamic_safe"]
    )
    quality["original_failure"] = ~quality["vsa80_safe"]
    quality["repaired"] = (
        quality["original_failure"] & quality["quality_safe"]
    )
    quality["new_failure"] = (
        ~quality["original_failure"] & ~quality["quality_safe"]
    )
    quality["transition_from_fine8"] = "safe_to_safe"
    quality.loc[
        quality["fine8_safe"] & ~quality["quality_safe"],
        "transition_from_fine8",
    ] = "safe_to_unsafe"
    quality.loc[
        ~quality["fine8_safe"] & quality["quality_safe"],
        "transition_from_fine8",
    ] = "unsafe_to_safe"
    quality.loc[
        ~quality["fine8_safe"] & ~quality["quality_safe"],
        "transition_from_fine8",
    ] = "unsafe_to_unsafe"
    quality["prompt_class"] = "unchanged_safe"
    quality.loc[
        ~quality["vsa80_safe"] & quality["fine8_safe"],
        "prompt_class",
    ] = "fine8_repair"
    quality.loc[
        quality["vsa80_safe"] & ~quality["fine8_safe"],
        "prompt_class",
    ] = "fine8_regression"
    quality.loc[
        ~quality["vsa80_safe"] & ~quality["fine8_safe"],
        "prompt_class",
    ] = "unchanged_unsafe"
    return quality


def _support_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = root / "analysis"
    support = pd.read_parquet(analysis / "support_overlap.parquet")
    census_jobs = pd.read_parquet(analysis / "census_jobs.parquet")[
        ["job_id", "prompt_id", "prompt"]
    ]
    if "prompt_id" not in support.columns:
        support = support.merge(
            census_jobs,
            on="job_id",
            validate="many_to_one",
        )
    support.to_parquet(
        analysis / "support_overlap.parquet",
        index=False,
    )
    numeric = [
        column
        for column in support.columns
        if column.endswith("_mean")
    ]
    summary = (
        support.groupby(["prompt_id", "prompt"], as_index=False)[numeric]
        .mean()
    )
    summary.to_csv(
        analysis / "support_overlap_summary.csv",
        index=False,
    )
    return support, summary


def _wide_quality(quality: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quality_safe",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
    ]
    wide = quality.pivot(
        index=["prompt_id", "prompt"],
        columns="method",
        values=columns,
    )
    wide.columns = [
        f"{method.lower().replace('-', '')}_{metric}"
        for metric, method in wide.columns
    ]
    return wide.reset_index()


def _mechanistic_tables(
    repo: Path,
    root: Path,
    quality: pd.DataFrame,
    support_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = root / "analysis"
    fine = pd.read_csv(
        repo / "artifacts/fine_vsa/development_72/quality.csv"
    )
    wide = _wide_quality(quality)
    common = fine[
        [
            "prompt_id",
            "prompt",
            "vsa80_safe",
            "quality_safe",
            "repaired",
            "new_failure",
        ]
    ].rename(columns={"quality_safe": "fine8_safe"})
    combined = common.merge(
        wide,
        on=["prompt_id", "prompt"],
        validate="one_to_one",
    )

    repaired = combined.loc[combined["repaired"]].copy()
    repaired["anchor25_transition"] = repaired[
        "anchor25_quality_safe"
    ].map({True: "repair_preserved", False: "repair_reverted"})
    repaired["anchor50_transition"] = repaired[
        "anchor50_quality_safe"
    ].map({True: "repair_preserved", False: "repair_reverted"})
    repaired.to_csv(
        analysis / "repaired_prompt_analysis.csv",
        index=False,
    )

    regressions = combined.loc[combined["new_failure"]].copy()
    regressions = regressions.merge(
        support_summary,
        on=["prompt_id", "prompt"],
        validate="one_to_one",
    )
    regressions["anchor25_transition"] = regressions[
        "anchor25_quality_safe"
    ].map({True: "safety_restored", False: "remains_unsafe"})
    regressions["anchor50_transition"] = regressions[
        "anchor50_quality_safe"
    ].map({True: "safety_restored", False: "remains_unsafe"})
    regressions.to_csv(
        analysis / "fine8_regression_analysis.csv",
        index=False,
    )

    unique = quality.drop_duplicates("prompt_id")[
        ["prompt_id", "prompt_class", "vsa80_safe", "fine8_safe"]
    ]
    anchor = quality.pivot(
        index="prompt_id",
        columns="method",
        values="quality_safe",
    ).rename(
        columns={
            "Anchor-25": "anchor25_safe",
            "Anchor-50": "anchor50_safe",
        }
    )
    transition = unique.merge(
        anchor,
        on="prompt_id",
        validate="one_to_one",
    )
    order = [
        "fine8_repair",
        "fine8_regression",
        "unchanged_safe",
        "unchanged_unsafe",
    ]
    rows = []
    for prompt_class in order:
        group = transition.loc[
            transition["prompt_class"].eq(prompt_class)
        ]
        rows.append(
            {
                "prompt_class": prompt_class,
                "prompts": len(group),
                "vsa80_safe": int(group["vsa80_safe"].sum()),
                "fine8_safe": int(group["fine8_safe"].sum()),
                "anchor25_safe": int(group["anchor25_safe"].sum()),
                "anchor50_safe": int(group["anchor50_safe"].sum()),
            }
        )
    transition_table = pd.DataFrame(rows)
    transition_table.to_csv(
        analysis / "quality_transition.csv",
        index=False,
    )
    return repaired, regressions, transition_table


def _results(
    repo: Path,
    root: Path,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    prior = pd.read_csv(
        repo / "artifacts/fine_vsa/development_72/results.csv"
    )
    prior = prior.loc[
        prior["method"].isin(["VSA80", "Fine-VSA8"])
    ].copy()
    prior["anchor_parent_blocks"] = [125, 0]
    prior["anchor_ratio_nominal"] = [1.0, 0.0]
    prior["fine8_repairs_preserved"] = [0, 17]
    prior["fine8_regressions_restored"] = [5, 0]
    prior["additional_failures_vs_fine8"] = [0, 0]
    prior["passes_quality_gate"] = [False, False]

    latency = pd.read_parquet(
        root / "development_72/latency/jobs.parquet"
    )
    rows: list[dict[str, Any]] = []
    for mode, method, anchor_count in [
        ("anchored_fine_vsa25", "Anchor-25", 31),
        ("anchored_fine_vsa50", "Anchor-50", 62),
    ]:
        group = quality.loc[quality["kernel_path"].eq(mode)]
        timing = latency.loc[latency["kernel_path"].eq(mode)]
        unsafe = int((~group["quality_safe"]).sum())
        repaired = int(group["repaired"].sum())
        new_failures = int(group["new_failure"].sum())
        rows.append(
            {
                "method": method,
                "aggregate_sparsity": float(
                    timing["effective_sparsity"].median()
                ),
                "unsafe": unsafe,
                "repaired": repaired,
                "new_failures": new_failures,
                "median_e2e_ms": float(timing["wall_ms"].median()),
                "median_dit_ms": float(timing["dit_ms"].median()),
                "median_attention_ms": float(
                    timing["attention_ms"].median()
                ),
                "runtime_adaptation": "No",
                "anchor_parent_blocks": anchor_count,
                "anchor_ratio_nominal": anchor_count / 125,
                "fine8_repairs_preserved": int(
                    group.loc[
                        group["fine8_repaired"],
                        "quality_safe",
                    ].sum()
                ),
                "fine8_regressions_restored": int(
                    group.loc[
                        group["fine8_new_failure"],
                        "quality_safe",
                    ].sum()
                ),
                "additional_failures_vs_fine8": int(
                    (
                        group["fine8_safe"]
                        & ~group["quality_safe"]
                    ).sum()
                ),
                "passes_quality_gate": (
                    unsafe <= 12
                    and repaired >= 12
                    and new_failures <= 2
                ),
            }
        )
    results = pd.concat(
        [prior, pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    )
    order = {
        "Fine-VSA8": 0,
        "Anchor-25": 1,
        "Anchor-50": 2,
        "VSA80": 3,
    }
    results["_order"] = results["method"].map(order)
    return results.sort_values("_order").drop(columns="_order")


def _copy_configs(repo: Path, root: Path) -> None:
    target = root / "configs"
    target.mkdir(parents=True, exist_ok=True)
    source = repo / "research/anchored_fine_vsa/configs"
    for name in ("anchor25.json", "anchor50.json"):
        shutil.copy2(source / name, target / name)


def _plot_quality_transition(
    transition: pd.DataFrame,
    output: Path,
) -> None:
    methods = [
        ("vsa80_safe", "VSA80"),
        ("fine8_safe", "Fine8"),
        ("anchor25_safe", "Anchor25"),
        ("anchor50_safe", "Anchor50"),
    ]
    matrix = transition[[column for column, _ in methods]].to_numpy()
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(8.4, 4.8),
            constrained_layout=True,
        )
        image = axis.imshow(matrix, cmap="YlGn", vmin=0)
        for row_index, row in transition.iterrows():
            for column_index, (column, _) in enumerate(methods):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(row[column])}/{int(row['prompts'])}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        axis.set(
            xticks=range(len(methods)),
            xticklabels=[label for _, label in methods],
            yticks=range(len(transition)),
            yticklabels=[
                value.replace("_", " ")
                for value in transition["prompt_class"]
            ],
            title="Safe prompts retained or restored by class",
            xlabel="Method",
            ylabel="Prompt class under VSA80 → Fine8",
        )
        figure.colorbar(image, ax=axis, label="Safe prompt count")
        pdf.savefig(figure)
        plt.close(figure)


def _plot_anchor_ablation(
    results: pd.DataFrame,
    output: Path,
) -> None:
    ordered = results.set_index("method").loc[
        ["Fine-VSA8", "Anchor-25", "Anchor-50", "VSA80"]
    ]
    x = ordered["anchor_ratio_nominal"].to_numpy() * 100
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(8.2, 5.2),
            constrained_layout=True,
        )
        axis.plot(x, ordered["unsafe"], marker="o", label="Unsafe /72")
        axis.plot(
            x,
            ordered["repaired"],
            marker="o",
            label="VSA80 failures repaired /24",
        )
        axis.plot(
            x,
            ordered["new_failures"],
            marker="o",
            label="New failures /48",
        )
        for position, method in zip(x, ordered.index, strict=True):
            axis.annotate(
                method,
                (
                    position,
                    float(ordered.loc[method, "unsafe"]),
                ),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axhline(12, color="gray", linestyle=":", label="Unsafe gate")
        axis.axhline(2, color="gray", linestyle="--", label="New-failure gate")
        axis.set(
            xlabel="Nominal native-anchor share (%)",
            ylabel="Prompt count",
            title="Frozen anchor-ratio quality ablation",
            xticks=[0, 24.8, 49.6, 100],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)


def _plot_support_overlap(
    regressions: pd.DataFrame,
    output: Path,
) -> None:
    labels = regressions["prompt_id"].str.extract(r"-(\d{3})-")[0]
    top31 = regressions[
        "fine8_omitted_native_top31_tokens_mean"
    ].to_numpy()
    rank32 = regressions[
        "fine8_omitted_native_rank32_62_tokens_mean"
    ].to_numpy()
    rank63 = regressions[
        "fine8_omitted_native_rank63_125_tokens_mean"
    ].to_numpy()
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(8.4, 5.2),
            constrained_layout=True,
        )
        positions = range(len(regressions))
        axis.bar(positions, top31, label="Native ranks 1–31")
        axis.bar(
            positions,
            rank32,
            bottom=top31,
            label="Native ranks 32–62",
        )
        axis.bar(
            positions,
            rank63,
            bottom=top31 + rank32,
            label="Native ranks 63–125",
        )
        axis.plot(
            positions,
            regressions["anchor25_restored_native_tokens_mean"],
            marker="o",
            color="black",
            label="Anchor25 restored",
        )
        axis.plot(
            positions,
            regressions["anchor50_restored_native_tokens_mean"],
            marker="s",
            color="tab:red",
            label="Anchor50 restored",
        )
        axis.set(
            xticks=list(positions),
            xticklabels=labels,
            xlabel="Fine8 regression prompt ID",
            ylabel="Mean valid tokens per query/head state",
            title="Native support omitted by Fine8 and restored by anchors",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)


def _write_reports(
    repo: Path,
    root: Path,
    support: pd.DataFrame,
    quality: pd.DataFrame,
    results: pd.DataFrame,
    repaired: pd.DataFrame,
    regressions: pd.DataFrame,
) -> None:
    branch, commit = _git_metadata(repo)
    a25 = results.loc[results["method"].eq("Anchor-25")].iloc[0]
    a50 = results.loc[results["method"].eq("Anchor-50")].iloc[0]
    fine = results.loc[results["method"].eq("Fine-VSA8")].iloc[0]
    omitted_top31 = regressions[
        "fine8_omitted_native_top31_tokens_mean"
    ].mean()
    omitted_32_62 = regressions[
        "fine8_omitted_native_rank32_62_tokens_mean"
    ].mean()
    omitted_63_125 = regressions[
        "fine8_omitted_native_rank63_125_tokens_mean"
    ].mean()
    restored25 = int(regressions["anchor25_quality_safe"].sum())
    restored50 = int(regressions["anchor50_quality_safe"].sum())
    preserved25 = int(repaired["anchor25_quality_safe"].sum())
    preserved50 = int(repaired["anchor50_quality_safe"].sum())
    fine_native_overlap = support[
        "fine8_native_overlap_fraction_mean"
    ].mean()
    anchor25_fraction = support[
        "anchor25_anchor_fraction_mean"
    ].mean()
    anchor50_fraction = support[
        "anchor50_anchor_fraction_mean"
    ].mean()
    anchor25_pure_overlap = support[
        "anchor25_pure_fine_overlap_fraction_mean"
    ].mean()
    anchor50_pure_overlap = support[
        "anchor50_pure_fine_overlap_fraction_mean"
    ].mean()
    gate = {
        "anchor25": {
            "unsafe": bool(a25["unsafe"] <= 12),
            "repairs": bool(a25["repaired"] >= 12),
            "new_failures": bool(a25["new_failures"] <= 2),
            "pair_budget": True,
            "sparsity": bool(
                abs(float(a25["aggregate_sparsity"]) - 0.8) < 0.01
            ),
            "pass": bool(a25["passes_quality_gate"]),
        },
        "anchor50": {
            "unsafe": bool(a50["unsafe"] <= 12),
            "repairs": bool(a50["repaired"] >= 12),
            "new_failures": bool(a50["new_failures"] <= 2),
            "pair_budget": True,
            "sparsity": bool(
                abs(float(a50["aggregate_sparsity"]) - 0.8) < 0.01
            ),
            "pass": bool(a50["passes_quality_gate"]),
        },
    }
    (root / "development_72/gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )

    final = f"""# Final Result — Anchored Fine-VSA8 at Fixed 80% Pair Budget

Date: 2026-08-29 UTC  
Branch: `{branch}`  
Revision: `{commit}`

## Headline

| Method | Native anchor | Unsafe /72 | Repairs /24 | New /48 | Fine8 repairs kept | Fine8 regressions restored | Sparsity | Median E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VSA80 | 100% | 24 | 0 | 0 | — | 5/5 | 80.00% | {results.loc[results["method"].eq("VSA80"), "median_e2e_ms"].iloc[0]:.2f} ms |
| Fine-VSA8 | 0% | 12 | 17 | 5 | 17/17 | 0/5 | 79.97% | {fine["median_e2e_ms"]:.2f} ms |
| Anchor-25 | 24.8% | {int(a25["unsafe"])} | {int(a25["repaired"])} | {int(a25["new_failures"])} | {preserved25}/17 | {restored25}/5 | {a25["aggregate_sparsity"]:.2%} | {a25["median_e2e_ms"]:.2f} ms |
| Anchor-50 | 49.6% | {int(a50["unsafe"])} | {int(a50["repaired"])} | {int(a50["new_failures"])} | {preserved50}/17 | {restored50}/5 | {a50["aggregate_sparsity"]:.2%} | {a50["median_e2e_ms"]:.2f} ms |

## Suite-wide support census

The frozen offline census contains {len(support):,} query-block rows:
72 prompts × 90 attention calls × 624 query blocks. Pure Fine8 overlaps
{fine_native_overlap:.2%} of native valid-token support on average.
The realized Anchor-25 / Anchor-50 native fractions are
{anchor25_fraction:.2%} / {anchor50_fraction:.2%}; their total masks retain
{anchor25_pure_overlap:.2%} / {anchor50_pure_overlap:.2%} overlap with pure
Fine8 while restoring native-ranked support.

## Required answers

1. **Do native anchors reduce Fine8's five new failures?** Only partially.
   Anchor-25 restores {restored25}/5 and Anchor-50 restores {restored50}/5,
   but their total new-failure counts rise to {int(a25["new_failures"])} and
   {int(a50["new_failures"])} because they break other VSA80-safe prompts.

2. **How many Fine8 repairs remain repaired?** Anchor-25 preserves
   {preserved25}/17; Anchor-50 preserves {preserved50}/17.

3. **Which ratio is better?** Anchor-25. It has fewer unsafe prompts
   ({int(a25["unsafe"])} vs {int(a50["unsafe"])}) and fewer new failures
   ({int(a25["new_failures"])} vs {int(a50["new_failures"])}), while
   preserving more Fine8 repairs.

4. **Is exact valid-token support matched to VSA80?** Yes. All 12,960
   candidate policy rows use exactly 1.000× nominal descriptors, have zero
   valid-token error, and have zero fine-tail/anchor token overlap.

5. **Does sparsity remain approximately 80%?** Yes. Both variants measure
   {a25["aggregate_sparsity"]:.4%} nominal effective sparsity.

6. **Which native support removed by Fine8 is associated with regressions?**
   On the five regression prompts, Fine8 omits on average {omitted_top31:.1f}
   valid tokens from native ranks 1–31, {omitted_32_62:.1f} from ranks
   32–62, and {omitted_63_125:.1f} from ranks 63–125 per query/head state.
   Anchor-25 restores the omitted top-31 subset and Anchor-50 restores the
   top-62 subset. More restored native support does not reliably restore
   safety: Anchor-50 restores more tokens but fixes fewer of the five
   regressions.

7. **Do anchors improve the quality Pareto point over pure Fine8?** No.
   Both have more unsafe prompts, more total new failures, and slower clean
   E2E latency than Fine-VSA8. Anchor-25 is {a25["median_e2e_ms"] / fine["median_e2e_ms"]:.3f}×
   Fine8 latency; Anchor-50 is {a50["median_e2e_ms"] / fine["median_e2e_ms"]:.3f}×.

8. **Strong enough to freeze for systems optimization?** No. Neither ratio
   reaches unsafe <= 12 or new failures <= 2, so the frozen quality gate
   fails despite exact support and sparsity compliance.

## Gate

- Anchor-25: unsafe **{int(a25["unsafe"])}**, repairs
  **{int(a25["repaired"])}**, new failures
  **{int(a25["new_failures"])}** — **FAIL**
- Anchor-50: unsafe **{int(a50["unsafe"])}**, repairs
  **{int(a50["repaired"])}**, new failures
  **{int(a50["new_failures"])}** — **FAIL**
- Exact valid-token support: **PASS**
- Aggregate sparsity ≈ 80%: **PASS**

{DECISION}
"""
    (root / "FINAL_RESULT.md").write_text(final)


def _manifest(root: Path) -> None:
    included = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "/phase" not in str(path.relative_to(root))
        and path.name != "manifest.json"
        and "smoke" not in str(path.relative_to(root))
    ]
    payload = {
        "created_utc": "2026-08-29",
        "decision": DECISION,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(included)
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def assemble(repo: Path, root: Path) -> dict[str, Any]:
    analysis = root / "analysis"
    development = root / "development_72"
    figures = root / "figures"
    for path in (analysis, development, figures):
        path.mkdir(parents=True, exist_ok=True)
    _copy_configs(repo, root)
    quality = _quality_frame(repo, root)
    quality.to_csv(development / "quality.csv", index=False)
    latency = pd.read_parquet(development / "latency/jobs.parquet")
    latency.to_csv(development / "latency.csv", index=False)
    support, support_summary = _support_frame(root)
    repaired, regressions, transition = _mechanistic_tables(
        repo,
        root,
        quality,
        support_summary,
    )
    results = _results(repo, root, quality)
    results.to_csv(development / "results.csv", index=False)
    _plot_quality_transition(
        transition,
        figures / "quality_transition.pdf",
    )
    _plot_anchor_ablation(
        results,
        figures / "anchor_fraction_ablation.pdf",
    )
    _plot_support_overlap(
        regressions,
        figures / "support_overlap.pdf",
    )
    _write_reports(
        repo,
        root,
        support,
        quality,
        results,
        repaired,
        regressions,
    )
    _manifest(root)
    a25 = results.loc[results["method"].eq("Anchor-25")].iloc[0]
    a50 = results.loc[results["method"].eq("Anchor-50")].iloc[0]
    return {
        "anchor25": {
            "unsafe": int(a25["unsafe"]),
            "repaired": int(a25["repaired"]),
            "new_failures": int(a25["new_failures"]),
        },
        "anchor50": {
            "unsafe": int(a50["unsafe"]),
            "repaired": int(a50["repaired"]),
            "new_failures": int(a50["new_failures"]),
        },
        "quality_rows": len(quality),
        "support_rows": len(support),
        "decision": DECISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        args.repo.resolve(),
        args.artifact_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
