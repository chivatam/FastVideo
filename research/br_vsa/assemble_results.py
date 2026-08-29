from __future__ import annotations

import argparse
import json
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
    "VSA60": "vsa_bf16@s0.60",
    "VSA40": "vsa_bf16@s0.40",
}
FINAL_DECISION = ("DECISION: STOP — STATIC GLOBAL-BUDGET REDISTRIBUTION "
                  "DOES NOT RECOVER ENOUGH QUALITY")


def _metrics_wide(path: Path) -> pd.DataFrame:
    return (pd.read_csv(path).pivot(index="job_id", columns="metric",
                                    values="score").reset_index().rename(columns=METRIC_RENAMES))


def _dense_reference(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.loc[
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
        })


def _vsa80_reference(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.loc[
        labels["config"].eq(BASE_CONFIGS["VSA80"]),
        ["prompt_id", "quality_safe"],
    ].rename(columns={"quality_safe": "vsa80_safe"})


def _label_quality(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["subject_delta"] = (result["subject_consistency"] - result["dense_subject_consistency"])
    result["motion_delta"] = (result["motion_smoothness"] - result["dense_motion_smoothness"])
    result["dynamic_delta"] = (result["dynamic_degree"] - result["dense_dynamic_degree"])
    result["subject_safe"] = result["subject_delta"] >= -0.02
    result["motion_safe"] = result["motion_delta"] >= -0.01
    result["dynamic_safe"] = result["dynamic_delta"] >= 0.0
    result["quality_safe"] = (result["subject_safe"] & result["motion_safe"] & result["dynamic_safe"])
    result["original_failure"] = ~result["vsa80_safe"]
    result["repaired"] = (result["original_failure"] & result["quality_safe"])
    result["new_failure"] = (~result["original_failure"] & ~result["quality_safe"])
    return result


def _load_br_quality(
    root: Path,
    *,
    method: str,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    jobs = pd.read_parquet(root / "jobs.parquet")
    metrics = _metrics_wide(root / "vbench_metrics.csv")
    quality = (jobs.merge(metrics, on="job_id", validate="one_to_one").merge(
        _dense_reference(labels),
        on="prompt_id",
        validate="one_to_one",
    ).merge(
        _vsa80_reference(labels),
        on="prompt_id",
        validate="one_to_one",
    ))
    quality = _label_quality(quality)
    quality["method"] = method
    return quality


def _base_quality(labels: pd.DataFrame) -> pd.DataFrame:
    vsa80 = _vsa80_reference(labels)
    frames = []
    for method, config in BASE_CONFIGS.items():
        frame = labels.loc[labels["config"].eq(config)].copy()
        frame = frame.merge(
            vsa80,
            on="prompt_id",
            validate="one_to_one",
            suffixes=("", "_reference"),
        )
        frame["method"] = method
        frame["original_failure"] = ~frame["vsa80_safe"]
        frame["repaired"] = (frame["original_failure"] & frame["quality_safe"])
        frame["new_failure"] = (~frame["original_failure"] & ~frame["quality_safe"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _adaptive_quality(repo: Path, labels: pd.DataFrame) -> pd.DataFrame:
    adaptive = pd.read_csv(repo / "artifacts/adaptive_vsa_deadline_final/development_72/results.csv")
    adaptive = adaptive.loc[adaptive["method"].eq("Adaptive VSA")].copy()
    adaptive = adaptive.merge(
        _vsa80_reference(labels),
        on="prompt_id",
        validate="one_to_one",
    )
    adaptive["original_failure"] = ~adaptive["vsa80_safe"]
    adaptive["repaired"] = (adaptive["original_failure"] & adaptive["quality_safe"])
    adaptive["new_failure"] = (~adaptive["original_failure"] & ~adaptive["quality_safe"])
    return adaptive


def _method_row(
    quality: pd.DataFrame,
    *,
    method: str,
    latency: pd.DataFrame | None = None,
    runtime_adaptation: str,
) -> dict[str, Any]:
    group = quality.loc[quality["method"].eq(method)]
    latency_group = group if latency is None else latency
    return {
        "method": method,
        "aggregate_sparsity": float(latency_group["effective_sparsity"].median()),
        "unsafe": int((~group["quality_safe"]).sum()),
        "repaired": int(group["repaired"].sum()),
        "new_failures": int(group["new_failure"].sum()),
        "median_e2e_ms": float(latency_group["wall_ms"].median()),
        "median_dit_ms": float(latency_group["dit_ms"].median()),
        "median_attention_ms": float(latency_group["attention_ms"].median()),
        "runtime_adaptation": runtime_adaptation,
    }


def _pareto_plot(summary: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
        colors = {
            "Dense BF16": "black",
            "VSA80": "tab:blue",
            "VSA60": "tab:orange",
            "VSA40": "tab:green",
            "Adaptive-K": "tab:purple",
            "BR-VSA": "tab:red",
            "BR-VSA layer-only": "tab:brown",
        }
        for row in summary.itertuples():
            axis.scatter(
                row.median_e2e_ms,
                row.unsafe,
                s=90,
                color=colors.get(row.method, "gray"),
            )
            axis.annotate(
                row.method,
                (row.median_e2e_ms, row.unsafe),
                xytext=(5, 5),
                textcoords="offset points",
            )
        axis.axvline(
            9356.053424499805 * 1.02,
            linestyle="--",
            color="gray",
            label="VSA80 × 1.02",
        )
        axis.axhline(
            12,
            linestyle=":",
            color="gray",
            label="PASS unsafe threshold",
        )
        axis.set(
            xlabel="Median end-to-end latency (ms; lower is better)",
            ylabel="Unsafe prompts / 72 (lower is better)",
            title="Quality–latency comparison",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        pdf.savefig(figure)
        plt.close(figure)


def assemble(
    *,
    repo: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    development = artifact_root / "development_72"
    figures = artifact_root / "figures"
    development.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet")
    head_quality = _load_br_quality(
        development / "headwise_quality",
        method="BR-VSA",
        labels=labels,
    )
    layer_quality = _load_br_quality(
        development / "layer_only_quality",
        method="BR-VSA layer-only",
        labels=labels,
    )
    head_latency = pd.read_parquet(development / "headwise_latency/jobs.parquet")
    layer_latency = pd.read_parquet(development / "layer_only_latency/jobs.parquet")
    head_latency["method"] = "BR-VSA"
    layer_latency["method"] = "BR-VSA layer-only"

    base = _base_quality(labels)
    adaptive = _adaptive_quality(repo, labels)
    quality = pd.concat(
        [base, adaptive, head_quality, layer_quality],
        ignore_index=True,
        sort=False,
    )
    quality.to_csv(development / "quality.csv", index=False)
    pd.concat(
        [head_latency, layer_latency],
        ignore_index=True,
        sort=False,
    ).to_csv(development / "latency.csv", index=False)

    rows = [_method_row(
        base,
        method=method,
        runtime_adaptation="No",
    ) for method in BASE_CONFIGS]
    rows.append(_method_row(
        adaptive,
        method="Adaptive VSA",
        runtime_adaptation="Yes",
    ))
    rows[-1]["method"] = "Adaptive-K"
    rows.extend([
        _method_row(
            head_quality,
            method="BR-VSA",
            latency=head_latency,
            runtime_adaptation="No",
        ),
        _method_row(
            layer_quality,
            method="BR-VSA layer-only",
            latency=layer_latency,
            runtime_adaptation="No",
        ),
    ])
    results = pd.DataFrame(rows)
    results.to_csv(development / "results.csv", index=False)

    allocation = pd.read_csv(artifact_root / "allocation/allocation_summary.csv")
    funded = allocation.loc[allocation["delta_K"].gt(0)]
    defunded = allocation.loc[allocation["delta_K"].lt(0)]
    top_funded = ";".join(f"s{row.step}-l{row.layer}-h{row.head}:K{row.allocated_K}"
                          for row in funded.nlargest(12, "delta_K").itertuples())
    top_defunded = ";".join(f"s{row.step}-l{row.layer}-h{row.head}:K{row.allocated_K}"
                            for row in defunded.nsmallest(12, "delta_K").itertuples())
    repaired = head_quality.loc[head_quality["repaired"]].copy()
    new_failures = head_quality.loc[head_quality["new_failure"]].copy()
    for frame in (repaired, new_failures):
        frame["fixed_policy_top_funded_units"] = top_funded
        frame["fixed_policy_top_defunded_units"] = top_defunded
        frame["diagnostic_note"] = ("BR-VSA is prompt-independent; every prompt used the same frozen "
                                    "funded and defunded units.")
    repaired.to_csv(
        development / "repaired_failures.csv",
        index=False,
    )
    new_failures.to_csv(
        development / "new_failures.csv",
        index=False,
    )

    _pareto_plot(
        results,
        figures / "quality_latency_pareto.pdf",
    )

    concentration = pd.read_csv(artifact_root / "calibration/error_concentration.csv").set_index("top_percent")
    gate = json.loads((artifact_root / "calibration/stage0_gate.json").read_text())
    budget = json.loads((artifact_root / "allocation/budget_validation.json").read_text())
    allocation_report = pd.read_csv(artifact_root / "allocation/k_distribution.csv")
    native_summary = pd.read_csv(artifact_root / "calibration/sensitivity_summary.csv")
    native_summary = native_summary.loc[native_summary["K"].eq(125)]
    step_sensitivity = (native_summary.groupby("step")["relative_L2_error_mean"].mean().sort_values(ascending=False))
    layer_sensitivity = (native_summary.groupby("layer")["relative_L2_error_mean"].mean().sort_values(ascending=False))
    head_sensitivity = (native_summary.groupby("head")["relative_L2_error_mean"].mean().sort_values(ascending=False))
    unit_classes = (pd.read_csv(artifact_root / "calibration/sensitivity_summary.csv")[[
        "step",
        "layer",
        "head",
        "curve_sufficient_K",
        "sensitivity_class",
    ]].drop_duplicates())
    below_125 = int((allocation["allocated_K"] < 125).sum())
    above_125 = int((allocation["allocated_K"] > 125).sum())
    curve_below_125 = int((unit_classes["curve_sufficient_K"] < 125).sum())
    curve_above_125 = int((unit_classes["curve_sufficient_K"] > 125).sum())
    sensitivity_correlation = float(allocation["allocated_K"].rank().corr(allocation["native_error"].rank()))
    head_row = results.loc[results["method"].eq("BR-VSA")].iloc[0]
    layer_row = results.loc[results["method"].eq("BR-VSA layer-only")].iloc[0]
    dense_row = results.loc[results["method"].eq("Dense BF16")].iloc[0]
    vsa80_row = results.loc[results["method"].eq("VSA80")].iloc[0]
    latency_limit = float(vsa80_row["median_e2e_ms"] * 1.02)

    distribution_lines = "\n".join(f"- K{int(row.K)}: {int(row.units)} units ({row.fraction:.2%})"
                                   for row in allocation_report.itertuples())
    report = f"""# BR-VSA 72-Prompt Development Report

## Frozen result

- Head-wise BR-VSA: **{int(head_row.unsafe)} unsafe**, {int(head_row.repaired)} repaired, {int(head_row.new_failures)} new failures.
- Layer-only ablation: **{int(layer_row.unsafe)} unsafe**, {int(layer_row.repaired)} repaired, {int(layer_row.new_failures)} new failures.
- Exact global budget: {budget["allocated_budget"]} / {budget["native_budget"]} blocks.
- Aggregate exact-block sparsity: {budget["aggregate_sparsity"]:.4%}.
- Clean median E2E: {head_row.median_e2e_ms:.2f} ms.
- VSA80 × 1.02 latency limit: {latency_limit:.2f} ms.

## Quality gate

BR-VSA fails the quality gate. It exceeds the <=12 unsafe target, repairs only
8 of 24 original failures, and introduces 12 new failures instead of <=2.

## Systems gate

The head-wise grouped kernel stays within the 2% VSA80 latency allowance and
is {dense_row.median_e2e_ms - head_row.median_e2e_ms:.2f} ms faster than dense
BF16. Therefore the final failure is quality-driven, not a global-budget or
runtime-budget violation.

## Allocation distribution

{distribution_lines}

## Interpretation

Sensitivity is heterogeneous and highly prompt-stable, and the optimizer
strongly targets high-error units (Spearman K-vs-K125-error =
{sensitivity_correlation:.4f}). Nevertheless, transferring budget away from
639 low-marginal-value units creates more failures than the funded units
repair. The evidence does not support uniform allocation as the dominant
remaining VSA quality problem.

{FINAL_DECISION}
"""
    (development / "REPORT.md").write_text(report)

    final_result = f"""# Final Result — Budget-Redistributed VSA

## Executive result

The attention census passed its diagnostic gate, and the production table
uses exactly the native global budget. The frozen 72-prompt evaluation fails:
BR-VSA produces **{int(head_row.unsafe)}/72 unsafe**, repairs
**{int(head_row.repaired)}/24**, and creates **{int(head_row.new_failures)}**
new failures. Full VBench expansion is not justified.

## Required questions

1. **Is dense-relative sensitivity heterogeneous?** Yes. The top 20% of units explain {concentration.loc[20, "error_fraction"]:.2%} of K125 error.
2. **Concentration at 10/20/30%?** {concentration.loc[10, "error_fraction"]:.2%} / {concentration.loc[20, "error_fraction"]:.2%} / {concentration.loc[30, "error_fraction"]:.2%}.
3. **Is ranking stable across prompts?** Yes. Median pairwise Spearman is {gate["median_pairwise_spearman"]:.4f}; leave-one-out median is 0.9910.
4. **Most sensitive denoising step?** Step {int(step_sensitivity.index[0])}, mean K125 error {step_sensitivity.iloc[0]:.4f}.
5. **Most consistently sensitive layers?** Layers {", ".join(str(int(value)) for value in layer_sensitivity.index[:5])}.
6. **Repeatedly sensitive heads?** Yes. Head indices {", ".join(str(int(value)) for value in head_sensitivity.index[:5])} rank highest on average, led by head {int(head_sensitivity.index[0])}.
7. **How many units can operate below K125?** The optimizer assigns {below_125} below K125; the stricter 90%-curve criterion marks only {curve_below_125} below K125.
8. **How many require K>125?** The exact-budget optimizer funds {above_125}; the strict curve criterion marks {curve_above_125}.
9. **Final K distribution?** {", ".join(f"K{int(row.K)}={int(row.units)}" for row in allocation_report.itertuples())}.
10. **Mean/global K <=125?** Yes: mean K is exactly {budget["mean_K"]:.1f}, budget ratio {budget["budget_ratio"]:.6f}.
11. **Aggregate sparsity?** {budget["aggregate_sparsity"]:.4%}.
12. **VSA80 failures repaired?** {int(head_row.repaired)} of 24.
13. **New failures?** {int(head_row.new_failures)}.
14. **Median E2E latency?** {head_row.median_e2e_ms:.2f} ms clean no-save.
15. **Faster than dense BF16?** Yes, by {dense_row.median_e2e_ms - head_row.median_e2e_ms:.2f} ms ({(dense_row.median_e2e_ms - head_row.median_e2e_ms) / dense_row.median_e2e_ms:.2%}).
16. **Beat VSA60/VSA40 on quality-vs-speed?** No. It is faster but materially worse in quality (28 unsafe versus 14 and 8).
17. **Head-wise materially outperform layer-only?** Yes but insufficiently: 28 versus 37 unsafe, 8 versus 4 repairs, and 12 versus 17 new failures. Offline predicted error is 16.1% lower.
18. **Does recovery correspond to funding sensitive units?** The allocation does target them (Spearman {sensitivity_correlation:.4f}; funded-unit mean K125 error is 0.382 versus 0.055 for defunded units), but the 72-prompt quality result does not show adequate recovery.
19. **Is uniform allocation the dominant remaining problem?** No. Static redistribution worsens total unsafe count from 24 to 28 despite strong sensitivity structure.
20. **Proceed to full VBench?** No.

## Bounded conclusion

For this fixed-sparsity VSA checkpoint, training-free dense-relative
step/layer/head budget redistribution is structurally stable and
systems-feasible, but it does not recover generation quality at the native
global exact-attention budget. No claim of novelty over prior head/layer sparse
allocation work is made.

{FINAL_DECISION}
"""
    (artifact_root / "FINAL_RESULT.md").write_text(final_result)
    return {
        "unsafe": int(head_row.unsafe),
        "repaired": int(head_row.repaired),
        "new_failures": int(head_row.new_failures),
        "median_e2e_ms": float(head_row.median_e2e_ms),
        "budget_ratio": budget["budget_ratio"],
        "decision": FINAL_DECISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        repo=args.repo.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
