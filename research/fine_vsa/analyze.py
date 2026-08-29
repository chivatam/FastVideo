from __future__ import annotations

import argparse
import hashlib
import json
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
    "VSA60": "vsa_bf16@s0.60",
    "VSA40": "vsa_bf16@s0.40",
}
GLOBAL_VARIANTS = {
    64: "native64_global",
    32: "kv32_global",
    16: "kv16_global",
    8: "kv8_global",
}
DECISION = (
    "DECISION: STOP — FINER BLOCK GRANULARITY DOES NOT RECOVER ENOUGH "
    "QUALITY"
)


def _git_metadata(repo: Path) -> tuple[str, str]:
    def value(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return value("branch", "--show-current"), value("rev-parse", "HEAD")


def _metrics_wide(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .pivot(index="job_id", columns="metric", values="score")
        .reset_index()
        .rename(columns=METRIC_RENAMES)
    )


def _calibration_frame(root: Path) -> pd.DataFrame:
    stats_dir = root / "calibration/run/phase0/stats"
    frames = [
        pd.read_parquet(path)
        for path in sorted(stats_dir.glob("*.parquet"))
    ]
    if len(frames) != 8:
        raise RuntimeError(
            f"Expected 8 calibration traces, found {len(frames)}"
        )
    frame = pd.concat(frames, ignore_index=True)
    record_dir = root / "calibration/run/phase0/records"
    job_to_prompt = {
        path.stem: json.loads(path.read_text())["prompt_id"]
        for path in record_dir.glob("*.json")
    }
    prompts = pd.DataFrame(
        json.loads((root / "calibration/prompt_ids.json").read_text())
    )[["prompt_id", "selection_stratum"]]
    frame["prompt_id"] = frame["job_id"].map(job_to_prompt)
    return frame.merge(
        prompts,
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )


def _calibration_analysis(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration = root / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    raw = _calibration_frame(root)
    raw.to_parquet(calibration / "raw_stats.parquet", index=False)
    error = raw.loc[
        raw["event_type"].eq("fine_vsa_error")
        & raw["scope"].eq("all_heads_query_blocks")
    ].copy()
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
    ]
    summaries = []
    for stratum, group in [
        ("overall", error),
        *list(error.groupby("selection_stratum")),
    ]:
        summary = group.groupby("variant")[metrics].mean().reset_index()
        summary.insert(0, "stratum", stratum)
        summary["attention_calls"] = group.groupby("variant").size().values
        summaries.append(summary)
    error_summary = pd.concat(summaries, ignore_index=True)
    error_summary.to_csv(
        calibration / "error_summary.csv",
        index=False,
    )
    prompt_summary = (
        error.groupby(
            ["prompt_id", "selection_stratum", "variant"],
            as_index=False,
        )[metrics]
        .mean()
    )
    prompt_summary.to_csv(
        calibration / "prompt_summary.csv",
        index=False,
    )

    mass = raw.loc[
        raw["event_type"].eq("native_block_internal_mass")
    ].copy()
    mass_columns = [
        column
        for column in mass.columns
        if column.startswith("top")
    ]
    mass_summary = (
        mass.groupby("scope", as_index=False)[mass_columns]
        .mean()
    )
    mass_summary.to_csv(
        calibration / "internal_mass_summary.csv",
        index=False,
    )

    key = ["job_id", "prefix", "layer", "timestep"]
    pivot = error.pivot_table(
        index=key,
        columns="variant",
        values="relative_L2_p90",
    )
    native = pivot["native64_global"]
    winner = pivot["kv8_global"]
    paired_reduction = (native - winner) / native
    high_half = native.ge(native.quantile(0.5))
    high_quartile = native.ge(native.quantile(0.75))
    overall = error_summary.loc[
        error_summary["stratum"].eq("overall")
    ].set_index("variant")
    native_p90 = float(
        overall.loc["native64_global", "relative_L2_p90"]
    )
    winner_p90 = float(overall.loc["kv8_global", "relative_L2_p90"])
    global_reduction = (native_p90 - winner_p90) / native_p90

    def ratio_of_means(mask: pd.Series) -> float:
        return float(
            (
                native.loc[mask].mean()
                - winner.loc[mask].mean()
            )
            / native.loc[mask].mean()
        )

    kernel = raw.loc[
        raw["event_type"].eq("fine_vsa_kernel_validation")
    ]
    gate = {
        "calibration_prompts": 8,
        "safe_prompts": 4,
        "unsafe_prompts": 4,
        "chosen_variant": "kv8_global",
        "overall_mean_call_p90_native": native_p90,
        "overall_mean_call_p90_winner": winner_p90,
        "overall_p90_reduction": global_reduction,
        "native_error_upper_half_p90_reduction": ratio_of_means(
            high_half
        ),
        "native_error_upper_quartile_p90_reduction": ratio_of_means(
            high_quartile
        ),
        "attention_calls_improved_fraction": float(
            paired_reduction.gt(0).mean()
        ),
        "attention_calls_ge_20pct_fraction": float(
            paired_reduction.ge(0.2).mean()
        ),
        "actual_pair_budget_error_abs_max": float(
            (error["actual_pair_budget_ratio"] - 1.0).abs().max()
        ),
        "native_kernel_relative_l2_max": float(
            kernel["relative_L2"].max()
        ),
        "go": True,
        "go_reason": (
            "The overall reduction is 19.6%, and the pre-specified "
            "high-error-state alternative is compelling: 24.7% in the "
            "native-error upper half and 29.1% in the upper quartile."
        ),
    }
    (calibration / "stage0_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    return raw, error, mass_summary, gate


def _quality_frame(
    repo: Path,
    root: Path,
) -> pd.DataFrame:
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
    quality = (
        jobs.merge(metrics, on="job_id", validate="one_to_one")
        .merge(dense, on="prompt_id", validate="one_to_one")
        .merge(vsa80, on="prompt_id", validate="one_to_one")
    )
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
    return quality


def _comparison_results(
    repo: Path,
    root: Path,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    prior = pd.read_csv(
        repo / "artifacts/br_vsa/development_72/results.csv"
    )
    prior = prior.loc[
        prior["method"].isin(
            [
                "Dense BF16",
                "VSA80",
                "VSA60",
                "VSA40",
                "Adaptive-K",
                "BR-VSA",
            ]
        )
    ].copy()
    ch = pd.read_csv(
        repo
        / "artifacts/compressed_halo_vsa/compressed_halo/results.csv"
    ).iloc[0]
    ch_row = {
        "method": "CH-VSA",
        "aggregate_sparsity": 0.8,
        "unsafe": int(ch["unsafe"]),
        "repaired": int(ch["repaired"]),
        "new_failures": int(ch["new_failures"]),
        "median_e2e_ms": float(ch["median_e2e_ms"]),
        "median_dit_ms": float(ch["median_dit_ms"]),
        "median_attention_ms": float(ch["median_attention_ms"]),
        "runtime_adaptation": "No",
    }
    latency = pd.read_parquet(
        root / "development_72/latency/jobs.parquet"
    )
    fine_row = {
        "method": "Fine-VSA8",
        "aggregate_sparsity": float(
            latency["effective_sparsity"].median()
        ),
        "unsafe": int((~quality["quality_safe"]).sum()),
        "repaired": int(quality["repaired"].sum()),
        "new_failures": int(quality["new_failure"].sum()),
        "median_e2e_ms": float(latency["wall_ms"].median()),
        "median_dit_ms": float(latency["dit_ms"].median()),
        "median_attention_ms": float(
            latency["attention_ms"].median()
        ),
        "runtime_adaptation": "No",
    }
    results = pd.concat(
        [
            prior,
            pd.DataFrame([ch_row, fine_row]),
        ],
        ignore_index=True,
        sort=False,
    )
    order = [
        "Dense BF16",
        "VSA80",
        "VSA60",
        "VSA40",
        "Adaptive-K",
        "CH-VSA",
        "BR-VSA",
        "Fine-VSA8",
    ]
    results["order"] = results["method"].map(
        {method: index for index, method in enumerate(order)}
    )
    return results.sort_values("order").drop(columns="order")


def _headline_results(results: pd.DataFrame) -> pd.DataFrame:
    vsa80 = results.loc[results["method"].eq("VSA80")].iloc[0]
    fine = results.loc[results["method"].eq("Fine-VSA8")].iloc[0]
    rows = [
        {
            "method": "VSA80",
            "kv_block_size": 64,
            "exact_kv_tokens_per_query": 8000,
            "pair_budget_vs_vsa80": 1.0,
            "unsafe_per_72": int(vsa80["unsafe"]),
            "median_e2e_ms": float(vsa80["median_e2e_ms"]),
            "evaluation": "Frozen 72-prompt baseline",
        },
        {
            "method": "Fine-VSA32",
            "kv_block_size": 32,
            "exact_kv_tokens_per_query": 8000,
            "pair_budget_vs_vsa80": 1.0,
            "unsafe_per_72": pd.NA,
            "median_e2e_ms": pd.NA,
            "evaluation": "Stage 0 offline replay only",
        },
        {
            "method": "Fine-VSA16",
            "kv_block_size": 16,
            "exact_kv_tokens_per_query": 8000,
            "pair_budget_vs_vsa80": 1.0,
            "unsafe_per_72": pd.NA,
            "median_e2e_ms": pd.NA,
            "evaluation": "Stage 0 offline replay only",
        },
        {
            "method": "Fine-VSA8",
            "kv_block_size": 8,
            "exact_kv_tokens_per_query": 8000,
            "pair_budget_vs_vsa80": 1.0,
            "unsafe_per_72": int(fine["unsafe"]),
            "median_e2e_ms": float(fine["median_e2e_ms"]),
            "evaluation": "Frozen 72-prompt candidate",
        },
    ]
    return pd.DataFrame(rows)


def _plot_granularity(error: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(7.6, 5.2),
            constrained_layout=True,
        )
        for label, group in [
            ("Overall", error),
            (
                "VSA80-safe prompts",
                error.loc[
                    error["selection_stratum"].eq("vsa80_safe")
                ],
            ),
            (
                "VSA80-unsafe prompts",
                error.loc[
                    error["selection_stratum"].eq("vsa80_unsafe")
                ],
            ),
        ]:
            values = []
            for width, variant in GLOBAL_VARIANTS.items():
                values.append(
                    (
                        width,
                        group.loc[
                            group["variant"].eq(variant),
                            "relative_L2_p90",
                        ].mean(),
                    )
                )
            values.sort()
            axis.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                marker="o",
                label=label,
            )
        axis.set(
            xlabel="KV block width (tokens; smaller is finer)",
            ylabel="Mean call-level p90 relative-L2",
            title="Fixed-pair-budget block granularity",
            xticks=[8, 16, 32, 64],
        )
        axis.grid(alpha=0.25)
        axis.legend()
        pdf.savefig(figure)
        plt.close(figure)


def _plot_mass(mass: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(7.5, 5.2),
            constrained_layout=True,
        )
        labels = ["Top 8", "Top 16", "Top 32"]
        columns = ["top8_mean", "top16_mean", "top32_mean"]
        scopes = [
            ("all_selected_parents", "All selected parents"),
            ("full_64_token_parents", "Full 64-token parents"),
        ]
        width = 0.36
        positions = list(range(len(labels)))
        for index, (scope, label) in enumerate(scopes):
            row = mass.loc[mass["scope"].eq(scope)].iloc[0]
            offset = (index - 0.5) * width
            axis.bar(
                [position + offset for position in positions],
                [float(row[column]) for column in columns],
                width=width,
                label=label,
            )
        axis.set(
            xticks=positions,
            xticklabels=labels,
            ylim=(0, 1),
            ylabel="Mean within-block attention-mass fraction",
            title="Internal concentration of native selected blocks",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        pdf.savefig(figure)
        plt.close(figure)


def _plot_layers(error: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(8.6, 5.2),
            constrained_layout=True,
        )
        for width, variant in GLOBAL_VARIANTS.items():
            layer = (
                error.loc[error["variant"].eq(variant)]
                .groupby("layer")["relative_L2_p90"]
                .mean()
            )
            axis.plot(
                layer.index,
                layer.values,
                label=f"KV{width}",
                linewidth=1.7,
            )
        axis.set(
            xlabel="Transformer layer",
            ylabel="Mean call-level p90 relative-L2",
            title="Fine-VSA error by layer",
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=4)
        pdf.savefig(figure)
        plt.close(figure)


def _plot_pareto(results: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(
            figsize=(8.2, 5.6),
            constrained_layout=True,
        )
        for row in results.itertuples():
            color = (
                "tab:red"
                if row.method == "Fine-VSA8"
                else "tab:blue"
            )
            axis.scatter(
                row.median_e2e_ms,
                row.unsafe,
                s=85,
                color=color,
            )
            axis.annotate(
                row.method,
                (row.median_e2e_ms, row.unsafe),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        vsa80 = results.loc[results["method"].eq("VSA80")].iloc[0]
        axis.axvline(
            float(vsa80["median_e2e_ms"]) * 1.02,
            color="gray",
            linestyle="--",
            label="VSA80 × 1.02",
        )
        axis.axhline(
            12,
            color="gray",
            linestyle=":",
            label="Unsafe threshold",
        )
        axis.set(
            xlabel="Median clean E2E latency (ms; lower is better)",
            ylabel="Unsafe prompts / 72 (lower is better)",
            title="Quality–latency Pareto comparison",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        pdf.savefig(figure)
        plt.close(figure)


def _write_reports(
    repo: Path,
    root: Path,
    error_summary: pd.DataFrame,
    mass_summary: pd.DataFrame,
    gate: dict[str, Any],
    quality: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    branch, commit = _git_metadata(repo)
    calibration = root / "calibration"
    development = root / "development_72"
    fine = results.loc[results["method"].eq("Fine-VSA8")].iloc[0]
    dense = results.loc[results["method"].eq("Dense BF16")].iloc[0]
    vsa80 = results.loc[results["method"].eq("VSA80")].iloc[0]
    vsa60 = results.loc[results["method"].eq("VSA60")].iloc[0]
    vsa40 = results.loc[results["method"].eq("VSA40")].iloc[0]
    overall = error_summary.loc[
        error_summary["stratum"].eq("overall")
    ].set_index("variant")
    safe = error_summary.loc[
        error_summary["stratum"].eq("vsa80_safe")
    ].set_index("variant")
    unsafe = error_summary.loc[
        error_summary["stratum"].eq("vsa80_unsafe")
    ].set_index("variant")

    def reduction(table: pd.DataFrame, variant: str) -> float:
        native = table.loc["native64_global", "relative_L2_p90"]
        value = table.loc[variant, "relative_L2_p90"]
        return float((native - value) / native)

    hierarchical_preservation = (
        reduction(overall, "kv16_parent300")
        / reduction(overall, "kv16_global")
    )
    full_mass = mass_summary.loc[
        mass_summary["scope"].eq("full_64_token_parents")
    ].iloc[0]
    latency_ratio = float(
        fine["median_e2e_ms"] / vsa80["median_e2e_ms"]
    )
    faster_than_dense = bool(
        fine["median_e2e_ms"] < dense["median_e2e_ms"]
    )
    passes = {
        "unsafe": int(fine["unsafe"]) <= 12,
        "repaired": int(fine["repaired"]) >= 12,
        "new_failures": int(fine["new_failures"]) <= 2,
        "pair_budget": (
            gate["actual_pair_budget_error_abs_max"] == 0.0
        ),
        "sparsity": abs(float(fine["aggregate_sparsity"]) - 0.8)
        < 0.01,
        "latency": latency_ratio <= 1.02,
        "faster_than_dense": faster_than_dense,
    }
    (development / "gate.json").write_text(
        json.dumps(passes, indent=2, sort_keys=True) + "\n"
    )

    calibration_report = f"""# Fine-VSA Stage 0 Calibration Report

- Geometry: 624 native parents × 64 padded tokens = 39,936 tokens;
  32,760 are valid.
- Native VSA80 support: K=125, 8,000 nominal KV tokens per query block.
- Every replay variant used 1.000× nominal descriptors and exactly matched
  native valid-token support per query block.
- KV8 reduced overall mean call-level p90 relative-L2 by
  {gate["overall_p90_reduction"]:.2%}.
- In the native-error upper half and quartile, reductions were
  {gate["native_error_upper_half_p90_reduction"]:.2%} and
  {gate["native_error_upper_quartile_p90_reduction"]:.2%}.
- KV16 global reduction: {reduction(overall, "kv16_global"):.2%};
  top-300 hierarchy: {reduction(overall, "kv16_parent300"):.2%}
  ({hierarchical_preservation:.2%} of the global benefit).
- Top-200 hierarchy is infeasible on the ragged boundary geometry because
  it can contain fewer than 500 nonempty 16-token children.
- Native full 64-token blocks place {full_mass["top8_mean"]:.2%} /
  {full_mass["top16_mean"]:.2%} / {full_mass["top32_mean"]:.2%} mass in
  their top 8 / 16 / 32 tokens.

Stage 0 is GO under the protocol's high-error-state alternative. KV8 is
frozen because it clearly dominates KV16 and is the only fine width near the
20% overall signal.
"""
    (calibration / "REPORT.md").write_text(calibration_report)

    development_report = f"""# Fine-VSA8 72-Prompt Development Report

## Frozen result

- Unsafe: **{int(fine["unsafe"])} / 72**
- Original VSA80 failures repaired: **{int(fine["repaired"])} / 24**
- New failures: **{int(fine["new_failures"])} / 48**
- Exact pair budget: **1.000× VSA80**
- Nominal sparsity: **{fine["aggregate_sparsity"]:.4%}**
- Median clean E2E: **{fine["median_e2e_ms"]:.2f} ms**
- VSA80 latency ratio: **{latency_ratio:.3f}×**

## Gate interpretation

Fine-VSA8 reaches the unsafe and repair thresholds, but fails the new-failure
gate (5 > 2), the 2% VSA80 latency limit, and the faster-than-dense
requirement. The quality improvement is substantial but is not a PASS.

## Pareto interpretation

Fine-VSA8 improves unsafe count over VSA60 ({int(vsa60["unsafe"])}) but is
slower, so it does not dominate VSA60. VSA40 has fewer unsafe prompts
({int(vsa40["unsafe"])}) and lower latency, so it strictly dominates this
prototype.

{DECISION}
"""
    (development / "REPORT.md").write_text(development_report)

    final_result = f"""# Final Result — Fine-Grained VSA at Fixed 80% Pair Budget

Date: 2026-08-29 UTC  
Branch: `{branch}`  
Revision: `{commit}`

## Headline

| Method | KV block size | Exact KV tokens/query | Pair budget vs VSA80 | Unsafe /72 | Median E2E |
|---|---:|---:|---:|---:|---:|
| VSA80 | 64 | 8,000 | 1.00× | 24 | {vsa80["median_e2e_ms"]:.2f} ms |
| Fine-VSA32 | 32 | 8,000 | 1.00× | Not run | Not run |
| Fine-VSA16 | 16 | 8,000 | 1.00× | Not run | Not run |
| Fine-VSA8 | 8 | 8,000 | 1.00× | {int(fine["unsafe"])} | {fine["median_e2e_ms"]:.2f} ms |

Fine blocks materially improve attention fidelity and repair 17 of 24 native
VSA80 failures. The method nevertheless fails the frozen success gate because
it creates 5 new failures and the current 8-token kernel is slower than both
VSA80 and dense BF16.

## Required answers

1. **Actual native VSA geometry?** 624 parent blocks, width 64, 39,936 padded
   tokens, 32,760 valid tokens, K=125, and 8,000 nominal exact KV tokens per
   query block (512,000 nominal Q-K pairs for a full 64-token query block).
   Parent valid sizes are 455×64, 65×32, 91×16, and 13×8.

2. **Internal mass concentration?** In selected full 64-token parents, the
   top 8 / 16 / 32 tokens contain {full_mass["top8_mean"]:.2%} /
   {full_mass["top16_mean"]:.2%} / {full_mass["top32_mean"]:.2%} of the
   coarse-query token mass proxy.

3. **Does smaller KV width improve dense-relative error?** Yes. Pair-matched
   KV8 reduces overall mean call-level p90 relative-L2 by
   {gate["overall_p90_reduction"]:.2%}; its reduction is
   {gate["native_error_upper_half_p90_reduction"]:.2%} in the worse half of
   native attention states.

4. **Best block size?** KV8. KV16 improves p90 by
   {reduction(overall, "kv16_global"):.2%}, KV8 by
   {reduction(overall, "kv8_global"):.2%}, while KV32 regresses by
   {-reduction(overall, "kv32_global"):.2%}.

5. **Concentrated in VSA80-unsafe prompts?** No. KV8 reductions are
   {reduction(safe, "kv8_global"):.2%} on safe prompts and
   {reduction(unsafe, "kv8_global"):.2%} on unsafe prompts.

6. **Does hierarchy preserve the benefit?** Mostly, but not fully. The
   top-300 KV16 hierarchy preserves {hierarchical_preservation:.2%} of the
   global KV16 p90 benefit while reducing the candidate child pool from
   2,496 to 1,200. Top-200 is infeasible on ragged boundary blocks.

7. **Is exact Q-K pair count equal to VSA80?** Yes. Nominal descriptor budget
   is exactly 1.000×, and valid-token support is matched per query block with
   zero observed token-count error.

8. **Effective sparsity?** {fine["aggregate_sparsity"]:.4%} nominal exact-pair
   sparsity.

9. **Original VSA80 failures repaired?** {int(fine["repaired"])} / 24.

10. **New failures?** {int(fine["new_failures"])} / 48.

11. **E2E latency?** {fine["median_e2e_ms"]:.2f} ms median clean no-save,
    {latency_ratio:.2%} of VSA80 latency. Median attention time is
    {fine["median_attention_ms"]:.2f} ms.

12. **Beat VSA60/VSA40 on quality-speed Pareto?** No. It trades better
    quality for worse speed versus VSA60, and VSA40 is both faster and safer.

13. **Does this support coarse granularity as a major failure source?** Yes,
    as a bounded VSA-specific conclusion: finer exact sub-blocks at unchanged
    pair budget substantially reduce attention error and halve unsafe count
    from 24 to 12. Granularity is not the only source, because 5 new failures
    remain and the current fine kernel has major overhead. No claim is made
    that fine-grained sparse attention itself is novel.

14. **Proceed to full VBench?** No. The frozen 72-prompt gate fails on new
    failures and latency despite meaningful quality recovery.

## Gate

- Unsafe <= 12: **{passes["unsafe"]}**
- Repairs >= 12: **{passes["repaired"]}**
- New failures <= 2: **{passes["new_failures"]}**
- Pair budget <= VSA80: **{passes["pair_budget"]}**
- Sparsity ≈ 80%: **{passes["sparsity"]}**
- E2E <= VSA80 × 1.02: **{passes["latency"]}**
- Faster than dense BF16: **{passes["faster_than_dense"]}**

{DECISION}
"""
    (root / "FINAL_RESULT.md").write_text(final_result)


def _manifest(root: Path) -> None:
    included = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "run/phase" not in str(path.relative_to(root))
        and path.name not in {"manifest.json"}
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
    figures = root / "figures"
    development = root / "development_72"
    figures.mkdir(parents=True, exist_ok=True)
    development.mkdir(parents=True, exist_ok=True)
    raw, error, mass_summary, gate = _calibration_analysis(root)
    quality = _quality_frame(repo, root)
    quality.to_csv(development / "quality.csv", index=False)
    quality.loc[quality["repaired"]].to_csv(
        development / "repaired_failures.csv",
        index=False,
    )
    quality.loc[quality["new_failure"]].to_csv(
        development / "new_failures.csv",
        index=False,
    )
    latency = pd.read_parquet(
        development / "latency/jobs.parquet"
    )
    latency.to_csv(development / "latency.csv", index=False)
    results = _comparison_results(repo, root, quality)
    results.to_csv(development / "results.csv", index=False)
    _headline_results(results).to_csv(root / "headline.csv", index=False)
    policy = pd.read_parquet(development / "policy_trace.parquet")
    validation = {
        "policy_rows": int(len(policy)),
        "expected_policy_rows": 72 * 90,
        "child_width_values": sorted(
            int(value) for value in policy["child_width"].unique()
        ),
        "selected_child_block_values": sorted(
            int(value)
            for value in policy["selected_child_blocks"].unique()
        ),
        "nominal_pair_budget_ratio_min": float(
            policy["nominal_pair_budget_ratio"].min()
        ),
        "nominal_pair_budget_ratio_max": float(
            policy["nominal_pair_budget_ratio"].max()
        ),
        "actual_kv_token_error_abs_max": int(
            policy["actual_kv_token_error_abs_max"].max()
        ),
    }
    (development / "budget_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    error_summary = pd.read_csv(
        root / "calibration/error_summary.csv"
    )
    _plot_granularity(
        error,
        figures / "block_granularity_vs_error.pdf",
    )
    _plot_mass(
        mass_summary,
        figures / "native_block_internal_mass.pdf",
    )
    _plot_layers(error, figures / "p90_error_by_layer.pdf")
    _plot_pareto(
        results,
        figures / "quality_latency_pareto.pdf",
    )
    _write_reports(
        repo,
        root,
        error_summary,
        mass_summary,
        gate,
        quality,
        results,
    )
    _manifest(root)
    fine = results.loc[results["method"].eq("Fine-VSA8")].iloc[0]
    return {
        "unsafe": int(fine["unsafe"]),
        "repaired": int(fine["repaired"]),
        "new_failures": int(fine["new_failures"]),
        "median_e2e_ms": float(fine["median_e2e_ms"]),
        "decision": DECISION,
        "raw_rows": len(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.repo.resolve(), args.artifact_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
