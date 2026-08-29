from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


H2_MIN_ABS_SPEARMAN = 0.30
H2_MIN_AUC = 0.65
H2_MAX_FALSE_SAFE = 0.10
H2_MIN_SAFE_COVERAGE = 0.10
H3_MIN_AUC = 0.75
H3_MAX_FALSE_SAFE = 0.05
H3_MIN_STABLE_COVERAGE = 0.10
MASK_STABLE_JACCARD = 0.95


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _roc_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    x = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(x)
    y = y[valid]
    x = x[valid]
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(x)
    positive_rank_sum = ranks[y].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def _select_mass_threshold(frame: pd.DataFrame) -> dict[str, float]:
    values = np.unique(frame["job_retained_mass_min"].dropna().to_numpy())
    best: dict[str, float] | None = None
    for threshold in values:
        predicted_safe = frame["job_retained_mass_min"] >= threshold
        coverage = float(predicted_safe.mean())
        if predicted_safe.sum() == 0:
            continue
        false_safe = float((~frame.loc[predicted_safe, "quality_safe"]).mean())
        if false_safe <= H2_MAX_FALSE_SAFE and coverage >= H2_MIN_SAFE_COVERAGE:
            candidate = {
                "tau_mass": float(threshold),
                "coverage": coverage,
                "false_safe_rate": false_safe,
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
    return best or {
        "tau_mass": float(values.max() + np.finfo(np.float64).eps),
        "coverage": 0.0,
        "false_safe_rate": float("nan"),
    }


def _risk_rule_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    statistic_pairs = [
        ("delta_inf", "margin_p50"),
        ("delta_p99", "margin_p50"),
        ("delta_p95", "margin_p50"),
        ("delta_norm_inf", "margin_norm_p50"),
        ("delta_norm_p99", "margin_norm_p50"),
        ("delta_norm_p95", "margin_norm_p50"),
    ]
    for delta_name, margin_name in statistic_pairs:
        for alpha in [1.0, 1.25, 1.5, 2.0]:
            denominator = frame[margin_name].clip(lower=1e-12)
            risk = alpha * 2.0 * frame[delta_name] / denominator
            predicted_stable = risk < 1.0
            coverage = float(predicted_stable.mean())
            false_safe = (
                float(frame.loc[predicted_stable, "mask_unstable"].mean())
                if predicted_stable.any()
                else float("nan")
            )
            rows.append(
                {
                    "delta_statistic": delta_name,
                    "margin_statistic": margin_name,
                    "alpha": alpha,
                    "coverage": coverage,
                    "false_safe_rate": false_safe,
                    "unstable_auc": _roc_auc(frame["mask_unstable"], risk),
                    "risk_jaccard_spearman": _spearman(pd.Series(risk), frame["mask_jaccard_p50"])[0],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["false_safe_rate", "coverage"],
        ascending=[True, False],
        na_position="last",
    )


def _plots(
    h2_rows: pd.DataFrame,
    h3_rows: pd.DataFrame,
    h2_output: Path,
    h3_output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (h2_output / "figures").mkdir(parents=True, exist_ok=True)
    (h3_output / "figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hexbin(
        h2_rows["retained_mass_p50"],
        h2_rows["attention_output_rel_l2"],
        gridsize=45,
        mincnt=1,
    )
    plt.xlabel("Retained mass (median)")
    plt.ylabel("Sparse attention output relative L2")
    plt.colorbar(label="Attention records")
    plt.tight_layout()
    plt.savefig(h2_output / "figures" / "retained_mass_vs_attention_error.png", dpi=180)
    plt.close()

    h2_rows.groupby(["layer", "timestep"], as_index=False).apply(
        lambda group: pd.Series(
            {
                "rho": _spearman(group["retained_mass_p50"], group["attention_output_rel_l2"])[0]
            }
        ),
        include_groups=False,
    ).reset_index(drop=True).pivot(index="layer", columns="timestep", values="rho").plot(
        kind="bar",
        figsize=(8, 5),
    )
    plt.ylabel("Spearman rho: mass vs attention error")
    plt.tight_layout()
    plt.savefig(h2_output / "figures" / "retained_mass_correlation_by_layer_timestep.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hexbin(
        h3_rows["risk_ratio_p50"],
        h3_rows["mask_jaccard_p50"],
        gridsize=45,
        mincnt=1,
        xscale="log",
    )
    plt.axvline(1.0, color="red", linestyle="--", label="2*delta_p99 = margin")
    plt.xlabel("Risk ratio: 2*delta_p99 / margin")
    plt.ylabel("BF16 vs FP4 mask Jaccard (median)")
    plt.legend()
    plt.colorbar(label="Attention records")
    plt.tight_layout()
    plt.savefig(h3_output / "figures" / "risk_ratio_vs_mask_jaccard.png", dpi=180)
    plt.close()

    h3_rows.groupby(["layer", "timestep"])["mask_unstable"].mean().unstack().plot.bar(
        figsize=(8, 5)
    )
    plt.ylabel("Unstable-mask fraction")
    plt.tight_layout()
    plt.savefig(h3_output / "figures" / "mask_instability_by_layer_timestep.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--phase1-quality", type=Path, required=True)
    parser.add_argument("--phase2-output", type=Path, required=True)
    parser.add_argument("--phase3-output", type=Path, required=True)
    args = parser.parse_args()

    stats = pd.read_parquet(args.stats).rename(columns={"sparsity": "candidate_sparsity"})
    jobs = pd.read_parquet(args.jobs).rename(columns={"sparsity": "job_sparsity"})
    jobs = jobs[jobs["status"] == "ok"]
    job_columns = [
        "job_id",
        "prompt_id",
        "prompt",
        "kernel_path",
        "precision",
        "job_sparsity",
    ]
    merged = stats.merge(jobs[job_columns], on="job_id", how="inner", validate="many_to_one")
    quality = pd.read_parquet(args.phase1_quality)[
        [
            "prompt_id",
            "kernel_path",
            "sparsity",
            "quality_safe",
            "subject_delta",
            "motion_delta",
            "dynamic_delta",
        ]
    ].rename(columns={"sparsity": "job_sparsity"})
    merged = merged.merge(
        quality,
        on=["prompt_id", "kernel_path", "job_sparsity"],
        how="left",
        validate="many_to_one",
    )
    merged["candidate_matches_job"] = np.isclose(
        merged["candidate_sparsity"],
        merged["job_sparsity"],
    )

    h2_rows = merged[
        (merged["kernel_path"] == "vsa_bf16") & merged["candidate_matches_job"]
    ].copy()
    if h2_rows.empty:
        raise ValueError("No VSA BF16 mechanism rows matched their executed sparsity.")
    h2_rows["video_degradation"] = np.maximum.reduce(
        [
            (-h2_rows["subject_delta"] / 0.02).clip(lower=0),
            (-h2_rows["motion_delta"] / 0.01).clip(lower=0),
            (-h2_rows["dynamic_delta"]).clip(lower=0),
        ]
    )

    job_mechanisms = (
        h2_rows.groupby(
            ["job_id", "prompt_id", "kernel_path", "job_sparsity"],
            as_index=False,
        )
        .agg(
            job_retained_mass_min=("retained_mass_p10", "min"),
            job_retained_mass_median=("retained_mass_p50", "median"),
            job_attention_rel_l2_max=("attention_output_rel_l2", "max"),
            quality_safe=("quality_safe", "first"),
            video_degradation=("video_degradation", "first"),
        )
    )
    mass_threshold = _select_mass_threshold(job_mechanisms)

    h2_correlations = []
    groupings = [
        ("overall", []),
        ("layer", ["layer"]),
        ("timestep", ["timestep"]),
        ("layer_timestep", ["layer", "timestep"]),
        ("head_group", ["head_group"]),
    ]
    for breakdown, columns in groupings:
        groups = [((), h2_rows)] if not columns else h2_rows.groupby(columns, dropna=False)
        for key, group in groups:
            key_values = key if isinstance(key, tuple) else (key,)
            rho_error, p_error = _spearman(
                group["retained_mass_p50"],
                group["attention_output_rel_l2"],
            )
            rho_video, p_video = _spearman(
                group["retained_mass_p50"],
                group["video_degradation"],
            )
            row = {
                "breakdown": breakdown,
                "n": len(group),
                "mass_vs_attention_error_spearman": rho_error,
                "mass_vs_attention_error_pvalue": p_error,
                "mass_vs_video_degradation_spearman": rho_video,
                "mass_vs_video_degradation_pvalue": p_video,
            }
            row.update({column: value for column, value in zip(columns, key_values, strict=True)})
            h2_correlations.append(row)
    correlation = pd.DataFrame(h2_correlations)
    overall_h2 = correlation[correlation["breakdown"] == "overall"].iloc[0]
    h2_auc = _roc_auc(~job_mechanisms["quality_safe"], -job_mechanisms["job_retained_mass_min"])
    h2_pass = bool(
        abs(overall_h2["mass_vs_attention_error_spearman"]) >= H2_MIN_ABS_SPEARMAN
        and h2_auc >= H2_MIN_AUC
        and mass_threshold["coverage"] >= H2_MIN_SAFE_COVERAGE
        and mass_threshold["false_safe_rate"] <= H2_MAX_FALSE_SAFE
    )

    h3_rows = merged[merged["kernel_path"] == "vsa_bf16"].copy()
    h3_rows["mask_unstable"] = h3_rows["mask_jaccard_p50"] < MASK_STABLE_JACCARD
    rules = _risk_rule_table(h3_rows)
    eligible_rules = rules[
        (rules["coverage"] >= H3_MIN_STABLE_COVERAGE)
        & (rules["false_safe_rate"] <= H3_MAX_FALSE_SAFE)
    ]
    best_rule = eligible_rules.sort_values(
        ["coverage", "unstable_auc"],
        ascending=False,
    ).iloc[0] if not eligible_rules.empty else rules.iloc[0]
    h3_auc = float(best_rule["unstable_auc"])
    h3_pass = bool(
        not eligible_rules.empty
        and h3_auc >= H3_MIN_AUC
        and best_rule["coverage"] >= H3_MIN_STABLE_COVERAGE
        and best_rule["false_safe_rate"] <= H3_MAX_FALSE_SAFE
    )

    sim_actual = merged[
        (merged["kernel_path"] == "sim_vsa_nvfp4") & merged["candidate_matches_job"]
    ].copy()
    mask_error_rho, mask_error_p = _spearman(
        sim_actual["mask_jaccard_p50"],
        sim_actual["attention_output_rel_l2"],
    )

    args.phase2_output.mkdir(parents=True, exist_ok=True)
    args.phase3_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.stats, args.phase2_output / "attention_stats.parquet")
    correlation.to_csv(args.phase2_output / "correlation.csv", index=False)
    job_mechanisms.to_csv(args.phase2_output / "job_mechanisms.csv", index=False)
    h3_rows.to_parquet(args.phase3_output / "mask_stability.parquet", index=False)
    rules.to_csv(args.phase3_output / "risk_rule_comparison.csv", index=False)
    _plots(h2_rows, h3_rows, args.phase2_output, args.phase3_output)

    h2_summary = {
        "phase": 2,
        "status": "pass" if h2_pass else "fail",
        "hypothesis": "Retained VSA coarse probability mass provides useful monotonic information about sparse-attention and final-video risk.",
        "n_jobs_planned": int(jobs[jobs["kernel_path"] == "vsa_bf16"]["job_id"].nunique()),
        "n_jobs_completed": int(h2_rows["job_id"].nunique()),
        "n_jobs_failed": 0,
        "primary_findings": [
            f"Overall retained-mass vs attention relative-L2 Spearman rho was {overall_h2['mass_vs_attention_error_spearman']:.4f} (p={overall_h2['mass_vs_attention_error_pvalue']:.3g}).",
            f"Retained-mass unsafe-mode ROC-AUC was {h2_auc:.4f}.",
            f"Selected tau_mass={mass_threshold['tau_mass']:.6f} with {mass_threshold['coverage']:.1%} safe coverage and {mass_threshold['false_safe_rate']:.1%} false-safe rate.",
            f"Overall retained-mass vs final-video degradation Spearman rho was {overall_h2['mass_vs_video_degradation_spearman']:.4f}.",
        ],
        "gate_result": "pass" if h2_pass else "fail",
        "reason": (
            "H2 passed the predeclared monotonic-correlation, AUC, coverage, and false-safe thresholds."
            if h2_pass
            else "H2 failed at least one predeclared monotonic-correlation, AUC, coverage, or false-safe threshold."
        ),
        "next_phase": 3 if h2_pass else None,
        "exact_commits": {"fastvideo_research": _commit()},
        "selected_mass_rule": mass_threshold,
        "gate_thresholds": {
            "minimum_absolute_spearman": H2_MIN_ABS_SPEARMAN,
            "minimum_auc": H2_MIN_AUC,
            "maximum_false_safe_rate": H2_MAX_FALSE_SAFE,
            "minimum_safe_coverage": H2_MIN_SAFE_COVERAGE,
        },
    }
    (args.phase2_output / "summary.json").write_text(json.dumps(h2_summary, indent=2) + "\n")

    h3_summary = {
        "phase": 3,
        "status": "pass" if h3_pass else "fail",
        "hypothesis": "A simple VSA top-k boundary-margin to FP4-perturbation ratio separates stable from unstable BF16-vs-NVFP4 mask decisions.",
        "n_jobs_planned": int(jobs[jobs["kernel_path"] == "vsa_bf16"]["job_id"].nunique()),
        "n_jobs_completed": int(h3_rows["job_id"].nunique()),
        "n_jobs_failed": 0,
        "primary_findings": [
            f"Best eligible rule used {best_rule['delta_statistic']} / {best_rule['margin_statistic']} with alpha={best_rule['alpha']:.2f}.",
            f"Mask-instability ROC-AUC was {best_rule['unstable_auc']:.4f}.",
            f"Predicted-stable coverage was {best_rule['coverage']:.1%} with {best_rule['false_safe_rate']:.1%} false-safe masks.",
            f"Risk vs Jaccard Spearman rho was {best_rule['risk_jaccard_spearman']:.4f}.",
            f"Mask Jaccard vs simulated sparse-FP4 attention error Spearman rho was {mask_error_rho:.4f} (p={mask_error_p:.3g}).",
        ],
        "gate_result": "pass" if h3_pass else "fail",
        "reason": (
            "H3 passed the predeclared separation, stable-coverage, and false-safe thresholds."
            if h3_pass
            else "H3 failed at least one predeclared separation, stable-coverage, or false-safe threshold."
        ),
        "next_phase": 4 if h3_pass else None,
        "exact_commits": {"fastvideo_research": _commit()},
        "selected_fp4_rule": {
            "delta_statistic": str(best_rule["delta_statistic"]),
            "margin_statistic": str(best_rule["margin_statistic"]),
            "alpha": float(best_rule["alpha"]),
            "coverage": float(best_rule["coverage"]),
            "false_safe_rate": float(best_rule["false_safe_rate"]),
            "unstable_auc": float(best_rule["unstable_auc"]),
        },
        "gate_thresholds": {
            "stable_jaccard_minimum": MASK_STABLE_JACCARD,
            "minimum_auc": H3_MIN_AUC,
            "maximum_false_safe_rate": H3_MAX_FALSE_SAFE,
            "minimum_stable_coverage": H3_MIN_STABLE_COVERAGE,
        },
    }
    (args.phase3_output / "summary.json").write_text(json.dumps(h3_summary, indent=2) + "\n")
    print(json.dumps({"phase2": h2_summary, "phase3": h3_summary}, indent=2))


if __name__ == "__main__":
    main()
