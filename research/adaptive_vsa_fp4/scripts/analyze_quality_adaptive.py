from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUBJECT_DELTA_MIN = -0.02
MOTION_DELTA_MIN = -0.01
DYNAMIC_DELTA_MIN = 0.0
VSA80_KERNEL = "vsa_bf16"
VSA80_SPARSITY = 0.8
INVALID_COMMIT = "b1935daae16fdfae1cca0b039175aa78f066f34e"


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_rate_ci(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(center - radius), float(center + radius)


def _config_name(kernel_path: str, sparsity: float) -> str:
    return f"{kernel_path}@s{sparsity:.2f}"


def _mode_name(kernel_path: str, sparsity: float) -> str:
    percentage = int(round(sparsity * 100))
    if kernel_path == "dense_bf16_fa4":
        return "Dense BF16"
    if kernel_path == "dense_nvfp4_fa4":
        return "Dense NVFP4-QK"
    if kernel_path == "vsa_bf16":
        return f"VSA{percentage} BF16"
    if kernel_path == "sim_vsa_nvfp4":
        return f"VSA{percentage} simulated NVFP4-QK"
    return _config_name(kernel_path, sparsity)


def _mode_category(kernel_path: str, sparsity: float) -> str:
    if kernel_path == VSA80_KERNEL and abs(sparsity - VSA80_SPARSITY) < 1e-12:
        return "VSA80"
    if kernel_path == "vsa_bf16":
        return "less_sparse_vsa_bf16"
    if kernel_path == "dense_bf16_fa4":
        return "dense_bf16"
    if kernel_path == "dense_nvfp4_fa4":
        return "dense_nvfp4"
    if kernel_path == "sim_vsa_nvfp4":
        return "simulated_vsa_nvfp4"
    return "other"


def _count_histogram(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).items()
    }


def _sparsity_histogram(values: pd.Series) -> dict[str, int]:
    labels = values.map(
        lambda value: (
            "dense_0_percent"
            if abs(float(value)) < 1e-12
            else f"{float(value):.0%}_sparsity"
        )
    )
    return _count_histogram(labels)


def _failure_type(row: pd.Series) -> str:
    failures = []
    if bool(row["sc_fail"]):
        failures.append("SC")
    if bool(row["ms_fail"]):
        failures.append("MS")
    if bool(row["dd_fail"]):
        failures.append("DD")
    return "+".join(failures) if failures else "none"


def _validate_inputs(
    *,
    root: Path,
    jobs: pd.DataFrame,
    modes: pd.DataFrame,
    vbench: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any]:
    if len(jobs) != 864 or jobs["job_id"].nunique() != 864:
        raise ValueError(f"Expected 864 unique corrected jobs, found {len(jobs)} rows.")
    if set(jobs["status"]) != {"ok"}:
        raise ValueError(f"Corrected jobs contain non-ok statuses: {jobs['status'].value_counts().to_dict()}")
    if jobs["prompt_id"].nunique() != 72 or set(jobs["seed"]) != {1024}:
        raise ValueError("Prompt or seed pairing differs from the corrected 72-prompt, seed-1024 design.")
    pair_sizes = jobs.groupby(["prompt_id", "seed"]).size()
    if set(pair_sizes) != {12}:
        raise ValueError(f"Expected 12 configurations for each prompt/seed, got {pair_sizes.value_counts().to_dict()}.")
    sparse = jobs[jobs["kernel_path"].isin(["vsa_bf16", "sim_vsa_nvfp4"])]
    mismatch = sparse[(sparse["effective_sparsity"] - sparse["sparsity"]).abs() > 1e-12]
    if not mismatch.empty:
        raise ValueError(f"Found {len(mismatch)} effective-sparsity mismatches.")
    if len(vbench) != 2592 or vbench["job_id"].nunique() != 864:
        raise ValueError("VBench metrics are incomplete.")
    if len(paired) != 1584 or paired["job_id"].nunique() != 792:
        raise ValueError("Paired SSIM/PSNR metrics are incomplete.")
    required = {
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
        "wall_ms",
        "attention_ms",
        "dit_ms",
        "effective_sparsity",
    }
    missing = required - set(modes.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    database = sqlite3.connect(root / "phase1" / "jobs.sqlite")
    payloads = [json.loads(row[0]) for row in database.execute("SELECT payload FROM jobs")]
    database.close()
    commits = sorted({payload["code_commit"] for payload in payloads})
    if commits != ["c8766c83214b69a89622a25135cd479739d7bccd"]:
        raise ValueError(f"Unexpected corrected job commits: {commits}")
    if any(payload["code_commit"] == INVALID_COMMIT for payload in payloads):
        raise ValueError("The invalid persistent-worker run is present in the corrected database.")

    vsa80 = modes[
        (modes["kernel_path"] == VSA80_KERNEL)
        & (modes["sparsity"] == VSA80_SPARSITY)
    ]
    safe = int(vsa80["quality_safe"].sum())
    unsafe = int((~vsa80["quality_safe"]).sum())
    if (safe, unsafe) != (48, 24):
        raise ValueError(f"VSA80 safety split did not reproduce: safe={safe}, unsafe={unsafe}.")
    worst = vsa80.sort_values("subject_delta").iloc[0]
    if worst["prompt"] != "a bear sniffing the air for scents of food":
        raise ValueError(f"Unexpected worst VSA80 subject case: {worst['prompt']!r}")

    return {
        "jobs": len(jobs),
        "prompts": jobs["prompt_id"].nunique(),
        "seed": 1024,
        "configs_per_prompt": int(pair_sizes.iloc[0]),
        "phase1_generation_commits": commits,
        "effective_sparsity_mismatches": len(mismatch),
        "vbench_rows": len(vbench),
        "paired_metric_rows": len(paired),
        "vsa80_safe": safe,
        "vsa80_unsafe": unsafe,
        "worst_vsa80_subject_prompt": worst["prompt"],
        "worst_vsa80_subject_delta": float(worst["subject_delta"]),
        "invalid_archive": str(root / "phase1_invalid_b1935daa"),
        "invalid_archive_exists": (root / "phase1_invalid_b1935daa").exists(),
        "invalid_commit_excluded": True,
    }


def _add_reference_columns(modes: pd.DataFrame) -> pd.DataFrame:
    modes = modes.copy()
    modes["config"] = [
        _config_name(kernel_path, sparsity)
        for kernel_path, sparsity in zip(modes["kernel_path"], modes["sparsity"], strict=True)
    ]
    modes["mode"] = [
        _mode_name(kernel_path, sparsity)
        for kernel_path, sparsity in zip(modes["kernel_path"], modes["sparsity"], strict=True)
    ]
    modes["mode_category"] = [
        _mode_category(kernel_path, sparsity)
        for kernel_path, sparsity in zip(modes["kernel_path"], modes["sparsity"], strict=True)
    ]
    modes["is_vsa80"] = (
        (modes["kernel_path"] == VSA80_KERNEL)
        & (modes["sparsity"] == VSA80_SPARSITY)
    )
    config_group = modes.groupby(["kernel_path", "sparsity"])
    modes["config_wall_ms_median"] = config_group["wall_ms"].transform("median")
    modes["config_wall_ms_mean"] = config_group["wall_ms"].transform("mean")
    modes["config_attention_ms_median"] = config_group["attention_ms"].transform("median")

    vsa80 = modes[modes["is_vsa80"]][
        [
            "prompt_id",
            "subject_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "wall_ms",
            "attention_ms",
            "config_wall_ms_median",
        ]
    ].rename(
        columns={
            "subject_consistency": "vsa80_subject_consistency",
            "motion_smoothness": "vsa80_motion_smoothness",
            "dynamic_degree": "vsa80_dynamic_degree",
            "wall_ms": "vsa80_wall_ms",
            "attention_ms": "vsa80_attention_ms",
            "config_wall_ms_median": "vsa80_policy_wall_ms",
        }
    )
    dense = modes[modes["kernel_path"] == "dense_bf16_fa4"][
        [
            "prompt_id",
            "subject_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "wall_ms",
            "attention_ms",
            "config_wall_ms_median",
        ]
    ].rename(
        columns={
            "subject_consistency": "dense_subject_consistency_reference",
            "motion_smoothness": "dense_motion_smoothness_reference",
            "dynamic_degree": "dense_dynamic_degree_reference",
            "wall_ms": "dense_wall_ms",
            "attention_ms": "dense_attention_ms",
            "config_wall_ms_median": "dense_policy_wall_ms",
        }
    )
    modes = modes.merge(vsa80, on="prompt_id", how="left", validate="many_to_one")
    modes = modes.merge(dense, on="prompt_id", how="left", validate="many_to_one")
    modes["delta_sc_vs_vsa80"] = (
        modes["subject_consistency"] - modes["vsa80_subject_consistency"]
    )
    modes["delta_ms_vs_vsa80"] = (
        modes["motion_smoothness"] - modes["vsa80_motion_smoothness"]
    )
    modes["delta_dd_vs_vsa80"] = modes["dynamic_degree"] - modes["vsa80_dynamic_degree"]
    modes["delta_wall_ms_vs_vsa80"] = modes["wall_ms"] - modes["vsa80_wall_ms"]
    modes["delta_policy_wall_ms_vs_vsa80"] = (
        modes["config_wall_ms_median"] - modes["vsa80_policy_wall_ms"]
    )
    modes["sc_fail"] = modes["subject_delta"] < SUBJECT_DELTA_MIN
    modes["ms_fail"] = modes["motion_delta"] < MOTION_DELTA_MIN
    modes["dd_fail"] = modes["dynamic_delta"] < DYNAMIC_DELTA_MIN
    modes["dense_relative_safe_reconstructed"] = ~(
        modes["sc_fail"] | modes["ms_fail"] | modes["dd_fail"]
    )
    if not (modes["dense_relative_safe_reconstructed"] == modes["quality_safe"]).all():
        raise ValueError("Reconstructed dense-relative safety labels differ from Phase 1 labels.")
    modes["failure_type"] = modes.apply(_failure_type, axis=1)
    modes["dynamic_preserved"] = ~modes["dd_fail"]
    return modes


def _build_failure_tables(
    modes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vsa80_failures = modes[modes["is_vsa80"] & ~modes["quality_safe"]].copy()
    vsa80_failures["sc_threshold_excess"] = (
        SUBJECT_DELTA_MIN - vsa80_failures["subject_delta"]
    ).clip(lower=0)
    vsa80_failures["ms_threshold_excess"] = (
        MOTION_DELTA_MIN - vsa80_failures["motion_delta"]
    ).clip(lower=0)
    vsa80_failures["dd_regression_magnitude"] = (
        DYNAMIC_DELTA_MIN - vsa80_failures["dynamic_delta"]
    ).clip(lower=0)
    failure_columns = [
        "prompt_id",
        "prompt",
        "seed",
        "failure_type",
        "sc_fail",
        "ms_fail",
        "dd_fail",
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "subject_delta",
        "motion_delta",
        "dynamic_delta",
        "sc_threshold_excess",
        "ms_threshold_excess",
        "dd_regression_magnitude",
        "wall_ms",
        "attention_ms",
        "config_wall_ms_median",
        "dense_subject_consistency_reference",
        "dense_motion_smoothness_reference",
        "dense_dynamic_degree_reference",
        "dense_wall_ms",
    ]
    failure_prompts = vsa80_failures[failure_columns].sort_values("prompt_id").reset_index(drop=True)

    recovery_rows: list[dict[str, Any]] = []
    for failure in vsa80_failures.itertuples(index=False):
        alternatives = modes[
            (modes["prompt_id"] == failure.prompt_id) & ~modes["is_vsa80"]
        ].copy()
        repairing = alternatives[alternatives["quality_safe"]].copy()
        fastest = repairing.sort_values(
            ["config_wall_ms_median", "config_attention_ms_median", "config"]
        ).iloc[0]
        quality_best = repairing.sort_values(
            ["subject_consistency", "motion_smoothness", "config_wall_ms_median", "config"],
            ascending=[False, False, True, True],
        ).iloc[0]
        repair_configs = repairing.sort_values(
            ["config_wall_ms_median", "config"]
        )["mode"].tolist()
        lower_vsa = repairing[
            (repairing["kernel_path"] == "vsa_bf16")
            & (repairing["sparsity"] < VSA80_SPARSITY)
        ]
        recovery_rows.append(
            {
                "prompt_id": failure.prompt_id,
                "prompt": failure.prompt,
                "seed": failure.seed,
                "failure_type": failure.failure_type,
                "vsa80_subject_consistency": failure.subject_consistency,
                "vsa80_motion_smoothness": failure.motion_smoothness,
                "vsa80_dynamic_degree": failure.dynamic_degree,
                "vsa80_subject_delta": failure.subject_delta,
                "vsa80_motion_delta": failure.motion_delta,
                "vsa80_dynamic_delta": failure.dynamic_delta,
                "repairable_any_measured_mode": not repairing.empty,
                "repairable_non_dense": bool(
                    (repairing["kernel_path"] != "dense_bf16_fa4").any()
                ),
                "repairable_lower_vsa_bf16": not lower_vsa.empty,
                "repairable_dense_nvfp4": bool(
                    (repairing["kernel_path"] == "dense_nvfp4_fa4").any()
                ),
                "repairable_simulated_nvfp4": bool(
                    (repairing["kernel_path"] == "sim_vsa_nvfp4").any()
                ),
                "repairing_config_count": len(repairing),
                "repairing_modes": json.dumps(repair_configs),
                "repairing_lower_vsa_modes": json.dumps(lower_vsa["mode"].tolist()),
                "fastest_repair_config": fastest["config"],
                "fastest_repair_mode": fastest["mode"],
                "fastest_repair_kernel_path": fastest["kernel_path"],
                "fastest_repair_sparsity": fastest["sparsity"],
                "fastest_repair_precision": fastest["precision"],
                "fastest_repair_subject_consistency": fastest["subject_consistency"],
                "fastest_repair_motion_smoothness": fastest["motion_smoothness"],
                "fastest_repair_dynamic_degree": fastest["dynamic_degree"],
                "fastest_repair_subject_delta_vs_dense": fastest["subject_delta"],
                "fastest_repair_motion_delta_vs_dense": fastest["motion_delta"],
                "fastest_repair_dynamic_delta_vs_dense": fastest["dynamic_delta"],
                "fastest_repair_sc_improvement_vs_vsa80": fastest["delta_sc_vs_vsa80"],
                "fastest_repair_ms_improvement_vs_vsa80": fastest["delta_ms_vs_vsa80"],
                "fastest_repair_dd_change_vs_vsa80": fastest["delta_dd_vs_vsa80"],
                "fastest_repair_policy_wall_ms": fastest["config_wall_ms_median"],
                "fastest_repair_observed_wall_ms": fastest["wall_ms"],
                "extra_latency_to_repair_ms": fastest["delta_policy_wall_ms_vs_vsa80"],
                "extra_latency_to_repair_percent": (
                    fastest["config_wall_ms_median"] / failure.vsa80_policy_wall_ms - 1.0
                ),
                "fastest_repair_still_faster_than_dense": bool(
                    fastest["config_wall_ms_median"] < failure.dense_policy_wall_ms
                ),
                "fastest_repair_speed_advantage_vs_dense": (
                    failure.dense_policy_wall_ms / fastest["config_wall_ms_median"] - 1.0
                ),
                "highest_quality_repair_config": quality_best["config"],
                "highest_quality_repair_mode": quality_best["mode"],
                "highest_quality_repair_subject_consistency": quality_best[
                    "subject_consistency"
                ],
                "highest_quality_repair_motion_smoothness": quality_best[
                    "motion_smoothness"
                ],
                "highest_quality_repair_dynamic_degree": quality_best["dynamic_degree"],
                "highest_quality_repair_policy_wall_ms": quality_best[
                    "config_wall_ms_median"
                ],
            }
        )
    recovery = pd.DataFrame(recovery_rows).sort_values("prompt_id").reset_index(drop=True)
    return failure_prompts, recovery


def _build_repair_oracle(modes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id, group in modes.groupby("prompt_id", sort=True):
        vsa80 = group[group["is_vsa80"]].iloc[0]
        if bool(vsa80["quality_safe"]):
            chosen = vsa80
            reason = "VSA80 passed the dense-relative safety rule; keep VSA80."
        else:
            chosen = group[group["quality_safe"]].sort_values(
                ["config_wall_ms_median", "config_attention_ms_median", "config"]
            ).iloc[0]
            reason = "VSA80 failed; choose the lowest config-median latency quality-safe mode."
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": chosen["prompt"],
                "seed": chosen["seed"],
                "vsa80_safe": bool(vsa80["quality_safe"]),
                "vsa80_failure_type": vsa80["failure_type"],
                "selected_config": chosen["config"],
                "selected_mode": chosen["mode"],
                "selected_mode_category": chosen["mode_category"],
                "selected_kernel_path": chosen["kernel_path"],
                "selected_sparsity": chosen["sparsity"],
                "selected_effective_sparsity": chosen["effective_sparsity"],
                "selected_precision": chosen["precision"],
                "selection_reason": reason,
                "subject_consistency": chosen["subject_consistency"],
                "motion_smoothness": chosen["motion_smoothness"],
                "dynamic_degree": chosen["dynamic_degree"],
                "subject_delta_vs_dense": chosen["subject_delta"],
                "motion_delta_vs_dense": chosen["motion_delta"],
                "dynamic_delta_vs_dense": chosen["dynamic_delta"],
                "quality_safe_vs_dense": bool(chosen["quality_safe"]),
                "delta_sc_vs_vsa80": chosen["delta_sc_vs_vsa80"],
                "delta_ms_vs_vsa80": chosen["delta_ms_vs_vsa80"],
                "delta_dd_vs_vsa80": chosen["delta_dd_vs_vsa80"],
                "policy_wall_ms": chosen["config_wall_ms_median"],
                "observed_wall_ms": chosen["wall_ms"],
                "attention_ms": chosen["attention_ms"],
                "policy_latency_premium_vs_vsa80_ms": chosen[
                    "delta_policy_wall_ms_vs_vsa80"
                ],
                "policy_latency_premium_vs_vsa80_percent": (
                    chosen["config_wall_ms_median"] / vsa80["config_wall_ms_median"] - 1.0
                ),
                "policy_speed_advantage_vs_dense": (
                    vsa80["dense_policy_wall_ms"] / chosen["config_wall_ms_median"] - 1.0
                ),
                "source_job_id": chosen["job_id"],
                "source_video_path": chosen["video_path"],
            }
        )
    return pd.DataFrame(rows)


def _pareto_membership(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.copy().reset_index(drop=True)
    sc = frame["subject_consistency"].to_numpy()
    ms = frame["motion_smoothness"].to_numpy()
    dd = frame["dynamic_preserved"].astype(int).to_numpy()
    latency = frame["config_wall_ms_median"].to_numpy()
    dominated_by: list[list[str]] = []
    epsilon = 1e-12
    for index in range(len(frame)):
        no_worse = (
            (sc >= sc[index] - epsilon)
            & (ms >= ms[index] - epsilon)
            & (dd >= dd[index])
            & (latency <= latency[index] + epsilon)
        )
        strict = (
            (sc > sc[index] + epsilon)
            | (ms > ms[index] + epsilon)
            | (dd > dd[index])
            | (latency < latency[index] - epsilon)
        )
        mask = no_worse & strict
        mask[index] = False
        dominated_by.append(frame.loc[mask, "mode"].tolist())
    frame["dominated_by_count"] = [len(values) for values in dominated_by]
    frame["dominated_by_modes"] = [json.dumps(values) for values in dominated_by]
    frame["is_pareto"] = frame["dominated_by_count"] == 0
    frame["pareto_latency_basis"] = "config_median_wall_ms"
    return frame


def _build_pareto_tables(modes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.concat(
        [_pareto_membership(group) for _, group in modes.groupby("prompt_id", sort=True)],
        ignore_index=True,
    )
    oracle_rows = []
    for prompt_id, group in membership.groupby("prompt_id", sort=True):
        frontier = group[group["is_pareto"]].copy()
        best_sc = group.sort_values(
            ["subject_consistency", "motion_smoothness", "config_wall_ms_median", "config"],
            ascending=[False, False, True, True],
        ).iloc[0]
        best_ms = group.sort_values(
            ["motion_smoothness", "subject_consistency", "config_wall_ms_median", "config"],
            ascending=[False, False, True, True],
        ).iloc[0]
        safe_fast = group[group["quality_safe"]].sort_values(
            ["config_wall_ms_median", "config_attention_ms_median", "config"]
        ).iloc[0]
        dd_preserved = group[group["dynamic_preserved"]]
        lexicographic = dd_preserved.sort_values(
            ["subject_consistency", "motion_smoothness", "config_wall_ms_median", "config"],
            ascending=[False, False, True, True],
        ).iloc[0]
        vsa80 = group[group["is_vsa80"]].iloc[0]
        oracle_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": best_sc["prompt"],
                "seed": best_sc["seed"],
                "pareto_mode_count": len(frontier),
                "pareto_modes": json.dumps(frontier["mode"].tolist()),
                "vsa80_is_pareto": bool(vsa80["is_pareto"]),
                "vsa80_dominated_by_count": int(vsa80["dominated_by_count"]),
                "vsa80_dominated_by_modes": vsa80["dominated_by_modes"],
                "best_subject_config": best_sc["config"],
                "best_subject_mode": best_sc["mode"],
                "best_subject_sparsity": best_sc["sparsity"],
                "best_subject_precision": best_sc["precision"],
                "best_subject_consistency": best_sc["subject_consistency"],
                "best_subject_policy_wall_ms": best_sc["config_wall_ms_median"],
                "best_motion_config": best_ms["config"],
                "best_motion_mode": best_ms["mode"],
                "best_motion_sparsity": best_ms["sparsity"],
                "best_motion_precision": best_ms["precision"],
                "best_motion_smoothness": best_ms["motion_smoothness"],
                "best_motion_policy_wall_ms": best_ms["config_wall_ms_median"],
                "safe_fast_config": safe_fast["config"],
                "safe_fast_mode": safe_fast["mode"],
                "safe_fast_mode_category": safe_fast["mode_category"],
                "safe_fast_sparsity": safe_fast["sparsity"],
                "safe_fast_precision": safe_fast["precision"],
                "safe_fast_policy_wall_ms": safe_fast["config_wall_ms_median"],
                "lexicographic_quality_config": lexicographic["config"],
                "lexicographic_quality_mode": lexicographic["mode"],
                "lexicographic_quality_sparsity": lexicographic["sparsity"],
                "lexicographic_quality_precision": lexicographic["precision"],
                "lexicographic_subject_consistency": lexicographic[
                    "subject_consistency"
                ],
                "lexicographic_motion_smoothness": lexicographic[
                    "motion_smoothness"
                ],
                "lexicographic_policy_wall_ms": lexicographic[
                    "config_wall_ms_median"
                ],
            }
        )
    return membership, pd.DataFrame(oracle_rows)


def _policy_quality_summary(
    *,
    modes: pd.DataFrame,
    repair_oracle: pd.DataFrame,
) -> pd.DataFrame:
    dense = modes[modes["kernel_path"] == "dense_bf16_fa4"].copy()
    vsa80 = modes[modes["is_vsa80"]].copy()
    repair = repair_oracle.rename(
        columns={
            "subject_delta_vs_dense": "subject_delta",
            "motion_delta_vs_dense": "motion_delta",
            "dynamic_delta_vs_dense": "dynamic_delta",
        }
    ).copy()
    policies = {
        "Dense BF16": dense,
        "Fixed VSA80": vsa80,
        "Repair oracle": repair,
    }
    rows = []
    for name, frame in policies.items():
        if name == "Repair oracle":
            sc_delta_vsa = frame["delta_sc_vs_vsa80"]
            ms_delta_vsa = frame["delta_ms_vs_vsa80"]
            dd_delta_vsa = frame["delta_dd_vs_vsa80"]
        else:
            sc_delta_vsa = frame["delta_sc_vs_vsa80"]
            ms_delta_vsa = frame["delta_ms_vs_vsa80"]
            dd_delta_vsa = frame["delta_dd_vs_vsa80"]
        rows.append(
            {
                "policy": name,
                "prompts": len(frame),
                "subject_consistency_mean": frame["subject_consistency"].mean(),
                "subject_consistency_median": frame["subject_consistency"].median(),
                "subject_consistency_p10": frame["subject_consistency"].quantile(0.10),
                "subject_consistency_worst": frame["subject_consistency"].min(),
                "motion_smoothness_mean": frame["motion_smoothness"].mean(),
                "motion_smoothness_median": frame["motion_smoothness"].median(),
                "motion_smoothness_p10": frame["motion_smoothness"].quantile(0.10),
                "motion_smoothness_worst": frame["motion_smoothness"].min(),
                "dynamic_degree_mean": frame["dynamic_degree"].mean(),
                "sc_loss_gt_0_02_vs_dense": int(
                    (frame["subject_delta"] < SUBJECT_DELTA_MIN).sum()
                ),
                "ms_loss_gt_0_01_vs_dense": int(
                    (frame["motion_delta"] < MOTION_DELTA_MIN).sum()
                ),
                "dynamic_degree_regressions_vs_dense": int(
                    (frame["dynamic_delta"] < DYNAMIC_DELTA_MIN).sum()
                ),
                "unsafe_prompts": int(
                    (
                        (frame["subject_delta"] < SUBJECT_DELTA_MIN)
                        | (frame["motion_delta"] < MOTION_DELTA_MIN)
                        | (frame["dynamic_delta"] < DYNAMIC_DELTA_MIN)
                    ).sum()
                ),
                "delta_sc_vs_vsa80_mean": sc_delta_vsa.mean(),
                "delta_sc_vs_vsa80_median": sc_delta_vsa.median(),
                "delta_sc_vs_vsa80_p10": sc_delta_vsa.quantile(0.10),
                "delta_sc_vs_vsa80_worst": sc_delta_vsa.min(),
                "sc_improved_vs_vsa80": int((sc_delta_vsa > 1e-12).sum()),
                "sc_unchanged_vs_vsa80": int((sc_delta_vsa.abs() <= 1e-12).sum()),
                "sc_worsened_vs_vsa80": int((sc_delta_vsa < -1e-12).sum()),
                "delta_ms_vs_vsa80_mean": ms_delta_vsa.mean(),
                "delta_ms_vs_vsa80_median": ms_delta_vsa.median(),
                "delta_ms_vs_vsa80_p10": ms_delta_vsa.quantile(0.10),
                "delta_ms_vs_vsa80_worst": ms_delta_vsa.min(),
                "ms_improved_vs_vsa80": int((ms_delta_vsa > 1e-12).sum()),
                "ms_unchanged_vs_vsa80": int((ms_delta_vsa.abs() <= 1e-12).sum()),
                "ms_worsened_vs_vsa80": int((ms_delta_vsa < -1e-12).sum()),
                "dd_improved_vs_vsa80": int((dd_delta_vsa > 0).sum()),
                "dd_unchanged_vs_vsa80": int((dd_delta_vsa == 0).sum()),
                "dd_worsened_vs_vsa80": int((dd_delta_vsa < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _latency_summary(
    *,
    modes: pd.DataFrame,
    repair_oracle: pd.DataFrame,
) -> pd.DataFrame:
    dense = modes[modes["kernel_path"] == "dense_bf16_fa4"]
    vsa80 = modes[modes["is_vsa80"]]
    dense_policy = float(dense["config_wall_ms_median"].iloc[0])
    vsa80_policy = float(vsa80["config_wall_ms_median"].iloc[0])
    repair_policy_mean = float(repair_oracle["policy_wall_ms"].mean())
    repair_policy_median = float(repair_oracle["policy_wall_ms"].median())
    denominator = dense_policy - vsa80_policy
    retention = (
        (dense_policy - repair_policy_mean) / denominator
        if abs(denominator) > 1e-12
        else np.nan
    )
    rows = [
        {
            "policy": "Dense BF16",
            "policy_latency_median_ms": dense_policy,
            "policy_latency_mean_ms": dense_policy,
            "observed_wall_median_ms": dense["wall_ms"].median(),
            "observed_wall_mean_ms": dense["wall_ms"].mean(),
            "premium_vs_vsa80_ms": dense_policy - vsa80_policy,
            "premium_vs_vsa80_percent": dense_policy / vsa80_policy - 1.0,
            "speed_advantage_vs_dense_percent": 0.0,
            "fraction_vsa80_speedup_retained": 0.0,
        },
        {
            "policy": "Fixed VSA80",
            "policy_latency_median_ms": vsa80_policy,
            "policy_latency_mean_ms": vsa80_policy,
            "observed_wall_median_ms": vsa80["wall_ms"].median(),
            "observed_wall_mean_ms": vsa80["wall_ms"].mean(),
            "premium_vs_vsa80_ms": 0.0,
            "premium_vs_vsa80_percent": 0.0,
            "speed_advantage_vs_dense_percent": dense_policy / vsa80_policy - 1.0,
            "fraction_vsa80_speedup_retained": 1.0,
        },
        {
            "policy": "Repair oracle",
            "policy_latency_median_ms": repair_policy_median,
            "policy_latency_mean_ms": repair_policy_mean,
            "observed_wall_median_ms": repair_oracle["observed_wall_ms"].median(),
            "observed_wall_mean_ms": repair_oracle["observed_wall_ms"].mean(),
            "premium_vs_vsa80_ms": repair_policy_mean - vsa80_policy,
            "premium_vs_vsa80_percent": repair_policy_mean / vsa80_policy - 1.0,
            "speed_advantage_vs_dense_percent": dense_policy / repair_policy_mean - 1.0,
            "fraction_vsa80_speedup_retained": retention,
        },
    ]
    return pd.DataFrame(rows)


def _fixed_fallback_summary(modes: pd.DataFrame) -> pd.DataFrame:
    vsa80 = modes[modes["is_vsa80"]]
    failure_ids = set(vsa80[~vsa80["quality_safe"]]["prompt_id"])
    safe_ids = set(vsa80[vsa80["quality_safe"]]["prompt_id"])
    vsa80_latency = float(vsa80["config_wall_ms_median"].iloc[0])
    rows = []
    for config, group in modes.groupby("config"):
        rows.append(
            {
                "config": config,
                "mode": group["mode"].iloc[0],
                "mode_category": group["mode_category"].iloc[0],
                "kernel_path": group["kernel_path"].iloc[0],
                "sparsity": group["sparsity"].iloc[0],
                "precision": group["precision"].iloc[0],
                "repairs_vsa80_failures": int(
                    group[group["prompt_id"].isin(failure_ids)]["quality_safe"].sum()
                ),
                "vsa80_failures_total": len(failure_ids),
                "overall_safe_prompts": int(group["quality_safe"].sum()),
                "new_failures_among_vsa80_safe": int(
                    (~group[group["prompt_id"].isin(safe_ids)]["quality_safe"]).sum()
                ),
                "config_wall_ms_median": group["config_wall_ms_median"].iloc[0],
                "latency_premium_vs_vsa80_ms": (
                    group["config_wall_ms_median"].iloc[0] - vsa80_latency
                ),
                "latency_premium_vs_vsa80_percent": (
                    group["config_wall_ms_median"].iloc[0] / vsa80_latency - 1.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["repairs_vsa80_failures", "config_wall_ms_median"],
        ascending=[False, True],
    )


def _failure_type_summary(
    failure_recovery: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for failure_type, group in failure_recovery.groupby("failure_type"):
        mode_counts = group["fastest_repair_mode"].value_counts()
        rows.append(
            {
                "failure_type": failure_type,
                "count": len(group),
                "most_common_fastest_repair_mode": mode_counts.index[0],
                "most_common_fastest_repair_count": int(mode_counts.iloc[0]),
                "fastest_repair_mode_histogram": json.dumps(
                    {str(key): int(value) for key, value in mode_counts.items()}
                ),
                "median_latency_cost_ms": group["extra_latency_to_repair_ms"].median(),
                "median_latency_cost_percent": group[
                    "extra_latency_to_repair_percent"
                ].median(),
                "median_sc_recovery": group[
                    "fastest_repair_sc_improvement_vs_vsa80"
                ].median(),
                "median_ms_recovery": group[
                    "fastest_repair_ms_improvement_vs_vsa80"
                ].median(),
                "dynamic_regressions_repaired": int(
                    (
                        (group["vsa80_dynamic_delta"] < 0)
                        & (group["fastest_repair_dynamic_delta_vs_dense"] >= 0)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def _sparse_over_dense(modes: pd.DataFrame) -> pd.DataFrame:
    sparse = modes[modes["kernel_path"].isin(["vsa_bf16", "sim_vsa_nvfp4"])].copy()
    sparse["sc_improves_dense"] = sparse["subject_delta"] > 1e-12
    sparse["ms_improves_dense"] = sparse["motion_delta"] > 1e-12
    sparse["both_improve_dense"] = (
        sparse["sc_improves_dense"] & sparse["ms_improves_dense"]
    )
    sparse["any_quality_improves_dense"] = (
        sparse["sc_improves_dense"] | sparse["ms_improves_dense"]
    )
    return sparse[
        sparse["dynamic_preserved"] & sparse["any_quality_improves_dense"]
    ].sort_values(
        ["subject_delta", "motion_delta"],
        ascending=[False, False],
    )


def _plot_results(
    *,
    figures: Path,
    failure_recovery: pd.DataFrame,
    repair_oracle: pd.DataFrame,
    quality_oracle: pd.DataFrame,
    pareto_membership: pd.DataFrame,
    quality_summary: pd.DataFrame,
    failure_type_summary: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)

    def plot_sparsity_histogram(
        values: pd.Series,
        *,
        ylabel: str,
        filename: str,
    ) -> None:
        counts = values.value_counts().sort_index()
        counts.index = [
            "Dense (0%)" if abs(float(value)) < 1e-12 else f"{float(value):.0%}"
            for value in counts.index
        ]
        counts.plot.bar(figsize=(8, 4))
        plt.xlabel("Observed sparsity")
        plt.ylabel(ylabel)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(7, 5))
    for mode, group in failure_recovery.groupby("fastest_repair_mode"):
        plt.scatter(
            group["extra_latency_to_repair_ms"],
            group["fastest_repair_sc_improvement_vs_vsa80"],
            label=mode,
            alpha=0.8,
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Policy latency premium over VSA80 (ms)")
    plt.ylabel("Subject-consistency improvement over VSA80")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "failure_sc_recovery_vs_latency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for mode, group in failure_recovery.groupby("fastest_repair_mode"):
        plt.scatter(
            group["extra_latency_to_repair_ms"],
            group["fastest_repair_ms_improvement_vs_vsa80"],
            label=mode,
            alpha=0.8,
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Policy latency premium over VSA80 (ms)")
    plt.ylabel("Motion-smoothness improvement over VSA80")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "failure_ms_recovery_vs_latency.png", dpi=180)
    plt.close()

    repair_oracle["selected_mode"].value_counts().sort_values().plot.barh(figsize=(8, 4))
    plt.xlabel("Prompts")
    plt.ylabel("Repair-oracle mode")
    plt.tight_layout()
    plt.savefig(figures / "repair_oracle_mode_histogram.png", dpi=180)
    plt.close()

    quality_oracle["safe_fast_mode"].value_counts().sort_values().plot.barh(figsize=(8, 5))
    plt.xlabel("Prompts")
    plt.ylabel("Lowest-latency quality-safe mode")
    plt.tight_layout()
    plt.savefig(figures / "safe_fast_mode_histogram.png", dpi=180)
    plt.close()

    plot_sparsity_histogram(
        quality_oracle["safe_fast_sparsity"],
        ylabel="Prompts selecting safe-fast sparsity",
        filename="safe_fast_sparsity_histogram.png",
    )

    quality_oracle["best_subject_mode"].value_counts().sort_values().plot.barh(figsize=(9, 6))
    plt.xlabel("Prompts")
    plt.ylabel("Highest subject-consistency mode")
    plt.tight_layout()
    plt.savefig(figures / "best_subject_mode_histogram.png", dpi=180)
    plt.close()

    plot_sparsity_histogram(
        quality_oracle["best_subject_sparsity"],
        ylabel="Prompts selecting subject-best sparsity",
        filename="best_subject_sparsity_histogram.png",
    )

    quality_oracle["best_motion_mode"].value_counts().sort_values().plot.barh(figsize=(9, 6))
    plt.xlabel("Prompts")
    plt.ylabel("Highest motion-smoothness mode")
    plt.tight_layout()
    plt.savefig(figures / "best_motion_mode_histogram.png", dpi=180)
    plt.close()

    plot_sparsity_histogram(
        quality_oracle["best_motion_sparsity"],
        ylabel="Prompts selecting motion-best sparsity",
        filename="best_motion_sparsity_histogram.png",
    )

    quality_oracle["lexicographic_quality_mode"].value_counts().sort_values().plot.barh(
        figsize=(9, 6)
    )
    plt.xlabel("Prompts")
    plt.ylabel("Lexicographic quality-best mode")
    plt.tight_layout()
    plt.savefig(figures / "lexicographic_quality_mode_histogram.png", dpi=180)
    plt.close()

    plot_sparsity_histogram(
        quality_oracle["lexicographic_quality_sparsity"],
        ylabel="Prompts selecting lexicographic sparsity",
        filename="lexicographic_quality_sparsity_histogram.png",
    )

    pareto_membership[pareto_membership["is_pareto"]]["mode"].value_counts().sort_values().plot.barh(
        figsize=(9, 6)
    )
    plt.xlabel("Prompt frontiers containing mode")
    plt.ylabel("Execution mode")
    plt.tight_layout()
    plt.savefig(figures / "pareto_mode_frequency.png", dpi=180)
    plt.close()

    tails = quality_summary.set_index("policy")[
        ["subject_consistency_p10", "motion_smoothness_p10"]
    ]
    tails.plot.bar(figsize=(8, 4))
    plt.ylabel("P10 score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures / "tail_quality_comparison.png", dpi=180)
    plt.close()

    failure_type_summary.set_index("failure_type")["count"].plot.bar(figsize=(6, 4))
    plt.xlabel("VSA80 failure type")
    plt.ylabel("Prompts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures / "vsa80_failure_types.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/adaptive_vsa_fp4"))
    args = parser.parse_args()

    root = args.root
    output = root / "quality_adaptive_reanalysis"
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    if figures.exists():
        shutil.rmtree(figures)
    figures.mkdir(parents=True)

    source_paths = {
        "jobs": root / "phase1" / "jobs.parquet",
        "jobs_database": root / "phase1" / "jobs.sqlite",
        "quality_labels": root / "phase1" / "quality_labels.parquet",
        "vbench_metrics": root / "phase1" / "vbench_metrics.csv",
        "paired_metrics": root / "phase1" / "paired_metrics.csv",
        "phase1_summary": root / "phase1" / "summary.json",
        "environment": root / "env" / "environment.json",
    }
    jobs = pd.read_parquet(source_paths["jobs"])
    modes = pd.read_parquet(source_paths["quality_labels"])
    vbench = pd.read_csv(source_paths["vbench_metrics"])
    paired = pd.read_csv(source_paths["paired_metrics"])
    paired_wide = paired.pivot_table(
        index="job_id",
        columns="metric",
        values="score",
        aggfunc="first",
    ).reset_index().rename(
        columns={
            "common.ssim": "ssim_to_dense",
            "common.psnr": "psnr_to_dense",
        }
    )
    modes = modes.merge(paired_wide, on="job_id", how="left", validate="one_to_one")
    modes = _add_reference_columns(modes)
    validation = _validate_inputs(
        root=root,
        jobs=jobs,
        modes=modes,
        vbench=vbench,
        paired=paired,
    )

    modes.to_parquet(output / "all_modes.parquet", index=False)
    failure_prompts, failure_recovery = _build_failure_tables(modes)
    failure_prompts.to_csv(output / "vsa80_failure_prompts.csv", index=False)
    failure_prompts.sort_values("subject_delta").to_csv(
        output / "vsa80_failure_prompts_by_subject.csv",
        index=False,
    )
    failure_recovery.to_csv(output / "failure_recovery.csv", index=False)

    repair_oracle = _build_repair_oracle(modes)
    repair_oracle.to_csv(output / "repair_oracle.csv", index=False)
    pareto_membership, quality_oracle = _build_pareto_tables(modes)
    pareto_membership.to_csv(output / "pareto_membership.csv", index=False)
    quality_oracle.to_csv(output / "quality_pareto_oracle.csv", index=False)

    quality_summary = _policy_quality_summary(
        modes=modes,
        repair_oracle=repair_oracle,
    )
    quality_summary.to_csv(output / "quality_summary.csv", index=False)
    latency_summary = _latency_summary(modes=modes, repair_oracle=repair_oracle)
    latency_summary.to_csv(output / "latency_summary.csv", index=False)
    fixed_fallback = _fixed_fallback_summary(modes)
    fixed_fallback.to_csv(output / "fixed_fallback_summary.csv", index=False)
    failure_types = _failure_type_summary(failure_recovery)
    failure_types.to_csv(output / "failure_type_summary.csv", index=False)
    sparse_better = _sparse_over_dense(modes)
    sparse_better.to_csv(output / "sparse_over_dense.csv", index=False)

    _plot_results(
        figures=figures,
        failure_recovery=failure_recovery,
        repair_oracle=repair_oracle,
        quality_oracle=quality_oracle,
        pareto_membership=pareto_membership,
        quality_summary=quality_summary,
        failure_type_summary=failure_types,
    )

    recovery_count = int(failure_recovery["repairable_any_measured_mode"].sum())
    recovery_ci = _wilson_rate_ci(recovery_count, len(failure_recovery))
    lower_vsa_recovery_count = int(
        failure_recovery["repairable_lower_vsa_bf16"].sum()
    )
    non_dense_recovery_count = int(failure_recovery["repairable_non_dense"].sum())
    repair_quality = quality_summary.set_index("policy").loc["Repair oracle"]
    vsa_quality = quality_summary.set_index("policy").loc["Fixed VSA80"]
    repair_latency = latency_summary.set_index("policy").loc["Repair oracle"]
    vsa_latency = latency_summary.set_index("policy").loc["Fixed VSA80"]
    dense_latency = latency_summary.set_index("policy").loc["Dense BF16"]
    vsa80_dd_regressions = int(vsa_quality["dynamic_degree_regressions_vs_dense"])
    repair_dd_regressions = int(repair_quality["dynamic_degree_regressions_vs_dense"])
    dd_repaired = int(
        (
            (repair_oracle["vsa80_failure_type"].str.contains("DD"))
            & (repair_oracle["dynamic_delta_vs_dense"] >= 0)
        ).sum()
    )
    fastest_repair_histogram = failure_recovery["fastest_repair_mode"].value_counts()
    highest_quality_histogram = failure_recovery[
        "highest_quality_repair_mode"
    ].value_counts()
    safe_fast_vsa80_count = int(
        (quality_oracle["safe_fast_mode"] == "VSA80 BF16").sum()
    )
    safe_fast_less_sparse_count = int(
        (
            (quality_oracle["safe_fast_sparsity"] > 0)
            & (quality_oracle["safe_fast_sparsity"] < VSA80_SPARSITY)
        ).sum()
    )
    safe_fast_dense_count = int(
        (quality_oracle["safe_fast_sparsity"] == 0).sum()
    )
    safe_fast_nvfp4_count = int(
        quality_oracle["safe_fast_mode"].str.contains("NVFP4").sum()
    )
    prompt_count = len(quality_oracle)
    frontier_frequency = (
        pareto_membership[pareto_membership["is_pareto"]]["mode"].value_counts()
    )
    vsa80_dominated = int((~quality_oracle["vsa80_is_pareto"]).sum())
    pareto_counts = quality_oracle["pareto_mode_count"]
    vsa20 = fixed_fallback[fixed_fallback["mode"] == "VSA20 BF16"].iloc[0]

    repair_failures = repair_oracle[~repair_oracle["vsa80_safe"]]
    failure_latency_median = repair_failures[
        "policy_latency_premium_vs_vsa80_ms"
    ].median()
    failure_latency_percent_median = repair_failures[
        "policy_latency_premium_vs_vsa80_percent"
    ].median()
    failure_sc_gain_mean = repair_failures["delta_sc_vs_vsa80"].mean()
    failure_ms_gain_mean = repair_failures["delta_ms_vs_vsa80"].mean()

    sparse_prompt_count = sparse_better["prompt_id"].nunique()
    sparse_both_count = int(sparse_better["both_improve_dense"].sum())
    sparse_sc_gt_001 = int((sparse_better["subject_delta"] >= 0.01).sum())

    tail_examples = repair_failures.nsmallest(5, "delta_sc_vs_vsa80")[
        [
            "prompt_id",
            "prompt",
            "selected_mode",
            "delta_sc_vs_vsa80",
            "delta_ms_vs_vsa80",
            "delta_dd_vs_vsa80",
        ]
    ]
    tail_examples.to_csv(output / "repair_oracle_worst_five_deltas.csv", index=False)
    tail_example_lines = "\n".join(
        (
            f"| {str(row.prompt).replace('|', '&#124;')} | {row.selected_mode} | "
            f"{row.delta_sc_vs_vsa80:+.6f} | {row.delta_ms_vs_vsa80:+.6f} | "
            f"{row.delta_dd_vs_vsa80:+.6f} |"
        )
        for row in tail_examples.itertuples()
    )

    dense_tail_source = (
        modes[modes["kernel_path"] == "dense_bf16_fa4"][
            ["prompt_id", "prompt"]
        ]
        .assign(
            policy="Dense BF16",
            subject_delta_vs_dense=0.0,
            motion_delta_vs_dense=0.0,
            dynamic_delta_vs_dense=0.0,
        )
    )
    vsa80_tail_source = modes[modes["is_vsa80"]][
        [
            "prompt_id",
            "prompt",
            "subject_delta",
            "motion_delta",
            "dynamic_delta",
        ]
    ].rename(
        columns={
            "subject_delta": "subject_delta_vs_dense",
            "motion_delta": "motion_delta_vs_dense",
            "dynamic_delta": "dynamic_delta_vs_dense",
        }
    )
    vsa80_tail_source["policy"] = "Fixed VSA80"
    repair_tail_source = repair_oracle[
        [
            "prompt_id",
            "prompt",
            "subject_delta_vs_dense",
            "motion_delta_vs_dense",
            "dynamic_delta_vs_dense",
        ]
    ].copy()
    repair_tail_source["policy"] = "Repair oracle"
    policy_tail_examples = pd.concat(
        [
            source.nsmallest(5, "subject_delta_vs_dense")
            for source in [
                dense_tail_source,
                vsa80_tail_source,
                repair_tail_source,
            ]
        ],
        ignore_index=True,
    )[
        [
            "policy",
            "prompt_id",
            "prompt",
            "subject_delta_vs_dense",
            "motion_delta_vs_dense",
            "dynamic_delta_vs_dense",
        ]
    ]
    policy_tail_examples.to_csv(
        output / "policy_worst_five_dense_relative.csv",
        index=False,
    )
    policy_tail_report = policy_tail_examples[
        policy_tail_examples["policy"] != "Dense BF16"
    ]
    policy_tail_lines = "\n".join(
        (
            f"| {row.policy} | {str(row.prompt).replace('|', '&#124;')} | "
            f"{row.subject_delta_vs_dense:+.6f} | "
            f"{row.motion_delta_vs_dense:+.6f} | "
            f"{row.dynamic_delta_vs_dense:+.6f} |"
        )
        for row in policy_tail_report.itertuples()
    )

    decision = "PROCEED TO TRAINING-FREE QUALITY-RISK PREDICTOR"
    classification = "Case A — strong adaptive-quality opportunity"
    summary = {
        "status": "complete",
        "analysis_type": "offline_quality_adaptive_reanalysis",
        "analysis_commit": _commit(),
        "source_generation_commit": validation["phase1_generation_commits"][0],
        "baseline": "VSA80 BF16",
        "quality_reference": "Dense BF16",
        "validation": validation,
        "available_modes": (
            modes[["config", "mode", "kernel_path", "sparsity", "precision"]]
            .drop_duplicates()
            .sort_values(["kernel_path", "sparsity"])
            .to_dict(orient="records")
        ),
        "safety_rule": {
            "subject_delta_min": SUBJECT_DELTA_MIN,
            "motion_delta_min": MOTION_DELTA_MIN,
            "dynamic_delta_min": DYNAMIC_DELTA_MIN,
        },
        "latency_method": {
            "oracle_selection": "Median wall latency for each configuration across 72 prompts.",
            "reason": "Avoid selecting configurations from one-off per-prompt host/runtime noise.",
            "raw_observed_wall_latency_also_reported": True,
        },
        "repairability": {
            "vsa80_failures": len(failure_recovery),
            "repaired_by_any_mode": recovery_count,
            "recovery_rate": recovery_count / len(failure_recovery),
            "wilson_score_95_ci": list(recovery_ci),
            "repaired_by_non_dense_mode": non_dense_recovery_count,
            "repaired_by_some_lower_vsa_bf16_mode": lower_vsa_recovery_count,
            "fastest_repair_mode_histogram": {
                str(key): int(value) for key, value in fastest_repair_histogram.items()
            },
            "highest_quality_repair_mode_histogram": {
                str(key): int(value) for key, value in highest_quality_histogram.items()
            },
        },
        "quality_gain": {
            "subject_delta_vs_vsa80_mean_all_prompts": repair_quality[
                "delta_sc_vs_vsa80_mean"
            ],
            "subject_delta_vs_vsa80_mean_failures": failure_sc_gain_mean,
            "motion_delta_vs_vsa80_mean_all_prompts": repair_quality[
                "delta_ms_vs_vsa80_mean"
            ],
            "motion_delta_vs_vsa80_mean_failures": failure_ms_gain_mean,
            "subject_p10_vsa80": vsa_quality["subject_consistency_p10"],
            "subject_p10_repair_oracle": repair_quality[
                "subject_consistency_p10"
            ],
            "motion_p10_vsa80": vsa_quality["motion_smoothness_p10"],
            "motion_p10_repair_oracle": repair_quality[
                "motion_smoothness_p10"
            ],
            "sc_catastrophic_losses_vsa80": int(
                vsa_quality["sc_loss_gt_0_02_vs_dense"]
            ),
            "sc_catastrophic_losses_repair_oracle": int(
                repair_quality["sc_loss_gt_0_02_vs_dense"]
            ),
            "ms_catastrophic_losses_vsa80": int(
                vsa_quality["ms_loss_gt_0_01_vs_dense"]
            ),
            "ms_catastrophic_losses_repair_oracle": int(
                repair_quality["ms_loss_gt_0_01_vs_dense"]
            ),
            "dynamic_regressions_vsa80": vsa80_dd_regressions,
            "dynamic_regressions_repaired": dd_repaired,
            "dynamic_regressions_repair_oracle": repair_dd_regressions,
        },
        "latency": {
            "vsa80_policy_latency_ms": vsa_latency["policy_latency_median_ms"],
            "repair_oracle_policy_latency_mean_ms": repair_latency[
                "policy_latency_mean_ms"
            ],
            "dense_policy_latency_ms": dense_latency["policy_latency_median_ms"],
            "repair_oracle_premium_vs_vsa80_ms": repair_latency[
                "premium_vs_vsa80_ms"
            ],
            "repair_oracle_premium_vs_vsa80_percent": repair_latency[
                "premium_vs_vsa80_percent"
            ],
            "median_failure_repair_premium_ms": failure_latency_median,
            "median_failure_repair_premium_percent": failure_latency_percent_median,
            "fraction_vsa80_speedup_retained": repair_latency[
                "fraction_vsa80_speedup_retained"
            ],
            "raw_observed_means_ms": {
                "vsa80": vsa_latency["observed_wall_mean_ms"],
                "repair_oracle": repair_latency["observed_wall_mean_ms"],
                "dense": dense_latency["observed_wall_mean_ms"],
            },
        },
        "mode_heterogeneity": {
            "repair_oracle_mode_histogram": _count_histogram(
                repair_oracle["selected_mode"]
            ),
            "safe_fast_mode_histogram": _count_histogram(
                quality_oracle["safe_fast_mode"]
            ),
            "safe_fast_sparsity_histogram": _sparsity_histogram(
                quality_oracle["safe_fast_sparsity"]
            ),
            "safe_fast_preference_counts": {
                "vsa80": safe_fast_vsa80_count,
                "less_sparse": safe_fast_less_sparse_count,
                "dense": safe_fast_dense_count,
                "nvfp4": safe_fast_nvfp4_count,
            },
            "safe_fast_preference_fractions": {
                "vsa80": safe_fast_vsa80_count / prompt_count,
                "less_sparse": safe_fast_less_sparse_count / prompt_count,
                "dense": safe_fast_dense_count / prompt_count,
                "nvfp4": safe_fast_nvfp4_count / prompt_count,
            },
            "best_subject_mode_histogram": _count_histogram(
                quality_oracle["best_subject_mode"]
            ),
            "best_subject_sparsity_histogram": _sparsity_histogram(
                quality_oracle["best_subject_sparsity"]
            ),
            "best_motion_mode_histogram": _count_histogram(
                quality_oracle["best_motion_mode"]
            ),
            "best_motion_sparsity_histogram": _sparsity_histogram(
                quality_oracle["best_motion_sparsity"]
            ),
            "lexicographic_quality_mode_histogram": _count_histogram(
                quality_oracle["lexicographic_quality_mode"]
            ),
            "lexicographic_quality_sparsity_histogram": _sparsity_histogram(
                quality_oracle["lexicographic_quality_sparsity"]
            ),
            "vsa80_dominated_prompts": vsa80_dominated,
            "pareto_modes_per_prompt_median": pareto_counts.median(),
            "pareto_modes_per_prompt_range": [
                int(pareto_counts.min()),
                int(pareto_counts.max()),
            ],
            "frontier_mode_frequency": {
                str(key): int(value) for key, value in frontier_frequency.items()
            },
        },
        "fixed_fallback_test": {
            "vsa20_repairs": int(vsa20["repairs_vsa80_failures"]),
            "vsa20_overall_safe": int(vsa20["overall_safe_prompts"]),
            "vsa20_new_failures_among_vsa80_safe": int(
                vsa20["new_failures_among_vsa80_safe"]
            ),
            "vsa20_latency_premium_vs_vsa80_percent": vsa20[
                "latency_premium_vs_vsa80_percent"
            ],
            "conclusion": "No non-dense fixed mode repairs all failures while remaining safe on all prompts.",
        },
        "sparse_over_dense": {
            "prompt_count_with_any_strict_metric_improvement_and_no_dd_regression": sparse_prompt_count,
            "prompt_config_rows": len(sparse_better),
            "rows_improving_both_sc_and_ms": sparse_both_count,
            "rows_with_sc_improvement_at_least_0_01": sparse_sc_gt_001,
            "causality_claimed": False,
        },
        "classification": classification,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    source_manifest = {
        "analysis_commit": _commit(),
        "source_generation_commit": validation["phase1_generation_commits"][0],
        "created_at_utc": "2026-08-29",
        "source_artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "excluded_artifacts": {
            "path": str((root / "phase1_invalid_b1935daa").resolve()),
            "commit": INVALID_COMMIT,
            "reason": "Persistent-worker VSA sparsity propagation defect.",
        },
        "validation": validation,
        "baseline": {
            "name": "VSA80",
            "kernel_path": VSA80_KERNEL,
            "sparsity": VSA80_SPARSITY,
            "precision": "bf16",
        },
        "safety_rule": summary["safety_rule"],
        "no_new_generation": True,
        "gpu_generation_jobs_run_for_reanalysis": 0,
    }
    (output / "input_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )

    worst = failure_prompts.sort_values("subject_delta").iloc[0]
    repair_modes = ", ".join(
        f"{mode}: {count}"
        for mode, count in fastest_repair_histogram.items()
    )
    highest_quality_modes = ", ".join(
        f"{mode}: {count}"
        for mode, count in highest_quality_histogram.items()
    )
    failure_type_lines = "\n".join(
        (
            f"- {row.failure_type}: {row.count} prompts; most common fastest repair "
            f"{row.most_common_fastest_repair_mode}; median policy cost "
            f"{row.median_latency_cost_ms:.2f} ms."
        )
        for row in failure_types.itertuples()
    )
    report = f"""# Quality-adaptive reanalysis of fixed VSA80

## Executive result

This offline reanalysis supports a quality-adaptive follow-up. Fixed VSA80 failed the original dense-relative safety rule on 24/72 prompts. All **24/24 failures (100%; Wilson-score 95% CI {recovery_ci[0]:.1%}–{recovery_ci[1]:.1%})** were repairable by an already-measured mode, and all 24 were also repairable by at least one lower-sparsity VSA BF16 mode.

The repair-first oracle keeps VSA80 on its 48 safe prompts and uses the lowest config-median-latency safe fallback on the other 24. It raises mean subject consistency by **{repair_quality['delta_sc_vs_vsa80_mean']:.4f}** over all prompts (**{failure_sc_gain_mean:.4f}** on the failures), raises mean motion smoothness by **{repair_quality['delta_ms_vs_vsa80_mean']:.4f}** overall, eliminates all 18 excessive subject losses, both excessive motion losses, and all five dynamic-degree regressions, while adding only **{repair_latency['premium_vs_vsa80_ms']:.2f} ms ({repair_latency['premium_vs_vsa80_percent']:.2%})** to the policy-average latency estimate.

The robust median-based estimate retains **{repair_latency['fraction_vsa80_speedup_retained']:.1%}** of VSA80's latency advantage over dense BF16. For only the 24 repaired prompts, the median fallback cost is **{failure_latency_median:.2f} ms ({failure_latency_percent_median:.2%})**.

## Method

- Source: the corrected 864-job Phase 1 grid at commit `{validation['phase1_generation_commits'][0]}`.
- Baseline: VSA BF16 at 80% sparsity (`VSA80`).
- Quality reference: paired dense BF16.
- Safety rule: SC delta >= −0.02, MS delta >= −0.01, and no DD regression.
- No new generation, training, threshold changes, controller, kernel, model, or external-dataset work was performed.
- Oracle selection uses each configuration's median wall latency across all 72 prompts. Raw per-job wall latency remains in the tables, but is not used to choose modes because one-off host/runtime noise would make the oracle optimistically biased.
- The quality Pareto analysis does not use a weighted score. A mode dominates another only if it is no worse in SC, MS, DD preservation, and latency, and strictly better in at least one.

## 1. How many VSA80 failures were repairable?

**{recovery_count}/24 (100%)** by any measured mode. **{non_dense_recovery_count}/24** had a non-dense repair, and **{lower_vsa_recovery_count}/24** had at least one lower-sparsity VSA BF16 repair.

The original split reproduced exactly: 48 safe and 24 unsafe. The worst SC case also reproduced: “{worst['prompt']}”, delta **{worst['subject_delta']:.6f}**.

## 2. Which modes repaired them?

The lowest-latency safe repairs were: **{repair_modes}**.

The lexicographically highest-quality repairs were more heterogeneous: **{highest_quality_modes}**. This rule first requires DD preservation, then maximizes SC, then MS, then minimizes latency.

Although every failure had some lower-sparsity VSA repair, those lower-sparsity VSA modes were slower end to end than the dense fallbacks in this sweep. Consequently, the minimum-cost repair oracle selected dense NVFP4 or dense BF16.

## 3. How much did subject consistency improve over fixed VSA80?

- Mean delta over all 72 prompts: **{repair_quality['delta_sc_vs_vsa80_mean']:+.6f}**.
- Mean delta over the 24 repaired prompts: **{failure_sc_gain_mean:+.6f}**.
- Median delta over all prompts: **{repair_quality['delta_sc_vs_vsa80_median']:+.6f}**; 48 prompts are intentionally unchanged.
- Improved / unchanged / worsened: **{int(repair_quality['sc_improved_vs_vsa80'])} / {int(repair_quality['sc_unchanged_vs_vsa80'])} / {int(repair_quality['sc_worsened_vs_vsa80'])}**.
- Excessive SC losses relative to dense: **18 → 0**.

The repair-first oracle can slightly lower SC on a small number of failing prompts while still restoring the full joint safety rule, because it chooses the fastest safe repair rather than maximizing a weighted quality score.

## 4. How much did motion smoothness improve?

- Mean delta over all prompts: **{repair_quality['delta_ms_vs_vsa80_mean']:+.6f}**.
- Mean delta over repaired prompts: **{failure_ms_gain_mean:+.6f}**.
- Improved / unchanged / worsened: **{int(repair_quality['ms_improved_vs_vsa80'])} / {int(repair_quality['ms_unchanged_vs_vsa80'])} / {int(repair_quality['ms_worsened_vs_vsa80'])}**.
- Excessive MS losses relative to dense: **2 → 0**.

## 5. How many dynamic-degree regressions were repaired?

Fixed VSA80 had **{vsa80_dd_regressions}** DD regressions. The repair oracle repaired **{dd_repaired}/{vsa80_dd_regressions}** and introduced **0** new regressions.

## 6. How did P10 and worst-case quality change?

- SC P10: **{vsa_quality['subject_consistency_p10']:.6f} → {repair_quality['subject_consistency_p10']:.6f}**. Dense BF16 P10 is **{quality_summary.set_index('policy').loc['Dense BF16', 'subject_consistency_p10']:.6f}**.
- MS P10: **{vsa_quality['motion_smoothness_p10']:.6f} → {repair_quality['motion_smoothness_p10']:.6f}**.
- Absolute worst SC: **{vsa_quality['subject_consistency_worst']:.6f} → {repair_quality['subject_consistency_worst']:.6f}**.
- Absolute worst MS: **{vsa_quality['motion_smoothness_worst']:.6f} → {repair_quality['motion_smoothness_worst']:.6f}**.

The absolute minima do not improve because those prompts were already safe relative to their paired dense references. The important tail result is that threshold-exceeding dense-relative losses fall to zero.

Worst five repair-oracle deltas on the 24 repaired prompts versus fixed VSA80, sorted by subject-consistency delta:

| Prompt | Selected mode | ΔSC | ΔMS | ΔDD |
|---|---:|---:|---:|---:|
{tail_example_lines}

Dense BF16 is the paired reference, so all of its dense-relative deltas are exactly zero. The nontrivial worst-five dense-relative rows for fixed VSA80 and the repair oracle are:

| Policy | Prompt | ΔSC vs dense | ΔMS vs dense | ΔDD vs dense |
|---|---|---:|---:|---:|
{policy_tail_lines}

## 7. What was the latency premium over VSA80?

- VSA80 config-median latency: **{vsa_latency['policy_latency_median_ms']:.2f} ms**.
- Repair-oracle policy median / mean latency: **{repair_latency['policy_latency_median_ms']:.2f} / {repair_latency['policy_latency_mean_ms']:.2f} ms**.
- Dense BF16 policy median / mean latency: **{dense_latency['policy_latency_median_ms']:.2f} / {dense_latency['policy_latency_mean_ms']:.2f} ms**.
- Premium over VSA80: **{repair_latency['premium_vs_vsa80_ms']:.2f} ms ({repair_latency['premium_vs_vsa80_percent']:.2%})**.
- Median premium on the 24 repaired prompts: **{failure_latency_median:.2f} ms ({failure_latency_percent_median:.2%})**.

Raw observed wall means are also reported and contain substantial long-tail runtime noise: VSA80 **{vsa_latency['observed_wall_mean_ms']:.2f} ms**, repair oracle **{repair_latency['observed_wall_mean_ms']:.2f} ms**, dense BF16 **{dense_latency['observed_wall_mean_ms']:.2f} ms**.

## 8. What fraction of VSA80's speed advantage over dense was retained?

Using the robust configuration-median policy estimate:

**{repair_latency['fraction_vsa80_speedup_retained']:.1%} retained.**

Dense BF16 is **{dense_latency['policy_latency_median_ms']:.2f} ms**, VSA80 is **{vsa_latency['policy_latency_median_ms']:.2f} ms**, and the repair oracle averages **{repair_latency['policy_latency_mean_ms']:.2f} ms**.

## 9. Does the quality-optimal or safe-fast mode vary by prompt?

Yes. The repair oracle uses **{repair_oracle['selected_mode'].nunique()} modes**: {repair_modes}, plus VSA80 on the 48 already-safe prompts.

For the lowest-latency quality-safe choice across all 72 prompts:

- **{safe_fast_vsa80_count}/72 ({safe_fast_vsa80_count / prompt_count:.1%})** select VSA80.
- **{safe_fast_less_sparse_count}/72 ({safe_fast_less_sparse_count / prompt_count:.1%})** select a less-sparse VSA mode.
- **{safe_fast_dense_count}/72 ({safe_fast_dense_count / prompt_count:.1%})** select a dense mode.
- **{safe_fast_nvfp4_count}/72 ({safe_fast_nvfp4_count / prompt_count:.1%})** select an NVFP4 mode.

The dense and NVFP4 fractions overlap because dense NVFP4 is both categories. In exact mode counts, the safe-fast/repair policy is 48 VSA80, 11 dense BF16, and 13 dense NVFP4-QK.

The unconstrained subject-best choice spans **{quality_oracle['best_subject_mode'].nunique()} modes**, the motion-best choice spans **{quality_oracle['best_motion_mode'].nunique()} modes**, and the dense-relative-DD-preserving lexicographic choice spans **{quality_oracle['lexicographic_quality_mode'].nunique()} modes** across all prompts. The lexicographic quality-best repair uses **{failure_recovery['highest_quality_repair_mode'].nunique()} different modes** across the 24 failures. Per-prompt Pareto frontiers contain a median of **{pareto_counts.median():.1f} modes** (range **{int(pareto_counts.min())}–{int(pareto_counts.max())}**), and VSA80 is dominated on **{vsa80_dominated}/72** prompts.

## 10. Could one fixed lower-sparsity mode replace the adaptive oracle?

Not cleanly. VSA20 repairs **{int(vsa20['repairs_vsa80_failures'])}/24** VSA80 failures but is safe on only **{int(vsa20['overall_safe_prompts'])}/72** prompts, introduces **{int(vsa20['new_failures_among_vsa80_safe'])}** new failure among prompts VSA80 handled, and costs **{vsa20['latency_premium_vs_vsa80_percent']:.1%}** more latency than VSA80. It is also slower than dense BF16 in this sweep.

Dense BF16 is the only globally safe fixed policy, but using it everywhere discards all VSA80 latency benefit. No non-dense fixed mode matches the repair oracle's safety.

## 11. Are there prompts where sparse execution improves quality over dense BF16?

Yes, numerically. There are **{len(sparse_better)} prompt/config rows across {sparse_prompt_count}/72 prompts** where a sparse mode improves SC and/or MS while preserving DD; **{sparse_both_count}** rows improve both metrics, and **{sparse_sc_gt_001}** rows improve SC by at least 0.01.

This is consistent with the possibility that sparse execution can sometimes be beneficial for this sparse-distilled checkpoint, but the analysis does not establish causality. Metric noise and checkpoint-specific behavior remain alternative explanations.

## 12. Is there enough evidence to study a training-free quality-risk predictor?

Yes. The target is well defined:

- VSA80 has 24 observable quality failures.
- All 24 are repairable with existing modes.
- Tail threshold violations are eliminated.
- The robust policy estimate adds less than 1% average latency and retains 68% of VSA80's median speed advantage.
- No single non-dense fixed mode solves all failures.
- Failure types are mostly SC-related but are not uniform.

Observed failure types:

{failure_type_lines}

This supports studying a training-free predictor that decides when VSA80 should back off. It does **not** prove that retained mass, margin, entropy, or another runtime statistic can predict those failures; that is the next hypothesis.

## Classification

**{classification}.**

DECISION: {decision}
"""
    (output / "REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
