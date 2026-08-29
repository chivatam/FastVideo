from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _bootstrap_ci(values: np.ndarray, *, seed: int = 1024, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _copy_video(row: pd.Series, output_dir: Path, label: str) -> str:
    source = Path(str(row["video_path"]))
    target = output_dir / f"{row['prompt_id']}__{label}.mp4"
    shutil.copy2(source, target)
    return target.name


def _write_skipped_phase_summaries(root: Path, commit: str) -> None:
    hypotheses = {
        2: "Retained VSA mass predicts sparse-attention risk.",
        3: "Boundary margin relative to FP4 perturbation predicts mask stability.",
        4: "A deterministic training-free controller captures material oracle advantage.",
        5: "A precise B200 latency LUT can drive a hardware-aware adaptive policy.",
        6: "A real B200 sparse-NVFP4 QK kernel beats the existing VSA BF16 path.",
        7: "The frozen policy transfers from Wan 1.3B to Wan 14B.",
        8: "The frozen policy transfers to LTX-2.3 and untouched T2V-CompBench prompts.",
    }
    for phase, hypothesis in hypotheses.items():
        output = root / f"phase{phase}" / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "phase": phase,
            "status": "partial",
            "hypothesis": hypothesis,
            "n_jobs_planned": 0,
            "n_jobs_completed": 0,
            "n_jobs_failed": 0,
            "primary_findings": [
                "Not run because corrected Phase 1 failed the predeclared H1 latency-opportunity gate."
            ],
            "gate_result": "fail",
            "reason": (
                "Fail-fast stop: the corrected per-prompt oracle improved mean end-to-end latency "
                "by 1.64%, below the predeclared 2% minimum."
            ),
            "next_phase": None,
            "exact_commits": {"fastvideo_research": commit},
        }
        output.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/adaptive_vsa_fp4"))
    args = parser.parse_args()

    root = args.root
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "representative_videos", "failure_cases"):
        target = final / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    commit = _git_commit()
    phase0 = json.loads((root / "phase0" / "summary.json").read_text())
    phase1 = json.loads((root / "phase1" / "summary.json").read_text())
    environment = json.loads((root / "env" / "environment.json").read_text())
    jobs = pd.read_parquet(root / "phase1" / "jobs.parquet")
    quality = pd.read_parquet(root / "phase1" / "quality_labels.parquet")
    oracle = pd.read_csv(root / "phase1" / "oracle.csv")
    config = pd.read_csv(root / "phase1" / "fixed_policy_summary.csv")
    paired = pd.read_csv(root / "phase1" / "paired_metrics.csv")

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
    all_results = quality.merge(paired_wide, on="job_id", how="left", validate="one_to_one")
    all_results["h1_gate_result"] = phase1["gate_result"]
    all_results["downstream_status"] = "not_run_h1_failed"
    all_results.to_parquet(final / "all_results.parquet", index=False)

    best_fixed = config[config["safe_rate"] == 1.0].sort_values(
        ["wall_ms_median", "attention_ms_median"]
    ).iloc[0]
    oracle_latency = float(oracle["oracle_wall_ms"].mean())
    oracle_advantage = float(
        (best_fixed["wall_ms_median"] - oracle_latency) / best_fixed["wall_ms_median"]
    )
    prompt_advantages = (
        (best_fixed["wall_ms_median"] - oracle["oracle_wall_ms"])
        / best_fixed["wall_ms_median"]
    ).to_numpy()
    advantage_ci = _bootstrap_ci(prompt_advantages)
    histogram = oracle["oracle_config"].value_counts()

    oracle_vs_controller = oracle[
        [
            "prompt_id",
            "prompt",
            "oracle_config",
            "oracle_kernel_path",
            "oracle_sparsity",
            "oracle_precision",
            "oracle_wall_ms",
            "speedup_over_best_all_safe_fixed",
            "best_all_safe_fixed_config",
            "best_all_safe_fixed_wall_ms",
        ]
    ].copy()
    oracle_vs_controller["controller_status"] = "not_run_h1_failed"
    oracle_vs_controller["controller_config"] = pd.NA
    oracle_vs_controller["controller_wall_ms"] = np.nan
    oracle_vs_controller["controller_regret_ms"] = np.nan
    oracle_vs_controller["oracle_capture"] = np.nan
    oracle_vs_controller.to_csv(final / "oracle_vs_controller.csv", index=False)

    summary_frame = all_results.copy()
    vbench_summary = (
        summary_frame.groupby(["config", "kernel_path", "sparsity"], as_index=False)
        .agg(
            samples=("prompt_id", "nunique"),
            safe_prompts=("quality_safe", "sum"),
            safe_rate=("quality_safe", "mean"),
            subject_consistency_median=("subject_consistency", "median"),
            motion_smoothness_median=("motion_smoothness", "median"),
            dynamic_degree_mean=("dynamic_degree", "mean"),
            subject_delta_median=("subject_delta", "median"),
            motion_delta_median=("motion_delta", "median"),
            dynamic_delta_mean=("dynamic_delta", "mean"),
            ssim_to_dense_median=("ssim_to_dense", "median"),
            psnr_to_dense_median=("psnr_to_dense", "median"),
            wall_ms_p50=("wall_ms", "median"),
            wall_ms_p90=("wall_ms", lambda values: values.quantile(0.90)),
            dit_ms_p50=("dit_ms", "median"),
            attention_ms_p50=("attention_ms", "median"),
        )
        .sort_values(["wall_ms_p50", "attention_ms_p50"])
    )
    vbench_summary["speedup_vs_dense_bf16"] = (
        float(best_fixed["wall_ms_median"]) / vbench_summary["wall_ms_p50"]
    )
    vbench_summary.to_csv(final / "vbench_summary.csv", index=False)

    wan14 = json.loads((root / "phase0" / "model_load_wan14.json").read_text())
    ltx23 = json.loads((root / "phase0" / "model_load_ltx23.json").read_text())
    model_summary = pd.DataFrame(
        [
            {
                "model_family": "Wan 2.1 1.3B",
                "model_id": jobs["model"].iloc[0],
                "revision": jobs["model_revision"].iloc[0],
                "status": "phase1_completed_h1_failed",
                "jobs_completed": len(jobs),
                "load_seconds": np.nan,
                "evaluation": "72 VBench prompts, one fixed seed, 12 configurations",
                "reason": "Oracle advantage 1.64%, below 2% go gate.",
            },
            {
                "model_family": "Wan 2.1 14B",
                "model_id": wan14["model"],
                "revision": wan14["revision"],
                "status": "load_probe_passed_downstream_not_run",
                "jobs_completed": 0,
                "load_seconds": wan14["load_seconds"],
                "evaluation": "Load compatibility only",
                "reason": "Frozen transfer was stopped after H1 failure.",
            },
            {
                "model_family": "LTX-2.3",
                "model_id": ltx23["model"],
                "revision": ltx23["revision"],
                "status": "load_probe_passed_compatibility_gate_not_run",
                "jobs_completed": 0,
                "load_seconds": ltx23["load_seconds"],
                "evaluation": "Load compatibility only",
                "reason": "Architecture-transfer gate was stopped after H1 failure.",
            },
        ]
    )
    model_summary.to_csv(final / "model_summary.csv", index=False)
    model_records = (
        model_summary.astype(object)
        .where(pd.notna(model_summary), None)
        .to_dict(orient="records")
    )

    pd.DataFrame(
        [
            {
                "dataset": "T2V-CompBench Motion Binding",
                "revision": environment["datasets"]["t2v_compbench_revision"],
                "status": "not_run_h1_failed",
                "samples": 0,
                "threshold_tuning_performed": False,
                "reason": "External validation is downstream of the frozen-controller gate; H1 failed.",
            }
        ]
    ).to_csv(final / "external_validation_summary.csv", index=False)

    latency_lut = (
        jobs.groupby(["model", "model_revision", "kernel_path", "precision", "sparsity"], as_index=False)
        .agg(
            samples=("job_id", "count"),
            attention_total_ms_p10=("attention_ms", lambda values: values.quantile(0.10)),
            attention_total_ms_p50=("attention_ms", "median"),
            attention_total_ms_p90=("attention_ms", lambda values: values.quantile(0.90)),
            attention_total_ms_p99=("attention_ms", lambda values: values.quantile(0.99)),
            dit_ms_p50=("dit_ms", "median"),
            wall_ms_p50=("wall_ms", "median"),
        )
        .sort_values("attention_total_ms_p50")
    )
    latency_lut["source"] = "phase1_full_generation_cuda_events"
    latency_lut["is_phase5_kernel_lut"] = False
    latency_lut["phase5_status"] = "not_run_h1_failed"
    latency_lut["notes"] = (
        "Accumulated attention time for a full 30-layer x 3-step generation; "
        "not the shape-resolved Phase 5 kernel LUT."
    )
    latency_lut.to_parquet(final / "latency_lut.parquet", index=False)

    for figure in (root / "phase1" / "figures").glob("*.png"):
        shutil.copy2(figure, final / "figures" / figure.name)

    easy_pid = oracle[oracle["oracle_config"] == "vsa_bf16@s0.80"].iloc[0]["prompt_id"]
    fp4_pid = oracle[oracle["oracle_config"] == "dense_nvfp4_fa4@s0.00"].iloc[0]["prompt_id"]
    representative_rows = []
    for prompt_id, configs in (
        (easy_pid, [("dense_bf16_fa4", 0.0), ("vsa_bf16", 0.8)]),
        (fp4_pid, [("dense_bf16_fa4", 0.0), ("dense_nvfp4_fa4", 0.0), ("vsa_bf16", 0.8)]),
    ):
        for kernel_path, sparsity in configs:
            row = quality[
                (quality["prompt_id"] == prompt_id)
                & (quality["kernel_path"] == kernel_path)
                & (quality["sparsity"] == sparsity)
            ].iloc[0]
            label = f"{kernel_path}_s{sparsity:.2f}"
            copied = _copy_video(row, final / "representative_videos", label)
            representative_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": row["prompt"],
                    "config": row["config"],
                    "quality_safe": bool(row["quality_safe"]),
                    "subject_delta": row["subject_delta"],
                    "motion_delta": row["motion_delta"],
                    "dynamic_delta": row["dynamic_delta"],
                    "file": copied,
                }
            )
    pd.DataFrame(representative_rows).to_csv(
        final / "representative_videos" / "index.csv",
        index=False,
    )

    vsa80 = quality[(quality["kernel_path"] == "vsa_bf16") & (quality["sparsity"] == 0.8)]
    worst_subject = vsa80.sort_values("subject_delta").iloc[0]
    worst_motion = vsa80.sort_values("motion_delta").iloc[0]
    dense_fp4 = quality[quality["kernel_path"] == "dense_nvfp4_fa4"]
    dynamic_failures = dense_fp4[dense_fp4["dynamic_delta"] < 0].sort_values("subject_delta")
    worst_dynamic = dynamic_failures.iloc[0]
    failure_specs = [
        ("vsa80_subject", worst_subject),
        ("vsa80_motion", worst_motion),
        ("dense_nvfp4_dynamic", worst_dynamic),
    ]
    failure_rows = []
    for case_name, candidate in failure_specs:
        dense = quality[
            (quality["prompt_id"] == candidate["prompt_id"])
            & (quality["kernel_path"] == "dense_bf16_fa4")
        ].iloc[0]
        dense_file = _copy_video(dense, final / "failure_cases", f"{case_name}_dense_bf16")
        candidate_file = _copy_video(
            candidate,
            final / "failure_cases",
            f"{case_name}_{candidate['kernel_path']}_s{candidate['sparsity']:.2f}",
        )
        failure_rows.append(
            {
                "case": case_name,
                "prompt_id": candidate["prompt_id"],
                "prompt": candidate["prompt"],
                "candidate_config": candidate["config"],
                "subject_delta": candidate["subject_delta"],
                "motion_delta": candidate["motion_delta"],
                "dynamic_delta": candidate["dynamic_delta"],
                "quality_safe": bool(candidate["quality_safe"]),
                "dense_video": dense_file,
                "candidate_video": candidate_file,
            }
        )
    pd.DataFrame(failure_rows).to_csv(final / "failure_cases" / "failure_cases.csv", index=False)

    manifest = {
        "status": "negative_result_h1_failed",
        "generated_at_utc": "2026-08-29",
        "repository": {
            "branch": "research",
            "current_commit": commit,
            "phase1_generation_commit": phase1["exact_commits"]["fastvideo_research"],
            "published_remote": "origin/research",
        },
        "models": model_records,
        "dataset": {
            "development": "VBench subject_consistency complete 72-prompt suite",
            "seed": 1024,
            "external": "T2V-CompBench Motion Binding, not run after H1 failure",
        },
        "phase_status": {
            "phase0": "pass",
            "phase1": "fail",
            "phase2": "not_run_h1_failed",
            "phase3": "not_run_h1_failed",
            "phase4": "not_run_h1_failed",
            "phase5": "not_run_h1_failed",
            "phase6": "not_run_h1_failed",
            "phase7": "not_run_h1_failed",
            "phase8": "not_run_h1_failed",
        },
        "jobs": {
            "phase0_completed": phase0["n_jobs_completed"],
            "phase1_planned": phase1["n_jobs_planned"],
            "phase1_completed": phase1["n_jobs_completed"],
            "phase1_failed": phase1["n_jobs_failed"],
            "phase1_attempts_per_job": 1,
        },
        "h1": {
            "gate_result": "fail",
            "oracle_histogram": {str(key): int(value) for key, value in histogram.items()},
            "best_all_safe_fixed_config": best_fixed["config"],
            "best_all_safe_fixed_wall_ms": float(best_fixed["wall_ms_median"]),
            "mean_oracle_wall_ms": oracle_latency,
            "mean_oracle_advantage": oracle_advantage,
            "bootstrap_95_ci": [advantage_ci[0], advantage_ci[1]],
            "minimum_required_advantage": 0.02,
            "dominant_oracle_share": float(histogram.iloc[0] / len(oracle)),
            "maximum_allowed_dominant_share": 0.90,
        },
        "quality_rule": phase1["quality_rule"],
        "invalidated_run": {
            "path": str(root / "phase1_invalid_b1935daa"),
            "reason": "Persistent-worker sparsity propagation defect; archived and excluded.",
            "fix_commit": "c8766c83",
        },
        "artifacts": [
            "executive_summary.md",
            "experiment_manifest.json",
            "all_results.parquet",
            "environment.json",
            "oracle_vs_controller.csv",
            "model_summary.csv",
            "vbench_summary.csv",
            "external_validation_summary.csv",
            "latency_lut.parquet",
            "figures/",
            "representative_videos/",
            "failure_cases/",
            "reproduction_commands.sh",
        ],
    }
    (final / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    shutil.copy2(root / "env" / "environment.json", final / "environment.json")

    executive_summary = f"""# Adaptive VSA + NVFP4 research result

## Decision

The project stopped at the corrected Phase 1 H1 gate. The per-prompt oracle was non-degenerate, but its mean measured end-to-end advantage was **{oracle_advantage:.2%}** (bootstrap 95% CI **{advantage_ci[0]:.2%}–{advantage_ci[1]:.2%}**), below the predeclared **2.00%** minimum. The data do not justify building an adaptive controller or a real sparse-NVFP4 kernel.

The corrected experiment completed 864/864 jobs on 72 published VBench subject-consistency prompts with seed 1024. Every VSA job recorded and verified the sparsity observed inside FastVideo attention. An earlier run at commit `b1935daa` was invalidated because persistent workers remained at the first sparsity; it is archived and excluded.

## Required questions

1. **Does the optimal sparsity/precision mode vary by input?** Yes, but not enough to pass H1. The oracle selected VSA BF16 at 80% sparsity for **48/72 (66.7%)** prompts, dense NVFP4 for **13/72 (18.1%)**, and dense BF16 for **11/72 (15.3%)**.
2. **How much faster is the oracle than the best fixed policy?** The best all-safe fixed policy was dense BF16 at **{best_fixed['wall_ms_median']:.2f} ms** median end-to-end latency. Mean oracle latency was **{oracle_latency:.2f} ms**, a saving of **{best_fixed['wall_ms_median'] - oracle_latency:.2f} ms** or **{oracle_advantage:.2%}**. This missed the 2% gate.
3. **Does retained VSA mass predict sparse error?** Not tested. Phase 2 was stopped after H1 failed.
4. **Does `m_k / delta` predict FP4 mask instability?** Not tested. Phase 3 was stopped after H1 failed.
5. **How much oracle advantage does the training-free controller capture?** No controller was built; capture and regret are unavailable.
6. **What is controller overhead?** Not measured because the controller was not built.
7. **Does the real sparse NVFP4 B200 kernel beat VSA BF16?** Not tested; Phase 6 kernel engineering was prohibited after H1 failed. Dense NVFP4 attention measured **375.43 ms** per full generation versus **592.11 ms** for dense BF16, but that is not a sparse-NVFP4 result.
8. **Does the frozen rule transfer from Wan 1.3B to Wan 14B?** Not tested. The pinned Wan 14B VSA checkpoint loaded successfully in **{wan14['load_seconds']:.2f} s**, then transfer work stopped.
9. **Does it transfer to LTX-2.3?** Not tested. The pinned LTX-2.3 checkpoint loaded successfully in **{ltx23['load_seconds']:.2f} s**, but its numerical VSA compatibility gate was not run.
10. **Does it hold on untouched T2V-CompBench Motion Binding prompts?** Not tested. External validation remained untouched because no controller was frozen.
11. **What are the strongest failure cases?** VSA BF16 at 80% sparsity was unsafe on **24/72** prompts; the worst subject-consistency delta was **{worst_subject['subject_delta']:.4f}** for “{worst_subject['prompt']}”. Dense NVFP4 was unsafe on **21/72** prompts under the joint rule, often because dynamic degree dropped. Simulated VSA+NVFP4 was slower than VSA BF16 and never became an all-safe fixed policy.
12. **What exact claim is justified?** A training-free, within-Wan-1.3B development-set characterization: quality-preserving modes vary across prompts, but the measured end-to-end adaptive opportunity was only 1.64% and failed the predeclared gate. Calibration-free routing, controller efficacy, sparse-NVFP4 speedup, cross-model transfer, LTX transfer, and external generalization are not supported.

## Quality and latency details

The empirical safe rule required subject consistency no more than 0.02 below paired dense BF16, motion smoothness no more than 0.01 below dense, and no dynamic-degree drop. No non-dense fixed configuration was safe for all 72 prompts:

- VSA BF16, 80% sparsity: **48/72 safe**, **9356.05 ms** median wall time, **274.96 ms** attention.
- Dense NVFP4 QK: **51/72 safe**, **9569.69 ms** median wall time, **375.43 ms** attention.
- Dense BF16: **72/72 safe**, **9586.64 ms** median wall time, **592.11 ms** attention.
- Simulated VSA+NVFP4, 80% sparsity: **50/72 safe**, **9710.02 ms** median wall time, **612.40 ms** attention.

## Candidate paper claim

On 72 published VBench prompts, the fastest quality-preserving attention mode varied across inputs, but an exhaustive oracle improved measured end-to-end latency by only 1.64% over the all-safe dense-BF16 policy. Under a predeclared 2% fail-fast threshold, this did not justify training-free adaptive routing or sparse-NVFP4 kernel engineering.

## Claims not supported

- A retained-mass safety predictor.
- An FP4 boundary-margin stability rule.
- A useful deterministic adaptive controller.
- A real sparse-NVFP4 B200 speedup.
- Wan 14B, LTX-2.3, or T2V-CompBench transfer.
- Calibration-free or cross-model generalization.
"""
    (final / "executive_summary.md").write_text(executive_summary)

    reproduction = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ec2-user/FastVideo
git checkout research
git checkout c8766c83214b69a89622a25135cd479739d7bccd

export HF_HOME=/mnt/fastvideo-gpu0/hf-cache
export HF_HUB_CACHE=/mnt/fastvideo-gpu0/hf-cache/hub
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export PYTHONPATH=/home/ec2-user/FastVideo

source /mnt/fastvideo-gpu0/venvs/adaptive-vsa-fp4/bin/activate
python -m research.adaptive_vsa_fp4.scripts.run_grid \
  --phase 1 \
  --db artifacts/adaptive_vsa_fp4/phase1/jobs.sqlite \
  --modes dense_bf16_fa4 vsa_bf16 sim_vsa_nvfp4 \
  --sparsities 0.2 0.4 0.6 0.7 0.8 \
  --num-workers 8

source /mnt/fastvideo-gpu0/venvs/adaptive-vsa-fp4-dense-nvfp4/bin/activate
python -m research.adaptive_vsa_fp4.scripts.run_grid \
  --phase 1 \
  --db artifacts/adaptive_vsa_fp4/phase1/jobs.sqlite \
  --modes dense_nvfp4_fa4 \
  --num-workers 8

source /mnt/fastvideo-gpu0/venvs/adaptive-vsa-fp4/bin/activate
python -m research.adaptive_vsa_fp4.scripts.collect \
  --db artifacts/adaptive_vsa_fp4/phase1/jobs.sqlite \
  --output artifacts/adaptive_vsa_fp4/phase1/jobs.parquet

export FASTVIDEO_EVAL_CACHE=/mnt/fastvideo-gpu0/eval-cache
export TORCH_HOME=/mnt/fastvideo-gpu0/eval-cache/torch
export PYTHONPATH=/home/ec2-user/FastVideo/fastvideo/third_party/eval/vbench:/home/ec2-user/FastVideo
python -m research.adaptive_vsa_fp4.scripts.evaluate \
  --jobs artifacts/adaptive_vsa_fp4/phase1/jobs.parquet \
  --output artifacts/adaptive_vsa_fp4/phase1/vbench_metrics.csv \
  --num-gpus 8
python -m research.adaptive_vsa_fp4.scripts.evaluate_paired \
  --jobs artifacts/adaptive_vsa_fp4/phase1/jobs.parquet \
  --output artifacts/adaptive_vsa_fp4/phase1/paired_metrics.csv \
  --num-gpus 8

export PYTHONPATH=/home/ec2-user/FastVideo
python -m research.adaptive_vsa_fp4.scripts.analyze_phase1 \
  --jobs artifacts/adaptive_vsa_fp4/phase1/jobs.parquet \
  --metrics artifacts/adaptive_vsa_fp4/phase1/vbench_metrics.csv \
  --output-dir artifacts/adaptive_vsa_fp4/phase1
python -m research.adaptive_vsa_fp4.scripts.build_final_package
"""
    reproduction_path = final / "reproduction_commands.sh"
    reproduction_path.write_text(reproduction)
    os.chmod(reproduction_path, 0o755)

    _write_skipped_phase_summaries(root, commit)
    print(f"final_package={final}")


if __name__ == "__main__":
    main()
