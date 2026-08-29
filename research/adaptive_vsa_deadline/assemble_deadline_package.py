from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
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
BASE_CONFIGS = {
    "Dense BF16": "dense_bf16_fa4@s0.00",
    "Fixed VSA80": "vsa_bf16@s0.80",
    "Fixed VSA60": "vsa_bf16@s0.60",
    "Fixed VSA40": "vsa_bf16@s0.40",
}
DECISION = "DECISION: STOP — ADAPTIVE VSA DOES NOT IMPROVE THE QUALITY/SPEED PARETO FRONT"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    wide = frame.pivot(index="job_id", columns="metric", values="score")
    wide = wide.rename(columns=METRIC_NAMES).reset_index()
    return wide


def _label_quality(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["subject_delta"] = (
        result["subject_consistency"] - result["dense_subject_consistency"]
    )
    result["motion_delta"] = (
        result["motion_smoothness"] - result["dense_motion_smoothness"]
    )
    result["dynamic_delta"] = (
        result["dynamic_degree"] - result["dense_dynamic_degree"]
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


def _bootstrap_interval(
    values: pd.Series,
    *,
    statistic: str = "mean",
    iterations: int = 20_000,
    seed: int = 20260829,
) -> dict[str, float]:
    data = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(iterations, len(data)))
    samples = data[indices]
    if statistic == "mean":
        draws = samples.mean(axis=1)
        point = data.mean()
    elif statistic == "median":
        draws = np.median(samples, axis=1)
        point = np.median(data)
    else:
        raise ValueError(statistic)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(len(data)),
        "statistic": statistic,
        "estimate": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _load_development(repo: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    base = pd.read_parquet(
        repo / "artifacts/adaptive_vsa_fp4/phase1/quality_labels.parquet"
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

    candidate_root = (
        repo
        / "artifacts/adaptive_vsa_deadline_final/development_72/candidates"
    )
    jobs = pd.read_parquet(candidate_root / "all_candidate_jobs.parquet")
    metrics = _metrics_wide(candidate_root / "vbench_metrics.csv")
    aggressive = jobs.merge(metrics, on="job_id", validate="one_to_one")

    conservative_jobs = pd.read_parquet(
        candidate_root / "p097_floor70/jobs.parquet"
    )
    conservative_jobs["candidate"] = "p097_floor70"
    conservative_metrics = _metrics_wide(
        candidate_root / "p097_floor70/vbench_metrics.csv"
    )
    conservative = conservative_jobs.merge(
        conservative_metrics,
        on="job_id",
        validate="one_to_one",
    )
    all_candidates = pd.concat(
        [aggressive, conservative],
        ignore_index=True,
    )
    all_candidates = all_candidates.merge(
        dense,
        on="prompt_id",
        validate="many_to_one",
    ).merge(
        vsa80,
        on="prompt_id",
        validate="many_to_one",
    )
    all_candidates = _label_quality(all_candidates)
    all_candidates["original_failure"] = ~all_candidates["vsa80_safe"]
    all_candidates["repaired"] = (
        all_candidates["original_failure"]
        & all_candidates["quality_safe"]
    )
    all_candidates["new_failure"] = (
        ~all_candidates["original_failure"]
        & ~all_candidates["quality_safe"]
    )

    summaries: list[dict[str, Any]] = []
    for candidate, group in all_candidates.groupby("candidate", sort=True):
        summaries.append(
            {
                "candidate": candidate,
                "retained_mass_threshold": float(group["adaptive_p"].iloc[0]),
                "maximum_sparsity": float(
                    group["adaptive_floor_sparsity"].iloc[0]
                ),
                "minimum_density_floor": float(
                    round(
                        1.0 - group["adaptive_floor_sparsity"].iloc[0],
                        10,
                    )
                ),
                "repaired_original_failures": int(group["repaired"].sum()),
                "new_failures": int(group["new_failure"].sum()),
                "unsafe": int((~group["quality_safe"]).sum()),
                "attention_ms_median": float(group["attention_ms"].median()),
                "wall_ms_median": float(group["wall_ms"].median()),
                "dit_ms_median": float(group["dit_ms"].median()),
                "effective_sparsity_mean": float(
                    group["effective_sparsity"].mean()
                ),
                "publication_valid_wall_timing": candidate
                in {"p099_floor80", "p097_floor70"},
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        [
            "repaired_original_failures",
            "new_failures",
            "attention_ms_median",
        ],
        ascending=[False, True, True],
        ignore_index=True,
    )
    selected_name = str(summary.iloc[0]["candidate"])
    if selected_name != "p097_floor70":
        raise RuntimeError(
            f"Expected the lexicographic winner p097_floor70, got {selected_name}"
        )
    selected = all_candidates.loc[
        all_candidates["candidate"].eq(selected_name)
    ].copy()
    return base, all_candidates, summary, selected


def _development_results(
    base: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    common = [
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
        "wall_ms",
        "dit_ms",
        "attention_ms",
        "effective_sparsity",
        "video_path",
    ]
    frames: list[pd.DataFrame] = []
    for method, config in BASE_CONFIGS.items():
        group = base.loc[base["config"].eq(config)].copy()
        group["method"] = method
        frames.append(group[["method", *common]])
    adaptive = selected.copy()
    adaptive["method"] = "Adaptive VSA"
    frames.append(adaptive[["method", *common]])
    return pd.concat(frames, ignore_index=True)


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in results.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "safe": int(group["quality_safe"].sum()),
                "unsafe": int((~group["quality_safe"]).sum()),
                "wall_ms_median": float(group["wall_ms"].median()),
                "dit_ms_median": float(group["dit_ms"].median()),
                "attention_ms_median": float(group["attention_ms"].median()),
                "effective_sparsity_mean": float(
                    group["effective_sparsity"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _literature_rows() -> list[dict[str, Any]]:
    return [
        {
            "paper": "VSA: Faster Video Diffusion with Trainable Sparse Attention",
            "method": "VSA fine-tuning",
            "venue_status": "arXiv v5",
            "publication_date": "2025-05-19",
            "model": "Wan2.1-1.3B",
            "task": "T2V",
            "resolution": "published Wan-1.3B setting",
            "frames": "not stated in cited table",
            "steps": "not stated in cited table",
            "gpu": "not stated in cited result",
            "gpu_count": "not stated",
            "training_free": False,
            "additional_training": "VSA sparse adaptation/fine-tuning",
            "calibration_required": False,
            "attention_mechanism": "learned coarse global estimator + fixed Top-K block sparsity",
            "fixed_or_adaptive": "fixed K",
            "reported_density_or_sparsity": "91.2% sparsity (8.8% density)",
            "benchmark": "VBench 1.0",
            "protocol": "paper Wan-1.3B protocol",
            "quality_metric": "Quality / Semantic / Total",
            "method_quality": "83.60 / 79.47 / 82.77",
            "dense_or_original_baseline": "Original Wan 83.71 / 77.98 / 82.56; full fine-tune 84.07 / 81.85 / 83.63",
            "quality_delta": "Total +0.21 vs original Wan; -0.86 vs full fine-tune",
            "attention_speedup": "not isolated in cited Wan result",
            "e2e_speedup": "1.72x calculated from 31 s / 18 s; described as 1.7x",
            "latency": "18 s DiT vs 31 s full attention",
            "source_table": "Table 3(a); Section 3.3",
            "source_pdf_page": "8",
            "source_url": "https://arxiv.org/abs/2505.13389",
            "comparability_note": "Same benchmark family, but our FastWan sparse-distilled checkpoint and 3-step protocol differ from the paper's VSA fine-tuning recipe.",
        },
        {
            "paper": "SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning",
            "method": "SpargeAttention2",
            "venue_status": "arXiv v1",
            "publication_date": "2026-02-15",
            "model": "Wan2.1-1.3B",
            "task": "T2V",
            "resolution": "480p",
            "frames": "not stated in cited table",
            "steps": "not stated in cited table",
            "gpu": "not stated in cited table",
            "gpu_count": "not stated",
            "training_free": False,
            "additional_training": "velocity-distillation sparse fine-tuning",
            "calibration_required": False,
            "attention_mechanism": "trainable hybrid Top-k + Top-p masker",
            "fixed_or_adaptive": "hybrid adaptive mask",
            "reported_density_or_sparsity": "95% sparsity",
            "benchmark": "selected VBench dimensions + reward/VQA",
            "protocol": "paper protocol",
            "quality_metric": "IQ / OC / AQ",
            "method_quality": "67.68 / 21.57 / 65.05",
            "dense_or_original_baseline": "63.67 / 20.27 / 64.41",
            "quality_delta": "+4.01 / +1.30 / +0.64",
            "attention_speedup": "16.17x calculated from 97 s / 6 s",
            "e2e_speedup": "2.34x calculated from 159 s / 68 s",
            "latency": "6 s attention; 68 s E2E",
            "source_table": "Table 4",
            "source_pdf_page": "7",
            "source_url": "https://arxiv.org/abs/2602.13515",
            "comparability_note": "Not a complete Quality/Semantic/Total result and requires additional training.",
        },
        {
            "paper": "RAPID: Reusing Attention Sparsity with Inter-step Adaptation for Efficient Video Diffusion",
            "method": "RAPID",
            "venue_status": "CVPR 2026 paper",
            "publication_date": "2026",
            "model": "Wan2.1-14B",
            "task": "T2V",
            "resolution": "768p",
            "frames": 81,
            "steps": "model default in paper",
            "gpu": "NVIDIA A100",
            "gpu_count": 1,
            "training_free": True,
            "additional_training": "none",
            "calibration_required": "manual warmup/threshold schedule",
            "attention_mechanism": "one-shot block importance estimation, cached masks, inter-step adaptive pruning",
            "fixed_or_adaptive": "inter-step adaptive",
            "reported_density_or_sparsity": "41.88% density",
            "benchmark": "VBench-2.0 prompts; dense-output similarity",
            "protocol": "augmented Wan prompts",
            "quality_metric": "PSNR / SSIM / LPIPS",
            "method_quality": "26.112 / 0.871 / 0.096",
            "dense_or_original_baseline": "full attention reference",
            "quality_delta": "similarity metrics only",
            "attention_speedup": "up to 3.2x reported",
            "e2e_speedup": "1.53x",
            "latency": "2601 s vs 3971 s",
            "source_table": "Table 1",
            "source_pdf_page": "7",
            "source_url": "https://openaccess.thecvf.com/",
            "comparability_note": "Different model scale, VBench version, resolution, metric, and GPU.",
        },
        {
            "paper": "HEART: Exploiting Head Heterogeneity in Sparse Attention for Video Diffusion",
            "method": "HEART (earlier HASTE)",
            "venue_status": "arXiv v2",
            "publication_date": "2026-05-19",
            "model": "Wan2.1-1.3B",
            "task": "T2V",
            "resolution": "480p",
            "frames": "paper protocol",
            "steps": "paper protocol",
            "gpu": "not stated in cited row",
            "gpu_count": "not stated",
            "training_free": True,
            "additional_training": "no model training",
            "calibration_required": True,
            "attention_mechanism": "head-wise sparsity allocation, temporal mask reuse, error-bound calibration",
            "fixed_or_adaptive": "head-adaptive",
            "reported_density_or_sparsity": "not reported in Table 2",
            "benchmark": "paper VBench/VReward protocol + dense-output similarity",
            "protocol": "Wan2.1-1.3B 480p paper protocol",
            "quality_metric": "VBench / VReward / PSNR / SSIM / LPIPS",
            "method_quality": "77.04% / 0.0585 / 21.61 / 0.6833 / 0.2337",
            "dense_or_original_baseline": "Dense VBench 77.15%; 195 s E2E",
            "quality_delta": "VBench -0.11 points",
            "attention_speedup": "2.23x",
            "e2e_speedup": "1.44x",
            "latency": "48 s attention; 135 s E2E",
            "source_table": "Table 2",
            "source_pdf_page": "7",
            "source_url": "https://arxiv.org/abs/2605.14513",
            "comparability_note": "Subset/weighted metric and calibration; not the official 16-dimension aggregate.",
        },
        {
            "paper": "FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation",
            "method": "FVAttn",
            "venue_status": "arXiv v1",
            "publication_date": "2026-07-17",
            "model": "Wan2.1-14B distilled",
            "task": "T2V",
            "resolution": "720p",
            "frames": 81,
            "steps": 4,
            "gpu": "NVIDIA H20",
            "gpu_count": 8,
            "training_free": True,
            "additional_training": "none beyond existing distilled LoRA",
            "calibration_required": False,
            "attention_mechanism": "Top-p routing + runtime head load balancing + slack-aware augmentation",
            "fixed_or_adaptive": "adaptive Top-p",
            "reported_density_or_sparsity": "Top-p=0.90",
            "benchmark": "complete VBench T2V set",
            "protocol": "946 test cases; paper reports scalar VBench",
            "quality_metric": "VBench scalar",
            "method_quality": "81.6%",
            "dense_or_original_baseline": "81.3%",
            "quality_delta": "+0.3 points",
            "attention_speedup": "paper reports DiT speedup",
            "e2e_speedup": "2.32x DiT",
            "latency": "14.96 s DiT vs 34.70 s",
            "source_table": "Table 1",
            "source_pdf_page": "9",
            "source_url": "https://arxiv.org/abs/2607.16190",
            "comparability_note": "Full prompts, but different 14B four-step model, 8xH20, and scalar aggregate.",
        },
        {
            "paper": "HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion",
            "method": "HyperVAttention",
            "venue_status": "arXiv v1",
            "publication_date": "2026-07-03",
            "model": "Wan2.2-14B",
            "task": "T2V",
            "resolution": "720p",
            "frames": 81,
            "steps": "paper protocol",
            "gpu": "paper hardware",
            "gpu_count": "paper protocol",
            "training_free": True,
            "additional_training": "none",
            "calibration_required": False,
            "attention_mechanism": "spatio-temporal query/key clustering and block merging",
            "fixed_or_adaptive": "input-dependent clustering",
            "reported_density_or_sparsity": "42.05% density",
            "benchmark": "paper VBench protocol",
            "protocol": "Wan2.2 T2V, n=75,600 reported",
            "quality_metric": "VBench scalar",
            "method_quality": "0.845",
            "dense_or_original_baseline": "0.845",
            "quality_delta": "0.000",
            "attention_speedup": "not isolated",
            "e2e_speedup": "1.72x",
            "latency": "paper aggregate",
            "source_table": "Table 1",
            "source_pdf_page": "8",
            "source_url": "https://arxiv.org/abs/2607.03012",
            "comparability_note": "Different Wan generation and protocol; scalar VBench only.",
        },
        {
            "paper": "SPADE: An Input-Adaptive Sparse Attention Engine for Fast Video Diffusion Models Inference",
            "method": "SPADE",
            "venue_status": "DAC 2026 / arXiv v1",
            "publication_date": "2026-08-05",
            "model": "Wan2.1",
            "task": "T2V",
            "resolution": "paper setting",
            "frames": "paper setting",
            "steps": "paper setting",
            "gpu": "paper hardware",
            "gpu_count": "paper protocol",
            "training_free": True,
            "additional_training": "none",
            "calibration_required": True,
            "attention_mechanism": "runtime scheme generation with SICS and head-wise sparse kernels",
            "fixed_or_adaptive": "input/head-adaptive",
            "reported_density_or_sparsity": "82.32% sparsity",
            "benchmark": "VBench scalar + dense-output similarity",
            "protocol": "100 prompts x2 videos stated by paper",
            "quality_metric": "VBench scalar / PSNR / SSIM / LPIPS",
            "method_quality": "0.79 / 25.87 / 0.87 / 0.10",
            "dense_or_original_baseline": "VBench 0.79",
            "quality_delta": "VBench 0.00",
            "attention_speedup": "2.62x",
            "e2e_speedup": "1.36x",
            "latency": "normalized speedup",
            "source_table": "Table 2",
            "source_pdf_page": "6",
            "source_url": "https://arxiv.org/abs/2608.03335",
            "comparability_note": "Different prompt/sample protocol and scalar score.",
        },
        {
            "paper": "LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration",
            "method": "LoSA",
            "venue_status": "arXiv v1",
            "publication_date": "2026-08-13",
            "model": "Wan2.1-T2V-1.3B",
            "task": "T2V",
            "resolution": "480p",
            "frames": 81,
            "steps": 50,
            "gpu": "NVIDIA H200",
            "gpu_count": 1,
            "training_free": True,
            "additional_training": "none",
            "calibration_required": False,
            "attention_mechanism": "near-lossless adaptive sparse attention",
            "fixed_or_adaptive": "adaptive",
            "reported_density_or_sparsity": "about 40% block area removed at the default 99% retained-mass threshold",
            "benchmark": "VBench",
            "protocol": "paper Wan1.3B protocol",
            "quality_metric": "Quality / Overall",
            "method_quality": "82.19 / 79.58",
            "dense_or_original_baseline": "82.45 / 79.64",
            "quality_delta": "Overall -0.06",
            "attention_speedup": "not isolated in Table 1",
            "e2e_speedup": "1.36x",
            "latency": "77 s vs 104 s",
            "source_table": "Table 1",
            "source_pdf_page": "6",
            "source_url": "https://arxiv.org/abs/2608.12032",
            "comparability_note": "Closest broad protocol match, but 50-step dense checkpoint and reported aggregate differ from our sparse-distilled 3-step checkpoint.",
        },
        {
            "paper": "Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation",
            "method": "SVG2",
            "venue_status": "arXiv v5",
            "publication_date": "2025-05-25",
            "model": "Wan2.1-14B",
            "task": "T2V",
            "resolution": "720p",
            "frames": "paper setting",
            "steps": "paper setting",
            "gpu": "NVIDIA H100",
            "gpu_count": 1,
            "training_free": True,
            "additional_training": "none",
            "calibration_required": False,
            "attention_mechanism": "semantic-aware Q/K clustering, permutation, centroid Top-p",
            "fixed_or_adaptive": "input-adaptive",
            "reported_density_or_sparsity": "29.51% density",
            "benchmark": "VBench scalar + similarity",
            "protocol": "30% dense warmup",
            "quality_metric": "VBench / PSNR / SSIM / LPIPS",
            "method_quality": "0.842 / 25.808 / 0.854 / 0.138",
            "dense_or_original_baseline": "VBench 0.846",
            "quality_delta": "VBench -0.004",
            "attention_speedup": "not isolated in cited row",
            "e2e_speedup": "1.60x",
            "latency": "normalized speedup",
            "source_table": "Table 1",
            "source_pdf_page": "9",
            "source_url": "https://arxiv.org/abs/2505.18875",
            "comparability_note": "Different model scale, steps, and scalar VBench convention.",
        },
        {
            "paper": "FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Fast Video Diffusion",
            "method": "FPSAttention",
            "venue_status": "arXiv v2",
            "publication_date": "2025-06-06",
            "model": "Wan1.3B",
            "task": "T2V",
            "resolution": "paper setting",
            "frames": "paper setting",
            "steps": "paper setting",
            "gpu": "paper hardware",
            "gpu_count": "paper protocol",
            "training_free": False,
            "additional_training": "joint FP8 and sparsity optimization",
            "calibration_required": True,
            "attention_mechanism": "training-aware FP8 + sparse attention co-design",
            "fixed_or_adaptive": "trained sparsity schedule",
            "reported_density_or_sparsity": "paper configurations",
            "benchmark": "full VBench metrics",
            "protocol": "paper reports five videos per prompt",
            "quality_metric": "Total",
            "method_quality": "0.8160",
            "dense_or_original_baseline": "paper baseline",
            "quality_delta": "paper comparison",
            "attention_speedup": "paper systems result",
            "e2e_speedup": "paper systems result",
            "latency": "paper systems result",
            "source_table": "Tables 5-7",
            "source_pdf_page": "14-16",
            "source_url": "https://arxiv.org/abs/2506.04648",
            "comparability_note": "Full VBench dimensions, but trained FP8+sparsity and a different sample protocol.",
        },
        {
            "paper": "XAttention: Block Sparse Attention with Antidiagonal Scoring",
            "method": "XAttention",
            "venue_status": "ICML 2025",
            "publication_date": "2025",
            "model": "HunyuanVideo",
            "task": "T2V",
            "resolution": "paper setting",
            "frames": "paper setting",
            "steps": "paper setting",
            "gpu": "paper hardware",
            "gpu_count": "paper protocol",
            "training_free": True,
            "additional_training": "none",
            "calibration_required": "threshold selection",
            "attention_mechanism": "antidiagonal block scoring",
            "fixed_or_adaptive": "thresholded block sparsity",
            "reported_density_or_sparsity": "80% sparsity cited setting",
            "benchmark": "dense-output similarity",
            "protocol": "paper video protocol",
            "quality_metric": "PSNR / SSIM / LPIPS / CLIP-T",
            "method_quality": "23.5 / 0.822 / 0.155 at threshold 0.95",
            "dense_or_original_baseline": "full attention reference",
            "quality_delta": "similarity metrics only",
            "attention_speedup": "not reported for the video-generation experiment",
            "e2e_speedup": "not reported for the video-generation experiment",
            "latency": "not reported for the video-generation experiment",
            "source_table": "Table 4",
            "source_pdf_page": "paper page 7",
            "source_url": "https://arxiv.org/abs/2503.16428",
            "comparability_note": "Different model and similarity metrics, not VBench aggregate.",
        },
        {
            "paper": "Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation",
            "method": "Radial Attention",
            "venue_status": "arXiv v2",
            "publication_date": "2025-06-24",
            "model": "Wan2.1-1.3B / Wan2.1-14B",
            "task": "T2V",
            "resolution": "paper settings",
            "frames": "paper settings",
            "steps": "paper settings",
            "gpu": "paper hardware",
            "gpu_count": "paper protocol",
            "training_free": True,
            "additional_training": "none",
            "calibration_required": False,
            "attention_mechanism": "radial energy-decay sparse pattern",
            "fixed_or_adaptive": "structured sparse schedule",
            "reported_density_or_sparsity": "323 PFLOPs vs 560 PFLOPs in the cited Wan2.1-14B row",
            "benchmark": "VBench and dense-output fidelity",
            "protocol": "paper protocol",
            "quality_metric": "PSNR / SSIM / LPIPS / Vision Reward",
            "method_quality": "23.9 / 0.842 / 0.163 / 0.128",
            "dense_or_original_baseline": "full attention",
            "quality_delta": "dense-output fidelity metrics",
            "attention_speedup": "not isolated in cited row",
            "e2e_speedup": "1.77x",
            "latency": "917 s vs 1630 s on Wan2.1-14B",
            "source_table": "Table 1",
            "source_pdf_page": "paper results page",
            "source_url": "https://arxiv.org/abs/2506.19852",
            "comparability_note": "Different protocol; broad systems comparison only.",
        },
    ]


def _novelty_matrix() -> pd.DataFrame:
    columns = [
        "method",
        "training_free_inference_adaptation",
        "requires_additional_sparse_attention_training",
        "uses_new_predictor_clustering_or_calibration",
        "reuses_existing_models_trained_sparse_estimator",
        "fixed_k",
        "adaptive_k",
        "top_p_retained_mass",
        "head_adaptive",
        "query_block_adaptive",
        "inter_step_reuse",
        "quality_recovery_objective",
        "evaluated_on_already_sparse_trained_checkpoint",
        "full_vbench",
        "multi_model_validation",
    ]
    rows = [
        ["VSA", False, True, True, True, True, False, False, False, True, False, False, True, True, True],
        ["SpargeAttention2", False, True, True, False, False, True, True, False, True, False, False, False, False, True],
        ["RAPID", True, False, True, False, False, True, True, False, False, True, False, False, False, True],
        ["HEART/HASTE", True, False, True, False, False, True, False, True, False, True, False, False, False, True],
        ["FVAttn", True, False, True, False, False, True, True, True, True, False, False, False, True, True],
        ["HyperVAttention", True, False, True, False, False, True, True, False, True, False, False, False, False, True],
        ["SPADE", True, False, True, False, False, True, False, True, True, False, False, False, False, True],
        ["LoSA", True, False, True, False, False, True, True, True, True, False, True, False, True, True],
        ["SVG2", True, False, True, False, False, True, True, False, True, False, False, False, False, True],
        ["Ours", True, False, False, True, False, True, True, False, True, False, True, True, False, False],
    ]
    return pd.DataFrame(rows, columns=columns)


def _save_literature(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    literature = pd.DataFrame(_literature_rows())
    literature_dir = output / "literature"
    literature_dir.mkdir(parents=True, exist_ok=True)
    literature.to_csv(literature_dir / "literature_results.csv", index=False)

    direct = literature.loc[
        literature["method"].isin(
            ["VSA fine-tuning", "LoSA", "FVAttn", "FPSAttention"]
        ),
        [
            "method",
            "model",
            "training_free",
            "fixed_or_adaptive",
            "benchmark",
            "quality_metric",
            "method_quality",
            "dense_or_original_baseline",
            "quality_delta",
            "e2e_speedup",
            "protocol",
            "comparability_note",
        ],
    ].copy()
    direct.insert(
        0,
        "directly_comparable_to_our_development_result",
        False,
    )
    direct.to_csv(
        literature_dir / "direct_vbench_comparison.csv",
        index=False,
    )

    broad = literature[
        [
            "method",
            "model",
            "attention_mechanism",
            "training_free",
            "additional_training",
            "benchmark",
            "quality_metric",
            "method_quality",
            "reported_density_or_sparsity",
            "attention_speedup",
            "e2e_speedup",
            "gpu",
            "comparability_note",
        ]
    ].copy()
    broad.to_csv(
        literature_dir / "broad_sparse_attention_comparison.csv",
        index=False,
    )
    novelty = _novelty_matrix()
    novelty.to_csv(literature_dir / "novelty_matrix.csv", index=False)

    source_lines = [
        "# Primary literature sources",
        "",
        "All numerical entries were transcribed from the cited primary paper PDF. "
        "Rows are not a leaderboard: protocols, models, steps, resolutions, sample counts, "
        "hardware, and metrics differ.",
        "",
    ]
    for row in literature.to_dict("records"):
        source_lines.extend(
            [
                f"## {row['method']}",
                "",
                f"- Paper: {row['paper']}",
                f"- Status/date: {row['venue_status']}; {row['publication_date']}",
                f"- Source: {row['source_url']}",
                f"- Location: {row['source_table']}, PDF page {row['source_pdf_page']}",
                f"- Extracted result: {row['quality_metric']} = {row['method_quality']}; "
                f"E2E speedup = {row['e2e_speedup']}",
                f"- Comparability: {row['comparability_note']}",
                "",
            ]
        )
    source_lines.extend(
        [
            "## Novelty conclusion",
            "",
            "The reviewed primary sources contain many adaptive Top-p, variable-density, "
            "head-adaptive, clustering, calibration, and inter-step-reuse mechanisms. "
            "None of the reviewed methods was found to repurpose the already-trained coarse "
            "estimator inside a fixed-budget VSA checkpoint as a no-new-model, inference-time "
            "quality-risk/compute-budget controller. This is a bounded literature conclusion, "
            "not a universal priority claim.",
            "",
        ]
    )
    (literature_dir / "literature_sources.md").write_text(
        "\n".join(source_lines)
    )
    return direct, broad


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _make_figures(
    output: Path,
    results: pd.DataFrame,
    method_summary: pd.DataFrame,
    trace: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})

    order = ["Dense BF16", "Fixed VSA80", "Fixed VSA60", "Fixed VSA40", "Adaptive VSA"]
    summary = method_summary.set_index("method").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    colors = ["#4c78a8", "#e45756", "#f58518", "#b279a2", "#54a24b"]
    bars = ax.bar(summary["method"], summary["unsafe"], color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Unsafe prompts (lower is better)")
    ax.set_title("72-prompt quality failures under the frozen dense-relative rule")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(summary["unsafe"]) * 1.2)
    fig.tight_layout()
    fig.savefig(figures / "failure_recovery.pdf")
    plt.close(fig)

    count_columns = [
        "decision_s70_count",
        "decision_s60_count",
        "decision_s40_count",
        "decision_s00_count",
    ]
    counts = trace[count_columns].sum()
    fractions = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    labels = ["VSA70", "VSA60", "VSA40", "Dense"]
    bars = ax.bar(labels, fractions.values * 100, color=["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756"])
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in fractions.values * 100], padding=3)
    ax.set_ylabel("Query-row decisions")
    ax.set_title("Frozen adaptive budget distribution (p=0.97, max sparsity=70%)")
    ax.set_ylim(0, max(fractions.values * 100) * 1.2)
    ax.text(
        0.5,
        -0.24,
        "VSA80 decisions are 0% because the selected conservative floor always backs off to at least VSA70.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(figures / "adaptive_budget_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for _, row in summary.iterrows():
        ax.scatter(
            row["wall_ms_median"] / 1000,
            row["unsafe"],
            s=75,
            color=colors[order.index(row["method"])],
        )
        ax.annotate(
            row["method"],
            (row["wall_ms_median"] / 1000, row["unsafe"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Median end-to-end latency (s, lower is better)")
    ax.set_ylabel("Unsafe prompts (lower is better)")
    ax.set_title("Development quality–speed frontier")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.96,
        "Dense BF16 dominates Adaptive VSA on both axes.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(figures / "quality_speed_pareto.pdf")
    plt.close(fig)

    per_job = trace.groupby("job_id").agg(
        effective_sparsity=("effective_sparsity", "mean"),
        native_retained_mass=("native_retained_mass_mean", "mean"),
        selected_retained_mass=("selected_retained_mass_mean", "mean"),
        **{column: (column, "sum") for column in count_columns},
    )
    per_job = per_job.merge(
        selected[["job_id", "prompt"]],
        left_index=True,
        right_on="job_id",
        validate="one_to_one",
    )
    denominator = per_job[count_columns].sum(axis=1)
    for column in count_columns:
        per_job[column] = per_job[column] / denominator
    examples = pd.concat(
        [
            per_job.nlargest(1, "effective_sparsity"),
            per_job.nsmallest(1, "effective_sparsity"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    for ax, (_, row), title in zip(
        axes,
        examples.iterrows(),
        ["More concentrated example", "More diffuse example"],
    ):
        values = [row[column] * 100 for column in count_columns]
        ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, 55)
        prompt = str(row["prompt"])
        if len(prompt) > 52:
            prompt = prompt[:49] + "..."
        ax.text(
            0.5,
            0.98,
            prompt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            wrap=True,
        )
        ax.text(
            0.5,
            0.76,
            f"native VSA80 retained mass={row['native_retained_mass']:.3f}\n"
            f"effective sparsity={row['effective_sparsity']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
        )
    axes[0].set_ylabel("Query-row decisions (%)")
    fig.suptitle("Measured controller behavior: concentrated scores spend less; diffuse scores back off")
    fig.tight_layout()
    fig.savefig(figures / "mechanism_example.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "Full 16-dimension VBench was not run",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Fail-fast gate: Adaptive VSA was 4.98% slower than dense BF16\n"
        "and had 2 unsafe prompts versus 0 for dense on the 72-prompt development suite.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.savefig(figures / "full_vbench_dimensions.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_tables(
    output: Path,
    method_summary: pd.DataFrame,
    direct: pd.DataFrame,
    broad: pd.DataFrame,
) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ordered = method_summary.set_index("method").loc[
        ["Dense BF16", "Fixed VSA80", "Fixed VSA60", "Fixed VSA40", "Adaptive VSA"]
    ].reset_index()
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Unsafe / 72 & E2E (ms) & Attention (ms) & Effective sparsity \\",
        r"\midrule",
    ]
    for row in ordered.to_dict("records"):
        lines.append(
            f"{row['method']} & {row['unsafe']} & "
            f"{row['wall_ms_median']:.1f} & {row['attention_ms_median']:.1f} & "
            f"{100 * row['effective_sparsity_mean']:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables / "ablation.tex").write_text("\n".join(lines))

    (tables / "main_full_vbench.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{lc}",
                r"\toprule",
                r"Benchmark & Status \\",
                r"\midrule",
                r"Full-prompt single-seed VBench 1.0 & Not run: development speed gate failed \\",
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        )
    )
    (tables / "transfer.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{lc}",
                r"\toprule",
                r"Model & Status \\",
                r"\midrule",
                r"Wan2.1 VSA 14B 720p & Not run: downstream transfer stopped after development gate \\",
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        )
    )
    direct.to_latex(
        tables / "literature_direct.tex",
        index=False,
        longtable=True,
        escape=True,
    )
    broad.to_latex(
        tables / "literature_broad.tex",
        index=False,
        longtable=True,
        escape=True,
    )


def _write_stopped_benchmarks(output: Path, commit: str) -> None:
    full = output / "full_vbench_1p3b"
    full.mkdir(parents=True, exist_ok=True)
    _write_json(
        full / "generation_manifest.json",
        {
            "status": "not_run",
            "benchmark": "VBench 1.0 T2V, all 16 dimensions",
            "planned_protocol": "full-prompt single-seed VBench evaluation",
            "methods": ["Dense BF16", "Fixed VSA80", "Adaptive VSA"],
            "reason": "Fail-fast development gate failed: selected Adaptive VSA was slower than dense BF16 and therefore did not preserve a positive quality/speed Pareto improvement.",
            "commit": commit,
        },
    )
    pd.DataFrame(
        columns=[
            "dimension",
            "dense_bf16",
            "fixed_vsa80",
            "adaptive_vsa",
            "adaptive_minus_vsa80",
            "status",
        ]
    ).to_csv(full / "per_dimension.csv", index=False)
    pd.DataFrame(
        columns=["method", "quality", "semantic", "total", "status"]
    ).to_csv(full / "aggregates.csv", index=False)
    pd.DataFrame(
        columns=["method", "wall_ms", "dit_ms", "attention_ms", "status"]
    ).to_csv(full / "latency.csv", index=False)
    pd.DataFrame(
        columns=["method", "effective_sparsity", "status"]
    ).to_csv(full / "effective_sparsity.csv", index=False)
    (full / "REPORT.md").write_text(
        "# Full VBench 1.3B\n\n"
        "Status: **not run**.\n\n"
        "The required fail-fast development gate stopped expansion. The frozen adaptive "
        "policy substantially recovered quality, but its median end-to-end latency was "
        "slower than dense BF16, so a 2,838-video single-seed full-suite generation would "
        "not test a positive quality/speed hypothesis. Empty CSVs are intentional and "
        "prevent missing dimensions from being misreported as zeros.\n"
    )

    transfer = output / "wan14b_transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    _write_json(
        transfer / "generation_manifest.json",
        {
            "status": "not_run",
            "model": "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
            "model_revision": "c505014d0f8fe673ddf2f8cc5307eabf791e1f9d",
            "native_sparsity": 0.90,
            "frozen_policy_mapping": {
                "retained_mass_threshold": 0.97,
                "maximum_sparsity": 0.70,
                "native_sparsity": 0.90,
                "candidate_sparsities": [0.90, 0.80, 0.70, 0.60, 0.40, 0.0],
            },
            "reason": "Not authorized by the fail-fast outcome: the primary 1.3B development gate failed.",
            "commit": commit,
        },
    )
    pd.DataFrame(
        columns=[
            "method",
            "prompt_id",
            "subject_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "quality_safe",
            "status",
        ]
    ).to_csv(transfer / "results.csv", index=False)
    pd.DataFrame(
        columns=["method", "wall_ms", "dit_ms", "attention_ms", "status"]
    ).to_csv(transfer / "latency.csv", index=False)
    (transfer / "REPORT.md").write_text(
        "# Wan-14B frozen-policy transfer\n\n"
        "Status: **not run**.\n\n"
        "The deadline protocol makes this downstream validation conditional on a positive "
        "1.3B development quality/speed gate. That gate failed, so no Wan-14B generation "
        "was launched. The manifest records the exact density-based mapping that would have "
        "been used without retuning.\n"
    )


def _write_development_report(
    output: Path,
    summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    selected: pd.DataFrame,
    trace: pd.DataFrame,
    statistics: dict[str, Any],
    commit: str,
) -> None:
    development = output / "development_72"
    methods = method_summary.set_index("method")
    adaptive = methods.loc["Adaptive VSA"]
    vsa80 = methods.loc["Fixed VSA80"]
    dense = methods.loc["Dense BF16"]
    vsa60 = methods.loc["Fixed VSA60"]
    vsa40 = methods.loc["Fixed VSA40"]
    repaired = int(selected["repaired"].sum())
    new_failures = int(selected["new_failure"].sum())
    original_unsafe = int(selected["original_failure"].sum())
    unsafe = int((~selected["quality_safe"]).sum())
    wall_over_vsa = adaptive["wall_ms_median"] - vsa80["wall_ms_median"]
    wall_over_dense = adaptive["wall_ms_median"] - dense["wall_ms_median"]
    raw_retention = (
        (dense["wall_ms_median"] - adaptive["wall_ms_median"])
        / (dense["wall_ms_median"] - vsa80["wall_ms_median"])
    )
    decision_columns = [
        "decision_s70_count",
        "decision_s60_count",
        "decision_s40_count",
        "decision_s00_count",
    ]
    counts = trace[decision_columns].sum()
    fractions = counts / counts.sum()
    weighted_sparsity = 1.0 - (
        0.30 * counts["decision_s70_count"]
        + 0.40 * counts["decision_s60_count"]
        + 0.60 * counts["decision_s40_count"]
        + 1.00 * counts["decision_s00_count"]
    ) / counts.sum()

    selection_table = summary[
        [
            "candidate",
            "retained_mass_threshold",
            "maximum_sparsity",
            "repaired_original_failures",
            "new_failures",
            "unsafe",
            "attention_ms_median",
            "effective_sparsity_mean",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    report = f"""# Adaptive VSA 72-prompt development report

## Outcome

The retained-mass controller is a strong **quality recovery mechanism** but not a
quality/speed Pareto improvement in this implementation.

- Fixed VSA80 unsafe: **{original_unsafe} / 72**
- Adaptive VSA unsafe: **{unsafe} / 72**
- Original failures repaired: **{repaired} / 24**
- New failures among originally safe prompts: **{new_failures} / 48**
- Frozen policy: **p=0.97**, maximum sparsity **70%** (minimum density **30%**)
- Commit: `{commit}`

## Minimal policy selection

Lexicographic objective: maximize repaired original failures, then minimize new
failures, then minimize measured attention time.

{selection_table}

The conservative floor won because it tied the best repair count (23) while
reducing new failures from 2 to 1. The floor excludes VSA80, so the selected
policy always spends at least the VSA70 budget.

## Systems result

| Method | Unsafe / 72 | Median E2E (ms) | Median DiT (ms) | Median attention (ms) | Mean effective sparsity |
|---|---:|---:|---:|---:|---:|
| Dense BF16 | {int(dense['unsafe'])} | {dense['wall_ms_median']:.1f} | {dense['dit_ms_median']:.1f} | {dense['attention_ms_median']:.1f} | 0.0% |
| Fixed VSA80 | {int(vsa80['unsafe'])} | {vsa80['wall_ms_median']:.1f} | {vsa80['dit_ms_median']:.1f} | {vsa80['attention_ms_median']:.1f} | 80.0% |
| Fixed VSA60 | {int(vsa60['unsafe'])} | {vsa60['wall_ms_median']:.1f} | {vsa60['dit_ms_median']:.1f} | {vsa60['attention_ms_median']:.1f} | 60.0% |
| Fixed VSA40 | {int(vsa40['unsafe'])} | {vsa40['wall_ms_median']:.1f} | {vsa40['dit_ms_median']:.1f} | {vsa40['attention_ms_median']:.1f} | 40.0% |
| Adaptive VSA | {int(adaptive['unsafe'])} | {adaptive['wall_ms_median']:.1f} | {adaptive['dit_ms_median']:.1f} | {adaptive['attention_ms_median']:.1f} | {100 * weighted_sparsity:.2f}% |

Adaptive overhead versus fixed VSA80 is **{wall_over_vsa:.1f} ms
({100 * wall_over_vsa / vsa80['wall_ms_median']:.2f}%)**. It is also
**{wall_over_dense:.1f} ms ({100 * wall_over_dense / dense['wall_ms_median']:.2f}%)
slower than dense BF16**. The raw retained fraction of VSA80's small median speed
advantage is **{100 * raw_retention:.1f}%** (clipped practical retention: 0%).

## Budget distribution

- VSA80: 0.00% (excluded by selected conservative floor)
- VSA70: {100 * fractions['decision_s70_count']:.2f}%
- VSA60: {100 * fractions['decision_s60_count']:.2f}%
- VSA40: {100 * fractions['decision_s40_count']:.2f}%
- Dense: {100 * fractions['decision_s00_count']:.2f}%
- Decision-weighted effective sparsity: {100 * weighted_sparsity:.2f}%

## Statistical notes

Paired prompt-level bootstrap intervals are stored in `statistics.json`.
The benchmark metrics are deterministic single-seed measurements on 72 published
VBench subject-consistency prompts. This is a development mechanism study, not a
leaderboard result.

## Gate

- Quality recovery: **pass** (23/24 repaired; 1 new failure).
- Positive speedup over dense: **fail**.
- Preserve most of fixed-VSA speed: **fail**.
- Full VBench expansion: **stopped**.
- Wan-14B transfer: **stopped**.

Dense BF16 has both fewer failures (0 versus 2) and lower median latency
({dense['wall_ms_median']:.1f} ms versus {adaptive['wall_ms_median']:.1f} ms), so
the selected adaptive implementation is dominated on the measured development
quality/speed plane.
"""
    (development / "REPORT.md").write_text(report)
    _write_json(development / "statistics.json", statistics)


def _write_final_result(
    output: Path,
    method_summary: pd.DataFrame,
    selected: pd.DataFrame,
    trace: pd.DataFrame,
    commit: str,
) -> None:
    methods = method_summary.set_index("method")
    adaptive = methods.loc["Adaptive VSA"]
    vsa80 = methods.loc["Fixed VSA80"]
    dense = methods.loc["Dense BF16"]
    vsa60 = methods.loc["Fixed VSA60"]
    vsa40 = methods.loc["Fixed VSA40"]
    repaired = int(selected["repaired"].sum())
    new_failures = int(selected["new_failure"].sum())
    unsafe = int((~selected["quality_safe"]).sum())
    count_columns = [
        "decision_s70_count",
        "decision_s60_count",
        "decision_s40_count",
        "decision_s00_count",
    ]
    counts = trace[count_columns].sum()
    weighted_sparsity = 1.0 - (
        0.30 * counts["decision_s70_count"]
        + 0.40 * counts["decision_s60_count"]
        + 0.60 * counts["decision_s40_count"]
        + counts["decision_s00_count"]
    ) / counts.sum()
    speedup_vs_dense = dense["wall_ms_median"] / adaptive["wall_ms_median"]
    overhead_vs_vsa = (
        adaptive["wall_ms_median"] / vsa80["wall_ms_median"] - 1.0
    )

    text = f"""# FINAL RESULT — Adaptive VSA deadline study

## Executive result

Adaptive VSA substantially improved quality over the checkpoint's native fixed
VSA80 point, reducing unsafe prompts from **24/72 to {unsafe}/72** and repairing
**{repaired}/24** original failures. It introduced **{new_failures}/48** new
failure. However, the selected controller was **{100 * overhead_vs_vsa:.2f}%**
slower than VSA80 and **{100 * (1 / speedup_vs_dense - 1):.2f}% slower than dense
BF16**. Dense had 0 unsafe prompts, so the adaptive implementation did not improve
the measured quality/speed Pareto front.

## Required answers

1. **Does Adaptive VSA improve quality over native fixed VSA?** Yes on the
   72-prompt development rule: unsafe prompts fell from 24 to {unsafe}.
2. **Original failures repaired:** {repaired}/24.
3. **New failures introduced:** {new_failures}/48 originally safe prompts.
4. **Frozen policy:** retained-mass threshold `p=0.97`; maximum sparsity `0.70`;
   minimum density floor `0.30`; candidate sparsities `[0.8, 0.7, 0.6, 0.4, 0.0]`;
   native reference sparsity `0.80`; commit `{commit}`.
5. **Complete 16-dimensional full VBench:** not run because the fail-fast
   development speed gate failed. Empty result CSVs are intentional.
6. **Full VBench Quality/Semantic/Total:** not measured for Dense, VSA80, or
   Adaptive under this study.
7. **Latency added over fixed VSA:** {adaptive['wall_ms_median'] - vsa80['wall_ms_median']:.1f}
   ms median E2E ({100 * overhead_vs_vsa:.2f}%).
8. **Fixed-VSA speed advantage retained:** 0% in practical terms; Adaptive was
   slower than dense (speedup versus dense = {speedup_vs_dense:.3f}x).
9. **Effective average sparsity:** {100 * weighted_sparsity:.2f}%,
   decision-weighted.
10. **Does it outperform simply choosing safer fixed sparsity?** It has fewer
    failures than VSA60 ({int(vsa60['unsafe'])}) and VSA40 ({int(vsa40['unsafe'])}),
    but is slower than both; dense is faster and has zero failures.
11. **Wan-14B transfer:** not run because it was conditional on a positive
    primary gate. The frozen density mapping is recorded in its manifest.
12. **Published methods:** exact primary-source rows for VSA, SpargeAttention2,
    RAPID, HEART/HASTE, FVAttn, HyperVAttention, SPADE, LoSA, SVG2,
    FPSAttention, XAttention, and Radial Attention are in
    `literature/literature_results.csv`.
13. **Directly comparable literature:** none is strictly directly comparable to
    this development result because we did not run full VBench and used a
    sparse-distilled 3-step checkpoint. VSA and LoSA are the closest Wan/VBench
    anchors, with explicit caveats.
14. **Non-comparable literature:** rows differ in model size, checkpoint,
    resolution, frame count, denoising steps, VBench version/subset, sample count,
    GPU, and quality metric. The broad table does not rank unlike metrics.
15. **Prior exact reuse of VSA's trained estimator:** none was identified in the
    reviewed primary sources. Many methods use adaptive Top-p or variable budgets,
    but not this exact reuse path.
16. **Surviving novelty claim:** *We test repurposing the trained coarse estimator
    of a fixed-budget VSA checkpoint as a no-new-model, training-free inference-time
    attention-budget controller.* The mechanism improves quality, but this
    implementation does not preserve the required systems advantage.

## Key limitations

- One model, one seed, 72 development prompts, and three VBench dimensions.
- Threshold/floor selected on the same development suite: training-free, not
  calibration-free.
- The selected conservative floor never uses native VSA80; it starts at VSA70.
- Full VBench and second-model transfer were correctly stopped, so no general
  quality or cross-model claim is supported.
- The current variable-mask path has too much measured end-to-end overhead.

## Artifact map

- Development data/report: `development_72/`
- Stopped full-benchmark manifest: `full_vbench_1p3b/`
- Stopped transfer manifest: `wan14b_transfer/`
- Primary-source extraction: `literature/`
- Publication figures: `figures/`
- LaTeX tables: `tables/`

{DECISION}
"""
    (output / "FINAL_RESULT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_vsa_deadline_final"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (
        args.output
        if args.output.is_absolute()
        else (repo / args.output)
    )
    output.mkdir(parents=True, exist_ok=True)
    for child in [
        "development_72",
        "full_vbench_1p3b",
        "wan14b_transfer",
        "literature",
        "figures",
        "tables",
    ]:
        (output / child).mkdir(parents=True, exist_ok=True)

    commit = _git(repo, "rev-parse", "HEAD")
    base, all_candidates, selection_summary, selected = _load_development(
        repo
    )
    results = _development_results(base, selected)
    method_summary = _method_summary(results)
    development = output / "development_72"
    results.to_csv(development / "results.csv", index=False)
    selection_summary.to_csv(
        development / "candidate_summary.csv",
        index=False,
    )

    trace_source = (
        output
        / "development_72/candidates/p097_floor70/policy_trace.parquet"
    )
    trace_destination = development / "policy_trace.parquet"
    if trace_source.resolve() != trace_destination.resolve():
        shutil.copy2(trace_source, trace_destination)
    trace = pd.read_parquet(trace_destination)

    frozen = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "retained_mass_threshold": 0.97,
        "maximum_sparsity": 0.70,
        "minimum_density_floor": 0.30,
        "candidate_sparsities": [0.8, 0.7, 0.6, 0.4, 0.0],
        "native_sparsity": 0.80,
        "selection_method": (
            "Lexicographic: maximize repaired original VSA80 failures; "
            "minimize new failures among original safe prompts; minimize measured "
            "attention latency."
        ),
        "development_prompts": 72,
        "prompt_suite": "VBench subject_consistency published prompts",
        "seed": 1024,
        "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "model_revision": "25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef",
        "code_commit": commit,
        "selected_candidate": "p097_floor70",
        "full_vbench_started": False,
        "note": "This conservative floor excludes VSA80 decisions and always selects at least VSA70.",
    }
    _write_json(output / "adaptive_policy_frozen.json", frozen)
    _write_json(development / "adaptive_policy_frozen.json", frozen)
    _write_json(
        development / "policy_selection.json",
        {
            **frozen,
            "candidate_ranking": selection_summary.to_dict("records"),
            "quality_rule": {
                "subject_consistency_delta_min": -0.02,
                "motion_smoothness_delta_min": -0.01,
                "dynamic_degree_delta_min": 0.0,
            },
            "gate": {
                "quality_recovery": "pass",
                "positive_speedup_vs_dense": "fail",
                "preserve_fixed_vsa_speed": "fail",
                "downstream_expansion": "stop",
            },
        },
    )

    adaptive = selected.set_index("prompt_id")
    vsa80 = base.loc[base["config"].eq("vsa_bf16@s0.80")].set_index(
        "prompt_id"
    )
    paired = adaptive.join(
        vsa80[
            [
                "subject_consistency",
                "motion_smoothness",
                "dynamic_degree",
                "wall_ms",
                "dit_ms",
                "attention_ms",
            ]
        ],
        rsuffix="_vsa80",
    )
    statistics = {
        "adaptive_minus_fixed_vsa80": {
            "subject_consistency_mean": _bootstrap_interval(
                paired["subject_consistency"]
                - paired["subject_consistency_vsa80"]
            ),
            "motion_smoothness_mean": _bootstrap_interval(
                paired["motion_smoothness"]
                - paired["motion_smoothness_vsa80"]
            ),
            "dynamic_degree_mean": _bootstrap_interval(
                paired["dynamic_degree"] - paired["dynamic_degree_vsa80"]
            ),
            "wall_ms_mean": _bootstrap_interval(
                paired["wall_ms"] - paired["wall_ms_vsa80"]
            ),
            "wall_ms_median": _bootstrap_interval(
                paired["wall_ms"] - paired["wall_ms_vsa80"],
                statistic="median",
            ),
            "dit_ms_mean": _bootstrap_interval(
                paired["dit_ms"] - paired["dit_ms_vsa80"]
            ),
            "attention_ms_mean": _bootstrap_interval(
                paired["attention_ms"] - paired["attention_ms_vsa80"]
            ),
        }
    }
    _write_development_report(
        output,
        selection_summary,
        method_summary,
        selected,
        trace,
        statistics,
        commit,
    )
    _write_stopped_benchmarks(output, commit)
    direct, broad = _save_literature(output)
    _make_figures(output, results, method_summary, trace, selected)
    _write_tables(output, method_summary, direct, broad)
    _write_final_result(output, method_summary, selected, trace, commit)
    print(f"assembled={output}")
    print(DECISION)


if __name__ == "__main__":
    main()
