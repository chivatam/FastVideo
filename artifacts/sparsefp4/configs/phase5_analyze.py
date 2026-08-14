"""Phase 5 paired similarity: every arm against DENSE-BF16, per prompt.

Reads the float16 decoded frames each generation saved (not the mp4 -- H.264
quantization noise is far larger than the effect Phase 2 predicts, so scoring
compressed video would manufacture a null) and computes, per (prompt, arm):

* PSNR / SSIM / LPIPS via FastVideo's own ``fastvideo.eval`` metrics
  (``common.psnr``, ``common.ssim``, ``common.lpips``) rather than a private
  reimplementation;
* mean absolute pixel difference and Pearson pixel correlation, which is what
  Phase 0 reported -- kept for continuity;
* max absolute pixel difference and the fraction of pixels that changed at all,
  which is the sharpest available answer to "is this visible".

The decisive contrast is computed twice over: every arm against DENSE-BF16, and
then SPARSE-FP4-NAIVE against SPARSE-FP4-ROUTE16 directly (NVFP4 router vs BF16
router at identical sparsity and identical compute). Both are written as raw
JSONL before any aggregation.

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_analyze.py \
        --run-id 20260814-031500-8208536-p5-main
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REFERENCE_ARM = "DENSE-BF16"
ARM_ORDER = (
    "DENSE-BF16",
    "DENSE-FP4",
    "SPARSE-BF16",
    "SPARSE-FP4-NAIVE",
    "SPARSE-FP4-ROUTE8",
    "SPARSE-FP4-ROUTE16",
)
# Pairs that answer a specific question, scored directly rather than differenced
# out of two vs-reference numbers (differencing would hide sign and scale).
DIRECT_PAIRS = (
    ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16", "H3 at the video level: NVFP4 router vs BF16 router"),
    ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "H3 at the video level: NVFP4 router vs FP8 router"),
    ("SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16", "H3 at the video level: FP8 router vs BF16 router"),
    ("SPARSE-BF16", "DENSE-BF16", "cost of sparsity itself, BF16 compute"),
    ("SPARSE-FP4-NAIVE", "DENSE-FP4", "cost of sparsity itself, NVFP4 Q/K compute"),
    ("DENSE-FP4", "DENSE-BF16", "cost of NVFP4 Q/K quantization alone"),
)
# The three arms whose compute has no native kernel here.
SIMULATED_ARMS = {"SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16"}


def load_frames(path: Path) -> torch.Tensor:
    """``[B, C, T, H, W]`` float16 on disk -> ``(T, C, H, W)`` float32 tensor."""
    array = np.load(path)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.dim() == 5:
        tensor = tensor[0]
    return tensor.permute(1, 0, 2, 3).contiguous()


def pixel_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Phase 0's pixel metrics plus the "is anything different at all" ones."""
    cand = candidate.double().flatten()
    ref = reference.double().flatten()
    diff = cand - ref
    cand_centered = cand - cand.mean()
    ref_centered = ref - ref.mean()
    denominator = cand_centered.norm() * ref_centered.norm()
    # 1/255 is one 8-bit code: below it, a difference cannot survive being
    # written to any ordinary video file, let alone be seen.
    return {
        "mean_abs_pixel_diff": float(diff.abs().mean().item()),
        "max_abs_pixel_diff": float(diff.abs().max().item()),
        "rms_pixel_diff": float(diff.pow(2).mean().sqrt().item()),
        "pixel_correlation": float((cand_centered @ ref_centered / denominator).item()) if denominator > 0 else None,
        "frac_pixels_changed": float((diff != 0).double().mean().item()),
        "frac_pixels_changed_gt_1_255": float((diff.abs() > 1.0 / 255.0).double().mean().item()),
        "identical": bool(torch.equal(candidate, reference)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--video-root", type=Path, default=Path("/mnt/scratch/sparsefp4-videos"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/sparsefp4/raw"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="similarity")
    args = parser.parse_args()

    from fastvideo.eval import create_evaluator

    raw_dir = args.raw_root / args.run_id
    video_dir = args.video_root / args.run_id
    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    summaries_all: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("run_summary_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("seed", -1)) != args.seed:
            continue
        summaries_all[record.get("tag") or f"{record['prompt_id']}_{record['arm']}"] = record
        if record["arm"] in ARM_ORDER:
            summaries[(record["prompt_id"], record["arm"])] = record
    prompts = sorted({prompt for prompt, _ in summaries})
    if not prompts:
        raise SystemExit(f"no run summaries for seed {args.seed} under {raw_dir}")

    evaluator = create_evaluator(metrics=["common.psnr", "common.ssim", "common.lpips"], device=args.device)

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    def frames_for(prompt: str, arm: str) -> torch.Tensor | None:
        record = summaries.get((prompt, arm))
        if record is None:
            exclusions.append({"prompt_id": prompt, "arm": arm, "reason": "no run summary"})
            return None
        path = Path(record["frame_path"]) if record.get("frame_path") else video_dir / f"{prompt}_{arm}_s{args.seed}.f16.npy"
        if not path.is_file():
            exclusions.append({"prompt_id": prompt, "arm": arm, "reason": f"missing frames {path}"})
            return None
        if not record.get("arm_receipt_written"):
            exclusions.append({
                "prompt_id": prompt,
                "arm": arm,
                "reason": "no arm receipt: the backend override may have been ignored (trap 1)",
            })
            return None
        return load_frames(path)

    def score(prompt: str, candidate_arm: str, reference_arm: str, kind: str, question: str | None = None) -> None:
        candidate = frames_for(prompt, candidate_arm)
        reference = frames_for(prompt, reference_arm)
        if candidate is None or reference is None:
            return
        scores = evaluator.evaluate(video=candidate, reference=reference)
        record = summaries[(prompt, candidate_arm)]
        rows.append({
            "record_type": kind,
            "run_id": args.run_id,
            "prompt_id": prompt,
            "seed": args.seed,
            "arm": candidate_arm,
            "reference_arm": reference_arm,
            "question": question,
            "requested_sparsity": record.get("requested_sparsity"),
            "realized_sparsity": record.get("realized_sparsity"),
            "attention_compute": record.get("attention_compute"),
            "router_precision": record.get("router_precision"),
            "native_or_simulated": record.get("native_or_simulated"),
            "numerical_only": candidate_arm in SIMULATED_ARMS,
            "native_latency_claim_allowed": record.get("native_latency_claim_allowed"),
            "psnr_db": scores["common.psnr"].score,
            "ssim": scores["common.ssim"].score,
            "lpips": scores["common.lpips"].score,
            **pixel_stats(candidate, reference),
        })

    for prompt in prompts:
        for arm in ARM_ORDER:
            score(prompt, arm, REFERENCE_ARM, "vs_reference", f"{arm} vs {REFERENCE_ARM}")
        for candidate_arm, reference_arm, question in DIRECT_PAIRS:
            score(prompt, candidate_arm, reference_arm, "direct_pair", question)

    raw_path = out_dir / f"phase5_{args.tag}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    # --- perturbation calibration ladder ---------------------------------
    # SPARSE-BF16-EPS runs are SPARSE-BF16 plus a measured per-call attention
    # perturbation. Scored against SPARSE-BF16 (their own unperturbed twin, not
    # DENSE-BF16) so the x-axis is exactly the injected perturbation and nothing
    # else. This is the curve that decides whether the pixel metrics above can
    # rank perturbation magnitudes at all.
    calibration_rows: list[dict[str, Any]] = []
    eps_records = [
        record for record in summaries_all.values()
        if record["arm"] == "SPARSE-BF16-EPS" and record.get("arm_receipt_written")
    ]
    for record in sorted(eps_records, key=lambda r: (r["prompt_id"], r.get("requested_perturb_rel_l2") or 0.0)):
        prompt = record["prompt_id"]
        baseline = frames_for(prompt, "SPARSE-BF16")
        path = Path(record["frame_path"]) if record.get("frame_path") else None
        if baseline is None or path is None or not path.is_file():
            exclusions.append({
                "prompt_id": prompt,
                "arm": "SPARSE-BF16-EPS",
                "requested_perturb_rel_l2": record.get("requested_perturb_rel_l2"),
                "reason": "missing SPARSE-BF16 twin or missing frames",
            })
            continue
        candidate = load_frames(path)
        scores = evaluator.evaluate(video=candidate, reference=baseline)
        calibration_rows.append({
            "record_type": "calibration",
            "run_id": args.run_id,
            "prompt_id": prompt,
            "seed": args.seed,
            "arm": "SPARSE-BF16-EPS",
            "reference_arm": "SPARSE-BF16",
            "requested_perturb_rel_l2": record.get("requested_perturb_rel_l2"),
            "realized_perturb_rel_l2": record.get("realized_perturb_rel_l2"),
            "realized_sparsity": record.get("realized_sparsity"),
            "native_or_simulated": "control",
            "numerical_only": True,
            "psnr_db": scores["common.psnr"].score,
            "ssim": scores["common.ssim"].score,
            "lpips": scores["common.lpips"].score,
            **pixel_stats(candidate, baseline),
        })
    if calibration_rows:
        calibration_path = out_dir / "phase5_calibration.jsonl"
        with calibration_path.open("w", encoding="utf-8") as handle:
            for row in calibration_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(subset)}
        for field in ("psnr_db", "ssim", "lpips", "mean_abs_pixel_diff", "max_abs_pixel_diff", "rms_pixel_diff",
                      "pixel_correlation", "frac_pixels_changed", "frac_pixels_changed_gt_1_255"):
            values = [row[field] for row in subset if row.get(field) is not None]
            if not values:
                continue
            out[f"{field}_median"] = statistics.median(values)
            out[f"{field}_mean"] = statistics.fmean(values)
            out[f"{field}_min"] = min(values)
            out[f"{field}_max"] = max(values)
            if len(values) > 1:
                out[f"{field}_stdev"] = statistics.stdev(values)
        out["all_identical"] = all(row.get("identical") for row in subset) if subset else None
        return out

    aggregates = {
        "vs_reference": {
            arm: aggregate([r for r in rows if r["record_type"] == "vs_reference" and r["arm"] == arm])
            for arm in ARM_ORDER
        },
        "direct_pair": {
            f"{cand}_vs_{ref}":
            {
                "question": question,
                **aggregate([
                    r for r in rows
                    if r["record_type"] == "direct_pair" and r["arm"] == cand and r["reference_arm"] == ref
                ]),
            }
            for cand, ref, question in DIRECT_PAIRS
        },
    }

    # The headline ratio: how large is the cost of sparsity next to the cost of
    # getting the routing precision "wrong"? Computed on the paired per-prompt
    # medians so it cannot be inflated by one outlier prompt.
    def median_of(kind: str, cand: str, ref: str, field: str) -> float | None:
        values = [
            row[field] for row in rows
            if row["record_type"] == kind and row["arm"] == cand and row["reference_arm"] == ref
            and row.get(field) is not None
        ]
        return statistics.median(values) if values else None

    ratios: dict[str, Any] = {}
    for field in ("mean_abs_pixel_diff", "rms_pixel_diff", "lpips"):
        sparsity_effect = median_of("direct_pair", "SPARSE-BF16", "DENSE-BF16", field)
        routing_effect = median_of("direct_pair", "SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16", field)
        quantization_effect = median_of("direct_pair", "DENSE-FP4", "DENSE-BF16", field)
        ratios[field] = {
            "sparsity_effect_median": sparsity_effect,
            "routing_precision_effect_median": routing_effect,
            "quantization_effect_median": quantization_effect,
            "sparsity_over_routing":
            (sparsity_effect / routing_effect) if (sparsity_effect and routing_effect) else None,
            "quantization_over_routing":
            (quantization_effect / routing_effect) if (quantization_effect and routing_effect) else None,
        }

    payload = {
        "run_id": args.run_id,
        "seed": args.seed,
        "prompts": prompts,
        "n_prompts": len(prompts),
        "reference_arm": REFERENCE_ARM,
        "metric_source": "fastvideo.eval common.psnr / common.ssim / common.lpips (LPIPS net=alex)",
        "input_source": "float16 decoded frames, not mp4 (compression noise would swamp the effect)",
        "development_set": True,
        "scope_note": ("10-prompt development set, 1 seed per run_id. No benchmark-wide claim. "
                       "SPARSE-FP4-* compute is simulated NVFP4 Q/K + BF16 PV; excluded from all latency tables."),
        "aggregates": aggregates,
        "effect_size_ratios": ratios,
        "calibration_ladder": [{
            "requested_perturb_rel_l2": row["requested_perturb_rel_l2"],
            "realized_perturb_rel_l2": row["realized_perturb_rel_l2"],
            "prompt_id": row["prompt_id"],
            "mean_abs_pixel_diff": row["mean_abs_pixel_diff"],
            "psnr_db": row["psnr_db"],
            "ssim": row["ssim"],
            "lpips": row["lpips"],
        } for row in calibration_rows],
        "calibration_note": ("SPARSE-BF16-EPS vs its own unperturbed SPARSE-BF16 twin. If the video difference "
                             "is already saturated at an injected attention perturbation far below the "
                             "routing-precision perturbation, the pixel metrics cannot rank magnitudes and the "
                             "six-arm table must be read as 'the trajectories diverged', not as an error size."),
        "exclusions": exclusions,
        "raw_path": str(raw_path),
        "rows": len(rows),
    }
    out_path = out_dir / f"phase5_{args.tag}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "aggregates"}, indent=2))
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
