"""Phase 5 VBench: temporal + quality dimensions per (prompt, arm).

Uses FastVideo's integrated ``fastvideo.eval`` VBench adapter -- the same 16
dimension modules and the same pinned upstream submodule the repo's own
regression tests score against -- rather than a separate VBench install.

Dimensions attempted, all of them temporal or perceptual-quality (the ones that
can actually respond to an attention change):

    vbench.subject_consistency      DINO ViT-B/16 temporal feature similarity
    vbench.background_consistency   CLIP temporal feature similarity
    vbench.temporal_flickering      local frame-to-frame absolute difference
    vbench.motion_smoothness        AMT frame interpolation error
    vbench.dynamic_degree           RAFT optical-flow magnitude
    vbench.imaging_quality          MUSIQ no-reference quality
    vbench.aesthetic_quality        LAION aesthetic predictor

The four GRiT dimensions (``color``, ``object_class``, ``multiple_objects``,
``spatial_relationship``) are **not** attempted: they need ``detectron2``, which
``pyproject.toml`` does not auto-install. Any dimension whose weights cannot be
fetched or whose dependency is missing is recorded with the reason rather than
silently dropped, per SKILL integrity rule 9.

Each dimension is loaded in its own subprocess-free pass but scored one metric at
a time, so one unavailable dimension cannot take the rest of the run down.

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_vbench.py \
        --run-id 20260814-032700-8208536-p5-main --metrics vbench.subject_consistency
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ARM_ORDER = (
    "DENSE-BF16",
    "DENSE-FP4",
    "SPARSE-BF16",
    "SPARSE-FP4-NAIVE",
    "SPARSE-FP4-ROUTE8",
    "SPARSE-FP4-ROUTE16",
)
DEFAULT_METRICS = (
    "vbench.subject_consistency",
    "vbench.background_consistency",
    "vbench.temporal_flickering",
    "vbench.motion_smoothness",
    "vbench.dynamic_degree",
    "vbench.imaging_quality",
    "vbench.aesthetic_quality",
)
# Recorded, not attempted: the four GRiT dimensions plus the ones that need a
# reference video or a prompt-conditioned ground truth this study does not have.
SKIPPED_BY_DESIGN = {
    "vbench.color": "needs detectron2 (GRiT); not auto-installed by pyproject [eval-vbench]",
    "vbench.object_class": "needs detectron2 (GRiT); not auto-installed by pyproject [eval-vbench]",
    "vbench.multiple_objects": "needs detectron2 (GRiT); not auto-installed by pyproject [eval-vbench]",
    "vbench.spatial_relationship": "needs detectron2 (GRiT); not auto-installed by pyproject [eval-vbench]",
    "vbench.human_action": ("prompt-category dimension: scores only VBench's human-action prompt list, "
                            "which this 10-prompt development set is not drawn from"),
    "vbench.appearance_style": "prompt-category dimension (style prompts); not applicable to this prompt set",
    "vbench.temporal_style": "prompt-category dimension (style prompts); not applicable to this prompt set",
    "vbench.scene": "prompt-category dimension (scene prompts) and needs AVoCaDO weights",
    "vbench.overall_consistency": ("text-video alignment via ViCLIP; measures prompt adherence rather than "
                                   "the attention change under test, and needs extra weights"),
}


def load_frames(path: Path) -> torch.Tensor:
    array = np.load(path)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.dim() == 5:
        tensor = tensor[0]
    return tensor.permute(1, 0, 2, 3).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--video-root", type=Path, default=Path("/mnt/scratch/sparsefp4-videos"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/sparsefp4/raw"))
    parser.add_argument("--tag", default="vbench")
    args = parser.parse_args()

    from fastvideo.eval import create_evaluator

    raw_dir = args.raw_root / args.run_id
    video_dir = args.video_root / args.run_id
    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("run_summary_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("seed", -1)) != args.seed or record["arm"] not in ARM_ORDER:
            continue
        if not record.get("arm_receipt_written"):
            continue
        summaries[(record["prompt_id"], record["arm"])] = record
    prompts = sorted({prompt for prompt, _ in summaries})
    if not prompts:
        raise SystemExit(f"no eligible run summaries under {raw_dir}")

    rows: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for metric_name in args.metrics:
        try:
            evaluator = create_evaluator(metrics=[metric_name], device=args.device)
        except Exception as error:  # noqa: BLE001
            unavailable[metric_name] = f"evaluator construction failed: {type(error).__name__}: {error}"
            print(f"UNAVAILABLE {metric_name}: {error}")
            continue
        metric_failed = False
        for prompt in prompts:
            for arm in ARM_ORDER:
                record = summaries.get((prompt, arm))
                if record is None or not record.get("frame_path"):
                    continue
                path = Path(record["frame_path"])
                if not path.is_file():
                    continue
                try:
                    scores = evaluator.evaluate(video=load_frames(path), fps=float(record.get("fps") or 16))
                except Exception as error:  # noqa: BLE001
                    unavailable[metric_name] = (f"{type(error).__name__}: {error}")
                    print(f"UNAVAILABLE {metric_name} on {prompt}/{arm}:\n{traceback.format_exc()[-800:]}")
                    metric_failed = True
                    break
                result = scores[metric_name]
                rows.append({
                    "run_id": args.run_id,
                    "record_type": "vbench",
                    "metric": metric_name,
                    "prompt_id": prompt,
                    "arm": arm,
                    "seed": args.seed,
                    "score": result.score,
                    "requested_sparsity": record.get("requested_sparsity"),
                    "realized_sparsity": record.get("realized_sparsity"),
                    "attention_compute": record.get("attention_compute"),
                    "router_precision": record.get("router_precision"),
                    "native_or_simulated": record.get("native_or_simulated"),
                })
            if metric_failed:
                break
        del evaluator
        torch.cuda.empty_cache()

    raw_path = out_dir / f"phase5_{args.tag}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    aggregates: dict[str, dict[str, Any]] = {}
    for metric_name in sorted({row["metric"] for row in rows}):
        aggregates[metric_name] = {}
        for arm in ARM_ORDER:
            values = [row["score"] for row in rows if row["metric"] == metric_name and row["arm"] == arm]
            if not values:
                continue
            aggregates[metric_name][arm] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
            }

    payload = {
        "run_id": args.run_id,
        "seed": args.seed,
        "prompts": prompts,
        "n_prompts": len(prompts),
        "metric_source": "fastvideo.eval VBench adapter (pinned submodule fastvideo/third_party/eval/vbench)",
        "requested_metrics": args.metrics,
        "scored_metrics": sorted({row["metric"] for row in rows}),
        "unavailable_metrics": unavailable,
        "skipped_by_design": SKIPPED_BY_DESIGN,
        "development_set": True,
        "scope_note": ("10-prompt development set, 1 seed. VBench dimensions are absolute quality scores, "
                       "not paired similarity; with n=10 the between-arm differences should be read against "
                       "the between-prompt spread, which is reported as stdev."),
        "aggregates": aggregates,
        "raw_path": str(raw_path),
        "rows": len(rows),
    }
    out_path = out_dir / f"phase5_{args.tag}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
