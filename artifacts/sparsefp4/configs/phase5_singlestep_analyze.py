"""Summarize the Phase 5 single-step trajectory control into the paired numbers
the video-level comparison has to be read against.

Two quantities, both computed per exactly-paired cell
``(prompt, layer, head, timestep, cfg_branch)`` and then aggregated:

1. **Per-arm attention error vs dense BF16** -- the ladder ``B``, ``C``, ``D``,
   ``D8``, ``E``, ``F8``, ``F16``, ``C_rand``. This is Phase 2's decomposition,
   resampled at all 30 layers along a real trajectory.
2. **The paired H3 reduction**: at each cell, how much smaller is the error with
   a higher-precision router, as a fraction of the NVFP4-router error. Because
   the cells are exactly paired, the median of the per-cell ratio is the right
   statistic, not the ratio of the medians.

The output is the reference the free-running video numbers are compared against
in ``PHASE5.md``: if the single-step routing effect is ~1e-4 relative while the
video pixel difference between the same two arms is a few percent, the video
metric is saturated and cannot rank magnitudes.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_singlestep_analyze.py \
        --run-id 20260814-034700-8208536-p5-singlestep
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any

CONFIGS = ("A", "B", "B_sim", "C", "D", "D8", "C_rand", "E", "F8", "F16")
# Phase 5 arm <-> single-step configuration correspondence. The video arms and
# the single-step configurations are the same six designs; naming them together
# is what makes the two tables comparable.
ARM_TO_CONFIG = {
    "DENSE-BF16": "A",
    "DENSE-FP4": "B",
    "SPARSE-BF16": "C",
    "SPARSE-FP4-NAIVE": "E",
    "SPARSE-FP4-ROUTE8": "F8",
    "SPARSE-FP4-ROUTE16": "F16",
}
# H3: does a higher-precision router reduce the error of the low-precision
# sparse-compute arm? Pre-registered support threshold is >= 20%.
H3_PAIRS = (("E", "F8", "NVFP4 router -> FP8 router"), ("E", "F16", "NVFP4 router -> BF16 router"))
# The same question with BF16 compute, which isolates the mask entirely.
MASK_PAIRS = (("D", "C", "NVFP4 wrong-mask term (D - C)"), ("D8", "C", "FP8 wrong-mask term (D8 - C)"),
              ("C_rand", "C", "equal-magnitude random wrong-mask term (C_rand - C)"))


def cell_key(row: dict[str, Any]) -> tuple:
    return (row["prompt_id"], row["layer"], row["head"], row["timestep"], row["cfg_branch"], row.get("sparsity"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/sparsefp4/raw"))
    parser.add_argument("--target-run-id", default=None, help="run_id the medians file is written under")
    args = parser.parse_args()

    raw_dir = args.raw_root / args.run_id
    out_dir = args.out_root / (args.target_run_id or args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_config: dict[str, list[float]] = {name: [] for name in CONFIGS}
    by_cell: dict[tuple, dict[str, float]] = {}
    prompts: set[str] = set()
    layers: set[int] = set()
    timesteps: set[int] = set()
    sparsities: set[float] = set()

    for shard in sorted(raw_dir.glob("p*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("record_type") != "error_decomposition" or row.get("rel_l2") is None:
                    continue
                config = row["config"]
                if config not in by_config:
                    continue
                by_config[config].append(float(row["rel_l2"]))
                by_cell.setdefault(cell_key(row), {})[config] = float(row["rel_l2"])
                prompts.add(row["prompt_id"])
                layers.add(row["layer"])
                timesteps.add(row["timestep"])
                if row.get("sparsity") is not None:
                    sparsities.add(float(row["sparsity"]))

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        out = {"n": len(values), "median": statistics.median(values), "mean": statistics.fmean(values)}
        if len(values) >= 10:
            quantiles = statistics.quantiles(values, n=10)
            out["p10"] = quantiles[0]
            out["p90"] = quantiles[-1]
        return out

    per_config = {name: stats(values) for name, values in by_config.items() if values}

    paired: dict[str, Any] = {}
    for high, low, label in H3_PAIRS:
        # ``high`` is the NVFP4-router arm, ``low`` the higher-precision-router
        # arm. Reduction is positive when the higher-precision router helps.
        reductions, improved = [], 0
        for cell in by_cell.values():
            if high not in cell or low not in cell or cell[high] <= 0:
                continue
            reduction = (cell[high] - cell[low]) / cell[high]
            reductions.append(reduction)
            improved += reduction > 0
        if not reductions:
            continue
        paired[f"{high}_to_{low}"] = {
            "question": label,
            "n_paired_cells": len(reductions),
            "median_relative_reduction": statistics.median(reductions),
            "mean_relative_reduction": statistics.fmean(reductions),
            "median_relative_reduction_pct": statistics.median(reductions) * 100.0,
            "fraction_of_cells_improved": improved / len(reductions),
            "pre_registered_threshold_pct": 20.0,
            "meets_threshold": statistics.median(reductions) * 100.0 >= 20.0,
        }

    mask_terms: dict[str, Any] = {}
    for high, low, label in MASK_PAIRS:
        excess = [cell[high] - cell[low] for cell in by_cell.values() if high in cell and low in cell]
        if not excess:
            continue
        mask_terms[f"{high}_minus_{low}"] = {"question": label, **stats(excess)}

    total = per_config.get("C", {}).get("median")
    quantization = per_config.get("B", {}).get("median")
    wrong_mask = mask_terms.get("D_minus_C", {}).get("median")
    payload: dict[str, Any] = {
        "run_id": args.run_id,
        "stage": "5-singlestep",
        "purpose": ("single-step exactly-paired attention error along the real dense-BF16 trajectory, at the "
                    "same sparsity as the Phase 5 free-running video runs; the reference the video-level "
                    "pixel numbers must be read against"),
        "prompts": sorted(prompts),
        "n_prompts": len(prompts),
        "n_layers": len(layers),
        "layers": sorted(layers),
        "timesteps": sorted(timesteps),
        "sparsities": sorted(sparsities),
        "n_paired_cells": len(by_cell),
        "per_config_rel_l2_vs_dense_bf16": per_config,
        "h3_paired_reduction": paired,
        "wrong_mask_terms": mask_terms,
        "error_budget_at_sparsity": {
            "sparsification_C": total,
            "quantization_B": quantization,
            "wrong_mask_D_minus_C": wrong_mask,
            "wrong_mask_as_fraction_of_sparsification": (wrong_mask / total) if (wrong_mask and total) else None,
        },
        # Keyed by Phase 5 arm name so fig4 can put the two curves on one axis.
        "by_arm": {
            arm: per_config[config]["median"]
            for arm, config in ARM_TO_CONFIG.items() if config in per_config
        },
        "arm_to_config": ARM_TO_CONFIG,
    }
    out_path = out_dir / "phase5_singlestep_medians.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Keep the raw shards with the report rather than only on ephemeral scratch:
    # they are the evidence for the decisive claim and compress well.
    for shard in sorted(raw_dir.glob("p*.jsonl")):
        target = out_dir / f"singlestep_{shard.stem}.jsonl.gz"
        if not target.is_file():
            with shard.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as sink:
                sink.writelines(source)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
