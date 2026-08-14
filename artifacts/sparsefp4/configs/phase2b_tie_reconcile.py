"""Settle the Phase-1-vs-Phase-2 boundary-tie discrepancy from archived data.

Phase 1 (`PHASE1.md` 7.1) reported **~104-115 exact boundary ties per cell** at
raster 128x64 and concluded that fp32 block scores "penalize the FP8 arm 1.6x
harder than NVFP4". Phase 2 (`PHASE2.md` 7), on real Wan Q/K at the same
geometry, measured **~1,400 ties per cell** and near-identical tie counts across
routers (1429/1430/1430), and reported that it could not reproduce the asymmetry.
Both cannot be right as stated, so this script recomputes both quantities from the
archived records rather than re-reading the two reports.

Two separable questions, answered separately:

1. **The ~13x tie-count scale gap.** Phase 1's ``boundary_ties`` is emitted
   *per head* (``routing_probe_attn.compare_masks``: ``(margin == 0).sum(dim=-1)``
   over ``n_q_blocks``), while Phase 2's is summed over the whole
   ``[head, query_block]`` grid (``precision_sparse_attn._emit_score_resolution_row``:
   ``(...).sum()``). With 12 heads that is a factor of 12 on identical data.
2. **The 1.6x FP8 asymmetry.** Phase 1's claim is about the *median Jaccard*
   moving when the scorer goes fp32 -> fp64, not about tie counts. It is measured
   here as an exactly paired per-cell difference between Phase 1's fp32 Stage-1 run
   and its fp64 Stage-1 control, which is the data the claim was made from.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_tie_reconcile.py \\
        --fp32-raw artifacts/sparsefp4/raw/20260814-013449-8208536-p1-stage1 \\
        --fp64-raw artifacts/sparsefp4/raw/20260814-015113-8208536-p1-stage1-fp64score \\
        --phase2-raw artifacts/sparsefp4/raw/20260814-025500-8208536-p2-main \\
        --out-tables artifacts/sparsefp4/tables/phase2b_geometry
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("bf16", "fp8_e4m3", "nvfp4")


def read_records(raw_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    files = sorted(list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no *.jsonl or *.jsonl.gz under {raw_dir}")
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            "" if row.get(key) is None else (f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key]))
            for key in keys) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase1_tie_scale(fp32: list[dict[str, Any]], fp64: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 1's per-head tie count, and what it becomes on Phase 2's denominator."""
    rows: list[dict[str, Any]] = []
    for label, records in (("fp32", fp32), ("fp64", fp64)):
        buckets: dict[tuple, list[int]] = defaultdict(list)
        heads: dict[tuple, set[int]] = defaultdict(set)
        qblocks: dict[tuple, set[int]] = defaultdict(set)
        for record in records:
            if record.get("boundary_ties") is None:
                continue
            key = (record["routing_precision"], record["sparsity"])
            buckets[key].append(int(record["boundary_ties"]))
            heads[key].add(record["head"])
            qblocks[key].add(record["n_q_blocks"])
        for (precision, sparsity), values in sorted(buckets.items()):
            n_heads = len(heads[(precision, sparsity)])
            n_q = sorted(qblocks[(precision, sparsity)])
            per_head = median([float(value) for value in values])
            rows.append({
                "source": f"phase1_stage1_{label}",
                "geometry": "128x64-raster",
                "score_dtype": label,
                "router_precision": precision,
                "sparsity": sparsity,
                "tie_counting_unit": "per (cell, head)",
                "ties_median_as_reported": per_head,
                "n_q_blocks": n_q[0] if len(n_q) == 1 else n_q,
                "n_heads_pooled_by_phase2": n_heads,
                "ties_rescaled_to_phase2_unit": None if per_head is None else per_head * n_heads,
                "tie_rate_of_query_blocks": None if per_head is None or not n_q else per_head / n_q[0],
                "n_records": len(values),
            })
    return rows


def phase2_tie_scale(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 2's per-cell tie count, and what it becomes on Phase 1's denominator."""
    resolution = [r for r in records if r["record_type"] == "score_resolution"]
    rows: list[dict[str, Any]] = []
    if not resolution:
        return rows
    n_heads = max(r["num_heads"] for r in resolution)
    n_q_blocks = max(r["n_q_blocks"] for r in resolution)
    sparsities = sorted({int(round(value * 100)) for r in records for value in (r.get("sparsity"), ) if value})
    for precision in ARMS:
        for sparsity in sparsities:
            for label in ("fp32", "fp64"):
                field = f"boundary_ties_{label}_{precision}_s{sparsity}"
                values = [float(r[field]) for r in resolution if field in r]
                if not values:
                    continue
                per_cell = median(values)
                rows.append({
                    "source":
                    "phase2_main",
                    "geometry":
                    "128x64-raster",
                    "score_dtype":
                    label,
                    "router_precision":
                    precision,
                    "sparsity":
                    sparsity / 100,
                    "tie_counting_unit":
                    "per cell, summed over all heads",
                    "ties_median_as_reported":
                    per_cell,
                    "n_q_blocks":
                    n_q_blocks,
                    "n_heads_pooled_by_phase2":
                    n_heads,
                    "ties_rescaled_to_phase1_unit":
                    None if per_cell is None else per_cell / n_heads,
                    "tie_rate_of_head_query_block_pairs":
                    (None if per_cell is None else per_cell / (n_heads * n_q_blocks)),
                    "n_records":
                    len(values),
                })
    return rows


def fp32_vs_fp64_jaccard(fp32: list[dict[str, Any]], fp64: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exactly paired per-cell fp32 -> fp64 Jaccard shift, per router arm.

    This is the quantity Phase 1's "1.6x harder on FP8" claim is about. Pairing is
    on ``(prompt, layer, head, timestep, cfg_branch, sparsity, routing_precision)``,
    which is well defined because the fp64 control re-ran Stage 1 with everything
    except the score matmul dtype held byte-identical.
    """

    def index(records: list[dict[str, Any]]) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for record in records:
            if record.get("jaccard") is None:
                continue
            key = (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"],
                   record["sparsity"], record["routing_precision"])
            out[key] = float(record["jaccard"])
        return out

    left, right = index(fp32), index(fp64)
    shared = sorted(set(left) & set(right))
    buckets: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for key in shared:
        buckets[(key[5], key[6])].append((left[key], right[key]))

    rows: list[dict[str, Any]] = []
    per_arm: dict[tuple, float] = {}
    for (sparsity, precision), pairs in sorted(buckets.items()):
        deltas = [right_value - left_value for left_value, right_value in pairs]
        median_delta = median(deltas)
        median_fp32 = median([pair[0] for pair in pairs])
        median_fp64 = median([pair[1] for pair in pairs])
        shift_of_medians = None if median_fp32 is None or median_fp64 is None else median_fp64 - median_fp32
        if shift_of_medians is not None:
            per_arm[(sparsity, precision)] = shift_of_medians
        rows.append({
            "geometry": "128x64-raster",
            "sparsity": sparsity,
            "router_precision": precision,
            "median_jaccard_fp32_scorer": median_fp32,
            "median_jaccard_fp64_scorer": median_fp64,
            "shift_of_medians_fp64_minus_fp32": shift_of_medians,
            "paired_delta_median": median_delta,
            "paired_delta_p10": quantile(deltas, 0.10),
            "paired_delta_p90": quantile(deltas, 0.90),
            "frac_cells_fp64_higher": sum(1 for value in deltas if value > 0) / len(deltas),
            "n_paired_cells": len(deltas),
        })
    for row in rows:
        fp8 = per_arm.get((row["sparsity"], "fp8_e4m3"))
        nvfp4 = per_arm.get((row["sparsity"], "nvfp4"))
        row["fp8_over_nvfp4_shift_ratio"] = (None if not fp8 or not nvfp4 else fp8 / nvfp4)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-raw", type=Path, required=True)
    parser.add_argument("--fp64-raw", type=Path, required=True)
    parser.add_argument("--phase2-raw", type=Path, required=True)
    parser.add_argument("--out-tables", type=Path, required=True)
    args = parser.parse_args()

    fp32 = read_records(args.fp32_raw)
    fp64 = read_records(args.fp64_raw)
    phase2 = read_records(args.phase2_raw)

    scale = phase1_tie_scale(fp32, fp64) + phase2_tie_scale(phase2)
    jaccard = fp32_vs_fp64_jaccard(fp32, fp64)
    write_table(args.out_tables / "table6_tie_count_denominator.csv", scale)
    write_table(args.out_tables / "table7_fp32_vs_fp64_jaccard_shift.csv", jaccard)

    phase1_08 = next(
        (row for row in scale
         if row["source"] == "phase1_stage1_fp32" and row["router_precision"] == "nvfp4" and row["sparsity"] == 0.80),
        None)
    phase2_08 = next((row for row in scale if row["source"] == "phase2_main" and row["score_dtype"] == "fp32"
                      and row["router_precision"] == "nvfp4" and row["sparsity"] == 0.80), None)
    summary = {
        "phase1_fp32_ties_per_head_at_s080": phase1_08 and phase1_08["ties_median_as_reported"],
        "phase1_fp32_ties_rescaled_to_phase2_unit": phase1_08 and phase1_08["ties_rescaled_to_phase2_unit"],
        "phase2_fp32_ties_per_cell_at_s080": phase2_08 and phase2_08["ties_median_as_reported"],
        "phase2_fp32_ties_rescaled_to_phase1_unit": phase2_08 and phase2_08["ties_rescaled_to_phase1_unit"],
        "phase1_tie_rate_of_query_blocks": phase1_08 and phase1_08["tie_rate_of_query_blocks"],
        "phase2_tie_rate_of_head_query_block_pairs": phase2_08 and phase2_08["tie_rate_of_head_query_block_pairs"],
        "fp8_over_nvfp4_jaccard_shift_ratio": {
            f"s{row['sparsity']}": row["fp8_over_nvfp4_shift_ratio"]
            for row in jaccard if row["router_precision"] == "fp8_e4m3"
        },
        "n_paired_cells_per_arm_sparsity": {
            f"{row['router_precision']}_s{row['sparsity']}": row["n_paired_cells"]
            for row in jaccard
        },
    }
    (args.out_tables / "tie_reconciliation_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
