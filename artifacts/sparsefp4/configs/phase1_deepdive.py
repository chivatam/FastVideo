"""Phase-1 deep dive: the questions the standard aggregate tables cannot answer.

`scripts/analyze_masks.py` produces the pre-registered aggregates. This script adds
the five things the Phase-1 verdict actually turns on:

1. **Tail shape**, not just median/IQR — the PIVOT test in `SKILL.md` is about
   whether overlap is "> 0.95 almost everywhere", which is a statement about the
   *fraction of cells* below a threshold, not about the median.
2. **Query-block-level disruption** (`frac_query_blocks_changed`), which separates
   "a few blocks each lost many" from "almost every block lost one".
3. **The margin mechanism** (EXPERIMENT_SPEC 5.3): routing changes should
   concentrate where the BF16 decision margin at the top-k boundary is small.
   Reported as a binned relationship, not asserted.
4. **Native vs simulated NVFP4**, paired per cell, as a harness cross-check on the
   simulated quantizer.
5. **Affected-cell counts** (EXPERIMENT_SPEC 5.5) at every granularity and at
   several thresholds, so the localization claim is a count and not an anecdote.

Usage::

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_deepdive.py \
        --raw /mnt/scratch/sparsefp4/<run_id> --out-tables artifacts/sparsefp4/tables/<tag>
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any
from collections.abc import Iterator

QUANTILES: tuple[float, ...] = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
AFFECTED_THRESHOLDS: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)
MIN_N = 20
N_MARGIN_BINS = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--out-tables", type=Path, required=True)
    return parser


def iter_records(roots: list[Path]) -> Iterator[dict[str, Any]]:
    paths: list[str] = []
    for root in roots:
        if root.is_file():
            paths.append(str(root))
            continue
        paths += sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True))
        paths += sorted(glob.glob(str(root / "**" / "*.jsonl.gz"), recursive=True))
    if not paths:
        raise SystemExit(f"no shards found under {roots}")
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[int(position)]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def cell_name(key: tuple[Any, ...], offset: int = 2) -> str:
    """Render a grouping key's trailing components as a compact cell label."""
    return "/".join(str(component) for component in key[offset:])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = build_parser().parse_args()
    out = args.out_tables
    out.mkdir(parents=True, exist_ok=True)

    jaccard: dict[tuple[float, str], list[float]] = collections.defaultdict(list)
    changed_blocks: dict[tuple[float, str], list[float]] = collections.defaultdict(list)
    ties: dict[tuple[float, str], list[float]] = collections.defaultdict(list)
    rho: dict[tuple[float, str], list[float]] = collections.defaultdict(list)
    # margin -> (1 - recall) pairs, for the mechanism check
    margin_pairs: dict[tuple[float, str], list[tuple[float, float]]] = collections.defaultdict(list)
    # per-cell jaccard keyed to allow a paired native-vs-simulated difference
    paired: dict[tuple[Any, ...], dict[str, float]] = collections.defaultdict(dict)
    # granularity -> cell key -> jaccard list
    granular: dict[str, dict[tuple[Any, ...], list[float]]] = {
        name: collections.defaultdict(list)
        for name in ("layer_timestep", "layer_head", "layer_head_timestep", "layer", "head", "timestep")
    }
    total = 0

    for record in iter_records(list(args.raw)):
        total += 1
        precision = str(record["routing_precision"])
        sparsity = float(record["sparsity"])
        value = float(record["jaccard"])
        key = (sparsity, precision)
        jaccard[key].append(value)
        if isinstance(record.get("frac_query_blocks_changed"), int | float):
            changed_blocks[key].append(float(record["frac_query_blocks_changed"]))
        if isinstance(record.get("boundary_ties"), int | float):
            ties[key].append(float(record["boundary_ties"]))
        if isinstance(record.get("spearman_rho"), int | float):
            rho[key].append(float(record["spearman_rho"]))
        if isinstance(record.get("decision_margin_reference"), int | float):
            margin_pairs[key].append((float(record["decision_margin_reference"]), 1.0 - float(record["recall"])))
        cell = (record["prompt_id"], record["layer"], record["head"], record["timestep"], record.get("cfg_branch"),
                sparsity)
        paired[cell][precision] = value
        if precision != "bf16":
            granular["layer_timestep"][(precision, sparsity, record["layer"], record["timestep"])].append(value)
            granular["layer_head"][(precision, sparsity, record["layer"], record["head"])].append(value)
            granular["layer_head_timestep"][(precision, sparsity, record["layer"], record["head"],
                                             record["timestep"])].append(value)
            granular["layer"][(precision, sparsity, record["layer"])].append(value)
            granular["head"][(precision, sparsity, record["head"])].append(value)
            granular["timestep"][(precision, sparsity, record["timestep"])].append(value)

    print(f"read {total} records")

    tail_rows: list[dict[str, Any]] = []
    for (sparsity, precision), values in sorted(jaccard.items()):
        ordered = sorted(values)
        row: dict[str, Any] = {
            "sparsity": sparsity,
            "retained_fraction": round(1.0 - sparsity, 4),
            "routing_precision": precision,
            "n": len(ordered),
        }
        for q in QUANTILES:
            row[f"jaccard_p{q * 100:g}"] = round(quantile(ordered, q), 6)
        row["jaccard_min"] = round(ordered[0], 6)
        row["jaccard_max"] = round(ordered[-1], 6)
        row["jaccard_mean"] = round(statistics.fmean(ordered), 6)
        for threshold in AFFECTED_THRESHOLDS:
            below = sum(1 for value in ordered if value < threshold)
            row[f"frac_records_below_{threshold:.2f}"] = round(below / len(ordered), 6)
        changed = sorted(changed_blocks.get((sparsity, precision), []))
        row["frac_query_blocks_changed_median"] = round(quantile(changed, 0.5), 6) if changed else ""
        row["frac_query_blocks_changed_p90"] = round(quantile(changed, 0.9), 6) if changed else ""
        tie_values = sorted(ties.get((sparsity, precision), []))
        row["boundary_ties_median"] = round(quantile(tie_values, 0.5), 3) if tie_values else ""
        rho_values = sorted(rho.get((sparsity, precision), []))
        row["spearman_rho_median"] = round(quantile(rho_values, 0.5), 6) if rho_values else ""
        row["n_spearman"] = len(rho_values)
        tail_rows.append(row)
    write_csv(out / "tail_by_sparsity_precision.csv", tail_rows)

    margin_rows: list[dict[str, Any]] = []
    for (sparsity, precision), pairs in sorted(margin_pairs.items()):
        if precision == "bf16" or not pairs:
            continue
        pairs.sort(key=lambda item: item[0])
        size = max(1, len(pairs) // N_MARGIN_BINS)
        for index in range(N_MARGIN_BINS):
            start = index * size
            stop = len(pairs) if index == N_MARGIN_BINS - 1 else min(len(pairs), start + size)
            chunk = pairs[start:stop]
            if not chunk:
                continue
            margins = [item[0] for item in chunk]
            changed_fracs = [item[1] for item in chunk]
            margin_rows.append({
                "sparsity": sparsity,
                "routing_precision": precision,
                "reference_margin_decile": index + 1,
                "margin_low": round(margins[0], 8),
                "margin_high": round(margins[-1], 8),
                "margin_median": round(statistics.median(margins), 8),
                "frac_decisions_changed_median": round(statistics.median(changed_fracs), 6),
                "frac_decisions_changed_mean": round(statistics.fmean(changed_fracs), 6),
                "n": len(chunk),
            })
    write_csv(out / "margin_decile_vs_changed.csv", margin_rows)

    native_rows: list[dict[str, Any]] = []
    by_sparsity: dict[float, list[float]] = collections.defaultdict(list)
    identical: collections.Counter[float] = collections.Counter()
    for cell, arms in paired.items():
        if "nvfp4" in arms and "nvfp4_sim" in arms:
            sparsity = float(cell[-1])
            difference = arms["nvfp4"] - arms["nvfp4_sim"]
            by_sparsity[sparsity].append(difference)
            if difference == 0.0:
                identical[sparsity] += 1
    for sparsity, differences in sorted(by_sparsity.items()):
        ordered = sorted(differences)
        native_rows.append({
            "sparsity": sparsity,
            "n_paired_cells": len(ordered),
            "frac_cells_identical": round(identical[sparsity] / len(ordered), 6),
            "mean_jaccard_native_minus_simulated": round(statistics.fmean(ordered), 8),
            "median_diff": round(quantile(ordered, 0.5), 8),
            "p1_diff": round(quantile(ordered, 0.01), 8),
            "p99_diff": round(quantile(ordered, 0.99), 8),
            "max_abs_diff": round(max(abs(ordered[0]), abs(ordered[-1])), 8),
        })
    write_csv(out / "native_vs_simulated_nvfp4.csv", native_rows)

    affected_rows: list[dict[str, Any]] = []
    for name, cells in granular.items():
        grouped: dict[tuple[str, float], list[tuple[tuple[Any, ...], float, int]]] = collections.defaultdict(list)
        for key, values in cells.items():
            grouped[(str(key[0]), float(key[1]))].append((key[2:], quantile(sorted(values), 0.5), len(values)))
        for (arm, arm_sparsity), entries in sorted(grouped.items()):
            eligible = [entry for entry in entries if entry[2] >= MIN_N]
            affected_row: dict[str, Any] = {
                "granularity": name,
                "routing_precision": arm,
                "sparsity": arm_sparsity,
                "cells_total": len(entries),
                "cells_eligible_n_ge_20": len(eligible),
                "cells_insufficient_n": len(entries) - len(eligible),
            }
            for threshold in AFFECTED_THRESHOLDS:
                affected = [entry for entry in eligible if entry[1] < threshold]
                affected_row[f"cells_below_{threshold:.2f}"] = len(affected)
                affected_row[f"frac_cells_below_{threshold:.2f}"] = (round(len(affected) /
                                                                           len(eligible), 6) if eligible else "")
            if eligible:
                by_median = sorted(eligible, key=lambda entry: entry[1])
                affected_row["worst_cell"] = cell_name(by_median[0][0], offset=0)
                affected_row["worst_cell_median_jaccard"] = round(by_median[0][1], 6)
                affected_row["worst_cell_n"] = by_median[0][2]
                affected_row["best_cell_median_jaccard"] = round(by_median[-1][1], 6)
                affected_row["cell_median_spread"] = round(by_median[-1][1] - by_median[0][1], 6)
            affected_rows.append(affected_row)
    write_csv(out / "affected_cell_counts.csv", affected_rows)

    ranked_rows: list[dict[str, Any]] = []
    for name in ("layer", "head", "layer_head", "timestep"):
        for key, values in granular[name].items():
            ordered = sorted(values)
            cell_label = cell_name(key)
            ranked_rows.append({
                "granularity": name,
                "routing_precision": key[0],
                "sparsity": key[1],
                "cell": cell_label,
                "n": len(ordered),
                "jaccard_median": round(quantile(ordered, 0.5), 6),
                "jaccard_q1": round(quantile(ordered, 0.25), 6),
                "jaccard_q3": round(quantile(ordered, 0.75), 6),
                "jaccard_p10": round(quantile(ordered, 0.10), 6),
                "jaccard_min": round(ordered[0], 6),
            })
    ranked_rows.sort(
        key=lambda row: (row["granularity"], row["routing_precision"], row["sparsity"], row["jaccard_median"]))
    write_csv(out / "ranked_cells.csv", ranked_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
