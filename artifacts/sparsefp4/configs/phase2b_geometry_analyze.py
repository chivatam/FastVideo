"""Phase 2B: compare the mechanism across three block geometries.

Phases 1 and 2 measured one geometry only — raster-order 128x64 blocks. VSA, the
deployed sparse backend, uses 64-token (4,4,4) spatio-temporal cubes in
tile-contiguous token order. Those differ in **two** ways at once (block size and
token-to-block assignment), so this reads three runs and reports the same
mechanism quantities for each, with the ``64x64-raster`` arm separating the two
factors:

``128x64-raster``  Phase 2's run, re-aggregated here for row-by-row comparison.
``64x64-raster``   block size changed, token order held.
``64x64-cube``     token order changed too; VSA's deployed geometry.

Every cell carries median, IQR and ``n``. Paired statistics are paired per
``(prompt, layer, head, timestep, cfg_branch, sparsity)`` within a geometry.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_geometry_analyze.py \\
        --raw 128x64-raster=artifacts/sparsefp4/raw/<phase2_main> \\
        --raw 64x64-raster=artifacts/sparsefp4/raw/<p2b_raster> \\
        --raw 64x64-cube=artifacts/sparsefp4/raw/<p2b_cube> \\
        --out-tables artifacts/sparsefp4/tables/phase2b_geometry \\
        --out-figures artifacts/sparsefp4/figures/phase2b_geometry
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GEOMETRY_ORDER = ("128x64-raster", "64x64-raster", "64x64-cube")
AFFECTED_LAYERS = (0, 1, 2, 27, 28, 29)
UNAFFECTED_LAYERS = (5, 6, 10, 11, 13)
NOTE = ("No latency claim anywhere: sparse NVFP4 compute has no native kernel here, and even the BF16 "
        "sparse arms are diagnostic. 'Low precision' means NVFP4 Q/K with BF16 PV.")


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


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def iqr(values: Sequence[float]) -> float | None:
    p25, p75 = quantile(values, 0.25), quantile(values, 0.75)
    return None if p25 is None or p75 is None else p75 - p25


def region_of(layer: int) -> str:
    if layer in AFFECTED_LAYERS:
        return "affected"
    if layer in UNAFFECTED_LAYERS:
        return "unaffected"
    return "broad"


def geometry_of(record: dict[str, Any]) -> str:
    """Geometry label, back-filling Phase 2's records which predate the field."""
    label = record.get("geometry")
    if label:
        return str(label)
    return f"{record['block_q']}x{record['block_k']}-raster"


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_table(path: Path, rows: list[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(columns) if columns else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(key)) for key in keys) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cell_key(record: dict[str, Any]) -> tuple:
    return (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"],
            record.get("sparsity"))


def verify(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path) -> dict[str, Any]:
    """Pre-analysis gate. No number below is quoted until this reports PASS.

    Checks what would silently invalidate a cross-geometry comparison: an fp32
    scorer sneaking in, a geometry label that disagrees with the recorded block
    sizes, unequal ``k`` across arms at a cell, a non-zero self-error for the
    reference, the null control not being an exact identity, and an incomplete arm
    set at any measured cell.
    """
    failures: list[str] = []
    per_geometry: dict[str, Any] = {}
    for label, records in by_geometry.items():
        errors = [r for r in records if r["record_type"] == "error_decomposition"]
        dtypes = {r["score_dtype"] for r in records}
        if dtypes != {"float64"}:
            failures.append(f"{label}: score_dtype must be float64 (trap 8); saw {sorted(dtypes)}")
        labels = {geometry_of(r) for r in records}
        if labels != {label}:
            failures.append(f"{label}: records carry geometry labels {sorted(labels)}")
        backends = {r["attention_backend"] for r in records}
        if backends != {"PRECISION_SPARSE_ATTN"}:
            failures.append(f"{label}: unexpected attention_backend {sorted(backends)}")
        if any(r["rel_l2"] not in (0.0, None) for r in errors if r["config"] == "A"):
            failures.append(f"{label}: configuration A has non-zero error against itself")

        budgets: dict[tuple, set[int]] = defaultdict(set)
        arms_at_cell: dict[tuple, frozenset] = {}
        raw_arms: dict[tuple, set[str]] = defaultdict(set)
        for record in errors:
            if record.get("sparsity") is None:
                continue
            raw_arms[cell_key(record)].add(record["config"])
            budgets[cell_key(record)].add(int(record["k_per_query_block"]))
        for key, values in raw_arms.items():
            arms_at_cell[key] = frozenset(values)
        k_disagreements = sum(1 for values in budgets.values() if len(values) > 1)
        if k_disagreements:
            failures.append(f"{label}: {k_disagreements} cells disagree on k across arms")
        arm_sets: dict[frozenset, int] = defaultdict(int)
        for value in arms_at_cell.values():
            arm_sets[value] += 1
        expected = max(arm_sets, key=lambda value: arm_sets[value]) if arm_sets else frozenset()
        incomplete = sum(count for value, count in arm_sets.items() if value != expected)
        if incomplete:
            failures.append(f"{label}: {incomplete} sparse cells have an arm set other than {sorted(expected)}")

        # Null control: C_null must reproduce C bit-for-bit at every cell.
        paired: dict[tuple, dict[str, float]] = defaultdict(dict)
        for record in errors:
            if record["config"] in ("C", "C_null") and record["rel_l2"] is not None:
                paired[cell_key(record)][record["config"]] = record["rel_l2"]
        null_pairs = [arms for arms in paired.values() if {"C", "C_null"} <= arms.keys()]
        null_deviations = sum(1 for arms in null_pairs if arms["C"] != arms["C_null"])
        null_jaccard = [
            r["mask_jaccard_vs_bf16"] for r in errors
            if r["config"] == "C_null" and r["mask_jaccard_vs_bf16"] is not None
        ]
        null_jaccard_deviations = sum(1 for value in null_jaccard if value != 1.0)
        if null_deviations or null_jaccard_deviations:
            failures.append(f"{label}: null control deviates ({null_deviations} rel_l2, "
                            f"{null_jaccard_deviations} jaccard)")
        if "C_null" in expected and not null_pairs:
            failures.append(f"{label}: C_null arm present but no paired null-control cells")

        simulated_latency = [
            r for r in errors if r["native_or_simulated"] == "simulated" and r["native_latency_claim_allowed"]
        ]
        if simulated_latency:
            failures.append(f"{label}: a simulated row allows a native latency claim")

        per_geometry[label] = {
            "records_total": len(records),
            "records_by_type": {
                key: sum(1 for r in records if r["record_type"] == key)
                for key in sorted({r["record_type"]
                                   for r in records})
            },
            "arms": sorted(expected),
            "prompts": sorted({r["prompt_id"]
                               for r in records}),
            "layers": sorted({r["layer"]
                              for r in records}),
            "timesteps": sorted({r["timestep"]
                                 for r in records}),
            "sparsities": sorted({r["sparsity"]
                                  for r in errors if r.get("sparsity") is not None}),
            "n_q_blocks": sorted({r["n_q_blocks"]
                                  for r in records}),
            "n_k_blocks": sorted({r["n_k_blocks"]
                                  for r in records}),
            "padded_seq_len": sorted({r.get("padded_seq_len")
                                      for r in records}),
            "seq_len": sorted({r["seq_len"]
                               for r in records}),
            "token_order": sorted({r["token_order"]
                                   for r in records}),
            "score_dtype": sorted(dtypes),
            "k_disagreements_across_arms": k_disagreements,
            "sparse_cells_total": len(arms_at_cell),
            "sparse_cells_incomplete": incomplete,
            "null_control_paired_cells": len(null_pairs),
            "null_control_rel_l2_deviations": null_deviations,
            "null_control_jaccard_deviations": null_jaccard_deviations,
            "null_metric_exclusions": sum(1 for r in errors if r["rel_l2"] is None),
        }

    payload = {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "geometries": per_geometry,
        "note": NOTE,
    }
    out_tables.mkdir(parents=True, exist_ok=True)
    (out_tables / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def table_error_arms(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path,
                     out_figures: Path) -> list[dict[str, Any]]:
    """Per-arm rel-L2 against dense BF16, by geometry and sparsity."""
    rows: list[dict[str, Any]] = []
    order = ["A", "B", "C", "C_null", "D", "D8", "C_rand", "E", "F8", "F16", "B_sim"]
    for label in GEOMETRY_ORDER:
        records = by_geometry.get(label)
        if not records:
            continue
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["record_type"] == "error_decomposition":
                groups[(record["config"], record.get("sparsity"))].append(record)
        for (config_id, sparsity), bucket in sorted(groups.items(),
                                                    key=lambda item:
                                                    (item[0][1] is not None, item[0][1] or 0.0, order.index(item[0][0])
                                                     if item[0][0] in order else 99)):
            rel = [r["rel_l2"] for r in bucket if r["rel_l2"] is not None]
            sample = bucket[0]
            rows.append({
                "geometry": label,
                "token_order": sample["token_order"],
                "n_k_blocks": sample["n_k_blocks"],
                "config": config_id,
                "isolates": sample["isolates"],
                "sparsity": sparsity,
                "k_per_query_block": sample.get("k_per_query_block"),
                "retained_block_fraction": (None if sparsity is None else 1.0 - sparsity),
                "retained_token_fraction": sample.get("retained_token_fraction"),
                "attention_compute_precision": sample["compute_precision_label"],
                "mask_source_precision": sample["mask_source_precision"],
                "native_or_simulated": sample["native_or_simulated"],
                "router_native_or_simulated": sample["router_native_or_simulated"],
                "numerical_only_no_latency_claim": sample["numerical_only"],
                "rel_l2_median": median(rel),
                "rel_l2_iqr": iqr(rel),
                "rel_l2_p10": quantile(rel, 0.10),
                "rel_l2_p90": quantile(rel, 0.90),
                "n": len(rel),
                "n_excluded_null_metric": len(bucket) - len(rel),
            })
    write_table(out_tables / "table2_error_arms_by_geometry.csv", rows)
    write_table(out_figures / "fig1_rel_l2_by_geometry_sparsity.csv", rows,
                ("geometry", "config", "sparsity", "rel_l2_median", "rel_l2_p10", "rel_l2_p90", "n"))
    return rows


def table_paired_excess(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path,
                        out_figures: Path) -> list[dict[str, Any]]:
    """The study's key isolation, per geometry: C_rand excess over D excess.

    Paired per cell against C so the sparsification error, which dominates by
    three orders of magnitude, cancels exactly. ``random_over_quantization_ratio``
    is the ratio of medians of the two excesses, which is the statistic Phase 2
    reported as 10-27x at 128x64.
    """
    rows: list[dict[str, Any]] = []
    for label in GEOMETRY_ORDER:
        records = by_geometry.get(label)
        if not records:
            continue
        per_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
        for record in records:
            if record["record_type"] != "error_decomposition" or record["rel_l2"] is None:
                continue
            if record["config"] in ("C", "D", "C_rand"):
                per_cell[cell_key(record)][record["config"]] = record["rel_l2"]
        buckets: dict[tuple, list[dict[str, float]]] = defaultdict(list)
        for key, arms in per_cell.items():
            if not {"C", "D", "C_rand"} <= arms.keys():
                continue
            entry = {
                "excess_quantization": arms["D"] - arms["C"],
                "excess_random": arms["C_rand"] - arms["C"],
                "sparsification": arms["C"],
            }
            buckets[(key[5], "all")].append(entry)
            buckets[(key[5], region_of(key[1]))].append(entry)
        for (sparsity, region), bucket in sorted(buckets.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])):
            quant = [entry["excess_quantization"] for entry in bucket]
            rand = [entry["excess_random"] for entry in bucket]
            sparsification = [entry["sparsification"] for entry in bucket]
            median_quant, median_rand, median_c = median(quant), median(rand), median(sparsification)
            rows.append({
                "geometry":
                label,
                "sparsity":
                sparsity,
                "region":
                region,
                "sparsification_C_median":
                median_c,
                "excess_quantization_mask_D_minus_C_median":
                median_quant,
                "excess_quantization_mask_iqr":
                iqr(quant),
                "excess_random_mask_Crand_minus_C_median":
                median_rand,
                "excess_random_mask_iqr":
                iqr(rand),
                "random_over_quantization_ratio":
                (None if not median_quant or median_rand is None else median_rand / median_quant),
                "wrong_mask_share_of_C": (None if not median_c or median_quant is None else median_quant / median_c),
                "frac_cells_random_worse":
                sum(1 for entry in bucket if entry["excess_random"] > entry["excess_quantization"]) / len(bucket),
                "n_paired_cells":
                len(bucket),
            })
    write_table(out_tables / "table3_paired_excess_by_geometry.csv", rows)
    write_table(
        out_figures / "fig2_random_over_quantization_by_geometry.csv", [row for row in rows if row["region"] == "all"],
        ("geometry", "sparsity", "excess_quantization_mask_D_minus_C_median", "excess_random_mask_Crand_minus_C_median",
         "random_over_quantization_ratio", "frac_cells_random_worse", "n_paired_cells"))
    return rows


def table_mask_stability(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path,
                         out_figures: Path) -> list[dict[str, Any]]:
    """Mask overlap between the NVFP4 and BF16 routers, per geometry.

    ``frac_query_blocks_changed`` and ``blocks_swapped_per_query_block`` are
    recorded per error row only from Phase 2B onward, so for the Phase 2 arm they
    are recomputed from that run's mechanism records and flagged as such — the
    mechanism sample is a deterministic 12-query-block lattice, not the full set.
    """
    rows: list[dict[str, Any]] = []
    for label in GEOMETRY_ORDER:
        records = by_geometry.get(label)
        if not records:
            continue
        errors = [
            r for r in records
            if r["record_type"] == "error_decomposition" and r["config"] == "D" and r.get("sparsity") is not None
        ]
        mech = [r for r in records if r["record_type"] == "mechanism"]
        by_sparsity: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for record in errors:
            by_sparsity[record["sparsity"]].append(record)
        mech_by_sparsity: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for record in mech:
            mech_by_sparsity[record["sparsity"]].append(record)
        for sparsity, bucket in sorted(by_sparsity.items()):
            jaccard = [r["mask_jaccard_vs_bf16"] for r in bucket if r["mask_jaccard_vs_bf16"] is not None]
            changed = [r["frac_query_blocks_changed"] for r in bucket if r.get("frac_query_blocks_changed") is not None]
            swapped = [
                r["blocks_swapped_per_query_block"] for r in bucket
                if r.get("blocks_swapped_per_query_block") is not None
            ]
            mech_bucket = mech_by_sparsity.get(sparsity, [])
            mech_swaps = [float(r["n_swapped"]) for r in mech_bucket]
            source = "error_rows" if changed else "mechanism_sample"
            rows.append({
                "geometry":
                label,
                "n_k_blocks":
                bucket[0]["n_k_blocks"],
                "n_q_blocks":
                bucket[0]["n_q_blocks"],
                "sparsity":
                sparsity,
                "k_per_query_block":
                bucket[0]["k_per_query_block"],
                "retained_token_fraction":
                bucket[0].get("retained_token_fraction"),
                "mask_jaccard_median":
                median(jaccard),
                "mask_jaccard_iqr":
                iqr(jaccard),
                "mask_jaccard_p10":
                quantile(jaccard, 0.10),
                "mask_jaccard_min":
                min(jaccard) if jaccard else None,
                "frac_cells_jaccard_below_0.95":
                (None if not jaccard else sum(1 for v in jaccard if v < 0.95) / len(jaccard)),
                "n_cells":
                len(jaccard),
                "churn_source":
                source,
                "frac_query_blocks_changed_median":
                (median(changed) if changed else
                 (None if not mech_bucket else sum(1 for r in mech_bucket if r["n_swapped"] > 0) / len(mech_bucket))),
                "blocks_swapped_per_query_block_mean": (statistics.fmean(swapped) if swapped else
                                                        (statistics.fmean(mech_swaps) if mech_swaps else None)),
                "n_query_block_observations":
                len(mech_bucket),
            })
    write_table(out_tables / "table4_mask_stability_by_geometry.csv", rows)
    write_table(out_figures / "fig3_mask_jaccard_by_geometry.csv", rows,
                ("geometry", "sparsity", "mask_jaccard_median", "mask_jaccard_iqr", "mask_jaccard_p10",
                 "frac_query_blocks_changed_median", "blocks_swapped_per_query_block_mean", "n_cells"))
    return rows


def table_mechanism(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path,
                    out_figures: Path) -> list[dict[str, Any]]:
    """Block-mass and score-margin mechanism quantities, per geometry.

    Conditional on the query block having at least one swap, so the masses are
    comparable with the per-block means rather than diluted by no-swap blocks —
    the same convention Phase 2 used.
    """
    columns = ("mass_dropped_mean", "mass_added_mean", "mass_agreed_mean", "mass_excluded_mean",
               "mass_random_dropped_mean", "mass_dropped_total", "mass_random_dropped_total", "mass_retained_total",
               "score_gap_swapped_norm", "score_gap_swapped_raw", "score_spread")
    rows: list[dict[str, Any]] = []
    for label in GEOMETRY_ORDER:
        records = by_geometry.get(label)
        if not records:
            continue
        mech = [r for r in records if r["record_type"] == "mechanism"]
        buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for record in mech:
            buckets[(record["sparsity"], "all")].append(record)
            buckets[(record["sparsity"], region_of(record["layer"]))].append(record)
        for (sparsity, region), bucket in sorted(buckets.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])):
            swapped = [r["n_swapped"] for r in bucket]
            with_swap = [r for r in bucket if r["n_swapped"] > 0]
            row: dict[str, Any] = {
                "geometry": label,
                "sparsity": sparsity,
                "region": region,
                "n_query_block_observations": len(bucket),
                "n_with_at_least_one_swap": len(with_swap),
                "frac_query_blocks_with_a_swap": sum(1 for value in swapped if value > 0) / len(bucket),
                "mean_n_swapped": statistics.fmean(swapped),
                "k_per_query_block": bucket[0].get("k_per_query_block"),
            }
            for column in columns:
                values = [r[column] for r in with_swap if r.get(column) is not None]
                row[f"{column}_median"] = median(values)
                row[f"{column}_iqr"] = iqr(values)
                row[f"{column}_n"] = len(values)
            agreed = row["mass_agreed_mean_median"] or 0.0
            dropped = row["mass_dropped_mean_median"] or 0.0
            random_dropped = row["mass_random_dropped_mean_median"] or 0.0
            row["agreed_over_dropped_mass_ratio"] = None if dropped == 0 else agreed / dropped
            row["dropped_over_agreed_mass_ratio"] = None if agreed == 0 else dropped / agreed
            row["random_over_quantization_dropped_mass_ratio"] = (None if dropped == 0 else random_dropped / dropped)
            for config_id in ("C", "D", "C_rand", "E", "F16"):
                values = [
                    r[f"qblock_rel_l2_{config_id}"] for r in bucket if r.get(f"qblock_rel_l2_{config_id}") is not None
                ]
                row[f"qblock_rel_l2_{config_id}_median"] = median(values)
            excess = [
                r["qblock_rel_l2_D"] - r["qblock_rel_l2_C"] for r in with_swap
                if r.get("qblock_rel_l2_D") is not None and r.get("qblock_rel_l2_C") is not None
            ]
            row["qblock_wrong_mask_excess_median"] = median(excess)
            row["qblock_wrong_mask_excess_n"] = len(excess)
            rows.append(row)
    write_table(out_tables / "table5_mechanism_by_geometry.csv", rows)
    write_table(out_figures / "fig4_block_mass_by_geometry.csv", [row for row in rows if row["region"] == "all"],
                ("geometry", "sparsity", "mass_dropped_mean_median", "mass_agreed_mean_median",
                 "mass_excluded_mean_median", "mass_random_dropped_mean_median", "agreed_over_dropped_mass_ratio",
                 "score_gap_swapped_norm_median", "n_with_at_least_one_swap"))
    return rows


def table_tie_diagnostic(by_geometry: dict[str, list[dict[str, Any]]], out_tables: Path) -> list[dict[str, Any]]:
    """Boundary ties at all three geometries, on both counting denominators.

    Emitted by the Phase 2B runs at every measured cell, so the Phase 1 (~110 per
    head) and Phase 2 (~1,400 per cell) figures can be compared without
    reconstructing either report's denominator.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for label, records in by_geometry.items():
        diagnostics = [r for r in records if r["record_type"] == "tie_diagnostic"]
        by_key: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for record in diagnostics:
            by_key[(record["diagnostic_geometry"], )].append(record)
        for (diagnostic_geometry, ), bucket in sorted(by_key.items()):
            sparsities = sorted({
                int(key.rsplit("_s", 1)[1])
                for record in bucket[:1]
                for key in record if key.startswith("ties_per_cell_")
            })
            for precision in ("bf16", "fp8_e4m3", "nvfp4"):
                for dtype in ("fp32", "fp64"):
                    for sparsity in sparsities:
                        tag = f"{precision}_{dtype}_s{sparsity}"
                        per_cell = [float(r[f"ties_per_cell_{tag}"]) for r in bucket if f"ties_per_cell_{tag}" in r]
                        if not per_cell:
                            continue
                        # Records are duplicated across the source runs (both emit
                        # all three diagnostic geometries); dedupe on the key.
                        identity = (diagnostic_geometry, precision, dtype, sparsity)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        per_head = [
                            float(r[f"ties_per_head_median_{tag}"]) for r in bucket
                            if f"ties_per_head_median_{tag}" in r
                        ]
                        margin_norm = [
                            float(r[f"margin_norm_median_{tag}"]) for r in bucket if f"margin_norm_median_{tag}" in r
                        ]
                        rows.append({
                            "measured_in_run_geometry":
                            label,
                            "diagnostic_geometry":
                            diagnostic_geometry,
                            "n_q_blocks":
                            bucket[0]["diagnostic_n_q_blocks"],
                            "n_k_blocks":
                            bucket[0]["diagnostic_n_k_blocks"],
                            "router_precision":
                            precision,
                            "score_dtype":
                            dtype,
                            "sparsity":
                            sparsity / 100,
                            "ties_per_cell_median":
                            median(per_cell),
                            "ties_per_head_median":
                            median(per_head),
                            "ties_per_head_over_n_q_blocks":
                            (None if not per_head else median(per_head) / bucket[0]["diagnostic_n_q_blocks"]),
                            "boundary_margin_norm_median":
                            median(margin_norm),
                            "n_cells":
                            len(per_cell),
                        })
    write_table(out_tables / "table8_tie_diagnostic_by_geometry.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",
                        action="append",
                        required=True,
                        metavar="GEOMETRY=DIR",
                        help="repeatable; e.g. --raw 64x64-cube=artifacts/sparsefp4/raw/<run_id>")
    parser.add_argument("--out-tables", type=Path, required=True)
    parser.add_argument("--out-figures", type=Path, required=True)
    args = parser.parse_args()

    by_geometry: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for spec in args.raw:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--raw expects GEOMETRY=DIR, got {spec!r}")
        if label not in GEOMETRY_ORDER:
            raise SystemExit(f"unknown geometry {label!r}; expected one of {GEOMETRY_ORDER}")
        by_geometry[label] = read_records(Path(path))
        sources[label] = path

    verification = verify(by_geometry, args.out_tables)
    arms = table_error_arms(by_geometry, args.out_tables, args.out_figures)
    excess = table_paired_excess(by_geometry, args.out_tables, args.out_figures)
    stability = table_mask_stability(by_geometry, args.out_tables, args.out_figures)
    mechanism = table_mechanism(by_geometry, args.out_tables, args.out_figures)
    ties = table_tie_diagnostic(by_geometry, args.out_tables)

    headline: list[dict[str, Any]] = []
    excess_all = {(row["geometry"], row["sparsity"]): row for row in excess if row["region"] == "all"}
    mech_all = {(row["geometry"], row["sparsity"]): row for row in mechanism if row["region"] == "all"}
    stability_all = {(row["geometry"], row["sparsity"]): row for row in stability}
    for label in GEOMETRY_ORDER:
        for sparsity in sorted({key[1] for key in excess_all if key[0] == label}):
            row = excess_all[(label, sparsity)]
            mech = mech_all.get((label, sparsity), {})
            stab = stability_all.get((label, sparsity), {})
            headline.append({
                "geometry":
                label,
                "token_order": ("vsa_tile_4x4x4_contiguous" if label.endswith("cube") else "raster_frame_y_x"),
                "sparsity":
                sparsity,
                "n_k_blocks":
                stab.get("n_k_blocks"),
                "k_per_query_block":
                stab.get("k_per_query_block"),
                "retained_token_fraction":
                stab.get("retained_token_fraction"),
                "mask_jaccard_median":
                stab.get("mask_jaccard_median"),
                "mask_jaccard_iqr":
                stab.get("mask_jaccard_iqr"),
                "frac_query_blocks_changed":
                stab.get("frac_query_blocks_changed_median"),
                "blocks_swapped_per_query_block":
                stab.get("blocks_swapped_per_query_block_mean"),
                "mass_swapped_out_median":
                mech.get("mass_dropped_mean_median"),
                "mass_agreed_median":
                mech.get("mass_agreed_mean_median"),
                "agreed_over_swapped_mass_ratio":
                mech.get("agreed_over_dropped_mass_ratio"),
                "score_gap_swapped_norm_median":
                mech.get("score_gap_swapped_norm_median"),
                "rel_l2_C_median":
                row["sparsification_C_median"],
                "excess_D_minus_C_median":
                row["excess_quantization_mask_D_minus_C_median"],
                "excess_Crand_minus_C_median":
                row["excess_random_mask_Crand_minus_C_median"],
                "Crand_over_D_excess_ratio":
                row["random_over_quantization_ratio"],
                "frac_cells_random_worse":
                row["frac_cells_random_worse"],
                "n_paired_cells":
                row["n_paired_cells"],
                "n_query_block_observations":
                mech.get("n_with_at_least_one_swap"),
                "numerical_only_no_latency_claim":
                True,
            })
    write_table(args.out_tables / "table1_three_geometry_headline.csv", headline)
    write_table(args.out_figures / "fig5_three_geometry_headline.csv", headline)

    summary = {
        "verification": verification,
        "sources": sources,
        "headline": headline,
        "tie_diagnostic_rows": len(ties),
        "arm_rows": len(arms),
        "note": NOTE,
    }
    (args.out_tables / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verification["verdict"],
                "failures": verification["failures"],
                "geometries": {
                    label: info["records_total"]
                    for label, info in verification["geometries"].items()
                },
                "headline_rows": len(headline),
            },
            indent=2))
    return 0 if verification["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
