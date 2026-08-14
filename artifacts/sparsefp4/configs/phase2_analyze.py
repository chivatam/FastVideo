"""Phase 2 analysis: A-F decomposition, the H3 verdict, and the mechanism tests.

Reads the raw JSONL (plain or gzipped) written by ``PrecisionSparseAttentionImpl``
and emits tables + figure CSVs. Aggregates are medians with IQR and always carry
``n``; the H3 comparison is **paired** at the
``(prompt, layer, head, timestep, cfg_branch, sparsity)`` level, so it reports the
per-cell paired difference distribution and the fraction of cells that improve,
not a difference of independently-pooled medians.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_analyze.py \
        --raw artifacts/sparsefp4/raw/<run_id> \
        --out-tables artifacts/sparsefp4/tables/<tag> \
        --out-figures artifacts/sparsefp4/figures/<tag>
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
from pathlib import Path
from typing import Any
from collections.abc import Sequence

AFFECTED_LAYERS = (0, 1, 2, 27, 28, 29)
UNAFFECTED_LAYERS = (5, 6, 10, 11, 13)
GEOMETRY_NOTE = ("All numbers are the raster-order 128x64 diagnostic geometry executed on the kernel's "
                 "64x64 grid, NOT VSA's 64-token (4,4,4) spatio-temporal cubes.")
PRECISION_NOTE = ("\"Low precision\" everywhere means NVFP4 Q/K with BF16 PV (the FA4 kernel's "
                  "qk_mode=nvfp4, pv_mode=bf16), never fully-FP4 attention.")
H3_THRESHOLD = 0.20
MIN_CELL_N = 20


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
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def iqr(values: Sequence[float]) -> float | None:
    p25, p75 = quantile(values, 0.25), quantile(values, 0.75)
    if p25 is None or p75 is None:
        return None
    return p75 - p25


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
    markdown = path.with_suffix(".md")
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(key)) for key in keys) + " |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def region_of(layer: int) -> str:
    if layer in AFFECTED_LAYERS:
        return "affected"
    if layer in UNAFFECTED_LAYERS:
        return "unaffected"
    return "broad"


def cell_key(record: dict[str, Any]) -> tuple:
    return (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"],
            record.get("sparsity"))


def table_h3(records: list[dict[str, Any]], out_tables: Path, out_figures: Path) -> dict[str, Any]:
    """The H3 test: paired per-cell comparison of E vs F8 vs F16.

    Everything except ``router_precision`` is held fixed — same sparsity, same
    ``k``, same simulated NVFP4 Q/K + BF16 PV compute — so the statistic is the
    per-cell paired difference. Reports the full distribution, the fraction of
    cells that improve, the affected/unaffected split, and the BF16-router null
    control (F16 vs itself, which must be exactly zero).
    """
    per_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["record_type"] != "error_decomposition" or record["rel_l2"] is None:
            continue
        if record["config"] in ("E", "F8", "F16"):
            per_cell[cell_key(record)][record["config"]] = record["rel_l2"]

    rows: list[dict[str, Any]] = []
    ecdf_rows: list[dict[str, Any]] = []
    for candidate in ("F8", "F16"):
        buckets: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        for key, arms in per_cell.items():
            if "E" not in arms or candidate not in arms:
                continue
            sparsity, layer = key[5], key[1]
            buckets[(sparsity, "all")].append((arms["E"], arms[candidate]))
            buckets[(sparsity, region_of(layer))].append((arms["E"], arms[candidate]))
        for (sparsity, region), pairs in sorted(buckets.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])):
            e_values = [pair[0] for pair in pairs]
            f_values = [pair[1] for pair in pairs]
            differences = [pair[0] - pair[1] for pair in pairs]
            relative = [(pair[0] - pair[1]) / pair[0] for pair in pairs if pair[0] > 0]
            median_e = median(e_values)
            median_f = median(f_values)
            reduction = None if not median_e or median_f is None else (median_e - median_f) / median_e
            improved = sum(1 for pair in pairs if pair[1] < pair[0])
            trimmed = sorted(relative)[len(relative) // 20:len(relative) - len(relative) // 20] or relative
            rows.append({
                "comparison": f"E_to_{candidate}",
                "router_E": "nvfp4",
                "router_candidate": "fp8_e4m3" if candidate == "F8" else "bf16",
                "sparsity": sparsity,
                "region": region,
                "attention_compute": "nvfp4_qk_bf16_pv (simulated; numerical-only)",
                "rel_l2_median_E": median_e,
                "rel_l2_median_candidate": median_f,
                "median_relative_reduction": reduction,
                "mean_paired_relative_reduction": (None if not relative else statistics.fmean(relative)),
                "trimmed5_mean_relative_reduction": (None if not trimmed else statistics.fmean(trimmed)),
                "paired_diff_p10": quantile(differences, 0.10),
                "paired_diff_p50": median(differences),
                "paired_diff_p90": quantile(differences, 0.90),
                "frac_cells_improved": improved / len(pairs),
                "n_paired_cells": len(pairs),
                "meets_20pct_threshold": (None if reduction is None else reduction >= H3_THRESHOLD),
            })
            if region == "all":
                for value in sorted(relative):
                    ecdf_rows.append({
                        "comparison": f"E_to_{candidate}",
                        "sparsity": sparsity,
                        "relative_reduction": value,
                    })

    # Null control: the BF16-router arm compared with itself must be identically
    # zero. This does not certify score resolution (STATUS.md trap 8) and is
    # reported as an arithmetic-identity check only.
    null_diffs = [arms["F16"] - arms["F16"] for arms in per_cell.values() if "F16" in arms]
    rows.append({
        "comparison": "F16_to_F16_null_control",
        "router_E": "bf16",
        "router_candidate": "bf16",
        "sparsity": None,
        "region": "all",
        "attention_compute": "nvfp4_qk_bf16_pv (simulated; numerical-only)",
        "rel_l2_median_E": None,
        "rel_l2_median_candidate": None,
        "median_relative_reduction": 0.0 if null_diffs else None,
        "paired_diff_p50": median(null_diffs),
        "frac_cells_improved": 0.0,
        "n_paired_cells": len(null_diffs),
        "meets_20pct_threshold": False,
    })
    write_table(out_tables / "table3_h3_paired.csv", rows)
    write_table(out_figures / "fig2_h3_paired_reduction_ecdf.csv", ecdf_rows)

    headline = [row for row in rows if row["region"] == "all" and row["comparison"] != "F16_to_F16_null_control"]
    best = max((row for row in headline if row["median_relative_reduction"] is not None),
               key=lambda row: row["median_relative_reduction"],
               default=None)
    return {
        "rows": rows,
        "best": best,
        "supported": bool(best and best["median_relative_reduction"] >= H3_THRESHOLD),
        "null_control_max_abs_diff": max((abs(value) for value in null_diffs), default=None),
    }


def table_decomposition_split(records: list[dict[str, Any]], out_tables: Path) -> list[dict[str, Any]]:
    """Attribution of E's error into quantization / sparsification / wrong-mask.

    These are differences of errors against the single reference A, not an exact
    additive decomposition — quantization and sparsification errors do not
    compose linearly (EXPERIMENT_SPEC 7.1 rule 5). Raw per-configuration errors
    are in table 1 alongside.
    """
    dense: dict[tuple, dict[str, float]] = defaultdict(dict)
    sparse: dict[tuple, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["record_type"] != "error_decomposition" or record["rel_l2"] is None:
            continue
        base = (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"])
        if record.get("sparsity") is None:
            dense[base][record["config"]] = record["rel_l2"]
        else:
            sparse[(base, record["sparsity"])][record["config"]] = record["rel_l2"]

    buckets: dict[float, list[dict[str, float]]] = defaultdict(list)
    for (base, sparsity), arms in sparse.items():
        if not {"C", "D", "E", "F16"} <= arms.keys() or "B" not in dense.get(base, {}):
            continue
        buckets[sparsity].append({
            "quantization_B": dense[base]["B"],
            "sparsification_C": arms["C"],
            "wrong_mask_D_minus_C": arms["D"] - arms["C"],
            "wrong_mask_D8_minus_C": arms.get("D8", arms["D"]) - arms["C"],
            "random_mask_Crand_minus_C": arms.get("C_rand", arms["C"]) - arms["C"],
            "combined_E": arms["E"],
            "router_recoverable_E_minus_F16": arms["E"] - arms["F16"],
            "residual_E_minus_B_minus_C": arms["E"] - dense[base]["B"] - arms["C"],
        })

    rows: list[dict[str, Any]] = []
    for sparsity, bucket in sorted(buckets.items()):
        row: dict[str, Any] = {"sparsity": sparsity, "retained_fraction": 1.0 - sparsity, "n": len(bucket)}
        for column in bucket[0]:
            values = [entry[column] for entry in bucket]
            row[f"{column}_median"] = median(values)
            row[f"{column}_iqr"] = iqr(values)
        total = row["combined_E_median"] or 1.0
        row["share_quantization_of_E"] = (row["quantization_B_median"] or 0.0) / total
        row["share_sparsification_of_E"] = (row["sparsification_C_median"] or 0.0) / total
        row["share_wrong_mask_of_E"] = (row["wrong_mask_D_minus_C_median"] or 0.0) / total
        rows.append(row)
    write_table(out_tables / "table4_error_attribution.csv", rows)
    return rows


def table_af(records: list[dict[str, Any]], out_tables: Path, out_figures: Path) -> list[dict[str, Any]]:
    """The A-F decomposition table: median/IQR rel-L2, cosine, max-abs, n."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["record_type"] != "error_decomposition":
            continue
        groups[(record["config"], record.get("sparsity"))].append(record)

    order = ["A", "B", "B_sim", "C", "D", "D8", "C_rand", "E", "F8", "F16"]
    rows: list[dict[str, Any]] = []
    for (config_id, sparsity), bucket in sorted(groups.items(),
                                                key=lambda item:
                                                (item[0][1] is not None, item[0][1] or 0.0, order.index(item[0][0]))):
        rel = [r["rel_l2"] for r in bucket if r["rel_l2"] is not None]
        cos = [r["cosine"] for r in bucket if r["cosine"] is not None]
        mx = [r["max_abs"] for r in bucket if r["max_abs"] is not None]
        sample = bucket[0]
        rows.append({
            "config": config_id,
            "isolates": sample["isolates"],
            "sparsity": sparsity,
            "retained_fraction": sample.get("retained_fraction"),
            "attention_compute_precision": sample["compute_precision_label"],
            "mask_source_precision": sample["mask_source_precision"],
            "native_or_simulated": sample["native_or_simulated"],
            "router_native_or_simulated": sample["router_native_or_simulated"],
            "numerical_only_no_latency_claim": sample["numerical_only"],
            "k_per_query_block": sample.get("k_per_query_block"),
            "rel_l2_median": median(rel),
            "rel_l2_iqr": iqr(rel),
            "rel_l2_p10": quantile(rel, 0.10),
            "rel_l2_p90": quantile(rel, 0.90),
            "cosine_median": median(cos),
            "max_abs_median": median(mx),
            "n": len(rel),
            "n_excluded_null_metric": len(bucket) - len(rel),
        })
    write_table(out_tables / "table1_af_decomposition.csv", rows)
    write_table(out_figures / "fig1_af_rel_l2_by_sparsity.csv", rows,
                ("config", "sparsity", "rel_l2_median", "rel_l2_p10", "rel_l2_p90", "n"))

    by_region: list[dict[str, Any]] = []
    region_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["record_type"] != "error_decomposition":
            continue
        region_groups[(record["config"], record.get("sparsity"), region_of(record["layer"]))].append(record)
    for (config_id, sparsity, region), bucket in sorted(region_groups.items(),
                                                        key=lambda item:
                                                        (item[0][1] or 0.0, item[0][2], order.index(item[0][0]))):
        rel = [r["rel_l2"] for r in bucket if r["rel_l2"] is not None]
        by_region.append({
            "config": config_id,
            "sparsity": sparsity,
            "region": region,
            "rel_l2_median": median(rel),
            "rel_l2_iqr": iqr(rel),
            "n": len(rel),
            "insufficient_n": len(rel) < MIN_CELL_N,
        })
    write_table(out_tables / "table2_af_by_region.csv", by_region)
    return rows


def table_mechanism(records: list[dict[str, Any]], out_tables: Path, out_figures: Path) -> list[dict[str, Any]]:
    """The decision-margin mechanism: what do swapped blocks actually carry?

    Compares, per query block, the exact dense attention mass of the blocks the
    NVFP4 mask swaps against (a) the blocks both masks agree on, (b) the average
    excluded block, and (c) the equal-magnitude random control's swapped blocks.
    """
    mech = [r for r in records if r["record_type"] == "mechanism"]
    rows: list[dict[str, Any]] = []
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in mech:
        buckets[(record["sparsity"], "all")].append(record)
        buckets[(record["sparsity"], region_of(record["layer"]))].append(record)

    columns = ("mass_dropped_mean", "mass_added_mean", "mass_agreed_mean", "mass_excluded_mean",
               "mass_random_dropped_mean", "mass_dropped_total", "mass_random_dropped_total", "mass_retained_total",
               "n_swapped", "score_gap_swapped_norm", "score_gap_swapped_raw")
    for (sparsity, region), bucket in sorted(buckets.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])):
        row: dict[str, Any] = {"sparsity": sparsity, "region": region, "n_query_blocks": len(bucket)}
        swapped = [r["n_swapped"] for r in bucket]
        row["frac_query_blocks_with_a_swap"] = sum(1 for value in swapped if value > 0) / len(bucket)
        row["mean_n_swapped"] = statistics.fmean(swapped)
        row["k_per_query_block"] = bucket[0].get("k_per_query_block")
        # Conditional on at least one swap, so the totals are comparable with the
        # per-block means rather than being diluted by no-swap query blocks.
        with_swap = [r for r in bucket if r["n_swapped"] > 0]
        for column in columns:
            values = [r[column] for r in with_swap if r.get(column) is not None]
            row[f"{column}_median"] = median(values)
        agreed = row["mass_agreed_mean_median"] or 0.0
        dropped = row["mass_dropped_mean_median"] or 0.0
        random_dropped = row["mass_random_dropped_mean_median"] or 0.0
        row["dropped_over_agreed_ratio"] = None if agreed == 0 else dropped / agreed
        row["random_over_quantization_dropped_ratio"] = None if dropped == 0 else random_dropped / dropped
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
        rows.append(row)
    write_table(out_tables / "table5_margin_mechanism.csv", rows)

    # Per-query-block output error vs the score gap of the swapped blocks, binned
    # by gap decile. This is the direct near-tie test.
    scatter: list[dict[str, Any]] = []
    with_gap = [r for r in mech if r.get("score_gap_swapped_norm") is not None and r.get("qblock_rel_l2_D") is not None]
    for sparsity in sorted({r["sparsity"] for r in with_gap}):
        subset = sorted((r for r in with_gap if r["sparsity"] == sparsity), key=lambda r: r["score_gap_swapped_norm"])
        if not subset:
            continue
        size = max(1, len(subset) // 10)
        for decile in range(10):
            chunk = subset[decile * size:(decile + 1) * size] if decile < 9 else subset[9 * size:]
            if not chunk:
                continue
            mass = [r["mass_dropped_mean"] for r in chunk if r.get("mass_dropped_mean") is not None]
            scatter.append({
                "sparsity":
                sparsity,
                "gap_decile":
                decile + 1,
                "score_gap_swapped_norm_median":
                median([r["score_gap_swapped_norm"] for r in chunk]),
                "mass_dropped_mean_median":
                median(mass),
                "qblock_rel_l2_D_median":
                median([r["qblock_rel_l2_D"] for r in chunk]),
                "qblock_rel_l2_C_median":
                median([r["qblock_rel_l2_C"] for r in chunk]),
                "wrong_mask_excess_median":
                median([r["qblock_rel_l2_D"] - r["qblock_rel_l2_C"] for r in chunk]),
                "n":
                len(chunk),
            })
    write_table(out_figures / "fig3_output_error_vs_score_gap.csv", scatter)
    write_table(out_tables / "table6_error_vs_score_gap_decile.csv", scatter)
    return rows


def table_random_contrast(records: list[dict[str, Any]], out_tables: Path, out_figures: Path) -> list[dict[str, Any]]:
    """Contrast control: same number of blocks changed, chosen at random.

    Paired at the cell level against configuration D (same BF16 compute, same
    budget, quantization-chosen swaps) and against C (the no-wrong-mask floor).
    If random swaps of equal magnitude hurt much more, the mechanism is isolated:
    quantization perturbs only the harmless top-k boundary.
    """
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
        }
        buckets[(key[5], "all")].append(entry)
        buckets[(key[5], region_of(key[1]))].append(entry)

    rows: list[dict[str, Any]] = []
    for (sparsity, region), bucket in sorted(buckets.items(), key=lambda item: (item[0][0] or 0.0, item[0][1])):
        quant = [entry["excess_quantization"] for entry in bucket]
        rand = [entry["excess_random"] for entry in bucket]
        median_quant, median_rand = median(quant), median(rand)
        rows.append({
            "sparsity":
            sparsity,
            "region":
            region,
            "excess_rel_l2_quantization_mask_median":
            median_quant,
            "excess_rel_l2_quantization_mask_iqr":
            iqr(quant),
            "excess_rel_l2_random_mask_median":
            median_rand,
            "excess_rel_l2_random_mask_iqr":
            iqr(rand),
            "random_over_quantization_ratio":
            (None if not median_quant or median_rand is None else median_rand / median_quant),
            "frac_cells_random_worse":
            sum(1 for entry in bucket if entry["excess_random"] > entry["excess_quantization"]) / len(bucket),
            "n_paired_cells":
            len(bucket),
        })
    write_table(out_tables / "table7_random_perturbation_contrast.csv", rows)
    write_table(out_figures / "fig4_random_vs_quantization_excess.csv", rows,
                ("sparsity", "region", "excess_rel_l2_quantization_mask_median", "excess_rel_l2_random_mask_median",
                 "random_over_quantization_ratio", "n_paired_cells"))
    return rows


def table_saturation_control(records: list[dict[str, Any]], out_tables: Path,
                             out_figures: Path) -> list[dict[str, Any]]:
    """Do the edge layers just have wider activations?

    Pairs each layer's e2m1 saturation and activation range against its wrong-mask
    excess error. If saturation tracks the "sensitive layer" ranking, the correct
    framing is "NVFP4 saturation follows activation range", not "routing is
    layer-sensitive".
    """
    activation: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if record["record_type"] == "activation_stats":
            activation[(record["prompt_id"], record["layer"], record["head"], record["timestep"],
                        record["cfg_branch"])] = record

    excess: dict[tuple, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["record_type"] != "error_decomposition" or record["rel_l2"] is None:
            continue
        if record["config"] in ("C", "D") and record.get("sparsity") is not None:
            key = (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"],
                   record["sparsity"])
            excess[key][record["config"]] = record["rel_l2"]

    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, arms in excess.items():
        if not {"C", "D"} <= arms.keys():
            continue
        stats = activation.get(key[:5])
        if stats is None:
            continue
        bucket = per_layer[key[1]]
        bucket["excess"].append(arms["D"] - arms["C"])
        bucket["relative_excess"].append((arms["D"] - arms["C"]) / arms["C"] if arms["C"] else 0.0)
        bucket["sat_q"].append(stats["sat_frac_q_nvfp4_layer"])
        bucket["sat_k"].append(stats["sat_frac_k_nvfp4_layer"])
        bucket["q_absmax"].append(stats["q_absmax"])
        bucket["k_absmax"].append(stats["k_absmax"])
        bucket["q_dyn"].append(stats["q_intra_group_dynamic_range"])

    rows: list[dict[str, Any]] = []
    for layer in sorted(per_layer):
        bucket = per_layer[layer]
        rows.append({
            "layer": layer,
            "region": region_of(layer),
            "wrong_mask_excess_rel_l2_median": median(bucket["excess"]),
            "wrong_mask_relative_excess_median": median(bucket["relative_excess"]),
            "sat_frac_q_nvfp4_median": median(bucket["sat_q"]),
            "sat_frac_k_nvfp4_median": median(bucket["sat_k"]),
            "q_absmax_median": median(bucket["q_absmax"]),
            "k_absmax_median": median(bucket["k_absmax"]),
            "q_intra_group_dynamic_range_median": median(bucket["q_dyn"]),
            "n": len(bucket["excess"]),
        })
    write_table(out_tables / "table8_saturation_vs_layer_sensitivity.csv", rows)
    write_table(out_figures / "fig5_saturation_vs_wrong_mask_excess.csv", rows)
    return rows


def table_score_resolution(records: list[dict[str, Any]], out_tables: Path) -> list[dict[str, Any]]:
    """Trap 8 documented on real Q/K rather than asserted."""
    resolution = [r for r in records if r["record_type"] == "score_resolution"]
    rows: list[dict[str, Any]] = []
    for precision in ("bf16", "fp8_e4m3", "nvfp4"):
        for sparsity in (0.80, 0.90, 0.95):
            tag = f"{precision}_s{int(sparsity * 100)}"
            ties32 = [r[f"boundary_ties_fp32_{tag}"] for r in resolution if f"boundary_ties_fp32_{tag}" in r]
            ties64 = [r[f"boundary_ties_fp64_{tag}"] for r in resolution if f"boundary_ties_fp64_{tag}" in r]
            changed = [
                r[f"fp32_vs_fp64_frac_changed_{tag}"] for r in resolution if f"fp32_vs_fp64_frac_changed_{tag}" in r
            ]
            if not ties32:
                continue
            rows.append({
                "router_precision": precision,
                "sparsity": sparsity,
                "score_abs_median": median([r[f"score_abs_median_{precision}"] for r in resolution]),
                "score_spread_median": median([r[f"score_spread_median_{precision}"] for r in resolution]),
                "boundary_ties_fp32_median": median([float(value) for value in ties32]),
                "boundary_ties_fp64_median": median([float(value) for value in ties64]),
                "frac_topk_decisions_fp32_differs_from_fp64": median(changed),
                "n_cells": len(ties32),
            })
    write_table(out_tables / "table9_score_resolution_trap8.csv", rows)
    return rows


def verify(records: list[dict[str, Any]], out_tables: Path) -> dict[str, Any]:
    """Pre-analysis gate. No aggregate is quoted until this reports PASS.

    Checks the things that would silently invalidate the decomposition: a layer
    falling off the research backend, an fp32 scorer sneaking in, unequal ``k``
    across arms at one cell, a non-zero self-error for the reference, and the
    dense/sparse configuration set being incomplete at any measured cell.
    """
    failures: list[str] = []
    errors = [r for r in records if r["record_type"] == "error_decomposition"]

    backends = {r["attention_backend"] for r in records}
    if backends != {"PRECISION_SPARSE_ATTN"}:
        failures.append(f"unexpected attention_backend values: {sorted(backends)}")

    dtypes = {r["score_dtype"] for r in records}
    if dtypes != {"float64"}:
        failures.append(f"score_dtype must be float64 everywhere (STATUS.md trap 8); saw {sorted(dtypes)}")

    self_errors = [r["rel_l2"] for r in errors if r["config"] == "A"]
    if any(value not in (0.0, None) for value in self_errors):
        failures.append("configuration A has non-zero error against itself")

    budgets: dict[tuple, set[int]] = defaultdict(set)
    configs_at_cell: dict[tuple, set[str]] = defaultdict(set)
    for record in errors:
        if record.get("sparsity") is None:
            continue
        base = (record["prompt_id"], record["layer"], record["head"], record["timestep"], record["cfg_branch"],
                record["sparsity"])
        budgets[base].add(int(record["k_per_query_block"]))
        configs_at_cell[base].add(record["config"])
    k_disagreements = sum(1 for values in budgets.values() if len(values) > 1)
    if k_disagreements:
        failures.append(f"{k_disagreements} cells have differing k across arms (equal-budget rule broken)")

    expected_sparse = {"C", "D", "D8", "C_rand", "E", "F8", "F16"}
    incomplete = sum(1 for values in configs_at_cell.values() if values != expected_sparse)
    if incomplete:
        failures.append(f"{incomplete} sparse cells are missing at least one configuration")

    seq_lens = {r["seq_len"] for r in records}
    if len(seq_lens) != 1:
        failures.append(f"seq_len is not constant: {sorted(seq_lens)}")

    simulated_with_latency_claim = [
        r for r in errors if r["native_or_simulated"] == "simulated" and r["native_latency_claim_allowed"]
    ]
    if simulated_with_latency_claim:
        failures.append("a simulated row is marked as allowing a native latency claim")

    payload = {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "records_total": len(records),
        "records_by_type": {
            key: sum(1 for r in records if r["record_type"] == key)
            for key in sorted({r["record_type"]
                               for r in records})
        },
        "configs_present": sorted({r["config"]
                                   for r in errors}),
        "prompts": sorted({r["prompt_id"]
                           for r in records}),
        "layers": sorted({r["layer"]
                          for r in records}),
        "timesteps": sorted({r["timestep"]
                             for r in records}),
        "sparsities": sorted({r["sparsity"]
                              for r in errors if r.get("sparsity") is not None}),
        "cfg_branches": sorted({r["cfg_branch"]
                                for r in records}),
        "heads": sorted({r["head"]
                         for r in records if r["head"] is not None}),
        "seq_len": sorted(seq_lens),
        "score_dtype": sorted(dtypes),
        "k_disagreements_across_arms": k_disagreements,
        "sparse_cells_incomplete": incomplete,
        "sparse_cells_total": len(configs_at_cell),
        "null_metric_exclusions": sum(1 for r in errors if r["rel_l2"] is None),
        "geometry_note": GEOMETRY_NOTE,
        "precision_note": PRECISION_NOTE,
    }
    out_tables.mkdir(parents=True, exist_ok=True)
    (out_tables / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-tables", type=Path, required=True)
    parser.add_argument("--out-figures", type=Path, required=True)
    args = parser.parse_args()

    records = read_records(args.raw)
    verification = verify(records, args.out_tables)
    af = table_af(records, args.out_tables, args.out_figures)
    h3 = table_h3(records, args.out_tables, args.out_figures)
    attribution = table_decomposition_split(records, args.out_tables)
    mechanism = table_mechanism(records, args.out_tables, args.out_figures)
    contrast = table_random_contrast(records, args.out_tables, args.out_figures)
    saturation = table_saturation_control(records, args.out_tables, args.out_figures)
    resolution = table_score_resolution(records, args.out_tables)

    summary = {
        "verification": verification,
        "h3_supported": h3["supported"],
        "h3_best_arm": h3["best"],
        "h3_threshold": H3_THRESHOLD,
        "h3_null_control_max_abs_diff": h3["null_control_max_abs_diff"],
        "af_rows": len(af),
        "attribution": attribution,
        "mechanism_all": [row for row in mechanism if row["region"] == "all"],
        "contrast_all": [row for row in contrast if row["region"] == "all"],
        "saturation_rows": saturation,
        "score_resolution": resolution,
        "geometry_note": GEOMETRY_NOTE,
        "precision_note": PRECISION_NOTE,
    }
    (args.out_tables / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verification["verdict"],
                "failures": verification["failures"],
                "records": verification["records_total"],
                "h3_supported": h3["supported"],
                "h3_best": h3["best"],
            },
            indent=2))
    return 0 if verification["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
