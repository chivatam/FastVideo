"""F1 aggregation: scorer-arithmetic tables and the F1.6 decision-rule verdicts.

Aggregation rules that matter for correctness, all from FOLLOWUP_SPEC:

* excesses are **paired differences** within a cell, so the median of the
  differences is reported — never the difference of medians;
* the isolation ratio is ``median(random_excess) / median(wrong_mask_excess)``,
  a ratio of medians. Per-cell quotients are not averaged, because the
  denominator is near zero in most cells and the mean of such quotients is
  dominated by the cells where the precision effect happens to vanish;
* ``wrong_mask_excess`` is **signed**. Study 1 found it can be negative — a
  perturbed mask sometimes lands marginally closer to dense than the fp64-optimal
  mask does. Clamping at zero would manufacture a positive effect out of noise, so
  the sign is preserved and the fraction of cells where the arm is *worse* is
  reported alongside;
* ``|wrong_mask_excess| / sparsification_error`` is reported as the relative
  attribution the phase actually decides on, since a mask-overlap number alone
  does not say whether the damage matters.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_aggregate.py \
        --shard "$FV_RAW_ROOT/<run-id>/p01_s1234.jsonl" \
        --tables artifacts/sparsefp4_followup/tables/f1_diagnostic \
        --stage F1-diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "artifacts/sparsefp4_followup/baseline_snapshot.json"
REFERENCE_ARM = "R0"
STUDY1_ARM = "R1"
LADDER = ("fp64", "fp32", "bf16_acc_fp32", "bf16_acc_bf16", "fp8", "nvfp4_like")

# Study 1's Phase 2 layer partition, frozen so regions are not re-chosen post hoc.
REGIONS = {
    "affected": (0, 1, 2, 27, 28, 29),
    "unaffected": (5, 6, 10, 11, 13),
    "broad": (8, 16, 20, 23, 24, 25),
}


def median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.median(clean) if clean else None


def iqr(values: Iterable[float | None]) -> float | None:
    clean = sorted(value for value in values if value is not None and not math.isnan(value))
    if len(clean) < 4:
        return None
    quantiles = statistics.quantiles(clean, n=4, method="inclusive")
    return quantiles[2] - quantiles[0]


def fraction(values: Sequence[bool]) -> float | None:
    return (sum(1 for value in values if value) / len(values)) if values else None


def region_of(layer: int) -> list[str]:
    names = [name for name, layers in REGIONS.items() if layer in layers]
    return names + ["all"]


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.4g}")
            else:
                rendered.append("" if value is None else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load(shards: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def arm_summary(rows: list[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group = grouped[key]
        first = group[0]
        wrong = [row["wrong_mask_excess"] for row in group]
        random_excess = [row["random_matched_excess"] for row in group]
        sparsification = [row["sparsification_error"] for row in group]
        relative = [
            abs(row["wrong_mask_excess"]) / row["sparsification_error"] for row in group
            if row["wrong_mask_excess"] is not None and row["sparsification_error"]
        ]
        median_wrong = median(wrong)
        median_random = median(random_excess)
        record: dict[str, Any] = dict(zip(group_keys, key, strict=False))
        record.update({
            "arm_label":
            first["arm_label"],
            "ladder_position":
            first["arithmetic_ladder_position"],
            "representation":
            first["representation_precision"],
            "native_or_simulated":
            first["native_or_simulated"],
            "n_cells":
            len(group),
            "jaccard_median":
            median(row["jaccard"] for row in group),
            "jaccard_iqr":
            iqr(row["jaccard"] for row in group),
            "jaccard_p05":
            (statistics.quantiles([row["jaccard"]
                                   for row in group], n=20, method="inclusive")[0] if len(group) >= 20 else None),
            "recall_median":
            median(row["recall"] for row in group),
            "spearman_rho_median":
            median(row.get("spearman_rho_vs_reference") for row in group),
            "frac_decisions_changed_median":
            median(row["frac_decisions_changed"] for row in group),
            "frac_query_blocks_changed_median":
            median(row["frac_query_blocks_changed"] for row in group),
            "swaps_per_query_block_median":
            median(row["swaps_per_query_block"] for row in group),
            "jaccard_pooling_only_median":
            median(row.get("jaccard_pooling_only") for row in group),
            "jaccard_vs_study1_arm_median":
            median(row.get("jaccard_vs_study1_arm") for row in group),
            "deployed_score_ties_median":
            median(row.get("deployed_score_ties") for row in group),
            "deployed_score_ties_max":
            max((row.get("deployed_score_ties") or 0 for row in group), default=None),
            "exact_ties_fp64_total":
            sum(row.get("exact_ties_fp64") or 0 for row in group),
            "reference_margin_norm_fp64_median":
            median(row["reference_margin_norm_fp64_median"] for row in group),
            "changed_pair_gap_fp64_median":
            median(row["changed_pair_gap_fp64_median"] for row in group),
            "rel_l2_median":
            median(row["rel_l2_vs_dense_bf16"] for row in group),
            "cosine_median":
            median(row["cosine_vs_dense_bf16"] for row in group),
            "sparsification_error_median":
            median(sparsification),
            "wrong_mask_excess_median":
            median_wrong,
            "wrong_mask_excess_iqr":
            iqr(wrong),
            "wrong_mask_excess_p90":
            (statistics.quantiles([value for value in wrong
                                   if value is not None], n=10, method="inclusive")[8] if len(group) >= 10 else None),
            "abs_wrong_mask_over_sparsification_median":
            median(relative),
            "abs_wrong_mask_over_sparsification_p90":
            (statistics.quantiles(relative, n=10, method="inclusive")[8] if len(relative) >= 10 else None),
            "frac_cells_arm_worse_than_reference":
            fraction([row["wrong_mask_excess"] > 0 for row in group if row["wrong_mask_excess"] is not None]),
            "random_matched_excess_median":
            median_random,
            # Ratio of medians per FOLLOWUP_SPEC, not a mean of per-cell ratios.
            "isolation_ratio_median_of_medians":
            (None if median_wrong is None or median_random is None or median_wrong == 0 else median_random /
             median_wrong),
            "isolation_ratio_abs":
            (None if median_wrong is None or median_random is None or median_wrong == 0 else abs(median_random) /
             abs(median_wrong)),
            "frac_cells_random_worse_than_arm":
            fraction([
                row["random_matched_excess"] > row["wrong_mask_excess"] for row in group
                if row["random_matched_excess"] is not None and row["wrong_mask_excess"] is not None
            ]),
            "pool_semantics":
            first["pool_semantics"],
            "score_semantics":
            first["score_semantics"],
        })
        out.append(record)
    return out


def decision_verdicts(by_arm: list[dict[str, Any]], sparsity: float) -> dict[str, Any]:
    """Evaluate F1.6 against the aggregated arms at one sparsity."""
    index = {(row["arm"], row["sparsity"]): row for row in by_arm}
    reference_study1 = index.get((STUDY1_ARM, sparsity))
    findings: list[dict[str, Any]] = []

    for arm_id in ("R2", "R3", "R4", "R5", "R4L", "R5L", "R6", "R7", "R8", "R9"):
        row = index.get((arm_id, sparsity))
        if row is None:
            continue
        relative = row["abs_wrong_mask_over_sparsification_median"]
        isolation = row["isolation_ratio_abs"]
        ratio_vs_study1 = None
        if (reference_study1 is not None and reference_study1["wrong_mask_excess_median"]
                and row["wrong_mask_excess_median"] is not None):
            ratio_vs_study1 = abs(row["wrong_mask_excess_median"]) / abs(reference_study1["wrong_mask_excess_median"])
        findings.append({
            "arm": arm_id,
            "ladder_position": row["ladder_position"],
            "arm_label": row["arm_label"],
            "jaccard_median": row["jaccard_median"],
            "abs_wrong_mask_over_sparsification_median": relative,
            "under_0_1pct_of_sparsification": (None if relative is None else relative < 0.001),
            "reaches_1pct_of_sparsification": (None if relative is None else relative >= 0.01),
            "isolation_ratio_abs": isolation,
            "random_at_least_10x_more_damaging": (None if isolation is None else isolation >= 10.0),
            "excess_ratio_vs_study1_arm_R1": ratio_vs_study1,
            "increases_excess_10x_vs_R1": (None if ratio_vs_study1 is None else ratio_vs_study1 >= 10.0),
        })

    survives = all(item["under_0_1pct_of_sparsification"] for item in findings
                   if item["under_0_1pct_of_sparsification"] is not None)
    isolation_holds = all(item["random_at_least_10x_more_damaging"] for item in findings
                          if item["random_at_least_10x_more_damaging"] is not None)
    needs_revision = any(item["reaches_1pct_of_sparsification"]
                         for item in findings if item["reaches_1pct_of_sparsification"] is not None) or any(
                             item["increases_excess_10x_vs_R1"]
                             for item in findings if item["increases_excess_10x_vs_R1"] is not None)

    if needs_revision:
        verdict = "SCOPE_REVISION_REQUIRED"
    elif survives and isolation_holds:
        verdict = "PAPER_SURVIVES_STRONGLY"
    else:
        verdict = "PARTIAL_SUPPORT"
    return {
        "sparsity": sparsity,
        "verdict": verdict,
        "all_arms_under_0_1pct_of_sparsification": survives,
        "isolation_ratio_at_least_10x_for_all_arms": isolation_holds,
        "any_arm_reaches_1pct_or_10x_R1": needs_revision,
        "per_arm": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    rows = load(args.shard)
    for row in rows:
        row["_regions"] = region_of(row["layer"])
    print(f"loaded {len(rows)} rows from {len(args.shard)} shard(s)")

    tables = args.tables
    tables.mkdir(parents=True, exist_ok=True)

    by_arm = arm_summary(rows, ("arm", "sparsity"))
    ladder_order = {position: index for index, position in enumerate(LADDER)}
    by_arm.sort(key=lambda row: (row["sparsity"], ladder_order.get(row["ladder_position"], 99), row["arm"]))
    write_table(tables / "table1_arm_headline", by_arm)

    by_region: list[dict[str, Any]] = []
    for name in ("affected", "unaffected", "broad", "all"):
        subset = [row for row in rows if name in row["_regions"]]
        for record in arm_summary(subset, ("arm", "sparsity")):
            record["region"] = name
            by_region.append(record)
    by_region.sort(key=lambda row: (row["sparsity"], row["region"], ladder_order.get(row["ladder_position"], 99)))
    write_table(tables / "table2_arm_by_region", by_region)

    write_table(tables / "table3_arm_by_timestep", arm_summary(rows, ("arm", "timestep", "sparsity")))
    write_table(tables / "table4_arm_by_cfg_branch", arm_summary(rows, ("arm", "cfg_branch", "sparsity")))
    write_table(tables / "table5_arm_by_layer", arm_summary(rows, ("arm", "layer", "sparsity")))

    verdicts = [decision_verdicts(by_arm, sparsity) for sparsity in sorted({row["sparsity"] for row in rows})]

    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.is_file() else {}
    frozen = {entry["metric"]: entry["value"] for family in baseline.get("families", {}).values() for entry in family}
    comparison: list[dict[str, Any]] = []
    for sparsity in sorted({row["sparsity"] for row in rows}):
        study1_arm = next((row for row in by_arm if row["arm"] == STUDY1_ARM and row["sparsity"] == sparsity), None)
        if study1_arm is None:
            continue
        comparison.append({
            "sparsity":
            sparsity,
            "quantity":
            "mask_jaccard_median[router=nvfp4]",
            "followup_R1_value":
            study1_arm["jaccard_median"],
            "study1_frozen_value":
            frozen.get(f"mask_jaccard_median[sparsity={sparsity},router=nvfp4]"),
            "note": ("R1 reproduces study 1's condition exactly (nvfp4 representation, fp64 scorer); "
                     "any gap is prompt/layer coverage, not method"),
        })
        comparison.append({
            "sparsity":
            sparsity,
            "quantity":
            "isolation_ratio",
            "followup_R1_value":
            study1_arm["isolation_ratio_abs"],
            "study1_frozen_value":
            frozen.get(f"random_over_quantization_ratio[sparsity={sparsity},region=all]"),
            "note":
            "ratio of medians in both studies",
        })
        comparison.append({
            "sparsity": sparsity,
            "quantity": "wrong_mask_excess_median",
            "followup_R1_value": study1_arm["wrong_mask_excess_median"],
            "study1_frozen_value": frozen.get(f"wrong_mask_D_minus_C[sparsity={sparsity}]"),
            "note": "signed paired difference",
        })
    write_table(tables / "table6_r1_vs_study1_baseline", comparison)

    summary = {
        "stage":
        args.stage,
        "shards": [str(shard) for shard in args.shard],
        "n_rows":
        len(rows),
        "run_ids":
        sorted({row["run_id"]
                for row in rows}),
        "prompts":
        sorted({row["prompt_id"]
                for row in rows}),
        "seeds":
        sorted({row["seed"]
                for row in rows}),
        "geometry":
        sorted({row["geometry"]
                for row in rows}),
        "arms":
        sorted({row["arm"]
                for row in rows}),
        "layers":
        sorted({row["layer"]
                for row in rows}),
        "timesteps":
        sorted({row["timestep"]
                for row in rows}),
        "sparsities":
        sorted({row["sparsity"]
                for row in rows}),
        "worker_tf32_state":
        sorted({(row.get("worker_allow_tf32_matmul"), row.get("worker_float32_matmul_precision"))
                for row in rows}),
        "decision_rules_F1_6":
        verdicts,
        "r1_vs_study1_baseline":
        comparison,
    }
    (tables / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\nwrote tables to {tables}")
    for verdict in verdicts:
        print(f"  sparsity={verdict['sparsity']}: {verdict['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
