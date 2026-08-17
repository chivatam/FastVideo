"""F2 aggregation: VSA-selector tables and the F2.8 decision-rule verdicts.

Aggregation follows F1's rules (paired within-cell differences, ratio-of-medians for
isolation, signed excess) — see ``f1_aggregate.py`` for why each is required. What is
specific to F2:

* The baseline is the **deployed VSA selector** (``V0``), not an fp64 ideal. "Wrong
  mask excess" therefore means "damage relative to what VSA actually ships", which is
  the quantity the paper's claim is about.
* ``V0_FP64`` is the *rescue* arm: it routes with an fp64 selector and executes the
  same VSA kernel. F2.8 asks whether that rescue is worth <1% of total VSA sparse
  error, so the rescue is reported as a signed fraction of the sparsification error,
  and its sign matters — a "rescue" that makes things worse is evidence against a
  precision-limited router.
* ``VC_GATE_NVFP4`` must be an exact null. It is reported in the tables as a visible
  invariant rather than hidden in the validator.
* The kernel's own budget deviation on k-th-value ties is carried through so no table
  silently averages over it.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f2_aggregate.py \
        --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
        --tables artifacts/sparsefp4_followup/tables/f2_full --stage F2-full
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts/sparsefp4_followup/configs"))

from f1_aggregate import REGIONS, fraction, iqr, median, region_of, write_table  # noqa: E402

DEPLOYED_ARM = "V0"
RESCUE_ARM = "V0_FP64"
GATE_ARM = "VC_GATE_NVFP4"
# Arms whose masks are derived from a *degraded* selector, i.e. the ones F2.8's
# "<0.1% of sparsification error" bound is about. V0 is the baseline, V0_FP64 is the
# rescue, VC is an invariant check, VD is a tie-break contrast — none are degradations.
DEGRADED_ARMS = ("VA_FP8", "VA_NVFP4", "VB_BF16_LOW", "VA_NVFP4_VB_FP64")


def signed_relative(rows: list[dict[str, Any]], numerator: str) -> list[float]:
    """Per-cell ``numerator / sparsification_error``, sign preserved."""
    return [
        row[numerator] / row["sparsification_error"] for row in rows
        if row.get(numerator) is not None and row.get("sparsification_error")
    ]


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
        median_wrong = median(wrong)
        median_random = median(random_excess)
        relative_abs = [
            abs(row["wrong_mask_excess"]) / row["sparsification_error"] for row in group
            if row["wrong_mask_excess"] is not None and row.get("sparsification_error")
        ]
        relative_signed = signed_relative(group, "wrong_mask_excess")

        record: dict[str, Any] = dict(zip(group_keys, key, strict=False))
        record.update({
            "intervention_kind":
            first["intervention_kind"],
            "purpose":
            first["purpose"],
            "representation":
            first["representation_precision"],
            "pool":
            first["pool_precision"],
            "score":
            first["score_precision"],
            "selection_rule":
            first["selection_rule"],
            "n_cells":
            len(group),
            "jaccard_median":
            median(row["jaccard"] for row in group),
            "jaccard_iqr":
            iqr(row["jaccard"] for row in group),
            "recall_median":
            median(row["recall"] for row in group),
            "spearman_rho_vs_exact_median":
            median(row.get("spearman_rho_vs_exact") for row in group),
            "swaps_per_query_block_median":
            median(row["swaps_per_query_block"] for row in group),
            "frac_query_blocks_changed_median":
            median(row["frac_query_blocks_changed"] for row in group),
            "frac_decisions_changed_median":
            median(row["frac_decisions_changed"] for row in group),
            "frac_cells_mask_identical_to_deployed":
            fraction([bool(row.get("mask_identical_to_deployed")) for row in group]),
            "changed_pair_gap_fp64_median":
            median(row["changed_pair_gap_fp64_median"] for row in group),
            "reference_margin_norm_fp64_median":
            median(row["reference_margin_norm_fp64_median"] for row in group),
            "rel_l2_median":
            median(row["rel_l2_vs_dense"] for row in group),
            "cosine_median":
            median(row["cosine_vs_dense"] for row in group),
            "sparsification_error_median":
            median(row["sparsification_error"] for row in group),
            "wrong_mask_excess_median":
            median_wrong,
            "wrong_mask_excess_iqr":
            iqr(wrong),
            "signed_wrong_mask_over_sparsification_median":
            median(relative_signed),
            "abs_wrong_mask_over_sparsification_median":
            median(relative_abs),
            "abs_wrong_mask_over_sparsification_p90":
            (statistics.quantiles(relative_abs, n=10, method="inclusive")[8] if len(relative_abs) >= 10 else None),
            "frac_cells_arm_worse_than_deployed":
            fraction([row["wrong_mask_excess"] > 0 for row in group if row["wrong_mask_excess"] is not None]),
            "random_matched_excess_median":
            median_random,
            "isolation_ratio_abs":
            (None if median_wrong is None or median_random is None or median_wrong == 0 else abs(median_random) /
             abs(median_wrong)),
            "frac_cells_random_worse_than_arm":
            fraction([
                row["random_matched_excess"] > row["wrong_mask_excess"] for row in group
                if row["random_matched_excess"] is not None and row["wrong_mask_excess"] is not None
            ]),
            "representation_saturation_frac_median":
            median(row.get("representation_saturation_frac") for row in group),
            "deployed_score_ties_median":
            median(row.get("deployed_score_ties") for row in group),
            "exact_ties_fp64_total":
            sum(row.get("exact_ties_fp64") or 0 for row in group),
            "selector_budget_violating_rows_total":
            sum(row.get("selector_budget_violating_rows") or 0 for row in group),
            "selector_budget_violating_frac_max":
            max((row.get("selector_budget_violating_frac") or 0.0 for row in group), default=None),
            "pool_semantics":
            first["pool_semantics"],
            "score_semantics":
            first["score_semantics"],
            "select_semantics":
            first["select_semantics"],
        })
        out.append(record)
    return out


def load_unresolved_thresholds() -> list[dict[str, Any]]:
    """F4's list of (arm, sparsity) whose isolation interval spans the 10x threshold.

    Empty if F4 has not been run yet, in which case verdicts are reported from point
    estimates alone and the summary says so.
    """
    path = REPO_ROOT / "artifacts/sparsefp4_followup/raw/f4_gates.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        item for item in payload.get("thresholds_not_resolved", {}).get("isolation_10x", [])
        if item.get("phase") == "F2"
    ]


def decision_verdicts(by_arm: list[dict[str, Any]], rows: list[dict[str, Any]], sparsity: float,
                      unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate F2.8 at one sparsity."""
    index = {(row["arm"], row["sparsity"]): row for row in by_arm}
    findings: list[dict[str, Any]] = []

    for arm_id in DEGRADED_ARMS:
        row = index.get((arm_id, sparsity))
        if row is None:
            continue
        relative = row["abs_wrong_mask_over_sparsification_median"]
        isolation = row["isolation_ratio_abs"]
        # "Changes selections at all" is a precondition of the paper's framing: if a
        # low-precision selector produced an identical mask there would be no routing
        # effect to characterize in either direction.
        changes_selections = (row["jaccard_median"] is not None and row["jaccard_median"] < 1.0)
        findings.append({
            "arm": arm_id,
            "purpose": row["purpose"],
            "intervention_kind": row["intervention_kind"],
            "jaccard_median": row["jaccard_median"],
            "changes_selections": changes_selections,
            "abs_wrong_mask_over_sparsification_median": relative,
            "under_0_1pct_of_sparsification": (None if relative is None else relative < 0.001),
            "reaches_1pct_of_sparsification": (None if relative is None else relative >= 0.01),
            "isolation_ratio_abs": isolation,
            "random_at_least_10x_more_damaging": (None if isolation is None else isolation >= 10.0),
        })

    # Rescue: what does routing at fp64 buy, as a fraction of total VSA sparse error?
    rescue = index.get((RESCUE_ARM, sparsity))
    rescue_rows = [row for row in rows if row["arm"] == RESCUE_ARM and row["sparsity"] == sparsity]
    rescue_signed = signed_relative(rescue_rows, "wrong_mask_excess")
    rescue_median_signed = median(rescue_signed)
    # A negative signed excess means the fp64 selector beat the deployed one, i.e. a
    # genuine rescue; magnitude is what the <1% / >=5% thresholds compare against.
    rescue_magnitude = None if rescue_median_signed is None else abs(rescue_median_signed)
    rescue_consistent = fraction([value < 0 for value in rescue_signed])
    rescue_report = {
        "arm":
        RESCUE_ARM,
        "signed_excess_over_sparsification_median":
        rescue_median_signed,
        "rescue_magnitude_over_sparsification_median":
        rescue_magnitude,
        "frac_cells_fp64_router_better_than_deployed":
        rescue_consistent,
        "jaccard_vs_deployed_median":
        None if rescue is None else rescue["jaccard_median"],
        "under_1pct_of_total_sparse_error": (None if rescue_magnitude is None else rescue_magnitude < 0.01),
        "at_least_5pct_and_consistent": (None if rescue_magnitude is None or rescue_consistent is None else
                                         (rescue_magnitude >= 0.05 and rescue_consistent >= 0.5)),
    }

    gate = index.get((GATE_ARM, sparsity))
    gate_report = {
        "arm": GATE_ARM,
        "frac_cells_mask_identical_to_deployed":
        None if gate is None else gate["frac_cells_mask_identical_to_deployed"],
        "gate_is_outside_selection_path":
        (None if gate is None else gate["frac_cells_mask_identical_to_deployed"] == 1.0),
        "meaning": "confirms VSA_GATE_MAP.md: quantizing gate_compress cannot move the mask",
    }

    changes = [item["changes_selections"] for item in findings]
    under = [
        item["under_0_1pct_of_sparsification"] for item in findings
        if item["under_0_1pct_of_sparsification"] is not None
    ]
    isolation_flags = [
        item["random_at_least_10x_more_damaging"] for item in findings
        if item["random_at_least_10x_more_damaging"] is not None
    ]
    reaches = [
        item["reaches_1pct_of_sparsification"] for item in findings
        if item["reaches_1pct_of_sparsification"] is not None
    ]

    generalizes = (any(changes) and all(under) and all(isolation_flags)
                   and rescue_report["under_1pct_of_total_sparse_error"] is True)
    must_revise = (any(reaches) or rescue_report["at_least_5pct_and_consistent"] is True
                   or (isolation_flags and not all(isolation_flags)))

    if must_revise:
        verdict = "SCOPE_REVISION_REQUIRED"
    elif generalizes:
        verdict = "GENERALIZES_TO_VSA"
    else:
        verdict = "PARTIAL_SUPPORT"

    # A verdict driven by an isolation ratio whose bootstrap interval spans 10x is not
    # actually determined by the evidence. F4 computes those intervals; when it says a
    # threshold is unresolved, the verdict is downgraded to explicitly indeterminate
    # rather than reported as though the point estimate settled it.
    unresolved_arms = sorted({item["arm"] for item in unresolved if item["sparsity"] == sparsity})
    decisive_arms = sorted({item["arm"] for item in findings if item["random_at_least_10x_more_damaging"] is False})
    verdict_hinges_on_unresolved = bool(unresolved_arms) and set(decisive_arms) <= set(unresolved_arms)
    if verdict_hinges_on_unresolved and verdict == "SCOPE_REVISION_REQUIRED" and not any(reaches):
        verdict = "INDETERMINATE_ISOLATION_THRESHOLD"

    return {
        "sparsity": sparsity,
        "verdict": verdict,
        "verdict_hinges_on_unresolved_threshold": verdict_hinges_on_unresolved,
        "arms_with_unresolved_isolation_interval": unresolved_arms,
        "arms_driving_a_revision_verdict": decisive_arms,
        "low_precision_selector_changes_selections": any(changes),
        "all_degraded_arms_under_0_1pct": bool(under) and all(under),
        "isolation_at_least_10x_for_all_degraded_arms": bool(isolation_flags) and all(isolation_flags),
        "any_degraded_arm_reaches_1pct": any(reaches),
        "rescue": rescue_report,
        "gate_invariant": gate_report,
        "per_arm": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for shard in args.shard:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    for row in rows:
        row["_regions"] = region_of(row["layer"])
    print(f"loaded {len(rows)} rows from {len(args.shard)} shard(s)")
    if not rows:
        raise SystemExit("no rows loaded")

    tables = args.tables
    tables.mkdir(parents=True, exist_ok=True)

    by_arm = arm_summary(rows, ("arm", "sparsity"))
    by_arm.sort(key=lambda row: (row["sparsity"], str(row["intervention_kind"]), str(row["arm"])))
    write_table(tables / "table1_vsa_arm_headline", by_arm)

    by_region: list[dict[str, Any]] = []
    for name in (*REGIONS, "all"):
        subset = [row for row in rows if name in row["_regions"]]
        for record in arm_summary(subset, ("arm", "sparsity")):
            record["region"] = name
            by_region.append(record)
    by_region.sort(key=lambda row: (row["sparsity"], str(row["region"]), str(row["arm"])))
    write_table(tables / "table2_vsa_arm_by_region", by_region)

    write_table(tables / "table3_vsa_arm_by_timestep", arm_summary(rows, ("arm", "timestep", "sparsity")))
    write_table(tables / "table4_vsa_arm_by_cfg_branch", arm_summary(rows, ("arm", "cfg_branch", "sparsity")))
    write_table(tables / "table5_vsa_arm_by_layer", arm_summary(rows, ("arm", "layer", "sparsity")))
    write_table(tables / "table6_vsa_arm_by_prompt", arm_summary(rows, ("arm", "prompt_id", "sparsity")))

    unresolved = load_unresolved_thresholds()
    verdicts = [
        decision_verdicts(by_arm, rows, sparsity, unresolved) for sparsity in sorted({row["sparsity"]
                                                                                      for row in rows})
    ]

    budget_rows = sum(row.get("selector_budget_violating_rows") or 0 for row in rows)
    budget_by_arm: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("selector_budget_violating_rows"):
            budget_by_arm[row["arm"]] += int(row["selector_budget_violating_rows"])

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
        "geometry":
        sorted({row["geometry"]
                for row in rows}),
        "trajectory_backend":
        sorted({row["model_trajectory_backend"]
                for row in rows}),
        "routing_interface":
        sorted({row["routing_interface"]
                for row in rows}),
        "vsa_sparsity_for_execution":
        sorted({row["vsa_sparsity_for_execution"]
                for row in rows}),
        "worker_autocast":
        sorted({(row.get("worker_autocast_enabled"), row.get("worker_autocast_dtype"))
                for row in rows}),
        "kernel_budget_deviation": {
            "total_violating_selector_rows": budget_rows,
            "by_arm": dict(sorted(budget_by_arm.items())),
            "cause": "fused_topk_mask fp32 bisection cannot resolve a k-th-value tie",
            "lesson": ".agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md",
        },
        "decision_rules_F2_8":
        verdicts,
        "isolation_intervals_source": ("artifacts/sparsefp4_followup/raw/f4_gates.json"
                                       if unresolved else "not available — verdicts reflect point estimates only"),
        "arms_with_unresolved_isolation_interval":
        unresolved,
    }
    (tables / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\nwrote tables to {tables}")
    for verdict in verdicts:
        print(f"  sparsity={verdict['sparsity']}: {verdict['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
