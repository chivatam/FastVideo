"""F2 gate: verify the VSA-selector record lattice and its reference invariants.

Run before aggregating F2. The checks here are F1's, adapted to what differs about
F2 — the selector under study is VSA's own kernel rather than a clean ``topk``, so
some invariants that were hard errors in F1 become quantities that must be small
and *recorded*:

- ``V0`` must be an exact null against itself (it is the deployed selector).
- ``VC_GATE_NVFP4`` must produce a mask bit-identical to ``V0``. This is the
  falsification test for ``VSA_GATE_MAP.md``'s central claim that ``gate_compress``
  never reaches the selector. If this check fails, the gate map is wrong and the F2
  interpretation collapses — so it is a hard gate, not a diagnostic.
- The retained budget must match across arms *except* where VSA's own
  ``fused_topk_mask`` over-selects on k-th-value ties (see
  ``.agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md``). That rate must
  stay negligible, and it is reported rather than assumed away.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f2_validate.py \
        --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
        --out artifacts/sparsefp4_followup/raw/f2_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "run_id",
    "arm",
    "layer",
    "timestep",
    "cfg_branch",
    "head",
    "sparsity",
    "retained_k",
    "jaccard",
    "recall",
    "num_swaps",
    "rel_l2_vs_dense",
    "sparsification_error",
    "wrong_mask_excess",
    "representation_precision",
    "pool_precision",
    "score_precision",
    "selection_rule",
    "intervention_kind",
    "pool_semantics",
    "score_semantics",
    "select_semantics",
    "gate_compress_in_selection_path",
    "mask_deployed_hash",
    "mask_candidate_hash",
    "model_trajectory_backend",
    "routing_interface",
    "selector_budget_violating_rows",
    "selector_budget_violating_frac",
)
DEPLOYED_ARM = "V0"
EXACT_ARM = "V0_FP64"
GATE_ARM = "VC_GATE_NVFP4"
# Observed rate is ~1 row in 7488 at one layer/timestep; 1e-3 leaves headroom while
# still failing loudly if the selector's budget ever breaks structurally.
MAX_BUDGET_VIOLATION_FRAC = 1e-3


def load_rows(shards: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    malformed.append({
                        "shard": shard.name,
                        "line": line_number,
                        "reason": f"json: {error}",
                        "bytes": len(line),
                    })
                    continue
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    malformed.append({"shard": shard.name, "line": line_number, "reason": f"missing {missing}"})
                    continue
                rows.append(record)
    return rows, malformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, malformed = load_rows(args.shard)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail is not None else ""))

    check("no_malformed_rows", not malformed, malformed[:5] if malformed else f"{len(rows)} rows parsed")
    if not rows:
        check("rows_present", False, "no parseable records")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"verdict": "FAIL", "checks": checks}, indent=2) + "\n", encoding="utf-8")
        return 1

    arms = sorted({row["arm"] for row in rows})
    prompts = sorted({row["prompt_id"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    timesteps = sorted({row["timestep"] for row in rows})
    branches = sorted({row["cfg_branch"] for row in rows})
    heads = sorted({row["head"] for row in rows})
    sparsities = sorted({row["sparsity"] for row in rows})
    expected = (len(arms) * len(prompts) * len(layers) * len(timesteps) * len(branches) * len(heads) * len(sparsities))

    def key_of(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["arm"], row["prompt_id"], row["layer"], row["timestep"], row["cfg_branch"], row["head"],
                row["sparsity"])

    seen = Counter(key_of(row) for row in rows)
    duplicates = {str(key): count for key, count in seen.items() if count > 1}
    holes = [(arm, prompt, layer, timestep, branch, head, sparsity) for arm in arms for prompt in prompts
             for layer in layers for timestep in timesteps for branch in branches for head in heads
             for sparsity in sparsities if (arm, prompt, layer, timestep, branch, head, sparsity) not in seen]
    check(
        "lattice_complete", not holes, {
            "expected":
            expected,
            "observed":
            len(rows),
            "n_holes":
            len(holes),
            "example_holes": [
                dict(zip(("arm", "prompt", "layer", "timestep", "cfg", "head", "sparsity"), hole, strict=False))
                for hole in holes[:8]
            ],
        })
    check("no_duplicate_cells", not duplicates, list(duplicates.items())[:5])

    deployed_rows = [row for row in rows if row["arm"] == DEPLOYED_ARM]
    deployed_exact = all(
        row["jaccard"] == 1.0 and row["num_swaps"] == 0 and row["recall"] == 1.0 and row["wrong_mask_excess"] == 0.0
        and row["null_control"] is True and row["mask_deployed_hash"] == row["mask_candidate_hash"]
        for row in deployed_rows)
    check(
        "deployed_arm_is_exact_null", deployed_exact, {
            "n_rows": len(deployed_rows),
            "jaccards": sorted({row["jaccard"]
                                for row in deployed_rows})[:5],
            "excesses": sorted({row["wrong_mask_excess"]
                                for row in deployed_rows})[:5],
        })

    # The gate map's falsification test. gate_compress is genuinely NVFP4-quantized in
    # this arm; if the mask still moves, the claim that the gate is outside the
    # selection path is false and F2's framing is invalid.
    gate_rows = [row for row in rows if row["arm"] == GATE_ARM]
    gate_identical = all(row.get("mask_identical_to_deployed") is True for row in gate_rows)
    gate_quantized = all(row.get("gate_compress_quantized") is True for row in gate_rows)
    gate_saturated = [row.get("gate_compress_saturation_frac") for row in gate_rows]
    check(
        "gate_quantization_cannot_move_the_mask",
        bool(gate_rows) and gate_identical and gate_quantized, {
            "n_rows":
            len(gate_rows),
            "all_masks_identical_to_deployed":
            gate_identical,
            "all_rows_actually_quantized_the_gate":
            gate_quantized,
            "gate_saturation_frac_range":
            ([min(x for x in gate_saturated if x is not None),
              max(x for x in gate_saturated if x is not None)] if any(x is not None for x in gate_saturated) else None),
            "meaning":
            "falsifies VSA_GATE_MAP.md if it fails",
        })
    check("gate_declared_outside_selection_path", all(row["gate_compress_in_selection_path"] is False for row in rows),
          None)

    # Budget: equal across arms except for the kernel's documented tie over-selection.
    by_cell: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        by_cell[(row["prompt_id"], row["layer"], row["timestep"], row["cfg_branch"], row["head"],
                 row["sparsity"])].add(row["retained_k"])
    unequal = {str(key): sorted(values) for key, values in by_cell.items() if len(values) > 1}
    check("retained_k_identical_across_arms", not unequal, list(unequal.items())[:5])

    violating_rows = sum(row["selector_budget_violating_rows"] for row in rows)
    worst = max(rows, key=lambda row: row["selector_budget_violating_frac"])
    worst_frac = float(worst["selector_budget_violating_frac"])
    by_arm: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["selector_budget_violating_rows"]:
            by_arm[row["arm"]] += 1
    check(
        "kernel_budget_deviation_negligible", worst_frac <= MAX_BUDGET_VIOLATION_FRAC, {
            "worst_row_violating_frac": worst_frac,
            "threshold": MAX_BUDGET_VIOLATION_FRAC,
            "worst_cell": {
                key: worst[key]
                for key in ("arm", "layer", "timestep", "cfg_branch", "sparsity")
            },
            "records_with_any_violation": dict(sorted(by_arm.items())),
            "total_violating_selector_rows": violating_rows,
            "cause": "fused_topk_mask fp32 bisection cannot resolve a k-th-value tie",
            "lesson": ".agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md",
        })

    hash_by_cell: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        hash_by_cell[(row["prompt_id"], row["layer"], row["timestep"], row["cfg_branch"], row["head"],
                      row["sparsity"])].add(row["mask_deployed_hash"])
    inconsistent = {str(key): sorted(values) for key, values in hash_by_cell.items() if len(values) > 1}
    check("deployed_mask_shared_within_cell", not inconsistent, list(inconsistent.items())[:5])

    mismatched = [{
        "arm": row["arm"],
        "layer": row["layer"],
        "num_swaps": row["num_swaps"],
        "random_matched_num_swaps": row["random_matched_num_swaps"],
    } for row in rows if row["arm"] != DEPLOYED_ARM and row.get("random_matched_num_swaps") != row["num_swaps"]]
    check("matched_random_swap_count_equals_arm", not mismatched, mismatched[:5])

    total_ties = sum(row.get("exact_ties_fp64") or 0 for row in rows)
    total_denominator = sum(row.get("tie_denominator_query_blocks") or 0 for row in rows)
    tie_rate = total_ties / total_denominator if total_denominator else 0.0
    check("fp64_shadow_ties_negligible", tie_rate < 1e-4, {
        "total_ties": total_ties,
        "denominator_query_blocks": total_denominator,
        "tie_rate": tie_rate,
    })

    # The trajectory must be real VSA — that is F2's entire external-validity claim.
    trajectories = sorted({row["model_trajectory_backend"] for row in rows})
    interfaces = sorted({row["routing_interface"] for row in rows})
    check("trajectory_is_real_vsa", all("VIDEO_SPARSE_ATTN" in value for value in trajectories), trajectories)
    check("routing_interface_is_actual_vsa", all("actual_vsa" in value for value in interfaces), interfaces)

    # The exact arm must actually be higher precision than the deployed one, else
    # "exact" is a label with nothing behind it.
    exact_rows = [row for row in rows if row["arm"] == EXACT_ARM]
    exact_declared = all("fp64" in row["pool_semantics"] and "fp64" in row["score_semantics"] for row in exact_rows)
    check("exact_arm_is_fp64_end_to_end",
          bool(exact_rows) and exact_declared, {
              "n_rows": len(exact_rows),
              "pool_semantics": sorted({row["pool_semantics"]
                                        for row in exact_rows}),
          })

    # A bf16 selection rule must be the kernel's own; higher precision must not be
    # pushed through the bf16 kernel (that would silently discard the intervention).
    rule_problems = [{
        "arm": row["arm"],
        "score_precision": row["score_precision"],
        "selection_rule": row["selection_rule"],
    } for row in rows if row["score_precision"] in ("fp32", "fp64") and row["selection_rule"] == "kernel"]
    check("higher_precision_arms_do_not_use_the_bf16_kernel", not rule_problems, rule_problems[:5])

    numeric = ("jaccard", "recall", "rel_l2_vs_dense", "sparsification_error", "wrong_mask_excess",
               "changed_pair_gap_fp64_median", "reference_margin_fp64_median", "spearman_rho_vs_exact")
    nan_counts = {
        field: sum(1 for row in rows if isinstance(row.get(field), float) and math.isnan(row[field]))
        for field in numeric
    }
    check("no_nans_in_reportable_columns", not any(nan_counts.values()), nan_counts)

    autocast_states = sorted({(row.get("worker_autocast_enabled"), row.get("worker_autocast_dtype")) for row in rows})
    check("ambient_autocast_state_recorded", all(row.get("worker_autocast_dtype") is not None for row in rows),
          {"observed_states": [list(state) for state in autocast_states]})

    n_failed = sum(1 for item in checks if not item["passed"])
    payload = {
        "verdict": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "shards": [str(shard) for shard in args.shard],
        "n_rows": len(rows),
        "n_expected": expected,
        "lattice": {
            "arms": arms,
            "prompts": prompts,
            "layers": layers,
            "timesteps": timesteps,
            "cfg_branches": branches,
            "heads": heads,
            "sparsities": sparsities,
        },
        "kernel_budget_deviation": {
            "total_violating_selector_rows": violating_rows,
            "worst_row_violating_frac": worst_frac,
            "records_with_any_violation_by_arm": dict(sorted(by_arm.items())),
        },
        "n_checks": len(checks),
        "n_failed": n_failed,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{payload['verdict']}: {len(checks) - n_failed}/{len(checks)} checks passed "
          f"({len(rows)}/{expected} rows)")
    print(f"wrote {args.out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
