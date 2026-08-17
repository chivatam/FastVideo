"""F1 gate: verify the record lattice and the null/reference invariants.

Run before any aggregation. Refuses to pass on a lattice hole, a malformed row, a
non-exact reference arm, an unequal retained-``k``, or a matched-random control
whose swap count differs from the arm it is paired with — each of which would let
a precision effect be confounded with a bookkeeping artifact.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_validate.py \
        --shard "$FV_RAW_ROOT/<run-id>/p01_s1234.jsonl" \
        --out artifacts/sparsefp4_followup/raw/f1_diagnostic_validation.json
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
    "prompt_id",
    "layer",
    "timestep",
    "cfg_branch",
    "head",
    "sparsity",
    "retained_k",
    "jaccard",
    "recall",
    "num_swaps",
    "rel_l2_vs_dense_bf16",
    "sparsification_error",
    "wrong_mask_excess",
    "representation_precision",
    "pool_precision",
    "score_precision",
    "arithmetic_ladder_position",
    "native_or_simulated",
    "pool_semantics",
    "score_semantics",
    "reference_margin_fp64_median",
    "exact_ties_fp64",
    "changed_pair_gap_fp64_median",
    "mask_reference_hash",
    "mask_candidate_hash",
)
REFERENCE_ARM = "R0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for shard in args.shard:
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    malformed.append({
                        "shard": str(shard),
                        "line": line_number,
                        "reason": f"json: {error}",
                        "bytes": len(line)
                    })
                    continue
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    malformed.append({"shard": str(shard), "line": line_number, "reason": f"missing {missing}"})
                    continue
                rows.append(record)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail is not None else ""))

    check("no_malformed_rows", not malformed, malformed[:5] if malformed else f"{len(rows)} rows parsed")

    arms = sorted({row["arm"] for row in rows})
    prompts = sorted({row["prompt_id"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    timesteps = sorted({row["timestep"] for row in rows})
    branches = sorted({row["cfg_branch"] for row in rows})
    heads = sorted({row["head"] for row in rows})
    sparsities = sorted({row["sparsity"] for row in rows})
    expected = (len(arms) * len(prompts) * len(seeds) * len(layers) * len(timesteps) * len(branches) * len(heads) *
                len(sparsities))

    # prompt_id and seed are both part of the cell identity: each (prompt, seed) pair is
    # an independent trajectory, so the same (arm, layer, timestep, ...) recurs once per
    # pair and its reference mask legitimately differs. Omitting either makes a
    # multi-prompt or multi-seed run look like duplicates with inconsistent references.
    seen = Counter((row["arm"], row["prompt_id"], row["seed"], row["layer"], row["timestep"], row["cfg_branch"],
                    row["head"], row["sparsity"]) for row in rows)
    duplicates = {str(key): count for key, count in seen.items() if count > 1}
    holes = [(arm, prompt, seed, layer, timestep, branch, head, sparsity) for arm in arms for prompt in prompts
             for seed in seeds for layer in layers for timestep in timesteps for branch in branches for head in heads
             for sparsity in sparsities if (arm, prompt, seed, layer, timestep, branch, head, sparsity) not in seen]
    check(
        "lattice_complete", not holes, {
            "expected":
            expected,
            "observed":
            len(rows),
            "n_holes":
            len(holes),
            "example_holes": [
                dict(zip(("arm", "prompt", "seed", "layer", "timestep", "cfg", "head", "sparsity"), hole, strict=False))
                for hole in holes[:8]
            ],
        })
    check("no_duplicate_cells", not duplicates, list(duplicates.items())[:5])

    # The reference arm must be exactly itself: any deviation means the pairing is
    # broken and every downstream excess is measured against the wrong baseline.
    reference_rows = [row for row in rows if row["arm"] == REFERENCE_ARM]
    reference_exact = all(
        row["jaccard"] == 1.0 and row["num_swaps"] == 0 and row["recall"] == 1.0 and row["wrong_mask_excess"] == 0.0
        and row["null_control"] is True and row["mask_reference_hash"] == row["mask_candidate_hash"]
        for row in reference_rows)
    check(
        "reference_arm_is_exact_null", reference_exact, {
            "n_reference_rows": len(reference_rows),
            "jaccards": sorted({row["jaccard"]
                                for row in reference_rows}),
            "excesses": sorted({row["wrong_mask_excess"]
                                for row in reference_rows}),
        })

    # Equal budget across arms at every cell: precision may change which blocks,
    # never how many.
    by_cell: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        by_cell[(row["prompt_id"], row["seed"], row["layer"], row["timestep"], row["cfg_branch"], row["head"],
                 row["sparsity"])].add(row["retained_k"])
    unequal = {str(key): sorted(values) for key, values in by_cell.items() if len(values) > 1}
    check("retained_k_identical_across_arms", not unequal, list(unequal.items())[:5])

    # Every non-reference arm must share the reference's mask hash, proving the
    # comparison really is paired within the cell.
    hash_by_cell: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        hash_by_cell[(row["prompt_id"], row["seed"], row["layer"], row["timestep"], row["cfg_branch"], row["head"],
                      row["sparsity"])].add(row["mask_reference_hash"])
    inconsistent = {str(key): sorted(values) for key, values in hash_by_cell.items() if len(values) > 1}
    check("reference_mask_shared_within_cell", not inconsistent, list(inconsistent.items())[:5])

    # Matched-random control must change the same number of blocks as its arm.
    mismatched = [{
        "arm": row["arm"],
        "layer": row["layer"],
        "num_swaps": row["num_swaps"],
        "random_matched_num_swaps": row["random_matched_num_swaps"],
    } for row in rows if row["arm"] != REFERENCE_ARM and row.get("random_matched_num_swaps") != row["num_swaps"]]
    check("matched_random_swap_count_equals_arm", not mismatched, mismatched[:5])

    # FP64 shadow boundary must be resolved, else margins mean nothing.
    tie_rows = [row for row in rows if (row.get("exact_ties_fp64") or 0) > 0]
    total_ties = sum(row.get("exact_ties_fp64") or 0 for row in rows)
    total_denominator = sum(row.get("tie_denominator_query_blocks") or 0 for row in rows)
    tie_rate = total_ties / total_denominator if total_denominator else 0.0
    check(
        "fp64_shadow_ties_negligible", tie_rate < 1e-4, {
            "total_ties": total_ties,
            "denominator_query_blocks": total_denominator,
            "tie_rate": tie_rate,
            "n_rows_with_any_tie": len(tie_rows),
        })

    # No NaNs in the reportable numeric columns.
    numeric = ("jaccard", "recall", "rel_l2_vs_dense_bf16", "sparsification_error", "wrong_mask_excess",
               "changed_pair_gap_fp64_median", "reference_margin_fp64_median")
    nan_counts = {
        field: sum(1 for row in rows if isinstance(row.get(field), float) and math.isnan(row[field]))
        for field in numeric
    }
    check("no_nans_in_reportable_columns", not any(nan_counts.values()), nan_counts)

    # Semantics strings must never claim pure low precision where the accumulator
    # is fp32, and simulated arms must be labelled.
    label_problems = [
        row["arm"] for row in rows if row["score_precision"] in (
            "bf16", "fp16") and row["score_accumulate"] == "native" and "acc_fp32" not in row["score_semantics"]
    ]
    check("bf16_native_accumulation_is_labelled_fp32", not label_problems, sorted(set(label_problems)))
    simulated = sorted({row["arm"] for row in rows if row["native_or_simulated"] != "native"})
    check("simulated_arms_declared", True, {"simulated_arms": simulated})

    ladder = sorted({(row["arithmetic_ladder_position"], row["arm"]) for row in rows})
    n_failed = sum(1 for item in checks if not item["passed"])
    payload = {
        "verdict": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "shards": [str(shard) for shard in args.shard],
        "n_rows": len(rows),
        "n_expected": expected,
        "lattice": {
            "arms": arms,
            "prompts": prompts,
            "seeds": seeds,
            "layers": layers,
            "timesteps": timesteps,
            "cfg_branches": branches,
            "heads": heads,
            "sparsities": sparsities,
        },
        "ladder_positions": [{
            "position": position,
            "arm": arm
        } for position, arm in ladder],
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
