"""F3A: seed robustness for the scorer-precision mechanism.

Reads F1-format shards spanning several seeds and answers the four F3A questions
directly, per seed and pooled:

1. is the mechanism's *direction* the same for every seed?
2. does any seed push wrong-mask damage to >=1% of sparsification error?
3. does the matched-random isolation ratio stay >5x for every seed?
4. does the higher-precision rescue stay <1% for every seed?

Seed is treated as the unit of replication: each seed is aggregated independently
and the verdict is the conjunction over seeds, so one seed cannot be averaged away
by two others. The pooled column is reported for reference only — it must never be
what a robustness claim rests on.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f3a_analyze.py \
        --shard "$FV_RAW_ROOT/<f1-run>"/*.jsonl "$FV_RAW_ROOT/<f3a-run>"/*.jsonl \
        --sparsity 0.90 \
        --tables artifacts/sparsefp4_followup/tables/f3a
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts/sparsefp4_followup/configs"))

from f1_aggregate import fraction, median, write_table  # noqa: E402

SEED_CONFIG = REPO_ROOT / "artifacts/sparsefp4_followup/configs/f3a_seeds.json"
# The mechanism arms: R1 is study 1's condition (nvfp4 representation, fp64 scorer),
# R8/R9 are the NVFP4-like scorer arms that F1 found most damaging. R2 (fp32 null)
# anchors the direction test.
MECHANISM_ARMS = ("R1", "R3", "R6", "R7", "R8", "R9")
RESCUE_FROM, RESCUE_TO = "R9", "R1"
MIN_ISOLATION = 5.0
MAX_DAMAGE_SHARE = 0.01
MAX_RESCUE_SHARE = 0.01


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [row["wrong_mask_excess"] for row in rows]
    random_excess = [row["random_matched_excess"] for row in rows]
    share = [
        abs(row["wrong_mask_excess"]) / row["sparsification_error"] for row in rows
        if row["wrong_mask_excess"] is not None and row.get("sparsification_error")
    ]
    median_wrong = median(wrong)
    median_random = median(random_excess)
    return {
        "n_cells":
        len(rows),
        "jaccard_median":
        median(row["jaccard"] for row in rows),
        "wrong_mask_excess_median":
        median_wrong,
        "sparsification_error_median":
        median(row["sparsification_error"] for row in rows),
        "abs_wrong_mask_over_sparsification_median":
        median(share),
        "isolation_ratio_abs":
        (None if median_wrong is None or median_random is None or median_wrong == 0 else abs(median_random) /
         abs(median_wrong)),
        "frac_cells_arm_worse_than_reference":
        fraction([row["wrong_mask_excess"] > 0 for row in rows if row["wrong_mask_excess"] is not None]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()

    declared = json.loads(SEED_CONFIG.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for shard in args.shard:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["sparsity"] == args.sparsity:
                    rows.append(record)
    if not rows:
        raise SystemExit(f"no rows at sparsity {args.sparsity}")

    seeds = sorted({row["seed"] for row in rows})
    print(f"loaded {len(rows)} rows at sparsity {args.sparsity}; seeds {seeds}")
    undeclared = [seed for seed in seeds if seed not in declared["seeds"]]
    if undeclared:
        raise SystemExit(f"seeds {undeclared} are not in the pre-declared list {declared['seeds']}; "
                         "post-hoc seed selection would invalidate the robustness claim")

    by_seed_arm: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed_arm[(row["seed"], row["arm"])].append(row)

    table: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in MECHANISM_ARMS:
            group = by_seed_arm.get((seed, arm))
            if not group:
                continue
            cell: dict[str, Any] = {"seed": seed, "arm": arm, "arm_label": group[0]["arm_label"]}
            cell.update(summarize(group))
            table.append(cell)
    # Pooled across seeds, for reference only.
    for arm in MECHANISM_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        pooled: dict[str, Any] = {"seed": "pooled", "arm": arm, "arm_label": group[0]["arm_label"]}
        pooled.update(summarize(group))
        table.append(pooled)
    write_table(args.tables / "table1_seed_by_arm", table)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        arms = {row["arm"]: row for row in table if row["seed"] == seed}
        damage = [
            row["abs_wrong_mask_over_sparsification_median"] for row in arms.values()
            if row["abs_wrong_mask_over_sparsification_median"] is not None
        ]
        isolation = [row["isolation_ratio_abs"] for row in arms.values() if row["isolation_ratio_abs"] is not None]
        # Rescue: how much of the sparsification error is recovered by moving the
        # scorer from the worst NVFP4-like arm back to study 1's high-precision one.
        worst = arms.get(RESCUE_FROM)
        best = arms.get(RESCUE_TO)
        rescue = None
        if (worst and best and worst["wrong_mask_excess_median"] is not None
                and best["wrong_mask_excess_median"] is not None and worst["sparsification_error_median"]):
            rescue = ((worst["wrong_mask_excess_median"] - best["wrong_mask_excess_median"]) /
                      worst["sparsification_error_median"])
        directions = {row["arm"]: (row["frac_cells_arm_worse_than_reference"] or 0) > 0.5 for row in arms.values()}
        per_seed.append({
            "seed": seed,
            "n_arms": len(arms),
            "max_damage_share": max(damage) if damage else None,
            "damage_share_under_1pct": (None if not damage else max(damage) < MAX_DAMAGE_SHARE),
            "min_isolation_ratio": min(isolation) if isolation else None,
            "isolation_above_5x": (None if not isolation else min(isolation) > MIN_ISOLATION),
            "rescue_share_R9_to_R1": rescue,
            "rescue_under_1pct": (None if rescue is None else abs(rescue) < MAX_RESCUE_SHARE),
            "arms_damaging_in_majority_of_cells": sorted(arm for arm, worse in directions.items() if worse),
            "jaccard_median_R9": arms["R9"]["jaccard_median"] if "R9" in arms else None,
        })
    write_table(args.tables / "table2_seed_verdicts", per_seed)

    real_seeds = [row for row in per_seed if row["seed"] in seeds]
    direction_sets = {tuple(row["arms_damaging_in_majority_of_cells"]) for row in real_seeds}
    same_direction = len(direction_sets) == 1
    damage_ok = all(row["damage_share_under_1pct"] for row in real_seeds if row["damage_share_under_1pct"] is not None)
    isolation_ok = all(row["isolation_above_5x"] for row in real_seeds if row["isolation_above_5x"] is not None)
    rescue_ok = all(row["rescue_under_1pct"] for row in real_seeds if row["rescue_under_1pct"] is not None)
    passed = same_direction and damage_ok and isolation_ok and rescue_ok

    summary = {
        "phase": "F3A",
        "verdict": "SEED_ROBUST" if passed else "SEED_SENSITIVE",
        "sparsity": args.sparsity,
        "seeds_declared": declared["seeds"],
        "seeds_observed": seeds,
        "n_rows": len(rows),
        "criteria": {
            "mechanism_direction_same_across_seeds": same_direction,
            "no_seed_reaches_1pct_of_sparsification": damage_ok,
            "isolation_above_5x_every_seed": isolation_ok,
            "rescue_under_1pct_every_seed": rescue_ok,
        },
        "direction_signatures": [list(item) for item in direction_sets],
        "per_seed": per_seed,
    }
    args.tables.mkdir(parents=True, exist_ok=True)
    (args.tables / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n{summary['verdict']}")
    for name, value in summary["criteria"].items():
        print(f"  [{'PASS' if value else 'FAIL'}] {name}")
    print(f"wrote {args.tables}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
