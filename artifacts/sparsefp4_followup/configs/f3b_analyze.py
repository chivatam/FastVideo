"""F3B/F3C: does the mechanism hold at a materially larger token count?

Compares the 720p run against the 480p baseline on the same axes, at matched
sparsity, using the same arms and the same aggregation rules. The comparison is
*between configurations*, so it deliberately does not pool them: each is summarized
independently and the report shows both plus the ratio.

Per F3B this configuration uses the controlled proxy scorer, not VSA, so every output
here is labelled **proxy generalization**.

F3C's required reporting (token count, block geometry, sparsity, Jaccard, wrong-mask
share, isolation ratio, higher-precision recovery) is emitted as one row per
configuration so the two can be read side by side.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f3b_analyze.py \
        --baseline-shard "$FV_RAW_ROOT/<f1-full>"/*.jsonl \
        --generalization-shard "$FV_RAW_ROOT/<f3b>"/*.jsonl \
        --tables artifacts/sparsefp4_followup/tables/f3b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts/sparsefp4_followup/configs"))

from f1_aggregate import fraction, median, write_table  # noqa: E402

F3B_CONFIG = REPO_ROOT / "artifacts/sparsefp4_followup/configs/f3b_config.json"
ARMS = ("R1", "R3", "R6", "R7", "R8", "R9")
RESCUE_FROM, RESCUE_TO = "R9", "R1"
# F3B says expand to 10 prompts "only if clean"; qualitative agreement is judged on
# direction plus these bounds, which are F3A's, reused so the two tiers are comparable.
MIN_ISOLATION = 5.0
MAX_DAMAGE_SHARE = 0.01


def load(shards: list[Path], sparsity: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["sparsity"] == sparsity:
                    rows.append(record)
    return rows


def arm_row(rows: list[dict[str, Any]], arm: str, label: str) -> dict[str, Any] | None:
    group = [row for row in rows if row["arm"] == arm]
    if not group:
        return None
    wrong = [row["wrong_mask_excess"] for row in group]
    random_excess = [row["random_matched_excess"] for row in group]
    share = [
        abs(row["wrong_mask_excess"]) / row["sparsification_error"] for row in group
        if row["wrong_mask_excess"] is not None and row.get("sparsification_error")
    ]
    median_wrong = median(wrong)
    median_random = median(random_excess)
    return {
        "configuration":
        label,
        "arm":
        arm,
        "arm_label":
        group[0]["arm_label"],
        "n_cells":
        len(group),
        "seq_len":
        group[0]["seq_len"],
        "resolution":
        group[0].get("resolution"),
        "geometry":
        group[0]["geometry"],
        "n_k_blocks":
        group[0].get("n_k_blocks"),
        "retained_k":
        group[0]["retained_k"],
        "jaccard_median":
        median(row["jaccard"] for row in group),
        "recall_median":
        median(row["recall"] for row in group),
        "swaps_per_query_block_median":
        median(row["swaps_per_query_block"] for row in group),
        "sparsification_error_median":
        median(row["sparsification_error"] for row in group),
        "wrong_mask_excess_median":
        median_wrong,
        "abs_wrong_mask_over_sparsification_median":
        median(share),
        "isolation_ratio_abs":
        (None if median_wrong is None or median_random is None or median_wrong == 0 else abs(median_random) /
         abs(median_wrong)),
        "frac_cells_arm_worse_than_reference":
        fraction([row["wrong_mask_excess"] > 0 for row in group if row["wrong_mask_excess"] is not None]),
    }


def configuration_report(rows: list[dict[str, Any]], label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = [row for row in (arm_row(rows, arm, label) for arm in ARMS) if row is not None]
    index = {row["arm"]: row for row in table}
    damage = [
        row["abs_wrong_mask_over_sparsification_median"] for row in table
        if row["abs_wrong_mask_over_sparsification_median"] is not None
    ]
    isolation = [row["isolation_ratio_abs"] for row in table if row["isolation_ratio_abs"] is not None]
    worst, best = index.get(RESCUE_FROM), index.get(RESCUE_TO)
    recovery = None
    if (worst and best and worst["wrong_mask_excess_median"] is not None
            and best["wrong_mask_excess_median"] is not None and worst["sparsification_error_median"]):
        recovery = ((worst["wrong_mask_excess_median"] - best["wrong_mask_excess_median"]) /
                    worst["sparsification_error_median"])
    first = rows[0]
    f3c = {
        "configuration":
        label,
        "labelling":
        "proxy generalization (controlled scorer, not VSA)",
        "resolution":
        first.get("resolution"),
        "frames":
        first.get("frames"),
        "token_count_padded_seq_len":
        first["seq_len"],
        "block_geometry":
        first["geometry"],
        "n_k_blocks":
        first.get("n_k_blocks"),
        "sparsity":
        first["sparsity"],
        "n_prompts":
        len({row["prompt_id"]
             for row in rows}),
        "n_rows":
        len(rows),
        "jaccard_median_R9":
        index[RESCUE_FROM]["jaccard_median"] if RESCUE_FROM in index else None,
        "jaccard_median_R1":
        index[RESCUE_TO]["jaccard_median"] if RESCUE_TO in index else None,
        "max_abs_wrong_mask_over_sparsification":
        max(damage) if damage else None,
        "min_isolation_ratio":
        min(isolation) if isolation else None,
        "higher_precision_router_recovery_share":
        recovery,
        "damage_under_1pct": (None if not damage else max(damage) < MAX_DAMAGE_SHARE),
        "isolation_above_5x": (None if not isolation else min(isolation) > MIN_ISOLATION),
        "arms_damaging_in_majority_of_cells":
        sorted(row["arm"] for row in table if (row["frac_cells_arm_worse_than_reference"] or 0) > 0.5),
    }
    return table, f3c


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-shard", type=Path, nargs="+", required=True)
    parser.add_argument("--generalization-shard", type=Path, nargs="+", required=True)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()

    declared = json.loads(F3B_CONFIG.read_text(encoding="utf-8"))
    baseline_rows = load(args.baseline_shard, args.sparsity)
    general_rows = load(args.generalization_shard, args.sparsity)
    if not baseline_rows or not general_rows:
        raise SystemExit(f"baseline={len(baseline_rows)} generalization={len(general_rows)} rows at "
                         f"sparsity {args.sparsity}; both are required")

    baseline_label = f"baseline {baseline_rows[0].get('resolution')}"
    general_label = f"generalization {general_rows[0].get('resolution')}"
    baseline_table, baseline_f3c = configuration_report(baseline_rows, baseline_label)
    general_table, general_f3c = configuration_report(general_rows, general_label)

    # Restrict the baseline to the same prompts so the contrast is token count, not
    # prompt coverage: F3B runs 5 prompts where F1 ran 10.
    shared_prompts = sorted({row["prompt_id"] for row in general_rows})
    matched_rows = [row for row in baseline_rows if row["prompt_id"] in shared_prompts]
    matched_table, matched_f3c = configuration_report(matched_rows, f"{baseline_label} (matched prompts)")

    write_table(args.tables / "table1_arms_by_configuration", baseline_table + matched_table + general_table)
    write_table(args.tables / "table2_f3c_required_metrics", [baseline_f3c, matched_f3c, general_f3c])

    token_ratio = general_f3c["token_count_padded_seq_len"] / matched_f3c["token_count_padded_seq_len"]
    same_direction = (
        general_f3c["arms_damaging_in_majority_of_cells"] == matched_f3c["arms_damaging_in_majority_of_cells"])
    holds = bool(same_direction and general_f3c["damage_under_1pct"] and general_f3c["isolation_above_5x"])

    comparison = []
    for arm in ARMS:
        left = next((row for row in matched_table if row["arm"] == arm), None)
        right = next((row for row in general_table if row["arm"] == arm), None)
        if left is None or right is None:
            continue
        comparison.append({
            "arm":
            arm,
            "jaccard_baseline":
            left["jaccard_median"],
            "jaccard_720p":
            right["jaccard_median"],
            "damage_share_baseline":
            left["abs_wrong_mask_over_sparsification_median"],
            "damage_share_720p":
            right["abs_wrong_mask_over_sparsification_median"],
            "damage_share_ratio":
            ((right["abs_wrong_mask_over_sparsification_median"] / left["abs_wrong_mask_over_sparsification_median"]) if
             (left["abs_wrong_mask_over_sparsification_median"]
              and right["abs_wrong_mask_over_sparsification_median"]) else None),
            "isolation_baseline":
            left["isolation_ratio_abs"],
            "isolation_720p":
            right["isolation_ratio_abs"],
        })
    write_table(args.tables / "table3_paired_arm_comparison", comparison)

    summary = {
        "phase": "F3B/F3C",
        "verdict": "GENERALIZES_ON_TOKEN_COUNT" if holds else "DOES_NOT_GENERALIZE_CLEANLY",
        "labelling": declared["labelling"],
        "choice": declared["choice"],
        "sparsity": args.sparsity,
        "token_count_ratio": token_ratio,
        "shared_prompts": shared_prompts,
        "criteria": {
            "same_damaging_arms_as_baseline": same_direction,
            "damage_under_1pct_of_sparsification": general_f3c["damage_under_1pct"],
            "isolation_above_5x": general_f3c["isolation_above_5x"],
        },
        "f3c_metrics": [baseline_f3c, matched_f3c, general_f3c],
        "paired_arm_comparison": comparison,
    }
    args.tables.mkdir(parents=True, exist_ok=True)
    (args.tables / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"{summary['verdict']}  (token count x{token_ratio:.3f}, {declared['labelling'].split('—')[0].strip()})")
    for name, value in summary["criteria"].items():
        print(f"  [{'PASS' if value else 'FAIL'}] {name}")
    print(f"wrote {args.tables}")
    return 0 if holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
