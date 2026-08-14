"""Phase 5: is the routing-precision difference distinguishable from zero?

The six-arm table shows the three ``SPARSE-FP4-*`` arms landing on top of each
other, but "the medians are close" is not a result. This computes, per metric and
per comparison, the paired statistics that answer the question directly:

* the paired per-prompt difference and its Wilcoxon signed-rank p-value (n=10,
  so the exact distribution is used, not a normal approximation);
* the same for the *sparsity* comparison, which is the effect that should be
  clearly non-null -- if the test cannot detect sparsity either, the test is
  underpowered and the routing null means nothing;
* the ratio of the two effect sizes, which is the number the report leads with.

With n=10 prompts the smallest attainable two-sided Wilcoxon p is 2/2^10 =
0.00195, so a non-significant routing result is reported as "not distinguishable
at n=10", never as "proven identical".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from scipy import stats

# (candidate, reference, label, expectation)
COMPARISONS = (
    ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16", "routing precision: NVFP4 router vs BF16 router", "null expected"),
    ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "routing precision: NVFP4 router vs FP8 router", "null expected"),
    ("SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16", "routing precision: FP8 router vs BF16 router", "null expected"),
    ("SPARSE-BF16", "DENSE-BF16", "sparsity 0.90 vs dense (positive control)", "clear effect expected"),
    ("DENSE-FP4", "DENSE-BF16", "NVFP4 Q/K quantization vs BF16 (positive control)", "clear effect expected"),
)


def wilcoxon(paired: list[tuple[float, float]]) -> dict[str, Any]:
    if len(paired) < 3:
        return {"n": len(paired), "p_value": None, "note": "too few pairs"}
    left = [a for a, _ in paired]
    right = [b for _, b in paired]
    differences = [a - b for a, b in paired]
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        return {
            "n": len(paired),
            "median_difference": 0.0,
            "p_value": 1.0,
            "note": "all paired differences exactly zero",
        }
    try:
        result = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided", method="exact")
        p_value = float(result.pvalue)
    except ValueError as error:
        return {"n": len(paired), "median_difference": statistics.median(differences), "p_value": None, "note": str(error)}
    return {
        "n": len(paired),
        "median_difference": statistics.median(differences),
        "mean_difference": statistics.fmean(differences),
        "median_left": statistics.median(left),
        "median_right": statistics.median(right),
        "n_nonzero_pairs": len(nonzero),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "min_attainable_two_sided_p": 2.0 / (2**len(nonzero)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/sparsefp4/raw"))
    args = parser.parse_args()

    raw_dir = args.raw_root / args.run_id
    merged = json.loads((raw_dir / "phase5_vbench_merged.json").read_text(encoding="utf-8"))

    vbench_rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("phase5_vbench_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            vbench_rows.extend(json.loads(line) for line in handle if line.strip())

    by_metric: dict[str, dict[tuple[str, str], float]] = {}
    for row in vbench_rows:
        by_metric.setdefault(row["metric"], {})[(row["prompt_id"], row["arm"])] = float(row["score"])

    similarity_rows: list[dict[str, Any]] = []
    with (raw_dir / "phase5_similarity.jsonl").open(encoding="utf-8") as handle:
        similarity_rows = [json.loads(line) for line in handle if line.strip()]
    pixel: dict[tuple[str, str], float] = {
        (row["prompt_id"], row["arm"]): float(row["mean_abs_pixel_diff"])
        for row in similarity_rows if row["record_type"] == "vs_reference" and row.get("mean_abs_pixel_diff") is not None
    }
    by_metric["pixel.mean_abs_diff_vs_DENSE-BF16"] = pixel

    tests: dict[str, dict[str, Any]] = {}
    for metric, scores in sorted(by_metric.items()):
        prompts = sorted({prompt for prompt, _ in scores})
        tests[metric] = {}
        for candidate, reference, label, expectation in COMPARISONS:
            paired = [(scores[(p, candidate)], scores[(p, reference)]) for p in prompts
                      if (p, candidate) in scores and (p, reference) in scores]
            tests[metric][f"{candidate}_vs_{reference}"] = {
                "question": label,
                "expectation": expectation,
                **wilcoxon(paired),
            }

    routing_key = "SPARSE-FP4-NAIVE_vs_SPARSE-FP4-ROUTE16"
    sparsity_key = "SPARSE-BF16_vs_DENSE-BF16"

    # 7 metrics x 3 routing comparisons = 21 routing tests. At alpha = 0.05 that
    # family is expected to produce ~1 false positive, so an uncorrected marginal
    # hit is not evidence. Holm-Bonferroni over the routing family only -- the
    # positive controls are not part of the hypothesis being tested.
    routing_tests = [(metric, key, result["p_value"]) for metric, per in tests.items() for key, result in per.items()
                     if "routing precision" in result["question"] and result.get("p_value") is not None]
    ordered = sorted(routing_tests, key=lambda item: item[2])
    family_size = len(ordered)
    holm: dict[str, Any] = {}
    running_max = 0.0
    for rank, (metric, key, p_value) in enumerate(ordered):
        adjusted = min(1.0, max(running_max, p_value * (family_size - rank)))
        running_max = adjusted
        holm[f"{metric}::{key}"] = {
            "raw_p": p_value,
            "holm_adjusted_p": adjusted,
            "significant_after_correction": adjusted < 0.05,
        }
    routing_family = {
        "family_size": family_size,
        "n_raw_significant_at_0.05": sum(1 for _, _, p in ordered if p < 0.05),
        "expected_false_positives_at_0.05": round(0.05 * family_size, 2),
        "n_significant_after_holm": sum(1 for entry in holm.values() if entry["significant_after_correction"]),
        "per_test": holm,
    }

    ratios: dict[str, Any] = {}
    for metric, per_comparison in tests.items():
        routing = per_comparison.get(routing_key, {}).get("median_difference")
        sparsity = per_comparison.get(sparsity_key, {}).get("median_difference")
        if routing is None or sparsity is None:
            continue
        ratios[metric] = {
            "routing_precision_median_difference": routing,
            "sparsity_median_difference": sparsity,
            "abs_ratio_sparsity_over_routing": (abs(sparsity) / abs(routing)) if routing else None,
            "routing_p_value": per_comparison[routing_key].get("p_value"),
            "sparsity_p_value": per_comparison[sparsity_key].get("p_value"),
        }

    payload = {
        "run_id": args.run_id,
        "n_prompts": 10,
        "test": "Wilcoxon signed-rank, two-sided, exact, paired by prompt",
        "power_note": ("n=10 paired prompts. The smallest attainable two-sided exact p is 2/2^10 = 0.00195. "
                       "A non-significant routing result means 'not distinguishable at this n', not 'identical'. "
                       "The sparsity and quantization comparisons are positive controls: if they are also "
                       "non-significant the test is underpowered and no null may be claimed."),
        "scope_note": ("10-prompt development set, 1 seed. SPARSE-FP4-* compute is simulated NVFP4 Q/K + BF16 PV. "
                       "No benchmark-wide claim."),
        "tests": tests,
        "routing_family_multiple_comparison": routing_family,
        "effect_size_ratios": ratios,
        "vbench_unavailable": merged.get("unavailable", {}),
        "vbench_skipped_by_design": merged.get("skipped_by_design", {}),
    }
    out_path = raw_dir / "phase5_significance.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for metric in sorted(tests):
        print(f"\n{metric}")
        for key, result in tests[metric].items():
            p_value = result.get("p_value")
            print(f"  {key:42s} n={result.get('n')} "
                  f"medΔ={result.get('median_difference'):+.6g} " if result.get("median_difference") is not None else
                  f"  {key:42s} n={result.get('n')} ", end="")
            print(f"p={p_value:.4g} {'SIG' if result.get('significant_at_0.05') else 'ns '} "
                  f"[{result['expectation']}]" if p_value is not None else f"p=n/a [{result['expectation']}]")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
