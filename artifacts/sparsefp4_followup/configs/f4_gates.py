"""F4 gates: statistical and numerical validation across the completed phases.

F4.1–F4.4 (lattice, pairing, null controls, fp64 shadow resolution) are enforced by
``f1_validate.py`` / ``f2_validate.py`` and are consumed here rather than re-derived —
this script reads their verdicts and refuses to pass if either failed.

What this script adds:

**F4.5 simulation fidelity.** Read from ``f4_representation_fidelity.json``, which
compares the native NVFP4 quantizer against the simulated one on real captured
activations and reports median/p90/max disagreement as required.

**Uncertainty on the decision thresholds.** F2's verdict turns on whether one arm's
isolation ratio sits above or below 10x, with point estimates at 9.04/9.97/10.49 —
close enough that reading a verdict off the point estimate alone would be an artifact
of sampling. The statistic is a ratio of medians, so its uncertainty comes from a
percentile bootstrap rather than a closed form, and **prompts** are resampled because
cells within a prompt share a trajectory and are not independent.

Operates on the compact ``.npz`` caches from ``build_stats_cache.py``, so the whole
gate runs in seconds and can be re-run freely.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f4_gates.py \
        --f1-cache artifacts/sparsefp4_followup/raw/cache/f1_full.npz \
        --f2-cache artifacts/sparsefp4_followup/raw/cache/f2_full.npz \
        --out artifacts/sparsefp4_followup/raw/f4_gates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "artifacts/sparsefp4_followup/raw"
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20260816
ISOLATION_THRESHOLD = 10.0
DAMAGE_THRESHOLD = 0.01


class Cache:
    """Column store loaded from ``build_stats_cache.py`` output."""

    def __init__(self, path: Path) -> None:
        self.data = np.load(path, allow_pickle=False)
        self.n = len(self.data["num_jaccard"])

    def num(self, name: str) -> np.ndarray:
        return self.data[f"num_{name}"]

    def codes(self, name: str) -> np.ndarray:
        return self.data[f"cat_{name}_codes"]

    def levels(self, name: str) -> np.ndarray:
        return self.data[f"cat_{name}_levels"]

    def code_of(self, name: str, value: str) -> int:
        levels = self.levels(name)
        found = np.nonzero(levels == value)[0]
        return int(found[0]) if len(found) else -1

    def level_values(self, name: str) -> list[str]:
        return [str(value) for value in self.levels(name)]


def bootstrap_ratio_and_share(cache: Cache, arm: str, sparsity: float,
                              rng: np.random.Generator) -> dict[str, Any] | None:
    """Percentile bootstrap of the isolation ratio and damage share, resampling prompts.

    Both statistics are medians (or ratios of medians), so each resample recomputes
    them on the pooled cells of the resampled prompts. Resampling *cells* instead would
    treat 30 layers x 12 heads within one prompt as independent replicates and yield an
    interval that is far too narrow.
    """
    arm_code = cache.code_of("arm", arm)
    if arm_code < 0:
        return None
    selected = (cache.codes("arm") == arm_code) & (cache.num("sparsity") == sparsity)
    if not selected.any():
        return None

    wrong = cache.num("wrong_mask_excess")[selected]
    random_excess = cache.num("random_matched_excess")[selected]
    sparsification = cache.num("sparsification_error")[selected]
    prompt_codes = cache.codes("prompt_id")[selected]

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.abs(wrong) / sparsification
    share[~np.isfinite(share)] = np.nan

    prompts = np.unique(prompt_codes)
    # Index cells by prompt once, so each resample is pure array indexing.
    by_prompt = [np.nonzero(prompt_codes == prompt)[0] for prompt in prompts]

    def statistics_of(index: np.ndarray) -> tuple[float, float]:
        median_wrong = np.nanmedian(wrong[index])
        median_random = np.nanmedian(random_excess[index])
        ratio = (np.abs(median_random) / np.abs(median_wrong)
                 if median_wrong not in (0.0, np.nan) and np.isfinite(median_wrong) and median_wrong != 0 else np.nan)
        return float(ratio), float(np.nanmedian(share[index]))

    point_ratio, point_share = statistics_of(np.arange(len(wrong)))

    ratios = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    shares = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        picks = rng.integers(0, len(prompts), size=len(prompts))
        index = np.concatenate([by_prompt[pick] for pick in picks])
        ratios[draw], shares[draw] = statistics_of(index)

    def interval(values: np.ndarray) -> tuple[float | None, float | None]:
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            return None, None
        return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))

    ratio_low, ratio_high = interval(ratios)
    share_low, share_high = interval(shares)

    def side(low: float | None, high: float | None, threshold: float) -> str | None:
        """Which side of ``threshold`` the whole interval falls on, if either."""
        if low is None or high is None:
            return None
        if low > threshold:
            return "above"
        if high < threshold:
            return "below"
        return "straddles"

    isolation_side = side(ratio_low, ratio_high, ISOLATION_THRESHOLD)
    damage_side = side(share_low, share_high, DAMAGE_THRESHOLD)
    return {
        "arm": arm,
        "sparsity": sparsity,
        "n_cells": int(selected.sum()),
        "n_prompts": int(len(prompts)),
        "isolation_ratio": {
            "point": point_ratio,
            "ci_low": ratio_low,
            "ci_high": ratio_high
        },
        "damage_share": {
            "point": point_share,
            "ci_low": share_low,
            "ci_high": share_high
        },
        "isolation_ci_resolves_10x": (None if isolation_side is None else isolation_side != "straddles"),
        "isolation_side": isolation_side,
        "damage_ci_resolves_1pct": (None if damage_side is None else damage_side != "straddles"),
        "damage_side": damage_side,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1-cache", type=Path, required=True)
    parser.add_argument("--f2-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool | None, detail: Any = None) -> None:
        checks.append({"check": name, "passed": None if passed is None else bool(passed), "detail": detail})
        mark = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"[{mark}] {name}" + (f" — {detail}" if detail is not None else ""))

    print("== F4.1-F4.4 (consumed from the phase validators) ==")
    upstream: dict[str, Any] = {}
    for label, path in (("F1", RAW / "f1_full_validation.json"), ("F2", RAW / "f2_full_validation.json")):
        if not path.is_file():
            check(f"{label.lower()}_validator_present", False, f"missing {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        upstream[label] = {
            "verdict": payload["verdict"],
            "n_rows": payload["n_rows"],
            "n_checks": payload["n_checks"],
            "n_failed": payload["n_failed"],
        }
        check(f"{label.lower()}_all_validator_checks_passed", payload["verdict"] == "PASS", upstream[label])
        by_name = {item["check"]: item for item in payload["checks"]}
        gates = [("F4.1_complete_lattice", "lattice_complete"), ("F4.2_pairing", "no_duplicate_cells"),
                 ("F4.4_fp64_shadow_resolution", "fp64_shadow_ties_negligible"),
                 ("F4.3_matched_random_equal_count", "matched_random_swap_count_equals_arm"),
                 ("F4.3_reference_null",
                  "reference_arm_is_exact_null" if label == "F1" else "deployed_arm_is_exact_null")]
        for gate, required in gates:
            item = by_name.get(required)
            check(f"{label.lower()}_{gate}", None if item is None else item["passed"], None)

    print("\n== F4.5 simulation fidelity ==")
    fidelity_path = RAW / "f4_representation_fidelity.json"
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8")) if fidelity_path.is_file() else None
    if fidelity is None:
        check("f4_5_native_vs_simulated_reported", False, f"missing {fidelity_path}")
    else:
        summary = fidelity["summary"]
        check(
            "f4_5_native_vs_simulated_reported", True, {
                "n_tensors": fidelity["n_tensors"],
                "representation_rel_median": summary["median_rel_disagreement"],
                "representation_rel_p90": summary["p90_rel_disagreement"],
                "representation_rel_max": summary["max_rel_disagreement"],
                "pooled_rel_median": summary["median_pooled_rel_disagreement"],
                "pooled_rel_max": summary["max_pooled_rel_disagreement"],
            })

    f1 = Cache(args.f1_cache)
    f2 = Cache(args.f2_cache)
    simulated_levels = f1.level_values("native_or_simulated")
    simulated_codes = {level: f1.code_of("native_or_simulated", level) for level in simulated_levels}
    arm_levels = f1.level_values("arm")
    simulated_arms = sorted({
        arm_levels[code]
        for code, sim in zip(f1.codes("arm"), f1.codes("native_or_simulated"), strict=False)
        if sim != simulated_codes.get("native", -1)
    })
    check("f4_5_simulated_arms_declared_in_records", bool(simulated_arms), {
        "simulated_arms": simulated_arms,
        "policy": "simulated arms are never used for latency claims",
    })

    print(f"\n== decision-threshold intervals ({BOOTSTRAP_RESAMPLES} bootstrap resamples over prompts) ==")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    intervals: list[dict[str, Any]] = []
    targets = [("F1", f1, ("R3", "R6", "R7", "R8", "R9")),
               ("F2", f2, ("VA_FP8", "VA_NVFP4", "VB_BF16_LOW", "VA_NVFP4_VB_FP64"))]
    for phase, cache, arms in targets:
        for sparsity in sorted(np.unique(cache.num("sparsity"))):
            for arm in arms:
                result = bootstrap_ratio_and_share(cache, arm, float(sparsity), rng)
                if result is None:
                    continue
                result["phase"] = phase
                intervals.append(result)
                iso, dmg = result["isolation_ratio"], result["damage_share"]
                print(f"  {phase} {arm:<18} sp={sparsity:.2f}  isolation {iso['point']:7.2f} "
                      f"[{iso['ci_low']:6.2f}, {iso['ci_high']:7.2f}] {result['isolation_side']:<9} "
                      f"damage {dmg['point']:.3e} [{dmg['ci_low']:.3e}, {dmg['ci_high']:.3e}] {result['damage_side']}")

    straddling_isolation = [{
        "phase": item["phase"],
        "arm": item["arm"],
        "sparsity": item["sparsity"],
        "point": item["isolation_ratio"]["point"],
        "ci": [item["isolation_ratio"]["ci_low"], item["isolation_ratio"]["ci_high"]],
    } for item in intervals if item["isolation_ci_resolves_10x"] is False]
    straddling_damage = [{
        "phase": item["phase"],
        "arm": item["arm"],
        "sparsity": item["sparsity"],
        "point": item["damage_share"]["point"],
        "ci": [item["damage_share"]["ci_low"], item["damage_share"]["ci_high"]],
    } for item in intervals if item["damage_ci_resolves_1pct"] is False]

    # Straddling is *reported*, not failed: an interval that spans the threshold is a
    # true statement about the evidence. What would be wrong is asserting a side.
    check("isolation_thresholds_resolved_by_intervals", None if straddling_isolation else True,
          straddling_isolation if straddling_isolation else "every isolation CI falls on one side of 10x")
    check("damage_thresholds_resolved_by_intervals", None if straddling_damage else True,
          straddling_damage if straddling_damage else "every damage CI falls on one side of 1%")

    hard_failures = [item for item in checks if item["passed"] is False]
    payload = {
        "phase": "F4",
        "verdict": "PASS" if not hard_failures else "FAIL",
        "upstream_validators": upstream,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "prompt",
            "why": "cells within a prompt share one denoising trajectory and are not independent",
            "interval": "percentile 2.5/97.5",
        },
        "representation_fidelity": fidelity["summary"] if fidelity else None,
        "decision_threshold_intervals": intervals,
        "thresholds_not_resolved": {
            "isolation_10x": straddling_isolation,
            "damage_1pct": straddling_damage,
        },
        "n_checks": len(checks),
        "n_failed": len(hard_failures),
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n{payload['verdict']}: {len(hard_failures)} hard failures of {len(checks)} checks")
    if straddling_isolation:
        print(f"NOTE: {len(straddling_isolation)} isolation ratio(s) straddle 10x — "
              "those verdicts must be reported as unresolved, not asserted")
    print(f"wrote {args.out}")
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
