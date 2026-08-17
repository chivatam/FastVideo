"""F0.2 baseline freeze: extract study 1's headline numbers from its committed tables.

The follow-up must not silently substitute remembered numbers for saved artifacts,
and later follow-up runs must not be able to move the comparison baseline. So every
value this study compares against is read out of study 1's CSV/JSON artifacts here,
once, and written with its source path, run ID, sample size, geometry and
native/simulated label to::

    artifacts/sparsefp4_followup/baseline_snapshot.json

If a source file or column is missing the entry is recorded as unavailable with the
reason rather than being filled in from the prose of REPORT.md — a discrepancy
between the report text and the tables is itself something the follow-up should
surface, not paper over.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/freeze_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY1 = REPO_ROOT / "artifacts/sparsefp4"
OUT = REPO_ROOT / "artifacts/sparsefp4_followup/baseline_snapshot.json"

# Run IDs that produced each family of tables, from study 1's REPORT.md §4/§6/§7.
RUN_PHASE1_STAGE2 = "20260814-014229-8208536-p1-stage2"
RUN_PHASE2_MAIN = "20260814-025500-8208536-p2-main"
RUN_PHASE2B_CUBE = "20260814-032500-8208536-p2b-64x64-cube"
RUN_PHASE5_MAIN = "20260814-032700-8208536-p5-main"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def entry(
    metric: str,
    value: Any,
    source: Path,
    run_id: str,
    n: Any,
    geometry: str,
    native_or_simulated: str,
    note: str = "",
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "metric": metric,
        "value": value,
        "source_path": str(source.relative_to(REPO_ROOT)) if source.is_absolute() else str(source),
        "source_run_id": run_id,
        "n": n,
        "geometry": geometry,
        "native_or_simulated": native_or_simulated,
    }
    if note:
        record["note"] = note
    record.update(extra)
    return record


def mask_overlap_by_sparsity() -> list[dict[str, Any]]:
    """Study 1's H1 headline: BF16<->candidate mask Jaccard, 10 prompts, seed 1234."""
    path = STUDY1 / "tables/main_stage2/agg_by_sparsity_precision.csv"
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        precision = row.get("routing_precision", "")
        out.append(
            entry(
                f"mask_jaccard_median[sparsity={row.get('sparsity')},router={precision}]",
                number(row.get("jaccard_median")),
                path,
                RUN_PHASE1_STAGE2,
                number(row.get("n")),
                "128x64-raster",
                "native" if precision in ("bf16", "nvfp4") else "simulated",
                note="equal-size masks: recall and jaccard are one measurement, not two",
                jaccard_iqr=number(row.get("jaccard_iqr")),
                recall_median=number(row.get("recall_median")),
            ))
    return out


def error_attribution() -> list[dict[str, Any]]:
    """``D - C`` wrong-mask excess, ``C_rand - C``, and the error shares."""
    path = STUDY1 / "tables/phase2_main/table4_error_attribution.csv"
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        sparsity = row.get("sparsity")
        n = number(row.get("n"))
        for metric, column, provenance in (
            ("quantization_B", "quantization_B_median", "native"),
            ("sparsification_C", "sparsification_C_median", "native"),
            ("wrong_mask_D_minus_C", "wrong_mask_D_minus_C_median", "native"),
            ("wrong_mask_D8_minus_C", "wrong_mask_D8_minus_C_median", "native compute / simulated fp8 router"),
            ("random_mask_Crand_minus_C", "random_mask_Crand_minus_C_median", "synthetic control"),
            ("router_recoverable_E_minus_F16", "router_recoverable_E_minus_F16_median", "simulated; numerical-only"),
            ("share_wrong_mask_of_E", "share_wrong_mask_of_E", "derived"),
            ("share_sparsification_of_E", "share_sparsification_of_E", "derived"),
        ):
            out.append(
                entry(f"{metric}[sparsity={sparsity}]", number(row.get(column)), path, RUN_PHASE2_MAIN, n,
                      "128x64-raster", provenance))
    return out


def random_contrast() -> list[dict[str, Any]]:
    """The isolation ratio — study 1's decisive mechanism evidence."""
    path = STUDY1 / "tables/phase2_main/table7_random_perturbation_contrast.csv"
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row.get("region") != "all":
            continue
        out.append(
            entry(
                f"random_over_quantization_ratio[sparsity={row.get('sparsity')},region=all]",
                number(row.get("random_over_quantization_ratio")),
                path,
                RUN_PHASE2_MAIN,
                number(row.get("n_paired_cells")),
                "128x64-raster",
                "native compute; random arm is a synthetic control",
                note="ratio of medians, not median of per-cell ratios",
                excess_quantization_median=number(row.get("excess_rel_l2_quantization_mask_median")),
                excess_random_median=number(row.get("excess_rel_l2_random_mask_median")),
                frac_cells_random_worse=number(row.get("frac_cells_random_worse")),
            ))
    return out


def h3_recovery() -> list[dict[str, Any]]:
    """Higher-precision-router recovery: the quantity F1/F2 must re-test."""
    path = STUDY1 / "tables/phase2_main/table3_h3_paired.csv"
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row.get("region") != "all":
            continue
        out.append(
            entry(
                f"h3_median_relative_reduction[{row.get('comparison')},sparsity={row.get('sparsity')},region=all]",
                number(row.get("median_relative_reduction")),
                path,
                RUN_PHASE2_MAIN,
                number(row.get("n_paired_cells")),
                "128x64-raster",
                "simulated; numerical-only",
                note="pre-registered support threshold was >=0.20 (i.e. 20%)",
                router_reference=row.get("router_E"),
                router_candidate=row.get("router_candidate"),
                frac_cells_improved=number(row.get("frac_cells_improved")),
                meets_20pct_threshold=row.get("meets_20pct_threshold"),
            ))
    return out


def geometry_headline() -> list[dict[str, Any]]:
    """Phase 2B at VSA's deployed cube geometry — the closest study 1 got to VSA."""
    path = STUDY1 / "tables/phase2b_geometry/table1_three_geometry_headline.csv"
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        geometry = row.get("geometry", "")
        run = RUN_PHASE2B_CUBE if "cube" in geometry else RUN_PHASE2_MAIN
        out.append(
            entry(
                f"cube_geometry_isolation_ratio[geometry={geometry},sparsity={row.get('sparsity')}]",
                number(row.get("Crand_over_D_excess_ratio")),
                path,
                run,
                number(row.get("n_paired_cells")),
                geometry,
                "native compute, research mean-pooled scorer — NOT VSA's own selector",
                note=("cube geometry matches VSA's (4,4,4) tiles and token ordering but the mask "
                      "came from the research mean-pooled scorer, not VSA's gate; F2 tests that"),
                mask_jaccard_median=number(row.get("mask_jaccard_median")),
                sparsification_C=number(row.get("rel_l2_C_median")),
                wrong_mask_excess=number(row.get("excess_D_minus_C_median")),
                random_excess=number(row.get("excess_Crand_minus_C_median")),
                agreed_over_swapped_mass_ratio=number(row.get("agreed_over_swapped_mass_ratio")),
                score_gap_swapped_norm_median=number(row.get("score_gap_swapped_norm_median")),
                frac_cells_random_worse=number(row.get("frac_cells_random_worse")),
            ))
    return out


def vbench_significance() -> list[dict[str, Any]]:
    """Study 1's end-to-end routing-precision null result.

    Read from the Phase 5 significance JSON rather than a CSV — study 1 emitted
    this family as JSON only, and inventing a CSV path would have silently
    produced an empty family.
    """
    path = STUDY1 / f"raw/{RUN_PHASE5_MAIN}/phase5_significance.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    family = payload.get("routing_family_multiple_comparison", {})
    n_prompts = payload.get("n_prompts")
    out: list[dict[str, Any]] = [
        entry(
            "vbench_routing_family_n_significant_after_holm",
            family.get("n_significant_after_holm"),
            path,
            RUN_PHASE5_MAIN,
            n_prompts,
            "n/a (end-to-end video)",
            "native BF16 compute; sparse-NVFP4 arms simulated",
            note=("the headline end-to-end null: routing precision changed no VBench "
                  "dimension after Holm correction across the declared family"),
            family_size=family.get("family_size"),
            n_raw_significant_at_0_05=family.get("n_raw_significant_at_0.05"),
            expected_false_positives_at_0_05=family.get("expected_false_positives_at_0.05"),
            test=payload.get("test"),
        )
    ]
    # The single most-cited paired test: NVFP4 router vs BF16 router on pixel MAE.
    for metric, comparisons in (payload.get("tests") or {}).items():
        for comparison, values in comparisons.items():
            if not isinstance(values, dict) or "routing precision" not in str(values.get("question", "")):
                continue
            out.append(
                entry(
                    f"vbench_routing_test[{metric}::{comparison}]",
                    values.get("p_value"),
                    path,
                    RUN_PHASE5_MAIN,
                    values.get("n"),
                    "n/a (end-to-end video)",
                    "native BF16 compute; sparse-NVFP4 arms simulated",
                    note=str(values.get("question")),
                    median_difference=values.get("median_difference"),
                    significant_at_0_05=values.get("significant_at_0.05"),
                    min_attainable_two_sided_p=values.get("min_attainable_two_sided_p"),
                ))
    return out


def dense_latency() -> list[dict[str, Any]]:
    """The only measured performance numbers study 1 is allowed to quote."""
    out: list[dict[str, Any]] = []
    probe = STUDY1 / "raw/phase0_nvfp4_kernel_probe.json"
    if probe.is_file():
        payload = json.loads(probe.read_text(encoding="utf-8"))
        out.append(
            entry("dense_attention_kernel_latency_probe",
                  payload,
                  probe,
                  "phase0_nvfp4_kernel_probe",
                  None,
                  "dense (no blocking)",
                  "native",
                  note="attention kernel in isolation; NOT an end-to-end speedup"))
    for arm in ("DENSE-BF16", "DENSE-FP4"):
        path = STUDY1 / f"raw/{RUN_PHASE5_MAIN}/perf/phase5_perf_p01_{arm}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                entry(f"dense_end_to_end_latency[{arm}]",
                      payload,
                      path,
                      RUN_PHASE5_MAIN,
                      payload.get("reps") if isinstance(payload, dict) else None,
                      "dense",
                      "native",
                      note="warmed, compile/CUDA-graphs off"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    families = {
        "mask_overlap_by_sparsity": mask_overlap_by_sparsity(),
        "error_attribution": error_attribution(),
        "random_matched_contrast": random_contrast(),
        "higher_precision_router_recovery": h3_recovery(),
        "deployed_cube_geometry": geometry_headline(),
        "vbench_routing_significance": vbench_significance(),
        "dense_measured_latency": dense_latency(),
    }
    missing = [name for name, values in families.items() if not values]

    snapshot = {
        "purpose":
        "Frozen study-1 baseline for the SparseFP4 paper-validation follow-up. Every "
        "value is read from a committed study-1 artifact, so later follow-up runs "
        "cannot move the comparison baseline.",
        "frozen_at_utc":
        __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study1_report":
        "artifacts/sparsefp4/REPORT.md",
        "study1_git_commit_at_report_time":
        "8208536cd1db7a1d32b68aaa6a679953ae23ab8b",
        "study1_run_ids": {
            "phase1_stage2": RUN_PHASE1_STAGE2,
            "phase2_main": RUN_PHASE2_MAIN,
            "phase2b_cube": RUN_PHASE2B_CUBE,
            "phase5_main": RUN_PHASE5_MAIN,
        },
        "study1_configuration": {
            "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
            "resolution": "480x832",
            "frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 3.0,
            "sp_size": 1,
            "seed": 1234,
            "layers": 30,
            "heads": 12,
            "head_dim": 128,
            "seq_len": 32760,
        },
        "binding_terminology":
        "'NVFP4' means NVFP4 Q/K with BF16 PV; the FA4 kernel is qk_mode=nvfp4, "
        "pv_mode=bf16. No fully-FP4 attention exists in this environment.",
        "known_limitations_this_followup_tests": [
            "scorer arithmetic precision was fp64 throughout study 1 (F1)",
            "the cube geometry control did NOT use VSA's own gate/selector (F2)",
            "single seed 1234, single model configuration (F3)",
        ],
        "families_missing_sources":
        missing,
        "families":
        families,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    total = sum(len(values) for values in families.values())
    print(f"wrote {args.out} — {total} frozen values across {len(families)} families")
    if missing:
        print(f"WARNING: no values found for: {missing}")
    for name, values in families.items():
        print(f"  {name}: {len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
