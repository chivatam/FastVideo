"""Render Phase 2 figures from the analysis CSVs.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_figures.py \
        --tables artifacts/sparsefp4/tables/phase2_main \
        --figures artifacts/sparsefp4/figures/phase2_main
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H3_THRESHOLD = 0.20


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def fig_decomposition(tables: Path, figures: Path) -> None:
    rows = [r for r in load(tables / "table1_af_decomposition.csv") if r["sparsity"]]
    order = ["C", "D", "D8", "C_rand", "E", "F8", "F16"]
    labels = {
        "C": "C  sparse BF16, BF16 mask",
        "D": "D  + NVFP4 mask",
        "D8": "D8 + FP8 mask",
        "C_rand": "C_rand + random mask",
        "E": "E  NVFP4 compute, NVFP4 router",
        "F8": "F8 NVFP4 compute, FP8 router",
        "F16": "F16 NVFP4 compute, BF16 router",
    }
    sparsities = sorted({float(r["sparsity"]) for r in rows})
    figure, axes = plt.subplots(figsize=(9.5, 5.2))
    width = 0.1
    for index, config_id in enumerate(order):
        values = [
            next((number(r, "rel_l2_median")
                  for r in rows if r["config"] == config_id and float(r["sparsity"]) == s), 0.0) for s in sparsities
        ]
        positions = [i + (index - len(order) / 2) * width for i in range(len(sparsities))]
        axes.bar(positions, values, width, label=labels[config_id])
    dense = next(
        (number(r, "rel_l2_median") for r in load(tables / "table1_af_decomposition.csv") if r["config"] == "B"), None)
    if dense is not None:
        axes.axhline(dense, color="black", linestyle="--", linewidth=1.2, label=f"B  quantization only ({dense:.3f})")
    axes.set_xticks(range(len(sparsities)))
    axes.set_xticklabels([f"sparsity {s:.2f}" for s in sparsities])
    axes.set_ylabel("median relative L2 vs dense BF16 (A)")
    axes.set_title("Phase 2: error decomposition — sparsification dominates, mask precision is invisible\n"
                   "n = 20,400 paired cells per bar; NVFP4 = NVFP4 Q/K + BF16 PV")
    axes.legend(fontsize=8, ncol=2)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "fig1_af_decomposition.png", dpi=150)
    plt.close(figure)


def fig_h3(tables: Path, figures: Path) -> None:
    rows = [
        r for r in load(tables / "table3_h3_paired.csv")
        if r["region"] == "all" and r["comparison"] != "F16_to_F16_null_control"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sparsities = sorted({float(r["sparsity"]) for r in rows})
    for comparison, colour in (("E_to_F8", "tab:orange"), ("E_to_F16", "tab:blue")):
        values = [
            next((number(r, "median_relative_reduction")
                  for r in rows if r["comparison"] == comparison and float(r["sparsity"]) == s), 0.0)
            for s in sparsities
        ]
        axes[0].plot(sparsities, values, "o-", color=colour, label=comparison.replace("E_to_", "NVFP4 -> "))
    axes[0].axhline(H3_THRESHOLD, color="red", linestyle="--", label="H3 threshold (20%)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sparsity")
    axes[0].set_ylabel("median relative error reduction")
    axes[0].set_title("H3: measured reduction vs pre-registered threshold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    ecdf = load(tables.parent.parent / "figures" / figures.name / "fig2_h3_paired_reduction_ecdf.csv")
    for comparison, colour in (("E_to_F8", "tab:orange"), ("E_to_F16", "tab:blue")):
        ordered = sorted(v for v in (number(r, "relative_reduction") for r in ecdf
                                     if r["comparison"] == comparison and r["sparsity"] == "0.9") if v is not None)
        if not ordered:
            continue
        axes[1].plot(ordered, [i / len(ordered) for i in range(len(ordered))],
                     color=colour,
                     label=comparison.replace("E_to_", "NVFP4 -> "))
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlim(-0.02, 0.02)
    axes[1].set_xlabel("per-cell relative error reduction (sparsity 0.90)")
    axes[1].set_ylabel("cumulative fraction of cells")
    axes[1].set_title("Per-cell distribution straddles zero\n(20% threshold is far off-scale to the right)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "fig2_h3_verdict.png", dpi=150)
    plt.close(figure)


def fig_mechanism(tables: Path, figures: Path) -> None:
    mech = [r for r in load(tables / "table5_margin_mechanism.csv") if r["region"] == "all"]
    contrast = [r for r in load(tables / "table7_random_perturbation_contrast.csv") if r["region"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    sparsities = [float(r["sparsity"]) for r in mech]
    series = (("mass_agreed_mean_median", "blocks both masks keep",
               "tab:green"), ("mass_random_dropped_mean_median", "dropped by RANDOM swap",
                              "tab:red"), ("mass_dropped_mean_median", "dropped by NVFP4 swap", "tab:orange"),
              ("mass_excluded_mean_median", "average excluded block", "tab:gray"))
    for key, label, colour in series:
        axes[0].plot(sparsities, [number(r, key) for r in mech], "o-", color=colour, label=label)
    axes[0].set_xlabel("sparsity")
    axes[0].set_ylabel("mean dense attention mass per key block")
    axes[0].set_title("Swapped blocks sit near the excluded population,\nnot the retained one")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    positions = range(len(contrast))
    width = 0.35
    axes[1].bar([p - width / 2 for p in positions],
                [number(r, "excess_rel_l2_quantization_mask_median") for r in contrast],
                width,
                color="tab:orange",
                label="NVFP4-chosen swaps (D - C)")
    axes[1].bar([p + width / 2 for p in positions], [number(r, "excess_rel_l2_random_mask_median") for r in contrast],
                width,
                color="tab:red",
                label="random swaps, same count (C_rand - C)")
    for index, row in enumerate(contrast):
        ratio = number(row, "random_over_quantization_ratio")
        if ratio is not None:
            axes[1].annotate(f"{ratio:.0f}x", (index, number(row, "excess_rel_l2_random_mask_median") or 0.0),
                             ha="center",
                             va="bottom",
                             fontsize=9)
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels([f"sparsity {float(r['sparsity']):.2f}" for r in contrast])
    axes[1].set_ylabel("excess relative L2 over C")
    axes[1].set_title("Equal-magnitude random contrast control:\nrandom swaps cost 10-27x more")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "fig3_mechanism.png", dpi=150)
    plt.close(figure)


def fig_saturation(tables: Path, figures: Path) -> None:
    rows = load(tables / "table8_saturation_vs_layer_sensitivity.csv")
    figure, axes = plt.subplots(figsize=(9, 4.6))
    colours = {"affected": "tab:red", "unaffected": "tab:blue", "broad": "tab:gray"}
    for region, colour in colours.items():
        subset = [r for r in rows if r["region"] == region]
        axes.scatter([number(r, "sat_frac_q_nvfp4_median") for r in subset],
                     [number(r, "wrong_mask_excess_rel_l2_median") for r in subset],
                     color=colour,
                     label=f"{region} layers",
                     s=60)
    for row in rows:
        axes.annotate(
            f"L{row['layer']}",
            (number(row, "sat_frac_q_nvfp4_median") or 0.0, number(row, "wrong_mask_excess_rel_l2_median") or 0.0),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points")
    axes.set_xlabel("NVFP4 e2m1 saturation fraction of Q (median)")
    axes.set_ylabel("wrong-mask excess relative L2 (D - C)")
    axes.set_title("Saturation does not explain the layer ranking (Spearman -0.25)\n"
                   "and the whole y-axis is ~1000x below the H3 threshold")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "fig5_saturation_control.png", dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig_decomposition(args.tables, args.figures)
    fig_h3(args.tables, args.figures)
    fig_mechanism(args.tables, args.figures)
    fig_saturation(args.tables, args.figures)
    print(f"wrote 4 figures to {args.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
