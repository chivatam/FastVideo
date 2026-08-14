"""Render the Phase 2B geometry figures from the analysis CSVs.

Each figure's plotted values are the CSV written next to it by
``phase2b_geometry_analyze.py``; the CSV is the archival artifact and the PNG is
derived from it.

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase2b_figures.py \\
        --figures artifacts/sparsefp4/figures/phase2b_geometry
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

GEOMETRY_ORDER = ("128x64-raster", "64x64-raster", "64x64-cube")
COLORS = {"128x64-raster": "#4c72b0", "64x64-raster": "#dd8452", "64x64-cube": "#55a868"}
LABELS = {
    "128x64-raster": "128x64 raster (Phase 2)",
    "64x64-raster": "64x64 raster (block size only)",
    "64x64-cube": "64x64 VSA cube (deployed)",
}


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def sparsities_of(rows: list[dict[str, Any]]) -> list[float]:
    values = {number(row, "sparsity") for row in rows}
    return sorted(value for value in values if value is not None)


def fig_isolation(figures: Path) -> None:
    """The study's key isolation, per geometry: random vs quantization excess."""
    rows = load(figures / "fig2_random_over_quantization_by_geometry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.26
    sparsities = sparsities_of(rows)
    for index, geometry in enumerate(GEOMETRY_ORDER):
        subset = {number(r, "sparsity"): r for r in rows if r["geometry"] == geometry}
        if not subset:
            continue
        positions = [i + (index - 1) * width for i in range(len(sparsities))]
        quant = [number(subset[s], "excess_quantization_mask_D_minus_C_median") or 0.0 for s in sparsities]
        rand = [number(subset[s], "excess_random_mask_Crand_minus_C_median") or 0.0 for s in sparsities]
        ratio = [number(subset[s], "random_over_quantization_ratio") or 0.0 for s in sparsities]
        axes[0].bar(positions, quant, width, label=f"{LABELS[geometry]} — NVFP4", color=COLORS[geometry])
        axes[0].bar(positions,
                    rand,
                    width,
                    bottom=None,
                    color=COLORS[geometry],
                    alpha=0.35,
                    hatch="//",
                    label=f"{LABELS[geometry]} — random")
        axes[1].bar(positions, ratio, width, label=LABELS[geometry], color=COLORS[geometry])
        for x, value in zip(positions, ratio, strict=True):
            axes[1].text(x, value, f"{value:.0f}x", ha="center", va="bottom", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("excess rel-L2 over C (median)")
    axes[0].set_title("Wrong-mask excess: quantization-chosen vs random")
    axes[1].set_ylabel("C_rand excess / D excess")
    axes[1].set_title("Isolation ratio (higher = swaps are cheaper than random)")
    for axis in axes:
        axis.set_xticks(range(len(sparsities)))
        axis.set_xticklabels([f"s={s:.2f}" for s in sparsities])
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=6, ncol=1)
    axes[1].legend(fontsize=7)
    fig.suptitle("Phase 2B — the near-tie mechanism at three block geometries (numerical only, no latency claim)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "fig2_random_over_quantization_by_geometry.png", dpi=150)
    plt.close(fig)


def fig_mass_and_margin(figures: Path) -> None:
    """Swapped-block mass vs agreed-block mass, and the boundary margin."""
    rows = load(figures / "fig4_block_mass_by_geometry.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    sparsities = sparsities_of(rows)
    for geometry in GEOMETRY_ORDER:
        subset = {number(r, "sparsity"): r for r in rows if r["geometry"] == geometry}
        if not subset:
            continue
        axes[0].plot(sparsities, [number(subset[s], "mass_agreed_mean_median") for s in sparsities],
                     "o-",
                     color=COLORS[geometry],
                     label=f"{LABELS[geometry]} — agreed")
        axes[0].plot(sparsities, [number(subset[s], "mass_dropped_mean_median") for s in sparsities],
                     "s--",
                     color=COLORS[geometry],
                     label=f"{LABELS[geometry]} — swapped out")
        axes[1].plot(sparsities, [number(subset[s], "agreed_over_dropped_mass_ratio") for s in sparsities],
                     "o-",
                     color=COLORS[geometry],
                     label=LABELS[geometry])
        axes[2].plot(sparsities, [number(subset[s], "score_gap_swapped_norm_median") for s in sparsities],
                     "o-",
                     color=COLORS[geometry],
                     label=LABELS[geometry])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("dense attention mass per block (median)")
    axes[0].set_title("Mass of swapped vs agreed blocks")
    axes[0].legend(fontsize=6)
    axes[1].set_ylabel("agreed mass / swapped mass")
    axes[1].set_title("Swapped blocks are this much less important")
    axes[1].legend(fontsize=7)
    axes[2].set_ylabel("normalized score gap of the swapped pair")
    axes[2].set_title("Swaps happen at a vanishing margin")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xlabel("sparsity")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "fig4_block_mass_by_geometry.png", dpi=150)
    plt.close(fig)


def fig_mask_stability(figures: Path) -> None:
    """Mask overlap and churn per geometry."""
    rows = load(figures / "fig3_mask_jaccard_by_geometry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    sparsities = sparsities_of(rows)
    for geometry in GEOMETRY_ORDER:
        subset = {number(r, "sparsity"): r for r in rows if r["geometry"] == geometry}
        if not subset:
            continue
        medians = [number(subset[s], "mask_jaccard_median") for s in sparsities]
        iqrs = [number(subset[s], "mask_jaccard_iqr") or 0.0 for s in sparsities]
        axes[0].errorbar(sparsities,
                         medians,
                         yerr=[value / 2 for value in iqrs],
                         fmt="o-",
                         capsize=3,
                         color=COLORS[geometry],
                         label=LABELS[geometry])
        axes[1].plot(sparsities, [number(subset[s], "blocks_swapped_per_query_block_mean") for s in sparsities],
                     "o-",
                     color=COLORS[geometry],
                     label=LABELS[geometry])
    axes[0].set_ylabel("mask Jaccard vs BF16 router (median, bar = IQR/2)")
    axes[0].set_title("Mask overlap is geometry-insensitive")
    axes[1].set_ylabel("blocks swapped per query block (mean)")
    axes[1].set_title("Churn is geometry-insensitive too")
    for axis in axes:
        axis.set_xlabel("sparsity")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fig3_mask_jaccard_by_geometry.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    fig_isolation(args.figures)
    fig_mass_and_margin(args.figures)
    fig_mask_stability(args.figures)
    print(f"wrote PNGs alongside their value CSVs under {args.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
