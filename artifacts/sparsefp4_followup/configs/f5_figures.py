"""F5: paper-ready figures A, B and C from the cached study columns.

Three figures, matching the F5 spec:

* **A — scorer arithmetic precision.** Three panels along the arithmetic ladder
  (fp64 → fp32 → bf16 → fp8 → NVFP4-like): mask Jaccard, wrong-mask excess as a
  percentage of sparsification error, and the matched-random / precision damage ratio.
  The ladder is drawn at *fixed representation*, with exact-BF16 and NVFP4 Q/K as
  separate series, because separating those two axes is the whole point of F1 — study 1
  varied only representation, so collapsing them would hide the axis this phase exists
  to isolate.
* **B — actual VSA selector.** The same three quantities for VSA's real selector, plus
  the ``VC_GATE_NVFP4`` invariant drawn explicitly so the falsification test is visible
  rather than merely described.
* **C — generalization.** Point-and-interval by seed and by configuration.

Intervals everywhere are 95% percentile bootstraps over *prompts*, computed by the
shared ``sparsefp4_stats`` module that ``f4_gates.py`` also uses — so a figure cannot
disagree with the gate that validated it.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f5_figures.py \
        --figures artifacts/sparsefp4_followup/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts/sparsefp4_followup/configs"))

import sparsefp4_stats as stats  # noqa: E402

FOLLOWUP = REPO_ROOT / "artifacts/sparsefp4_followup"
CACHE = FOLLOWUP / "raw/cache"
SPARSITY = 0.90
EXACT_COLOUR, NVFP4_COLOUR = "#1f77b4", "#d62728"

# F1's ladder at fixed representation: (label, exact-BF16 arm, NVFP4 arm).
LADDER = (
    ("fp64", "R0", "R1"),
    ("fp32", "R2", "R3"),
    ("bf16\n(fp32 acc)", "R4", "R5"),
    ("bf16\n(bf16 acc)", "R4L", "R5L"),
    ("fp8", "R6", "R7"),
    ("NVFP4\nlike", "R8", "R9"),
)
VSA_ARMS = (
    ("FP8\nrepr.", "VA_FP8"),
    ("NVFP4\nrepr.", "VA_NVFP4"),
    ("bf16 low\nselector", "VB_BF16_LOW"),
    ("NVFP4 repr.\n+fp64 sel.", "VA_NVFP4_VB_FP64"),
    ("fp64 selector\n(rescue)", "V0_FP64"),
)


def annotate_threshold(axis: Any, x: float, y: float, text: str, colour: str = "black") -> None:
    axis.annotate(text,
                  xy=(x, y),
                  xytext=(-4, 4),
                  textcoords="offset points",
                  fontsize=7,
                  color=colour,
                  ha="right",
                  va="bottom")


def figure_a(f1: stats.Cache, out: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    positions = np.arange(len(LADDER))
    collected: dict[str, list[dict[str, Any]]] = {"exact": [], "nvfp4": []}

    for key, arm_index, colour, label in (("exact", 1, EXACT_COLOUR, "exact BF16 Q/K"), ("nvfp4", 2, NVFP4_COLOUR,
                                                                                         "NVFP4 Q/K")):
        jaccards: list[float | None] = []
        share_x: list[float] = []
        share_y: list[float] = []
        zero_x: list[float] = []
        ratio_x: list[float] = []
        ratio_y: list[float] = []
        for position, entry in enumerate(LADDER):
            arm = entry[arm_index]
            group = stats.group_statistics(f1, f1.select(arm, SPARSITY))
            collected[key].append({"arm": arm, **group})
            jaccards.append(group["jaccard"])
            if position == 0:
                # Each representation's fp64 arm is its own reference: share and ratio
                # are structurally zero/undefined there, so the ladder starts at fp32.
                continue
            share = group["share"]
            if share == 0.0:
                zero_x.append(position)
            elif share is not None:
                share_x.append(position)
                share_y.append(100.0 * share)
            if group["isolation"] is not None:
                ratio_x.append(position)
                ratio_y.append(group["isolation"])

        axes[0].plot(positions, jaccards, "o-", color=colour, label=label)
        axes[1].plot(share_x, share_y, "o-", color=colour, label=label)
        axes[2].plot(ratio_x, ratio_y, "o-", color=colour, label=label)
        # A share of exactly zero is a real finding — bit-identical masks — that a log
        # axis cannot place. Annotate it instead of clipping it to the axis floor.
        for zero_position in zero_x:
            axes[1].annotate("exactly 0 —\nbit-identical to fp64",
                             xy=(zero_position, min(share_y)),
                             xytext=(zero_position + 0.35, min(share_y) * 2.2),
                             fontsize=7,
                             color=colour,
                             ha="left",
                             va="center",
                             arrowprops={
                                 "arrowstyle": "->",
                                 "color": colour,
                                 "lw": 0.9
                             })

    axes[0].set_ylabel("mask Jaccard vs fp64 reference")
    axes[0].set_title("A1  selector agreement")
    axes[1].set_ylabel("wrong-mask excess (% of sparsification error)")
    axes[1].set_yscale("log")
    axes[1].set_title("A2  damage, relative to sparsification")
    axes[1].axhline(0.1, color="grey", ls=":", lw=1)
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    annotate_threshold(axes[1], len(LADDER) - 1, 1.0, "1%  revision threshold")
    annotate_threshold(axes[1], len(LADDER) - 1, 0.1, "0.1%  strong-survival bound", "grey")
    axes[2].set_ylabel("matched-random / precision damage")
    axes[2].set_yscale("log")
    axes[2].set_title("A3  is it a boundary effect?")
    axes[2].axhline(10.0, color="black", ls="--", lw=1)
    annotate_threshold(axes[2], len(LADDER) - 1, 10.0, "10x isolation criterion")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels([name for name, _, _ in LADDER], fontsize=8)
        axis.set_xlabel("scorer arithmetic precision")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Figure A — scorer arithmetic precision at sparsity {SPARSITY} "
        "(Wan2.1-1.3B, 10 prompts x 30 layers x 12 heads x 6 timesteps x 2 CFG branches)",
        fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out / "figureA_scorer_arithmetic.png", dpi=200)
    figure.savefig(out / "figureA_scorer_arithmetic.pdf")
    plt.close(figure)
    return {"ladder": [name.replace("\n", " ") for name, _, _ in LADDER], **collected}


def figure_b(f2: stats.Cache, out: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    labels = [label for label, _ in VSA_ARMS]
    positions = np.arange(len(VSA_ARMS))
    values = [stats.group_statistics(f2, f2.select(arm, SPARSITY)) for _, arm in VSA_ARMS]
    gate = stats.group_statistics(f2, f2.select("VC_GATE_NVFP4", SPARSITY))
    gate_position = len(VSA_ARMS)

    # Markers rather than bars: every Jaccard lies between 0.94 and 1.0, so bars would
    # need a truncated baseline and would exaggerate small differences.
    axes[0].plot(positions, [item["jaccard"] for item in values], "o", color="#4c72b0", markersize=9, zorder=3)
    axes[0].axhline(1.0, color="black", lw=1)
    axes[0].plot([gate_position], [gate["jaccard"]],
                 "D",
                 color="#2ca02c",
                 markersize=10,
                 zorder=3,
                 label=f"gate_compress → NVFP4: Jaccard = {gate['jaccard']:.6f} (mask unchanged)")
    for position, item in zip(positions, values, strict=False):
        axes[0].annotate(f"{item['jaccard']:.4f}",
                         xy=(position, item["jaccard"]),
                         xytext=(0, -14),
                         textcoords="offset points",
                         fontsize=7,
                         ha="center")
    axes[0].set_ylim(0.93, 1.008)
    axes[0].set_ylabel("mask Jaccard vs deployed VSA selector")
    axes[0].set_title("B1  overlap with the shipped selector")
    axes[0].legend(fontsize=7, loc="lower center")

    axes[1].plot(positions, [100.0 * (item["share"] or 0.0) for item in values],
                 "o",
                 color="#4c72b0",
                 markersize=9,
                 zorder=3)
    axes[1].set_yscale("log")
    axes[1].axhline(0.1, color="grey", ls=":", lw=1)
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    annotate_threshold(axes[1], gate_position, 1.0, "1%  revision threshold")
    annotate_threshold(axes[1], gate_position, 0.1, "0.1%  strong-survival bound", "grey")
    axes[1].set_ylabel("wrong-mask excess (% of VSA sparsification error)")
    axes[1].set_title("B2  damage, relative to VSA's own sparsification")

    axes[2].plot(positions, [item["isolation"] or np.nan for item in values],
                 "o",
                 color="#4c72b0",
                 markersize=9,
                 zorder=3)
    axes[2].set_yscale("log")
    axes[2].axhline(10.0, color="black", ls="--", lw=1)
    annotate_threshold(axes[2], gate_position, 10.0, "10x isolation criterion")
    axes[2].set_ylabel("matched-random / precision damage")
    axes[2].set_title("B3  matched-random contrast")

    for axis in axes:
        axis.set_xticks(list(positions) + [gate_position])
        axis.set_xticklabels(labels + ["gate\ninvariant"], fontsize=8)
        axis.set_xlim(-0.5, gate_position + 0.5)
        axis.grid(alpha=0.25, axis="y")
    figure.suptitle(
        f"Figure B — VSA's real selector under precision intervention, sparsity {SPARSITY} "
        "(genuine VIDEO_SPARSE_ATTN trajectory; the fp64 selector is a rescue arm, not a degradation)",
        fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out / "figureB_vsa_selector.png", dpi=200)
    figure.savefig(out / "figureB_vsa_selector.pdf")
    plt.close(figure)
    return {"arms": [label.replace("\n", " ") for label in labels], "values": values, "gate_invariant": gate}


def figure_c(f1: stats.Cache, f2: stats.Cache, f3a: stats.Cache, f3b: stats.Cache, out: Path) -> dict[str, Any]:
    """Point-and-interval by seed and configuration, for the ladder's worst arm.

    R9 (NVFP4-like arithmetic on NVFP4 Q/K) is the informative arm: it is the floor of
    the ladder, so stability of its magnitude bounds everything above it.
    """
    groups: tuple[tuple[str, str, stats.Cache, str, int | None], ...] = (
        ("seed", "seed 1234\n480p proxy", f1, "R9", 1234),
        ("seed", "seed 2026\n480p proxy", f3a, "R9", 2026),
        ("seed", "seed 3407\n480p proxy", f3a, "R9", 3407),
        ("config", "720x1280\n75.6k tokens", f3b, "R9", None),
        ("config", "real VSA\nNVFP4 routing", f2, "VA_NVFP4", None),
    )
    rows: list[dict[str, Any]] = []
    for group, label, cache, arm, seed in groups:
        mask = cache.select(arm, SPARSITY, seed=seed)
        if not mask.any():
            continue
        share_statistic, isolation_statistic = stats.share_and_isolation_statistics(cache, mask)
        rows.append({
            "group": group,
            "label": label.replace("\n", " "),
            "plot_label": label,
            "arm": arm,
            "seed": seed,
            "share": stats.bootstrap_over_prompts(cache, mask, share_statistic),
            "isolation": stats.bootstrap_over_prompts(cache, mask, isolation_statistic),
        })

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    positions = np.arange(len(rows))
    colours = ["#4c72b0" if row["group"] == "seed" else "#dd8452" for row in rows]
    panels = ((axes[0], "share", "C1  wrong-mask excess / sparsification error", 0.01, "1%  revision threshold"),
              (axes[1], "isolation", "C2  matched-random isolation ratio", 10.0, "10x isolation criterion"))
    for axis, key, title, threshold, note in panels:
        for index, row in enumerate(rows):
            entry = row[key]
            if entry["ci_low"] is not None:
                axis.plot([index, index], [entry["ci_low"], entry["ci_high"]], color="black", lw=1.4, zorder=2)
            axis.plot([index], [entry["point"]], "o", color=colours[index], markersize=9, zorder=3)
        axis.axhline(threshold, color="black", ls="--", lw=1)
        annotate_threshold(axis, len(rows) - 0.6, threshold, note)
        axis.set_xticks(positions)
        axis.set_xticklabels([row["plot_label"] for row in rows], fontsize=7.5)
        axis.set_xlim(-0.5, len(rows) - 0.5)
        axis.set_yscale("log")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("share of sparsification error")
    axes[1].set_ylabel("ratio")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#4c72b0", label="seed replicate"),
        plt.Line2D([], [], marker="o", ls="", color="#dd8452", label="different configuration"),
        plt.Line2D([], [], color="black", lw=1.4, label="95% bootstrap interval over prompts"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left")
    figure.suptitle(
        "Figure C — generalization across seeds and configurations, worst arm "
        "(NVFP4-like scorer on NVFP4 Q/K; the VSA panel uses its NVFP4 routing arm)",
        fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out / "figureC_generalization.png", dpi=200)
    figure.savefig(out / "figureC_generalization.pdf")
    plt.close(figure)
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=Path, default=FOLLOWUP / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    f1 = stats.Cache(CACHE / "f1_full.npz")
    f2 = stats.Cache(CACHE / "f2_full.npz")
    f3a = stats.Cache(CACHE / "f3a.npz")
    f3b = stats.Cache(CACHE / "f3b.npz")

    data = {
        "sparsity": SPARSITY,
        "bootstrap": {
            "resamples": stats.BOOTSTRAP_RESAMPLES,
            "seed": stats.BOOTSTRAP_SEED,
            "resampling_unit": "prompt",
        },
        "figureA": figure_a(f1, args.figures),
        "figureB": figure_b(f2, args.figures),
        "figureC": figure_c(f1, f2, f3a, f3b, args.figures),
    }
    (args.figures / "figure_data.json").write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    for name in sorted(path.name for path in args.figures.glob("*.p*")):
        print(f"  wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
