"""Phase 5 figures. Every plotted value is dumped as CSV beside the PNG.

Four figures, each answering one question the report has to answer:

1. ``fig1_paired_similarity_by_arm`` -- per-arm paired similarity vs DENSE-BF16.
   The six-arm table, drawn.
2. ``fig2_routing_vs_sparsity_effect`` -- the effect-size comparison. Sparsity's
   cost next to routing precision's cost, on the same axis, so the ratio is
   visible rather than asserted.
3. ``fig3_perturbation_calibration`` -- the control that decides how figures 1
   and 2 may be read: final-video pixel difference as a function of a *known*
   injected per-call attention perturbation. If this curve is flat/saturated
   across four orders of magnitude of input perturbation, the pixel metrics
   cannot rank perturbation magnitudes and must be reported as saturated.
4. ``fig4_singlestep_vs_freerunning`` -- single-step exactly-paired attention
   error (Phase 2's quantity, resampled along a real trajectory) against the
   free-running pixel difference, per arm. The two disagree by orders of
   magnitude if and only if the trajectory is amplifying.

    source artifacts/sparsefp4/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_figures.py \
        --run-id 20260814-032700-8208536-p5-main
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = (
    "DENSE-BF16",
    "DENSE-FP4",
    "SPARSE-BF16",
    "SPARSE-FP4-NAIVE",
    "SPARSE-FP4-ROUTE8",
    "SPARSE-FP4-ROUTE16",
)
FP4_ARMS = ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fig1(rows: list[dict[str, Any]], out_dir: Path) -> None:
    vs_ref = [row for row in rows if row["record_type"] == "vs_reference" and row["arm"] != "DENSE-BF16"]
    if not vs_ref:
        return
    arms = [arm for arm in ARM_ORDER if arm != "DENSE-BF16" and any(r["arm"] == arm for r in vs_ref)]
    metrics = (("psnr_db", "PSNR (dB), higher = closer"), ("ssim", "SSIM, higher = closer"),
               ("lpips", "LPIPS, lower = closer"), ("mean_abs_pixel_diff", "mean |pixel diff|, lower = closer"))
    csv_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 4, figsize=(22, 5.4))
    for axis, (field, label) in zip(axes, metrics, strict=True):
        data = [[r[field] for r in vs_ref if r["arm"] == arm and r.get(field) is not None] for arm in arms]
        axis.boxplot(data, tick_labels=arms, showmeans=True)
        axis.set_title(label, fontsize=11)
        axis.tick_params(axis="x", rotation=38, labelsize=8)
        axis.grid(alpha=0.3, axis="y")
        for arm, values in zip(arms, data, strict=True):
            for value in values:
                csv_rows.append({"metric": field, "arm": arm, "value": value})
    figure.suptitle("Phase 5 fig 1 — paired per-prompt similarity vs DENSE-BF16 (10-prompt dev set, seed 1234, "
                    "sparsity 0.90). SPARSE-FP4-* compute is simulated; no latency claim.",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(out_dir / "fig1_paired_similarity_by_arm.png", dpi=130)
    plt.close(figure)
    write_csv(out_dir / "fig1_paired_similarity_by_arm.csv", csv_rows)


def fig2(rows: list[dict[str, Any]], out_dir: Path) -> None:
    pairs = (
        ("SPARSE-BF16", "DENSE-BF16", "sparsity 0.90\n(BF16 compute)"),
        ("DENSE-FP4", "DENSE-BF16", "NVFP4 Q/K\nquantization"),
        ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16", "routing precision\nNVFP4 vs BF16"),
        ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "routing precision\nNVFP4 vs FP8"),
        ("SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16", "routing precision\nFP8 vs BF16"),
    )
    direct = [row for row in rows if row["record_type"] == "direct_pair"]
    if not direct:
        return
    csv_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for axis, field, label in ((axes[0], "mean_abs_pixel_diff", "mean |pixel diff|"), (axes[1], "lpips", "LPIPS")):
        labels, medians, spreads = [], [], []
        for cand, ref, name in pairs:
            values = [
                r[field] for r in direct
                if r["arm"] == cand and r["reference_arm"] == ref and r.get(field) is not None
            ]
            if not values:
                continue
            labels.append(name)
            medians.append(statistics.median(values))
            spreads.append(statistics.stdev(values) if len(values) > 1 else 0.0)
            for value in values:
                csv_rows.append({"metric": field, "candidate": cand, "reference": ref, "value": value})
        colors = ["#c44e52" if "routing" not in name else "#4c72b0" for name in labels]
        axis.bar(range(len(labels)), medians, yerr=spreads, color=colors, capsize=4)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=8.5)
        axis.set_ylabel(f"median {label} (error bar = stdev over prompts)")
        axis.grid(alpha=0.3, axis="y")
        axis.set_title(label)
    figure.suptitle("Phase 5 fig 2 — effect sizes at the video level: sparsity and quantization (red) vs "
                    "routing precision (blue), identical sparsity and compute within each blue bar.",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(out_dir / "fig2_routing_vs_sparsity_effect.png", dpi=130)
    plt.close(figure)
    write_csv(out_dir / "fig2_routing_vs_sparsity_effect.csv", csv_rows)


def fig3(calibration: list[dict[str, Any]], routing_reference: dict[str, float], out_dir: Path) -> None:
    if not calibration:
        return
    csv_rows: list[dict[str, Any]] = []
    figure, axis = plt.subplots(figsize=(9.5, 6))
    by_eps: dict[float, list[float]] = {}
    for row in calibration:
        by_eps.setdefault(float(row["realized_perturb_rel_l2"]), []).append(float(row["mean_abs_pixel_diff"]))
    xs = sorted(by_eps)
    ys = [statistics.median(by_eps[x]) for x in xs]
    for x in xs:
        for value in by_eps[x]:
            csv_rows.append({"injected_attention_rel_l2": x, "mean_abs_pixel_diff": value})
    axis.plot(xs, ys, "o-", color="#4c72b0", label="injected attention perturbation (SPARSE-BF16-EPS)")
    for label, value in routing_reference.items():
        axis.axhline(value, linestyle="--", linewidth=1.2, label=f"{label} = {value:.4g}")
        csv_rows.append({"reference_line": label, "mean_abs_pixel_diff": value})
    axis.set_xscale("log")
    axis.set_xlabel("injected per-call attention-output relative L2 (measured, not requested)")
    axis.set_ylabel("resulting final-video mean |pixel diff| vs SPARSE-BF16")
    axis.set_title("Phase 5 fig 3 — perturbation calibration: how much video difference\n"
                   "does a known attention perturbation cause?",
                   fontsize=11)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "fig3_perturbation_calibration.png", dpi=130)
    plt.close(figure)
    write_csv(out_dir / "fig3_perturbation_calibration.csv", csv_rows)


def fig4(singlestep: dict[str, Any], rows: list[dict[str, Any]], calibration: list[dict[str, Any]],
         out_dir: Path) -> None:
    """Ratio compression: the same two effects, measured two ways.

    At the attention output, sparsification is ~4000x the routing-precision
    term. At the final video, the same two comparisons differ by ~5x. The gap is
    not a disagreement about physics -- it is the video metric hitting its floor,
    which the injected-perturbation control measures directly.
    """

    def video_median(cand: str, ref: str) -> float | None:
        values = [
            row["mean_abs_pixel_diff"] for row in rows
            if row["record_type"] == "direct_pair" and row["arm"] == cand and row["reference_arm"] == ref
            and row.get("mean_abs_pixel_diff") is not None
        ]
        return statistics.median(values) if values else None

    budget = singlestep.get("error_budget_at_sparsity", {})
    h3 = singlestep.get("h3_paired_reduction", {}).get("E_to_F16", {})
    sparsification = budget.get("sparsification_C")
    wrong_mask = budget.get("wrong_mask_D_minus_C")
    routing_singlestep = None
    if sparsification and h3.get("median_relative_reduction") is not None:
        # The routing-precision term expressed on the same absolute scale as the
        # sparsification error: the median fractional recovery times the arm's error.
        routing_singlestep = abs(h3["median_relative_reduction"]) * singlestep.get("by_arm", {}).get(
            "SPARSE-FP4-NAIVE", sparsification)

    video_sparsity = video_median("SPARSE-BF16", "DENSE-BF16")
    video_routing = video_median("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16")
    floor_values = [
        row["mean_abs_pixel_diff"] for row in calibration
        if row.get("realized_perturb_rel_l2") and float(row["realized_perturb_rel_l2"]) < 1e-5
    ]
    video_floor = statistics.median(floor_values) if floor_values else None

    csv_rows = [
        {
            "measurement": "single-step attention (exactly paired, n=%d cells)" % singlestep.get("n_paired_cells", 0),
            "sparsity_effect": sparsification,
            "routing_precision_effect": routing_singlestep,
            "wrong_mask_term": wrong_mask,
            "floor": None,
            "ratio_sparsity_over_routing": (sparsification / routing_singlestep) if
            (sparsification and routing_singlestep) else None,
        },
        {
            "measurement": "free-running final video (pixel MAE, n=10 prompts)",
            "sparsity_effect": video_sparsity,
            "routing_precision_effect": video_routing,
            "wrong_mask_term": None,
            "floor": video_floor,
            "ratio_sparsity_over_routing": (video_sparsity / video_routing) if (video_sparsity and video_routing) else
            None,
        },
    ]
    write_csv(out_dir / "fig4_ratio_compression.csv", csv_rows)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6))
    panels = (
        (axes[0], "single-step attention error\n(exactly paired, all 30 layers)", [
            ("sparsification\n(C vs dense)", sparsification, "#c44e52"),
            ("routing precision\n(NVFP4 vs BF16 router)", routing_singlestep, "#4c72b0"),
        ], "relative L2 vs dense BF16"),
        (axes[1], "free-running final video\n(10 prompts, seed 1234)", [
            ("sparsity 0.90\n(vs dense)", video_sparsity, "#c44e52"),
            ("routing precision\n(NVFP4 vs BF16 router)", video_routing, "#4c72b0"),
        ], "mean |pixel diff|"),
    )
    for axis, title, bars, ylabel in panels:
        labels = [name for name, value, _ in bars if value]
        values = [value for _, value, _ in bars if value]
        colors = [color for _, value, color in bars if value]
        axis.bar(range(len(values)), values, color=colors)
        axis.set_yscale("log")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=9)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.3, axis="y", which="both")
        for index, value in enumerate(values):
            axis.text(index, value * 1.25, f"{value:.3g}", ha="center", fontsize=9)
        if len(values) == 2:
            axis.text(0.5,
                      0.02,
                      f"ratio = {values[0] / values[1]:,.0f}x",
                      transform=axis.transAxes,
                      ha="center",
                      fontsize=12,
                      weight="bold")
    if video_floor:
        axes[1].axhline(video_floor,
                        linestyle="--",
                        color="#555555",
                        label=f"floor: 1e-6 injected perturbation = {video_floor:.4f}")
        axes[1].legend(fontsize=8.5, loc="upper right")
    figure.suptitle("Phase 5 fig 4 — ratio compression. The same two effects measured at the attention output and "
                    "at the final video.\nThe video ratio collapses because the video metric is at its floor "
                    "(dashed), not because routing precision matters end-to-end.",
                    fontsize=10.5)
    figure.tight_layout()
    figure.savefig(out_dir / "fig4_ratio_compression.png", dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/sparsefp4/raw"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sparsefp4/figures/phase5_main"))
    parser.add_argument("--similarity-tag", default="similarity")
    args = parser.parse_args()

    raw_dir = args.raw_root / args.run_id
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(raw_dir / f"phase5_{args.similarity_tag}.jsonl")
    fig1(rows, args.out_dir)
    fig2(rows, args.out_dir)

    calibration = read_jsonl(raw_dir / "phase5_calibration.jsonl")
    routing_reference: dict[str, float] = {}
    for cand, ref, label in (("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE16", "routing precision NVFP4 vs BF16"),
                             ("SPARSE-BF16", "DENSE-BF16", "sparsity 0.90 vs dense")):
        values = [
            row["mean_abs_pixel_diff"] for row in rows
            if row["record_type"] == "direct_pair" and row["arm"] == cand and row["reference_arm"] == ref
            and row.get("mean_abs_pixel_diff") is not None
        ]
        if values:
            routing_reference[label] = statistics.median(values)
    fig3(calibration, routing_reference, args.out_dir)

    singlestep_path = args.raw_root / args.run_id / "phase5_singlestep_medians.json"
    if singlestep_path.is_file():
        fig4(json.loads(singlestep_path.read_text(encoding="utf-8")), rows, calibration, args.out_dir)

    print(f"figures written to {args.out_dir}")
    for path in sorted(args.out_dir.iterdir()):
        print(f"  {path.name}  {path.stat().st_size / 1000:.1f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
