"""Generate figures for the geometry-alignment spin-out paper.

Methodology synthesized from four figure skills (see FIGURE_CONTRACT.md):
- nature-figure: contract-first design (claim -> evidence role -> archetype),
  hero panel, panel letters.
- ARIS paper-figure: no hardcoded data (read from data/*.csv), serif face
  matching paper body, no titles inside figures, render-then-verify.
- K-Dense scientific-visualization: honest encodings, n and uncertainty
  stated, Okabe-Ito accents, exact physical sizing, provenance manifest.
- publication-chart-skill: representation chosen by claim; exact values
  live in the paper's tables, not repeated inside figures.

Run:  /mnt/fastvideo/scratch/fv-venv/bin/python make_figures.py
"""

import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

# Okabe-Ito accents (see FIGURE_CONTRACT.md)
GREEN = "#009E73"      # ours
BLUE = "#0072B2"       # deployed baseline
VERMILLION = "#D55E00" # failure path
GREY = "#8F8F8F"       # neutral / dense
LIGHT = "#C9C9C9"
DARK = "#1A1A1A"

MM = 1 / 25.4
SINGLE_COL = 89 * MM   # ~3.5 in
DOUBLE_COL = 183 * MM  # ~7.2 in

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#666666",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

MANIFEST = {}


def save(fig, name, sources):
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    MANIFEST[name] = {
        "source_data": sources,
        "canonical_receipts": [
            "tables/c8_performance_v2.md",
            "tables/p4g_vs_p2_quality_bootstrap.md",
        ],
        "outputs": [f"{name}.png", f"{name}.pdf"],
    }
    print(f"{name} done")


def read_csv(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def panel_letter(ax_or_fig, x, y, letter, transform):
    ax_or_fig.text(x, y, letter, transform=transform, fontsize=10,
                   fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------- figure 1
def clustered_mask(n_q=16, n_k=16, keep=26, seed=0):
    """Spatially clustered ~10% mask (video-locality-like), illustrative."""
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(n_q, n_k))
    for _ in range(3):
        field = (np.roll(field, 1, 0) + field + np.roll(field, -1, 0)) / 3.0
        field = (np.roll(field, 1, 1) + field + np.roll(field, -1, 1)) / 3.0
    qi, ki = np.meshgrid(np.arange(n_q), np.arange(n_k), indexing="ij")
    field -= 0.35 * np.abs(qi - ki) / n_q
    thresh = np.sort(field.ravel())[-keep]
    return field >= thresh


def coarsen(mask, qf=4, kf=2):
    n_q, n_k = mask.shape
    return mask.reshape(n_q // qf, qf, n_k // kf, kf).any(axis=(1, 3))


def draw_mask(ax, m, color, superblocks=False):
    n_q, n_k = m.shape
    for i in range(n_q):
        for j in range(n_k):
            if m[i, j]:
                ax.add_patch(Rectangle((j, n_q - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="none"))
    if superblocks:
        for j in range(n_k + 1):
            ax.plot([j, j], [0, n_q], color="#ececec", lw=0.35, zorder=0)
        for i in range(n_q + 1):
            ax.plot([0, n_k], [i, i], color="#ececec", lw=0.35, zorder=0)
        for j in range(0, n_k + 1, 2):
            ax.plot([j, j], [0, n_q], color="#9a9a9a", lw=0.8)
        for i in range(0, n_q + 1, 4):
            ax.plot([0, n_k], [i, i], color="#9a9a9a", lw=0.8)
    else:
        for j in range(n_k + 1):
            ax.plot([j, j], [0, n_q], color="#dddddd", lw=0.45)
        for i in range(n_q + 1):
            ax.plot([0, n_k], [i, i], color="#dddddd", lw=0.45)
    ax.set_xlim(-0.2, n_k + 0.2)
    ax.set_ylim(-0.2, n_q + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def fig1_schematic():
    best = None
    for seed in range(400):
        m = clustered_mask(seed=seed)
        c = coarsen(m)
        infl = c.mean() / m.mean()
        if best is None or abs(infl - 2.4) < abs(best[2] - 2.4):
            best = (m, c, infl, seed)
    mask, cmask, infl, seed = best
    aligned = clustered_mask(n_q=4, n_k=8, keep=3, seed=7)

    fig = plt.figure(figsize=(DOUBLE_COL, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], wspace=0.40,
                          hspace=0.60)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_mask(ax_a, mask, BLUE, superblocks=True)
    ax_a.set_title("$\\bf{a}$  VSA selects on 64-token tiles \u2014 10% kept",
                   fontsize=9, pad=7, loc="left")

    draw_mask(ax_b, cmask, VERMILLION)
    ax_b.set_title("$\\bf{b}$  mask pooled onto kernel blocks \u2192 25% kept",
                   fontsize=8.5, color=VERMILLION, pad=4, loc="left")

    draw_mask(ax_c, aligned, GREEN)
    ax_c.set_title("$\\bf{c}$  selector re-aligned to 256-token tiles "
                   "\u2192 10% kept", fontsize=8.5, color=GREEN, pad=4,
                   loc="left")

    for target, color, rad in [(ax_b, VERMILLION, -0.18),
                               (ax_c, GREEN, 0.18)]:
        pa = ax_a.get_position()
        pt = target.get_position()
        arrow = FancyArrowPatch((pa.x1 + 0.004, (pa.y0 + pa.y1) / 2),
                                (pt.x0 - 0.007, (pt.y0 + pt.y1) / 2),
                                transform=fig.transFigure,
                                arrowstyle="-|>", mutation_scale=11,
                                color=color, lw=1.3,
                                connectionstyle=f"arc3,rad={rad}")
        fig.patches.append(arrow)
    save(fig, "fig1_geometry_schematic", ["illustrative masks (synthetic); "
         "2.4x inflation factor from P4_PERF_ROOT_CAUSE.md"])
    print(f"  fig1 toy mask: keep {mask.mean():.3f} -> {cmask.mean():.3f} "
          f"({infl:.2f}x), seed {seed}")


# ---------------------------------------------------------------- figure 2
def fig2_kernel():
    rows = read_csv("kernel_latency.csv")
    dense_ms = next(float(r["median_ms"]) for r in rows
                    if r["retained_fraction"] == "dense")
    sparse = [(float(r["retained_fraction"]), float(r["median_ms"]))
              for r in rows if r["retained_fraction"] != "dense"]
    retained = np.array([r for r, _ in sparse])
    sparse_ms = np.array([m for _, m in sparse])
    ideal = dense_ms * retained
    best_speedup = dense_ms / sparse_ms[np.argmin(retained)]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    ax.plot(retained, ideal, ls="--", color=LIGHT, lw=1.2)
    ax.plot(retained, sparse_ms, marker="o", ms=4.5, color=GREEN, lw=1.7)

    ax.text(0.62, 5.5, "measured", color=GREEN, fontsize=8.5,
            fontweight="bold", ha="center")
    ax.text(0.155, 1.00, "ideal", color="#9d9d9d", fontsize=8, ha="center")
    ax.annotate(f"{best_speedup:.1f}\u00d7 faster\nat 10% kept",
                xy=(retained[-1] * 1.02, sparse_ms[-1] * 1.08),
                xytext=(0.135, 4.0), fontsize=8.5, color=GREEN,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks(retained)
    ax.set_xticklabels([f"{r:.0%}" for r in retained])
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("fraction of attention kept")
    ax.set_ylabel("kernel time (ms)")
    save(fig, "fig2_kernel_scaling", ["data/kernel_latency.csv"])


# ---------------------------------------------------------------- figure 3
def fig3_e2e():
    rows = read_csv("e2e_performance.csv")
    panels = [("a", "480p"), ("b", "720p")]
    order = ["P0", "P2", "P4G"]
    colors = {"P0": GREY, "P2": BLUE, "P4G": GREEN}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 1.9), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, (letter, res) in zip(axes, panels):
        sub = {r["system"]: r for r in rows if r["resolution"] == res}
        vals = [float(sub[s]["e2e_s"]) for s in order]
        labels = [sub[s]["label"] for s in order]
        p0 = vals[0]
        y = np.arange(3)[::-1]
        ax.barh(y, vals, color=[colors[s] for s in order], height=0.58)
        for yi, v in zip(y, vals):
            sp = p0 / v
            best = v == min(vals)
            ax.text(v + p0 * 0.03, yi,
                    f"{sp:.2f}\u00d7" if sp != 1 else f"{v:.0f} s",
                    va="center", fontsize=9.5 if best else 8.5,
                    fontweight="bold" if best else "normal",
                    color=GREEN if best else "#666666")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.text(0.02, 1.06, f"$\\bf{{{letter}}}$  {res}",
                transform=ax.transAxes, fontsize=9, va="bottom")
        ax.set_xlim(0, p0 * 1.30)
        ax.set_xticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.5, -0.07, "end-to-end generation time "
             "(bar length; shorter is faster)", ha="center", fontsize=7.5,
             color="#888888")
    save(fig, "fig3_e2e_performance", ["data/e2e_performance.csv"])


# ---------------------------------------------------------------- figure 4
def fig4_quality():
    rows = read_csv("quality_p4g_vs_p2.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    y = np.arange(len(rows))[::-1]
    ylabels = []
    for yi, r in zip(y, rows):
        mean = float(r["mean_delta"])
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        sig = r["holm_significant"] == "yes"
        color = (GREEN if mean > 0 else VERMILLION) if sig else "#c4c4c4"
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot(mean, yi, "o", color=color, ms=5.5, zorder=3)
        ylabels.append(f"{r['dimension']}  ($n$={r['n']})")
    ax.axvline(0, color="#b5b5b5", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlim(-0.06, 0.24)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xticklabels(["\u22120.05", "0", "+0.05", "+0.10", "+0.15",
                        "+0.20"])
    ax.set_xlabel("VBench score change vs deployed VSA "
                  "(paired mean, 95% bootstrap CI)")
    ax.text(-0.0035, len(rows) - 0.35, "worse", fontsize=7.5,
            color="#999999", style="italic", ha="right")
    ax.text(0.0035, len(rows) - 0.35, "better", fontsize=7.5,
            color="#999999", style="italic", ha="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig4_quality_forest", ["data/quality_p4g_vs_p2.csv"])


if __name__ == "__main__":
    fig1_schematic()
    fig2_kernel()
    fig3_e2e()
    fig4_quality()
    with open(HERE / "provenance_manifest.json", "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print("provenance_manifest.json written")
