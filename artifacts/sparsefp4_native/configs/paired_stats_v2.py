"""Unified paired prompt-level bootstrap statistics for the V2 analysis.

One implementation serves every arm contrast (no per-contrast scripts):

  P1  - P0   dense NVFP4 quality effect
  P4  - P4G  sparse NVFP4 quality effect
  I = (P4-P4G) - (P1-P0)   factorial NVFP4 x sparsity interaction
  P4G - P2   geometry-alignment quality effect

For each VBench dimension (paired by prompt idx, only prompts present in
both/all arms): n, mean diff, median diff, 95% percentile bootstrap CI
(>=10k resamples), two-sided paired bootstrap p-value, Holm correction
across the 7 dimensions within each contrast family.

Pixel metrics (PSNR/SSIM/LPIPS) are measured against the P0 reference, so:
  - for P1 vs P0 we bootstrap P1's own PSNR/SSIM/LPIPS-vs-P0 level (no
    difference exists: P0 vs itself is degenerate);
  - for P4-P4G and P4G-P2 we bootstrap the paired per-prompt difference
    of each arm's similarity-to-P0;
  - the interaction is defined on VBench dimensions only.

Usage:
  paired_stats_v2.py --quality-dir artifacts/sparsefp4_native/raw/quality
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
PIXEL = ["psnr", "ssim", "lpips"]
N_BOOT = 10000
SEED = 20260817


def load_vbench(quality_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """metric -> arm -> {prompt idx: score}, deduped across shard files."""
    seen: set[tuple] = set()
    table: dict[str, dict[str, dict[int, float]]] = {}
    for f in sorted(quality_dir.glob("paper_vbench*.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "score" not in r:
                continue
            key = (r["metric"], r["idx"], r["arm"])
            if key in seen:
                continue
            seen.add(key)
            table.setdefault(r["metric"], {}).setdefault(r["arm"], {})[r["idx"]] = r["score"]
    return table


def load_pixel(quality_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """metric -> arm -> {prompt idx: value vs P0 reference}."""
    table: dict[str, dict[str, dict[int, float]]] = {m: {} for m in PIXEL}
    seen: set[tuple] = set()
    for f in sorted(quality_dir.glob("paper_paired.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            key = (r["idx"], r["arm"])
            if key in seen:
                continue
            seen.add(key)
            for m in PIXEL:
                table[m].setdefault(r["arm"], {})[r["idx"]] = r[m]
    return table


def bootstrap(d: np.ndarray, rng: np.random.Generator) -> dict:
    """Percentile bootstrap of the mean of paired differences d, plus a
    two-sided bootstrap p-value: p = 2*min(P(boot<=0), P(boot>=0))."""
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = min(1.0, max(p, 2.0 / N_BOOT))  # resolution floor: cannot claim p=0
    return dict(n=n, mean=float(d.mean()), median=float(np.median(d)),
                ci_low=float(lo), ci_high=float(hi), p_boot=float(p))


def holm(results: list[dict]) -> None:
    """Holm-Bonferroni step-down over the p_boot values, in place."""
    order = sorted(range(len(results)), key=lambda i: results[i]["p_boot"])
    m = len(results)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i]["p_boot"])
        adj = max(adj, prev)  # enforce monotonicity
        prev = adj
        results[i]["p_holm"] = adj


def contrast_diffs(a: dict[int, float], b: dict[int, float]) -> np.ndarray | None:
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    return np.array([a[i] - b[i] for i in common])


def run_contrast(vb, px, arm_a: str, arm_b: str, rng) -> dict:
    """Paired arm_a - arm_b on every VBench dim + pixel-metric differences."""
    out = {"contrast": f"{arm_a} - {arm_b}", "vbench": [], "pixel": []}
    for dim in DIMS:
        m = vb.get(dim, {})
        if arm_a not in m or arm_b not in m:
            continue
        d = contrast_diffs(m[arm_a], m[arm_b])
        if d is None:
            continue
        r = bootstrap(d, rng)
        r["metric"] = dim
        out["vbench"].append(r)
    holm(out["vbench"])
    for met in PIXEL:
        m = px.get(met, {})
        if arm_a not in m or arm_b not in m:
            continue
        d = contrast_diffs(m[arm_a], m[arm_b])
        if d is None:
            continue
        r = bootstrap(d, rng)
        r["metric"] = met
        out["pixel"].append(r)
    return out


def run_level(px, arm: str, rng) -> list[dict]:
    """Bootstrap CI of arm's own PSNR/SSIM/LPIPS level vs the P0 reference."""
    rows = []
    for met in PIXEL:
        vals = px.get(met, {}).get(arm)
        if not vals:
            continue
        d = np.array(sorted(vals.values()))
        r = bootstrap(d, rng)
        r.pop("p_boot")  # a level has no null hypothesis of zero
        r["metric"] = met
        rows.append(r)
    return rows


def run_interaction(vb, rng) -> dict:
    """I = (P4-P4G) - (P1-P0), paired per prompt where all 4 arms exist."""
    out = {"contrast": "(P4-P4G) - (P1-P0)", "vbench": []}
    for dim in DIMS:
        m = vb.get(dim, {})
        if any(a not in m for a in ("P0", "P1", "P4", "P4G")):
            continue
        common = sorted(set(m["P0"]) & set(m["P1"]) & set(m["P4"]) & set(m["P4G"]))
        if len(common) < 5:
            continue
        d = np.array([(m["P4"][i] - m["P4G"][i]) - (m["P1"][i] - m["P0"][i])
                      for i in common])
        r = bootstrap(d, rng)
        r["metric"] = dim
        out["vbench"].append(r)
    holm(out["vbench"])
    return out


def fmt_rows(rows: list[dict], holm_col: bool = True) -> list[str]:
    lines = []
    for r in rows:
        cells = [r["metric"], str(r["n"]), f"{r['mean']:+.4f}", f"{r['median']:+.4f}",
                 f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"]
        if "p_boot" in r:
            cells.append(f"{r['p_boot']:.4f}")
        if holm_col and "p_holm" in r:
            cells.append(f"{r['p_holm']:.4f}")
            cells.append("yes" if r["p_holm"] < 0.05 else "no")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


VB_HEADER = ("| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) "
             "| sig@0.05 |\n|---|---|---|---|---|---|---|---|")
PX_HEADER = "| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |\n|---|---|---|---|---|---|"
LV_HEADER = "| Metric | n | mean | median | 95% CI |\n|---|---|---|---|---|"


def write_table(path: Path, title: str, note: str, contrast: dict,
                level_rows: list[dict] | None = None,
                level_note: str = "") -> None:
    lines = [f"# {title}", "", note, "",
             f"Prompt-level percentile bootstrap, {N_BOOT} resamples, seed {SEED}; "
             "two-sided bootstrap p; Holm correction across the "
             f"{len(contrast['vbench'])} VBench dimensions.", "",
             "## VBench dimensions (paired Δ)", "", VB_HEADER]
    lines += fmt_rows(contrast["vbench"])
    if contrast.get("pixel"):
        lines += ["", "## Pixel metrics vs P0 reference (paired Δ of similarity-to-P0)",
                  "", PX_HEADER] + fmt_rows(contrast["pixel"], holm_col=False)
    if level_rows:
        lines += ["", level_note, "", LV_HEADER]
        for r in level_rows:
            lines.append(f"| {r['metric']} | {r['n']} | {r['mean']:+.4f} | "
                         f"{r['median']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-dir", type=Path,
                    default=Path("artifacts/sparsefp4_native/raw/quality"))
    ap.add_argument("--tables-dir", type=Path,
                    default=Path("artifacts/sparsefp4_native/tables"))
    ap.add_argument("--stats-dir", type=Path,
                    default=Path("artifacts/sparsefp4_native/raw/statistics"))
    args = ap.parse_args()
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    vb = load_vbench(args.quality_dir)
    px = load_pixel(args.quality_dir)

    # --- P1 - P0: dense NVFP4 effect ---
    c_dense = run_contrast(vb, px, "P1", "P0", rng)
    p1_levels = run_level(px, "P1", rng)
    c_dense_json = dict(c_dense, p1_pixel_levels=p1_levels)
    (args.stats_dir / "p1_vs_p0_bootstrap.json").write_text(
        json.dumps(c_dense_json, indent=2) + "\n")
    write_table(
        args.tables_dir / "p1_vs_p0_quality_bootstrap.md",
        "P1 - P0: dense NVFP4 quality effect (paired, 326-prompt protocol)",
        "Δ_dense = P1 - P0 per prompt; positive = NVFP4 scores higher. "
        "Pixel metrics are similarity of P1 to the P0 reference, so no paired "
        "pixel difference exists for this contrast; levels below.",
        # pixel diffs are undefined for this contrast (P0 vs itself degenerate)
        {**c_dense, "pixel": []},
        level_rows=p1_levels,
        level_note=("## P1 similarity to P0 (levels, bootstrap CI) — no null "
                    "hypothesis; describes how far dense NVFP4 output is from "
                    "dense BF16"))

    # --- P4 - P4G: sparse NVFP4 effect ---
    c_sparse = run_contrast(vb, px, "P4", "P4G", rng)
    (args.stats_dir / "p4_vs_p4g_bootstrap.json").write_text(
        json.dumps(c_sparse, indent=2) + "\n")
    write_table(
        args.tables_dir / "p4_vs_p4g_quality_bootstrap.md",
        "P4 - P4G: sparse NVFP4 quality effect (paired, 326-prompt protocol)",
        "Δ_sparse = P4 - P4G per prompt (identical VSA256 geometry, only fine QK "
        "precision differs); positive = NVFP4 scores higher. Pixel Δ compares "
        "each arm's similarity-to-P0.",
        c_sparse)

    # --- interaction ---
    inter = run_interaction(vb, rng)
    (args.stats_dir / "nvfp4_sparsity_interaction.json").write_text(
        json.dumps(inter, indent=2) + "\n")
    write_table(
        args.tables_dir / "nvfp4_sparsity_interaction.md",
        "Factorial interaction: does sparsity amplify the NVFP4 quality effect?",
        "I = (P4 - P4G) - (P1 - P0) per prompt (difference-in-differences). "
        "I < 0 means NVFP4 hurts MORE under VSA256 sparsity than under dense "
        "attention; I ~ 0 means the NVFP4 effect is the same in both regimes. "
        "Defined on VBench dimensions only (pixel metrics use P0 as reference, "
        "so the dense NVFP4 term is degenerate).",
        inter)

    # --- P4G - P2: geometry alignment ---
    c_geo = run_contrast(vb, px, "P4G", "P2", rng)
    (args.stats_dir / "p4g_vs_p2_bootstrap.json").write_text(
        json.dumps(c_geo, indent=2) + "\n")
    write_table(
        args.tables_dir / "p4g_vs_p2_quality_bootstrap.md",
        "P4G - P2: geometry-aligned VSA256/FA4 vs deployed VSA (paired, BF16)",
        "Δ_geometry = P4G - P2 per prompt (both BF16 sparse; selector tile "
        "geometry and fine kernel differ); positive = VSA256/FA4 scores higher. "
        "Pixel Δ compares each arm's similarity-to-P0. No non-inferiority "
        "margin is asserted; observed effects + CIs only.",
        c_geo)


if __name__ == "__main__":
    main()
