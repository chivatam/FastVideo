"""Aggregate C5 operator-matrix JSONL into the four-arm table (median/IQR/p10/p90/n)."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

METRICS = ("mse", "rel_l2", "cosine", "snr_db", "max_abs")


def q(vals, p):
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
    return vals[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    by_arm = collections.defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_arm[r["arm"]].append(r)

    lines = ["# C5 controlled 2x2 operator matrix — aggregate", ""]
    n_cells = len({r['cell'] for r in rows})
    keep64 = statistics.median(r['keep64'] for r in rows)
    keepfa4 = statistics.median(r['keep_fa4'] for r in rows)
    lines.append(f"Cells: {n_cells} (5 layers x 5 timesteps, Wan2.1-T2V-1.3B, "
                 f"480x832x81, VSA sparsity 0.90, seed 1234, positive CFG branch).")
    lines.append(f"Median retained fraction: 64x64 mask {keep64:.4f}; "
                 f"FA4 256x128 coarsened mask {keepfa4:.4f}.")
    lines.append("")
    lines.append("| Arm | n | metric | median | IQR | p10 | p90 |")
    lines.append("|---|---|---|---|---|---|---|")
    order = ["A0_vs_fp32_oracle", "B0_vs_A0", "C0_vs_A0", "D0_vs_A0", "D0_vs_C0",
             "C0_TRITON64_vs_A0", "D0_vs_dequant_oracle"]
    for arm in order:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        for m in METRICS:
            vals = [r[m] for r in rs if m in r]
            if not vals:
                continue
            med = statistics.median(vals)
            iqr = q(vals, 0.75) - q(vals, 0.25)
            lines.append(f"| {arm} | {len(vals)} | {m} | {med:.6g} | {iqr:.3g} "
                         f"| {q(vals, 0.10):.6g} | {q(vals, 0.90):.6g} |")
        fin = all(r.get("finite", False) for r in rs)
        lines.append(f"| {arm} | {len(rs)} | finite | {fin} | | | |")

    errs = [r for r in rows if "error" in r]
    if errs:
        lines.append("")
        lines.append(f"Errors: {len(errs)} rows — {[e['arm'] for e in errs[:5]]}")

    # per-timestep breakdown for D0_vs_A0 (explanatory only)
    lines.append("")
    lines.append("## D0 vs A0 by timestep (rel_l2 median)")
    lines.append("")
    lines.append("| timestep | B0 | C0 | D0 |")
    lines.append("|---|---|---|---|")
    steps = sorted({r["timestep"] for r in rows})
    for t in steps:
        vals = {}
        for arm in ("B0_vs_A0", "C0_vs_A0", "D0_vs_A0"):
            v = [r["rel_l2"] for r in by_arm[arm] if r["timestep"] == t]
            vals[arm] = statistics.median(v) if v else float("nan")
        lines.append(f"| {t} | {vals['B0_vs_A0']:.4g} | {vals['C0_vs_A0']:.4g} "
                     f"| {vals['D0_vs_A0']:.4g} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
