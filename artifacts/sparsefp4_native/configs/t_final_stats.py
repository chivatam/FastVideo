"""Final DQ-VSA recovery statistics: trained T3 gates vs teacher and untrained P4.

Contrasts (unified implementation from paired_stats_v2):
  T3c250 - P4G   residual gap to the BF16 teacher after 250 steps
  T3c500 - P4G   ... after 500 steps
  T3c250 - P4    recovery vs the untrained NVFP4 arm
  T3c500 - P4
Also re-reports P4 - P4G (the original deficit) for side-by-side reading.

Inputs: raw/quality/paper_vbench*.shard*.jsonl (V2 arms),
        raw/quality/t_final_vbench.shard*.jsonl (T3 arms),
        raw/quality/paper_paired.shard*.jsonl + t_final_paired.shard*.jsonl.
Writes: tables/t3_recovery_bootstrap.md, raw/statistics/t3_recovery.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paired_stats_v2 import (DIMS, PIXEL, SEED, VB_HEADER, PX_HEADER, fmt_rows,  # noqa: E402
                             run_contrast)

QUALITY = Path("artifacts/sparsefp4_native/raw/quality")
TABLE = Path("artifacts/sparsefp4_native/tables/t3_recovery_bootstrap.md")
STATS = Path("artifacts/sparsefp4_native/raw/statistics/t3_recovery.json")

CONTRASTS = [("T3c250", "P4G"), ("T3c500", "P4G"),
             ("T3c250", "P4"), ("T3c500", "P4"), ("P4", "P4G")]


def load_vbench_all() -> dict:
    """metric -> arm -> {idx: score} from V2 + t_final shard files."""
    seen: set[tuple] = set()
    table: dict = {}
    for pattern in ("paper_vbench*.shard*.jsonl", "t_final_vbench*.shard*.jsonl"):
        for f in sorted(QUALITY.glob(pattern)):
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


def load_pixel_all() -> dict:
    table: dict = {m: {} for m in PIXEL}
    seen: set[tuple] = set()
    for pattern in ("paper_paired.shard*.jsonl", "t_final_paired*.shard*.jsonl"):
        for f in sorted(QUALITY.glob(pattern)):
            for line in open(f):
                r = json.loads(line)
                key = (r["idx"], r["arm"])
                if key in seen:
                    continue
                seen.add(key)
                for m in PIXEL:
                    table[m].setdefault(r["arm"], {})[r["idx"]] = r[m]
    return table


def main() -> None:
    vb = load_vbench_all()
    px = load_pixel_all()
    rng = np.random.default_rng(SEED)

    out = {}
    lines = ["# DQ-VSA T3 recovery — paper-scale paired bootstrap (326-prompt protocol)",
             "",
             "T3 = velocity distillation from the frozen P4G-operator teacher with "
             "Attn-QAT-consistent backward, served through the native P4 path. "
             "Positive Δ = first arm scores higher. Same unified statistics as the "
             "V2 contrasts (10k bootstrap, Holm across 7 dims per contrast)."]
    for a, b in CONTRASTS:
        c = run_contrast(vb, px, a, b, rng)
        out[f"{a}-{b}"] = c
        lines += ["", f"## {a} - {b}", "", "### VBench", "", VB_HEADER]
        lines += fmt_rows(c["vbench"])
        if c.get("pixel"):
            lines += ["", "### Pixel similarity-to-P0 (paired Δ)", "", PX_HEADER]
            lines += fmt_rows(c["pixel"], holm_col=False)
    STATS.write_text(json.dumps(out, indent=2) + "\n")
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
