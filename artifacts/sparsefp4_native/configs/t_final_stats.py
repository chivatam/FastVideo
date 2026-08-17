"""Final DQ-VSA recovery statistics (paper-scale, 326-prompt protocol).

Candidates (all served through the NATIVE P4 path — VSA256/FA4 geometry,
exact 10% retention, BF16 selector, native NVFP4 QK, BF16 PV):

  T1  standard flow-matching / task-loss QAT                (not evaluated
      at paper scale: dev-gate motion collapse, see t_matrix_gates.md)
  T2  velocity distillation from frozen P4G teacher
      + fake-quant NVFP4 forward + naive/high-precision attention backward
  T3  velocity distillation from frozen P4G teacher
      + fake-quant NVFP4 forward + Attn-QAT-consistent backward semantics

Contrasts (unified implementation from paired_stats_v2): each candidate vs
P4G (residual gap to the BF16 teacher) and vs P4 (improvement over the
untrained arm), plus the original P4 - P4G deficit for reference.

Also reports the DESCRIPTIVE recovery fraction on the two pre-declared
targets (imaging_quality, dynamic_degree):
    recovery_fraction = 1 - |trained_mean - P4G_mean| / |P4_mean - P4G_mean|
computed on the prompts common to all three arms. Not a significance
statistic.

Writes: tables/dqvsa_recovery_bootstrap.md, raw/statistics/dqvsa_recovery.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paired_stats_v2 import (PIXEL, SEED, VB_HEADER, PX_HEADER, fmt_rows,  # noqa: E402
                             run_contrast)

QUALITY = Path("artifacts/sparsefp4_native/raw/quality")
TABLE = Path("artifacts/sparsefp4_native/tables/dqvsa_recovery_bootstrap.md")
STATS = Path("artifacts/sparsefp4_native/raw/statistics/dqvsa_recovery.json")

CANDIDATES = ("T2c250", "T3c250", "T3c500")
TARGET_DIMS = ("imaging_quality", "dynamic_degree")
VB_PATTERNS = ("paper_vbench*.shard*.jsonl", "t_final_vbench*.shard*.jsonl",
               "t2_final_vbench*.shard*.jsonl")
PX_PATTERNS = ("paper_paired.shard*.jsonl", "t_final_paired*.shard*.jsonl",
               "t2_final_paired*.shard*.jsonl")


def load_vbench_all() -> dict:
    seen: set[tuple] = set()
    table: dict = {}
    for pattern in VB_PATTERNS:
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
    for pattern in PX_PATTERNS:
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


def recovery_fractions(vb: dict) -> list[dict]:
    rows = []
    for cand in CANDIDATES:
        for dim in TARGET_DIMS:
            m = vb.get(dim, {})
            if any(a not in m for a in (cand, "P4", "P4G")):
                continue
            common = sorted(set(m[cand]) & set(m["P4"]) & set(m["P4G"]))
            if len(common) < 5:
                continue
            mu = {a: float(np.mean([m[a][i] for i in common])) for a in (cand, "P4", "P4G")}
            deficit = abs(mu["P4"] - mu["P4G"])
            frac = 1.0 - abs(mu[cand] - mu["P4G"]) / deficit if deficit > 1e-9 else float("nan")
            rows.append(dict(candidate=cand, dim=dim, n=len(common),
                             mean_trained=mu[cand], mean_P4=mu["P4"], mean_P4G=mu["P4G"],
                             recovery_fraction=frac))
    return rows


def main() -> None:
    vb = load_vbench_all()
    px = load_pixel_all()
    rng = np.random.default_rng(SEED)

    contrasts = [(c, ref) for c in CANDIDATES for ref in ("P4G", "P4")]
    contrasts.append(("P4", "P4G"))

    out: dict = {"definitions": {
        "T1": "standard flow-matching/task-loss QAT (excluded at paper scale: dev-gate motion collapse)",
        "T2": "velocity distillation from frozen P4G teacher + fake-quant NVFP4 fwd + naive backward",
        "T3": "velocity distillation from frozen P4G teacher + fake-quant NVFP4 fwd + Attn-QAT-consistent backward",
    }}
    lines = ["# DQ-VSA recovery — paper-scale paired bootstrap (326-prompt protocol)",
             "",
             "All candidates served through the NATIVE P4 path (VSA256/FA4, exact 10% "
             "retention, BF16 selector, native NVFP4 QK, BF16 PV; serving receipts in "
             "`DQVSA_NATIVE_SERVING_PROOF.md`).",
             "",
             "- **T2** = velocity distillation from frozen P4G teacher + fake-quant NVFP4 "
             "forward + **naive/high-precision-style attention backward**",
             "- **T3** = velocity distillation from frozen P4G teacher + fake-quant NVFP4 "
             "forward + **Attn-QAT-consistent backward semantics**",
             "- (T1 = task-loss flow-matching QAT; excluded from paper-scale eval after "
             "dev-gate motion collapse, `t_matrix_gates.md`.)",
             "",
             "Unified statistics: 10k prompt-level bootstrap, 95% CI, two-sided bootstrap "
             "p, Holm across the 7 VBench dimensions per contrast. Pixel metrics reported "
             "but NOT used for winner selection."]

    # descriptive recovery fractions first
    rf = recovery_fractions(vb)
    out["recovery_fractions"] = rf
    lines += ["", "## Descriptive recovery fractions (pre-declared targets; NOT a "
              "significance statistic)", "",
              "| Candidate | Dimension | n | trained mean | P4 mean | P4G mean | recovery |",
              "|---|---|---|---|---|---|---|"]
    for r in rf:
        lines.append(f"| {r['candidate']} | {r['dim']} | {r['n']} | "
                     f"{r['mean_trained']:.4f} | {r['mean_P4']:.4f} | {r['mean_P4G']:.4f} | "
                     f"{100 * r['recovery_fraction']:.0f}% |")

    for a, b in contrasts:
        c = run_contrast(vb, px, a, b, rng)
        out[f"{a}-{b}"] = c
        if not c["vbench"]:
            lines += ["", f"## {a} - {b}: MISSING DATA"]
            continue
        lines += ["", f"## {a} - {b}", "", "### VBench", "", VB_HEADER]
        lines += fmt_rows(c["vbench"])
        if c.get("pixel"):
            lines += ["", "### Pixel similarity-to-P0 (paired Δ; descriptive only)",
                      "", PX_HEADER] + fmt_rows(c["pixel"], holm_col=False)

    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(out, indent=2) + "\n")
    TABLE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
