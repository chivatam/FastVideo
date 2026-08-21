"""Phase-2 Parts B/D/E/F: decomposition stats, KV traffic, MMA work, intensity.

Reads the pair dataset from build_pair_dataset.py and reports, from REAL
masks only:

  B. shared / q0_private / q1_private / union distributions (overall and by
     layer / timestep / head),
  D. exact theoretical KV bytes for baseline vs union loading,
  E. MMA interaction counts for Strategy A, B1 (== baseline) and B2 (dense
     union) with compute inflation,
  F. relative arithmetic intensity (FLOPs / KV byte), baseline = 1.0x.

    python artifacts/vsa_local_reuse_phase2/analyze_decomposition.py \
        --dataset /mnt/nvme/outputs/phase2_pairs \
        --out artifacts/vsa_local_reuse_phase2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

# One 64-token Q block x one 64-token KV block x head_dim 128, multiply-add = 2 FLOPs:
#   QK^T: 2 * 64 * 64 * 128     P@V: 2 * 64 * 64 * 128
FLOPS_QK = 2 * 64 * 64 * 128
FLOPS_PV = 2 * 64 * 64 * 128
FLOPS_PER_INTERACTION = FLOPS_QK + FLOPS_PV
# One KV block in bf16: K[64,128] + V[64,128]
KV_BYTES_PER_BLOCK = 64 * 128 * 2 * 2

QUANTS = {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90, "p99": 99}


def qrow(x: np.ndarray) -> dict[str, float]:
    out = {"mean": float(np.mean(x)), "median": float(np.median(x))}
    out.update({name: float(np.percentile(x, q)) for name, q in QUANTS.items()})
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    d = np.load(os.path.join(args.dataset, "pair_counts.npz"))
    k = float(d["k_nominal"])
    inter = d["inter"].astype(np.float64)
    s0 = d["s0"].astype(np.float64)
    s1 = d["s1"].astype(np.float64)
    union = s0 + s1 - inter
    p0 = s0 - inter
    p1 = s1 - inter

    res_dir = os.path.join(args.out, "results")
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Part B: decomposition distributions ----
    decomp_rows = []
    for name, arr in (("shared", inter), ("q0_private", p0), ("q1_private", p1), ("union", union)):
        row = {
            "quantity": name,
            **{
                f"{k2}_abs": v
                for k2, v in qrow(arr).items()
            },
            **{
                f"{k2}_over_K": v
                for k2, v in qrow(arr / k).items()
            }
        }
        decomp_rows.append(row)
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(os.path.join(res_dir, "decomposition_summary.csv"), index=False)

    # Breakdowns (median of shared/K, union/K per layer / step / head)
    base = pd.DataFrame({
        "layer": d["layer"],
        "step": d["step"],
        "head": d["head"],
        "shared_over_K": inter / k,
        "union_over_K": union / k,
        "reuse": (s0 + s1) / union,
    })
    for dim, fname in (("layer", "layer_summary.csv"), ("step", "timestep_summary.csv"), ("head", "head_summary.csv")):
        agg = base.groupby(dim).agg(median_shared_over_K=("shared_over_K", "median"),
                                    median_union_over_K=("union_over_K", "median"),
                                    median_reuse=("reuse", "median"),
                                    n=("reuse", "size")).reset_index()
        agg.to_csv(os.path.join(res_dir, fname), index=False)

    # ---- Part D: exact KV traffic ----
    bytes_baseline = (s0 + s1) * KV_BYTES_PER_BLOCK
    bytes_union = union * KV_BYTES_PER_BLOCK
    saved = bytes_baseline - bytes_union
    pct_saved = saved / bytes_baseline
    traffic: dict[str, Any] = {
        "kv_bytes_per_block": KV_BYTES_PER_BLOCK,
        "total_bytes_baseline": float(bytes_baseline.sum()),
        "total_bytes_union": float(bytes_union.sum()),
        "total_pct_saved": float(saved.sum() / bytes_baseline.sum()),
        "per_pair_pct_saved": qrow(pct_saved),
        "per_pair_traffic_reduction_ratio": qrow(bytes_baseline / bytes_union),
    }

    # ---- Part E: MMA interactions ----
    inter_baseline = s0 + s1  # current PR#1719
    inter_a = 2 * inter + p0 + p1  # shared runs both Q, privates one
    inter_b1 = s0 + s1  # conditional: only member Q
    inter_b2 = 2 * union  # dense grouped: both Q for all
    assert np.array_equal(inter_a, inter_baseline), "Strategy A must match baseline FLOPs"
    b2_inflation = inter_b2 / inter_baseline - 1.0
    mma: dict[str, Any] = {
        "flops_per_interaction_64x64x128": FLOPS_PER_INTERACTION,
        "interactions_baseline_total": float(inter_baseline.sum()),
        "strategy_a_extra_flops_pct": 0.0,
        "strategy_b1_extra_flops_pct": 0.0,
        "strategy_b2_inflation": qrow(b2_inflation),
        "strategy_b2_total_inflation": float(inter_b2.sum() / inter_baseline.sum() - 1.0),
    }

    # ---- Part F: relative arithmetic intensity (FLOPs / KV byte) ----
    def intensity(interactions: np.ndarray, blocks_loaded: np.ndarray) -> float:
        return float((interactions.sum() * FLOPS_PER_INTERACTION) / (blocks_loaded.sum() * KV_BYTES_PER_BLOCK))

    i_base = intensity(inter_baseline, s0 + s1)
    intensity_rel = {
        "baseline": 1.0,
        "strategy_a": intensity(inter_a, union) / i_base,
        "strategy_b1": intensity(inter_b1, union) / i_base,
        "strategy_b2": intensity(inter_b2, union) / i_base,
    }

    costs = pd.DataFrame([
        {
            "strategy": "baseline_pr1719",
            "kv_blocks_loaded": float((s0 + s1).mean()),
            "interactions": float(inter_baseline.mean()),
            "extra_flops_pct": 0.0,
            "kv_traffic_saved_pct": 0.0,
            "rel_arithmetic_intensity": 1.0
        },
        {
            "strategy": "A_shared_private",
            "kv_blocks_loaded": float(union.mean()),
            "interactions": float(inter_a.mean()),
            "extra_flops_pct": 0.0,
            "kv_traffic_saved_pct": 100 * traffic["total_pct_saved"],
            "rel_arithmetic_intensity": intensity_rel["strategy_a"]
        },
        {
            "strategy": "B1_union_conditional",
            "kv_blocks_loaded": float(union.mean()),
            "interactions": float(inter_b1.mean()),
            "extra_flops_pct": 0.0,
            "kv_traffic_saved_pct": 100 * traffic["total_pct_saved"],
            "rel_arithmetic_intensity": intensity_rel["strategy_b1"]
        },
        {
            "strategy": "B2_union_dense",
            "kv_blocks_loaded": float(union.mean()),
            "interactions": float(inter_b2.mean()),
            "extra_flops_pct": 100 * mma["strategy_b2_total_inflation"],
            "kv_traffic_saved_pct": 100 * traffic["total_pct_saved"],
            "rel_arithmetic_intensity": intensity_rel["strategy_b2"]
        },
    ])
    costs.to_csv(os.path.join(res_dir, "strategy_costs.csv"), index=False)

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, arr, color in (("shared/K", inter / k, "tab:green"), ("q0_private/K", p0 / k, "tab:orange"),
                             ("q1_private/K", p1 / k, "tab:red")):
        ax.hist(arr, bins=100, alpha=0.55, label=name, color=color)
    ax.set(xlabel="fraction of K", ylabel="pairs", title="Shared / private decomposition (real pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "shared_private_distribution.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(union / k, bins=100, color="tab:blue")
    ax.axvline(np.median(union / k), color="red", ls="--", label=f"median={np.median(union / k):.3f}")
    ax.set(xlabel="|union| / K", ylabel="pairs", title="Union size (real pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "union_distribution.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(100 * pct_saved, bins=100, color="tab:purple")
    ax.axvline(np.median(100 * pct_saved), color="red", ls="--", label=f"median={np.median(100 * pct_saved):.1f}%")
    ax.set(xlabel="% KV bytes saved per pair", ylabel="pairs", title="KV traffic saved (union vs baseline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bytes_saved_distribution.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(100 * b2_inflation, bins=100, color="tab:brown")
    ax.axvline(np.median(100 * b2_inflation),
               color="red",
               ls="--",
               label=f"median={np.median(100 * b2_inflation):.1f}%")
    ax.set(xlabel="% extra MMA interactions (B2 vs baseline)", ylabel="pairs", title="Strategy B2 compute inflation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "compute_inflation_b2.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(intensity_rel)
    ax.bar(names, [intensity_rel[n] for n in names], color=["gray", "tab:green", "tab:blue", "tab:red"])
    ax.set(ylabel="relative FLOPs / KV byte", title="Arithmetic intensity (baseline = 1.0)")
    for i, n in enumerate(names):
        ax.text(i, intensity_rel[n] + 0.02, f"{intensity_rel[n]:.2f}x", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "arithmetic_intensity_comparison.png"), dpi=120)
    plt.close(fig)

    summary: dict[str, Any] = {
        "n_pairs": int(inter.size),
        "K": int(k),
        "decomposition_median_abs": {
            "shared": float(np.median(inter)),
            "q0_private": float(np.median(p0)),
            "q1_private": float(np.median(p1)),
            "union": float(np.median(union)),
        },
        "traffic": traffic,
        "mma": mma,
        "relative_arithmetic_intensity": intensity_rel,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(decomp_df.to_string(index=False))
    print(costs.to_string(index=False))
    print(
        json.dumps({k2: summary[k2]
                    for k2 in ("decomposition_median_abs", "relative_arithmetic_intensity")}, indent=2))
    print(f"traffic total_pct_saved: {traffic['total_pct_saved']:.4f}")
    print(f"B2 median inflation: {mma['strategy_b2_inflation']['median']:.4f}")


if __name__ == "__main__":
    main()
