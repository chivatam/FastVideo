"""Phase-1 analysis: local cross-query KV-block reuse in real VSA masks.

Reads compact top-k captures produced by ``run_capture.py`` (via
``fastvideo_kernel.vsa_capture``) and computes, for several query-block
pairing strategies and local group sizes, the theoretical KV-load reuse
available if paired/grouped query blocks shared their selected KV blocks.

    python artifacts/vsa_overlap_phase1/analyze_vsa_overlap.py \
        --capture-root /mnt/nvme/outputs/vsa_capture \
        --out artifacts/vsa_overlap_phase1

Outputs: summary.json, pair_strategy_summary.csv, layer_summary.csv,
timestep_summary.csv, head_summary.csv, group_size_summary.csv, plots/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Part B: q_block_id -> (t_tile, h_tile, w_tile)
# ---------------------------------------------------------------------------


def qblock_coords(num_tiles: tuple[int, int, int]) -> torch.Tensor:
    """[Nq, 3] (t, h, w) tile coords in FastVideo's VSA flattening order.

    ``get_tile_partition_indices`` iterates t-tile outer, then h-tile, then
    w-tile, so q_block_id = t*(n_h*n_w) + h*n_w + w. Verified against the
    real index builder in test_vsa_overlap_phase1.py.
    """
    n_t, n_h, n_w = num_tiles
    ids = torch.arange(n_t * n_h * n_w)
    t = ids // (n_h * n_w)
    h = (ids % (n_h * n_w)) // n_w
    w = ids % n_w
    return torch.stack([t, h, w], dim=1)


def coords_to_id(c: torch.Tensor, num_tiles: tuple[int, int, int]) -> torch.Tensor:
    n_t, n_h, n_w = num_tiles
    return c[..., 0] * (n_h * n_w) + c[..., 1] * n_w + c[..., 2]


# ---------------------------------------------------------------------------
# Part D: pairing strategies (lists of [P, 2] q-block-id pairs)
# ---------------------------------------------------------------------------


def build_pairs(num_tiles: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    """Pair lists per strategy (evaluated within-shard, within-head)."""
    coords = qblock_coords(num_tiles)
    n_t, n_h, n_w = num_tiles
    nq = coords.shape[0]

    def offset_pairs(dt: int, dh: int, dw: int) -> torch.Tensor:
        a = coords
        b = coords + torch.tensor([dt, dh, dw])
        valid = (b[:, 0] < n_t) & (b[:, 1] < n_h) & (b[:, 2] < n_w)
        return torch.stack([
            coords_to_id(a[valid], num_tiles),
            coords_to_id(b[valid], num_tiles),
        ], dim=1)

    ids = torch.arange(nq)
    pr1719 = torch.stack([ids[0::2], ids[1::2]], dim=1)  # mtile0=2p, mtile1=2p+1

    return {
        "pr1719_current": pr1719,
        "horizontal": offset_pairs(0, 0, 1),
        "vertical": offset_pairs(0, 1, 0),
        "temporal": offset_pairs(1, 0, 0),
    }


# ---------------------------------------------------------------------------
# Part E: aligned local groups (lists of [G, g] q-block ids)
# ---------------------------------------------------------------------------


def build_groups(num_tiles: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    """Aligned, non-overlapping local groups (what a g-wide CTA would own)."""
    n_t, n_h, n_w = num_tiles

    def block_groups(gt: int, gh: int, gw: int) -> torch.Tensor:
        groups = []
        for t0 in range(0, n_t - gt + 1, gt):
            for h0 in range(0, n_h - gh + 1, gh):
                for w0 in range(0, n_w - gw + 1, gw):
                    members = [(t0 + dt) * (n_h * n_w) + (h0 + dh) * n_w + (w0 + dw) for dt in range(gt)
                               for dh in range(gh) for dw in range(gw)]
                    groups.append(members)
        return torch.tensor(groups, dtype=torch.long)

    return {
        "g2_pr1719_1x1x2": block_groups(1, 1, 2),
        "g2_temporal_2x1x1": block_groups(2, 1, 1),
        "g4_spatial_1x2x2": block_groups(1, 2, 2),
        "g4_temporal_2x1x2": block_groups(2, 1, 2),
        "g4_wrun_1x1x4": block_groups(1, 1, 4),
        "g8_cube_2x2x2": block_groups(2, 2, 2),
        "g8_slab_1x2x4": block_groups(1, 2, 4),
    }


# ---------------------------------------------------------------------------
# Part C: overlap metrics
# ---------------------------------------------------------------------------


def masks_from_indices(indices: torch.Tensor, num_kv_blocks: int) -> torch.Tensor:
    """[H, Nq, K] int -> [H, Nq, Nk] bool. Invariant to per-row index order."""
    H, Nq, K = indices.shape
    m = torch.zeros(H, Nq, num_kv_blocks, dtype=torch.bool, device=indices.device)
    m.scatter_(-1, indices.long(), True)
    return m


def masks_from_shard(shard: dict, device: str) -> torch.Tensor:
    """Rebuild the [H, Nq, Nk] bool mask from either capture encoding."""
    nk = int(shard["num_kv_blocks"])
    if shard.get("indices") is not None:
        assert shard["indices"].shape[0] == 1, "expected batch=1 capture"
        return masks_from_indices(shard["indices"][0].to(device), nk)
    # Ragged fallback (rare fused_topk_mask tie): flat nonzero columns + row counts.
    H, Nq = int(shard["heads"]), int(shard["num_q_blocks"])
    counts = shard["row_counts"][0].to(device).long().reshape(H * Nq)
    cols = shard["indices_flat"].to(device).long()
    rows = torch.repeat_interleave(torch.arange(H * Nq, device=device), counts)
    m = torch.zeros(H * Nq, nk, dtype=torch.bool, device=device)
    m[rows, cols] = True
    return m.view(H, Nq, nk)


def pair_metrics(masks: torch.Tensor, pairs: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per (head, pair) metrics. masks: [H, Nq, Nk] bool; pairs: [P, 2].

    Set sizes come from the masks themselves (not nominal K), so rare
    ragged rows from top-k score ties are handled exactly.
    """
    ma = masks[:, pairs[:, 0], :]
    mb = masks[:, pairs[:, 1], :]
    inter = (ma & mb).sum(dim=-1).float()  # [H, P]
    size_sum = (ma.sum(dim=-1) + mb.sum(dim=-1)).float()
    union = size_sum - inter
    return {
        "intersection": inter,
        "overlap_k": inter / k,
        "jaccard": inter / union,
        "union": union,
        "union_over_k": union / k,
        "reuse_factor": size_sum / union,
        "shared_union_fraction": inter / union,
    }


def group_metrics(masks: torch.Tensor, groups: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per (head, group) union/reuse. groups: [G, g]."""
    mg = masks[:, groups, :]  # [H, G, g, Nk]
    union = mg.any(dim=2).sum(dim=-1).float()  # [H, G]
    size_sum = mg.sum(dim=-1).sum(dim=-1).float()  # [H, G] == g*K when rows are full
    return {
        "union_over_k": union / k,
        "reuse_factor": size_sum / union,
    }


# ---------------------------------------------------------------------------
# Shard loading and aggregation
# ---------------------------------------------------------------------------


def interior_qblocks(vbs: torch.Tensor, max_block: int) -> torch.Tensor:
    """Bool [Nq]: True where the tile is full (no padding). Sanity check #7."""
    return vbs.to(torch.long) == max_block


def load_shards(capture_root: str, run_glob: str = "*") -> list[dict]:
    shards = []
    for run_dir in sorted(glob.glob(os.path.join(capture_root, run_glob))):
        if not os.path.isdir(run_dir):
            continue
        meta_path = os.path.join(run_dir, "run_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                run_meta = json.load(f)
        else:
            run_meta = {}
        for shard_path in sorted(glob.glob(os.path.join(run_dir, "cap_*.pt"))):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            payload["run_id"] = os.path.basename(run_dir)
            payload["run_meta"] = run_meta
            payload["path"] = shard_path
            shards.append(payload)
    return shards


def quantiles(x: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "mean": float(np.mean(x)),
        "n": int(x.size),
    }


def analyze(capture_root: str, out_dir: str, device: str = "cuda", run_glob: str = "*") -> dict:

    shards = load_shards(capture_root, run_glob)
    if not shards:
        raise SystemExit(f"no capture shards under {capture_root}")
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pair_cache: dict[tuple, dict[str, torch.Tensor]] = {}
    group_cache: dict[tuple, dict[str, torch.Tensor]] = {}

    # Accumulators
    pair_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pair_vals_interior: dict[str, list] = defaultdict(list)
    group_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rows = []  # per (shard, head, strategy) medians for breakdowns
    sanity = {"counts_ok": 0, "counts_bad": 0, "reconstruct_ok": 0, "reconstruct_bad": 0}
    n_samples = 0

    for sh in shards:
        k = int(sh["topk"])
        num_tiles = tuple(sh["context"]["num_tiles"])
        vbs = sh["variable_block_sizes"]
        max_block = int(np.prod(sh["context"]["tile_size"]))
        sanity["counts_ok" if sh["counts_ok"] else "counts_bad"] += 1
        sanity["reconstruct_ok" if sh["reconstruct_ok"] else "reconstruct_bad"] += 1

        if num_tiles not in pair_cache:
            pair_cache[num_tiles] = build_pairs(num_tiles)
            group_cache[num_tiles] = build_groups(num_tiles)
        pairs_by_strategy = pair_cache[num_tiles]
        groups_by_geom = group_cache[num_tiles]
        interior = interior_qblocks(vbs, max_block)

        masks = masks_from_shard(sh, device)  # [H, Nq, Nk]
        n_heads = masks.shape[0]
        n_samples += 1

        layer = int(sh["layer_index"])
        step = int(sh["context"].get("timestep", -1))
        run_id = sh["run_id"]

        for strategy_name, pairs in pairs_by_strategy.items():
            m = pair_metrics(masks, pairs.to(device), k)
            for name in ("overlap_k", "jaccard", "union_over_k", "reuse_factor"):
                pair_vals[strategy_name][name].append(m[name].flatten().cpu().numpy())
            pin = interior[pairs[:, 0]] & interior[pairs[:, 1]]
            if pin.any():
                pair_vals_interior[strategy_name].append(m["reuse_factor"][:, pin.to(device)].flatten().cpu().numpy())
            reuse = m["reuse_factor"].cpu().numpy()  # [H, P]
            for h in range(n_heads):
                rows.append({
                    "run_id": run_id,
                    "layer": layer,
                    "step": step,
                    "head": h,
                    "strategy": strategy_name,
                    "median_reuse": float(np.median(reuse[h])),
                    "median_overlap_k": float(np.median(m["overlap_k"][h].cpu().numpy())),
                    "median_jaccard": float(np.median(m["jaccard"][h].cpu().numpy())),
                })

        for geom, groups in groups_by_geom.items():
            gm = group_metrics(masks, groups.to(device), k)
            for name in ("union_over_k", "reuse_factor"):
                group_vals[geom][name].append(gm[name].flatten().cpu().numpy())

    return _summarize(pair_vals, pair_vals_interior, group_vals, rows, sanity, n_samples, shards, out_dir, plots_dir)


def _summarize(pair_vals, pair_vals_interior, group_vals, rows, sanity, n_samples, shards, out_dir, plots_dir) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # ---- Primary table (Part C/D) ----
    strategy_rows = []
    for strategy_name, metrics in pair_vals.items():
        reuse = np.concatenate(metrics["reuse_factor"])
        row = {
            "strategy": strategy_name,
            "median_overlap_k": float(np.median(np.concatenate(metrics["overlap_k"]))),
            "median_jaccard": float(np.median(np.concatenate(metrics["jaccard"]))),
            "median_union_over_k": float(np.median(np.concatenate(metrics["union_over_k"]))),
            "median_reuse": float(np.median(reuse)),
            "p10_reuse": float(np.percentile(reuse, 10)),
            "p90_reuse": float(np.percentile(reuse, 90)),
            "n_pairs": int(reuse.size),
        }
        if strategy_name in pair_vals_interior:
            row["median_reuse_interior_only"] = float(np.median(np.concatenate(pair_vals_interior[strategy_name])))
        strategy_rows.append(row)
    strategy_df = pd.DataFrame(strategy_rows).sort_values("median_reuse", ascending=False)
    strategy_df.to_csv(os.path.join(out_dir, "pair_strategy_summary.csv"), index=False)

    # ---- Group table (Part E) ----
    group_rows = []
    for geom, metrics in group_vals.items():
        reuse = np.concatenate(metrics["reuse_factor"])
        group_rows.append({
            "geometry": geom,
            "group_size": int(geom[1]),
            "median_union_over_k": float(np.median(np.concatenate(metrics["union_over_k"]))),
            "median_reuse": float(np.median(reuse)),
            "p10_reuse": float(np.percentile(reuse, 10)),
            "p90_reuse": float(np.percentile(reuse, 90)),
            "n_groups": int(reuse.size),
        })
    group_df = pd.DataFrame(group_rows).sort_values(["group_size", "median_reuse"])
    group_df.to_csv(os.path.join(out_dir, "group_size_summary.csv"), index=False)

    # ---- Breakdowns (Part F) ----
    bd = pd.DataFrame(rows)
    for dim, fname in (("layer", "layer_summary.csv"), ("step", "timestep_summary.csv"), ("head", "head_summary.csv")):
        agg = (bd.groupby(["strategy", dim]).agg(median_reuse=("median_reuse", "median"),
                                                 median_overlap_k=("median_overlap_k", "median"),
                                                 median_jaccard=("median_jaccard", "median"),
                                                 n=("median_reuse", "size")).reset_index())
        agg.to_csv(os.path.join(out_dir, fname), index=False)

    # ---- Plots ----
    pr_reuse = np.concatenate(pair_vals["pr1719_current"]["reuse_factor"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(pr_reuse, bins=80, color="tab:blue", alpha=0.8)
    ax1.axvline(np.median(pr_reuse), color="red", ls="--", label=f"median={np.median(pr_reuse):.3f}x")
    ax1.set(xlabel="reuse_factor = 2K/|union|", ylabel="pairs", title="PR#1719 pairing: reuse factor")
    ax1.legend()
    xs = np.sort(pr_reuse)
    ax2.plot(xs, np.linspace(0, 1, xs.size))
    for thr in (1.2, 1.35, 1.5):
        ax2.axvline(thr, color="gray", ls=":", lw=0.8)
    ax2.set(xlabel="reuse_factor", ylabel="CDF", title="CDF (go/no-go thresholds dotted)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "reuse_factor_histogram.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [np.concatenate(pair_vals[s]["reuse_factor"]) for s in pair_vals]
    ax.boxplot(data, tick_labels=list(pair_vals), showfliers=False)
    ax.set(ylabel="reuse_factor", title="KV reuse by pairing strategy")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "overlap_by_pair_type.png"), dpi=120)
    plt.close(fig)

    for dim, fname in (("layer", "reuse_by_layer.png"), ("step", "reuse_by_timestep.png")):
        fig, ax = plt.subplots(figsize=(9, 4))
        for strategy_name in bd["strategy"].unique():
            sub = bd[bd["strategy"] == strategy_name].groupby(dim)["median_reuse"].median()
            ax.plot(sub.index, sub.values, marker="o", ms=3, label=strategy_name)
        ax.set(xlabel=dim, ylabel="median reuse_factor", title=f"KV reuse by {dim}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for geom, metrics in group_vals.items():
        g = int(geom[1])
        med = float(np.median(np.concatenate(metrics["reuse_factor"])))
        ax.scatter(g, med, label=geom)
        ax.annotate(geom[3:], (g, med), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set(xlabel="group size g", ylabel="median reuse = gK/|union|", title="Reuse vs local group size")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "group_size_reuse.png"), dpi=120)
    plt.close(fig)

    summary = {
        "n_capture_shards": n_samples,
        "runs": sorted({s["run_id"]
                        for s in shards}),
        "sanity": sanity,
        "pair_strategies": strategy_rows,
        "groups": group_rows,
        "pr1719_reuse": quantiles(pr_reuse),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(strategy_df.to_string(index=False))
    print(group_df.to_string(index=False))
    print(json.dumps({"sanity": sanity, "pr1719_reuse": summary["pr1719_reuse"]}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--run-glob", default="*", help="glob over run dirs, e.g. '720p_*'")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    analyze(args.capture_root, args.out, args.device, args.run_glob)


if __name__ == "__main__":
    main()
