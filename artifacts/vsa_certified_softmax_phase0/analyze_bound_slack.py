"""Phase-0 analysis: certified-bound slack + numerics on real Wan Q/K/V.

    python artifacts/vsa_certified_softmax_phase0/analyze_bound_slack.py \
        --qk-root /mnt/nvme/outputs/vsa_qk --out artifacts/vsa_certified_softmax_phase0
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certified_bound import (LOG2E, attn_exact_max, attn_fixed_u, attn_online, block_summaries, certified_u,
                             gather_valid_scores, simulate_online_rescales, true_row_max)

BLOCK = 64
QBLOCKS_PER_SHARD = 8
ROWS_PER_QBLOCK = 16
NUMERICS_QBLOCKS = 2  # per shard, full attention-output comparison

QUANTS = {"p50": 50, "p75": 75, "p90": 90, "p95": 95, "p99": 99, "p999": 99.9}


def process_shard(path: str, device: str, rng: np.random.Generator, records: list, numerics: list) -> None:
    sh = torch.load(path, map_location="cpu", weights_only=False)
    q = sh["q"][0].to(device)  # [H, S_pad, D] bf16
    k = sh["k"][0].to(device)
    v = sh["v"][0].to(device)
    idx = sh["indices"][0].to(device).long()  # [H, Nq, K]
    vbs = sh["variable_block_sizes"].to(device).int()
    H, S, D = q.shape
    nq = idx.shape[1]
    scale = 1.0 / math.sqrt(D)
    layer = int(sh["layer_index"])
    step = int(sh["context"]["timestep"])
    res = "720p" if vbs.numel() == 1440 else "480p"

    k_bar, rho = block_summaries(k, vbs)  # fp32

    qblocks = rng.choice(nq, size=min(QBLOCKS_PER_SHARD, nq), replace=False)
    for h in range(H):
        for j, qb in enumerate(qblocks):
            sel = idx[h, int(qb)]
            n_valid_q = int(vbs[int(qb)])
            rows = rng.choice(max(n_valid_q, 1), size=min(ROWS_PER_QBLOCK, n_valid_q), replace=False)
            q_rows = q[h].view(nq, BLOCK, D)[int(qb)][torch.from_numpy(rows).to(device)].float()
            scores, valid = gather_valid_scores(q_rows, k[h], sel, vbs, scale)
            m_true = true_row_max(scores)
            u = certified_u(q_rows, k_bar[h], rho[h], sel, scale)
            sim = simulate_online_rescales(scores, tile=128)
            delta = (u - m_true).cpu().numpy()
            for r in range(delta.shape[0]):
                records.append({
                    "res": res,
                    "layer": layer,
                    "step": step,
                    "head": h,
                    "delta": float(delta[r]),
                    "m_true": float(m_true[r].item()),
                    "u": float(u[r].item()),
                    "n_tiles": int(sim["n_tiles"]),
                    "n_max_updates": int(sim["n_max_updates"][r]),
                    "n_rescales": int(sim["n_nontrivial_rescales"][r]),
                })
            if j < NUMERICS_QBLOCKS:
                v_tok = v[h].view(-1, BLOCK, D)[sel].reshape(-1, D)
                for bf16_p in (False, True):
                    o_ref = attn_exact_max(scores, v_tok, bf16_p=False)  # fp32 stable ref
                    o_fix = attn_fixed_u(scores, v_tok, u, bf16_p=bf16_p)
                    o_onl = attn_online(scores, v_tok, bf16_p=bf16_p)
                    for name, o in (("fixed_u", o_fix), ("online", o_onl)):
                        d = (o - o_ref).double()
                        numerics.append({
                            "res":
                            res,
                            "layer":
                            layer,
                            "step":
                            step,
                            "head":
                            h,
                            "mode":
                            "bf16p" if bf16_p else "fp32",
                            "variant":
                            name,
                            "max_abs":
                            float(d.abs().max()),
                            "mean_abs":
                            float(d.abs().mean()),
                            "rel_l2":
                            float(d.norm() / o_ref.double().norm().clamp(min=1e-30)),
                            "cosine":
                            float(
                                torch.nn.functional.cosine_similarity(o.double().flatten(),
                                                                      o_ref.double().flatten(),
                                                                      dim=0)),
                        })


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qk-root", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    res_dir = os.path.join(args.out, "results")
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.qk_root, "*", "qk_*.pt")))
    if not shards:
        raise SystemExit(f"no qk shards under {args.qk_root}")
    rng = np.random.default_rng(0)
    records: list = []
    numerics: list = []
    for p in shards:
        process_shard(p, args.device, rng, records, numerics)
    df = pd.DataFrame(records)
    nm = pd.DataFrame(numerics)
    df.to_csv(os.path.join(res_dir, "bound_slack.csv"), index=False)

    # Part 4 gate: zero violations (tolerance for fp roundoff)
    viol = df[df.delta < -1e-4]
    assert len(viol) == 0, f"BOUND VIOLATIONS: {len(viol)} rows, min delta {df.delta.min()}"

    def qrow(x: np.ndarray) -> dict:
        d = {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())}
        d.update({k2: float(np.percentile(x, v)) for k2, v in QUANTS.items()})
        return d

    slack = qrow(df.delta.values)
    # Part 6: kernel exp2 domain -- top-token exponent argument is -delta*log2(e)
    exp_arg = -df.delta.values * LOG2E
    frac_bf16_zero = float((exp_arg < -133).mean())  # bf16 subnormal floor
    frac_fp32_ftz = float((exp_arg < -126).mean())  # FTZ boundary (ex2_approx is FTZ)
    frac_tiny = float((exp_arg < -60).mean())

    for dim, fname in (("layer", "bound_slack_by_layer.csv"), ("step", "bound_slack_by_step.csv"),
                       ("head", "bound_slack_by_head.csv"), ("res", "bound_slack_by_res.csv")):
        agg = df.groupby(dim).delta.agg(
            ["median", "mean", lambda x: np.percentile(x, 90), lambda x: np.percentile(x, 99), "max", "size"])
        agg.columns = ["p50", "mean", "p90", "p99", "max", "n"]
        agg.to_csv(os.path.join(res_dir, fname))

    (nm.groupby(["mode", "variant"]).agg(max_abs=("max_abs", "max"),
                                         mean_abs=("mean_abs", "mean"),
                                         rel_l2_max=("rel_l2", "max"),
                                         cosine_min=("cosine", "min"),
                                         n=("max_abs",
                                            "size")).reset_index().to_csv(os.path.join(res_dir, "numerical_parity.csv"),
                                                                          index=False))

    resc = {
        "n_tiles_median": float(df.n_tiles.median()),
        "max_updates": qrow(df.n_max_updates.values),
        "nontrivial_rescales": qrow(df.n_rescales.values),
        "frac_rows_with_any_rescale": float((df.n_rescales > 0).mean())
    }
    df[["res", "layer", "step", "head", "n_tiles", "n_max_updates",
        "n_rescales"]].to_csv(os.path.join(res_dir, "rescale_frequency.csv"), index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(df.delta, bins=120, color="tab:blue")
    a1.axvline(slack["p99"], color="red", ls="--", label=f"P99={slack['p99']:.2f}")
    a1.set(xlabel="delta = U - m_true (score units)", ylabel="rows", title="Certified bound slack")
    a1.legend()
    xs = np.sort(df.delta.values)
    a2.plot(xs, np.linspace(0, 1, xs.size))
    a2.set(xlabel="delta", ylabel="CDF", title="Slack CDF")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bound_slack_cdf.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    med = df.groupby("layer").delta.median()
    p99 = df.groupby("layer").delta.quantile(0.99)
    ax.plot(med.index, med.values, marker="o", label="P50")
    ax.plot(p99.index, p99.values, marker="s", label="P99")
    ax.set(xlabel="layer", ylabel="delta", title="Slack by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "slack_by_layer.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.n_rescales,
            bins=range(0,
                       int(df.n_max_updates.max()) + 2),
            alpha=0.6,
            label="nontrivial rescales (threshold=8)")
    ax.hist(df.n_max_updates, bins=range(0, int(df.n_max_updates.max()) + 2), alpha=0.5, label="raw max updates")
    ax.set(xlabel="events per row", ylabel="rows", title="Online-softmax counterfactual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "online_rescale_count.png"), dpi=120)
    plt.close(fig)

    summary = {
        "n_rows": int(len(df)),
        "n_shards": len(shards),
        "bound_violations": 0,
        "slack_score_units": slack,
        "exp2_domain": {
            "exp_arg_p99_log2": float(np.percentile(exp_arg, 1)),
            "frac_top_token_fp32_ftz_zero": frac_fp32_ftz,
            "frac_top_token_bf16_zero": frac_bf16_zero,
            "frac_exp_arg_below_-60": frac_tiny
        },
        "rescale_counterfactual": resc
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(nm.groupby(["mode", "variant"]).max_abs.max())


if __name__ == "__main__":
    main()
