"""Phase-2 Part C: numerical equivalence of strategies A/B1/B2 vs baseline.

Evaluates the streaming references over real captured (Q0, Q1) index pairs
at the true workload shape (64-token blocks, head_dim 128, K=144), in both
fp32 and kernel-realistic bf16 mode. The baseline is the current PR#1719
semantics (per-Q sorted walk, online softmax).

    python artifacts/vsa_local_reuse_phase2/run_reference_eval.py \
        --sample-pairs /mnt/nvme/outputs/phase2_pairs/sample_pairs.pt \
        --out artifacts/vsa_local_reuse_phase2 --n-pairs 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_attention import (baseline_pair, error_metrics, make_pair_tensors, strategy_a_pair, strategy_b_pair)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-pairs", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = torch.load(args.sample_pairs, weights_only=False)[:args.n_pairs]
    rows = []
    for i, s in enumerate(samples):
        q0_idx, q1_idx = s["q0_idx"].long(), s["q1_idx"].long()
        q0, q1, kv_k, kv_v, id_to_slot, scale = make_pair_tensors(q0_idx, q1_idx, seed=1000 + i, device=args.device)
        for bf16 in (False, True):
            ob = baseline_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16)
            outs = {
                "A_shared_private":
                strategy_a_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16),
                "B1_union_conditional":
                strategy_b_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16, dense=False),
                "B2_union_dense":
                strategy_b_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16, dense=True),
            }
            for strategy_name, (o0, o1) in outs.items():
                for qi, (ref, test) in enumerate(((ob[0], o0), (ob[1], o1))):
                    rows.append({
                        "pair": i,
                        "mode": "bf16" if bf16 else "fp32",
                        "strategy": strategy_name,
                        "q": qi,
                        "layer": s["layer"],
                        "step": s["step"],
                        "head": s["head"],
                        **error_metrics(ref, test),
                    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    df.to_csv(os.path.join(args.out, "results", "numerics_per_pair.csv"), index=False)
    agg = (df.groupby(["mode", "strategy"]).agg(max_abs_err=("max_abs_err", "max"),
                                                mean_abs_err=("mean_abs_err", "mean"),
                                                rel_l2_max=("rel_l2", "max"),
                                                cosine_min=("cosine_sim", "min"),
                                                n=("max_abs_err", "size")).reset_index())
    agg.to_csv(os.path.join(args.out, "results", "numerics_summary.csv"), index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
