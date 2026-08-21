"""Phase-2 Part A: build the real PR#1719 query-pair dataset from Phase-1 captures.

For every 720p capture shard (prompt x layer x step) and every head, takes the
kernel's actual CTA pairs (q_block 2p, 2p+1) and records per-pair set counts:

    |S0|, |S1|, |intersection|  ->  shared / privates / union derivable

Also exports a stratified sample of raw (q0_idx, q1_idx) index rows for the
exactness reference (Part C) and the metadata microbench (Part G).

    python artifacts/vsa_local_reuse_phase2/build_pair_dataset.py \
        --capture-root /mnt/nvme/outputs/vsa_capture \
        --out /mnt/nvme/outputs/phase2_pairs

Outputs:
    pair_counts.npz   int16 arrays [n_records]: inter, s0, s1, layer, step,
                      head, prompt (categorical), pair index
    sample_pairs.pt   list of dicts with raw q0_idx/q1_idx (int16 [K]) rows
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def load_720p_shards(capture_root: str) -> list[dict]:
    shards = []
    for run_dir in sorted(glob.glob(os.path.join(capture_root, "720p_*"))):
        for f in sorted(glob.glob(os.path.join(run_dir, "cap_*.pt"))):
            payload = torch.load(f, map_location="cpu", weights_only=False)
            payload["run_id"] = os.path.basename(run_dir)
            shards.append(payload)
    return shards


def masks_from_shard(shard: dict, device: str) -> torch.Tensor:
    nk = int(shard["num_kv_blocks"])
    if shard.get("indices") is not None:
        idx = shard["indices"][0].to(device)
        m = torch.zeros(*idx.shape[:2], nk, dtype=torch.bool, device=device)
        m.scatter_(-1, idx.long(), True)
        return m
    H, Nq = int(shard["heads"]), int(shard["num_q_blocks"])
    counts = shard["row_counts"][0].to(device).long().reshape(H * Nq)
    cols = shard["indices_flat"].to(device).long()
    rows = torch.repeat_interleave(torch.arange(H * Nq, device=device), counts)
    m = torch.zeros(H * Nq, nk, dtype=torch.bool, device=device)
    m[rows, cols] = True
    return m.view(H, Nq, nk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-pairs", type=int, default=512, help="raw index rows for Part C/G")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    shards = load_720p_shards(args.capture_root)
    if not shards:
        raise SystemExit("no 720p capture shards found")

    run_ids = sorted({s["run_id"] for s in shards})
    run_to_id = {r: i for i, r in enumerate(run_ids)}

    recs: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("inter", "s0", "s1", "layer", "step", "head", "prompt", "pair")
    }
    samples: list[dict] = []
    rng = np.random.default_rng(0)
    k_nominal = None

    for sh in shards:
        masks = masks_from_shard(sh, args.device)  # [H, Nq, Nk]
        H, Nq, _ = masks.shape
        k_nominal = int(sh["topk"])
        pairs = torch.arange(Nq, device=args.device).view(-1, 2)  # (2p, 2p+1)
        m0 = masks[:, pairs[:, 0], :]
        m1 = masks[:, pairs[:, 1], :]
        inter = (m0 & m1).sum(-1).to(torch.int16).cpu().numpy()  # [H, P]
        s0 = m0.sum(-1).to(torch.int16).cpu().numpy()
        s1 = m1.sum(-1).to(torch.int16).cpu().numpy()
        P = pairs.shape[0]
        hgrid, pgrid = np.meshgrid(np.arange(H, dtype=np.int16), np.arange(P, dtype=np.int16), indexing="ij")
        recs["inter"].append(inter.ravel())
        recs["s0"].append(s0.ravel())
        recs["s1"].append(s1.ravel())
        recs["head"].append(hgrid.ravel())
        recs["pair"].append(pgrid.ravel())
        recs["layer"].append(np.full(H * P, sh["layer_index"], dtype=np.int16))
        recs["step"].append(np.full(H * P, sh["context"]["timestep"], dtype=np.int16))
        recs["prompt"].append(np.full(H * P, run_to_id[sh["run_id"]], dtype=np.int16))

        # Stratified raw-pair sample: a few pairs per shard for Part C / G.
        if sh.get("indices") is not None and len(samples) < args.sample_pairs:
            idx = sh["indices"][0]  # [H, Nq, K] int16
            for _ in range(2):
                h = int(rng.integers(H))
                p = int(rng.integers(P))
                samples.append({
                    "q0_idx": idx[h, 2 * p].clone(),
                    "q1_idx": idx[h, 2 * p + 1].clone(),
                    "layer": int(sh["layer_index"]),
                    "step": int(sh["context"]["timestep"]),
                    "head": h,
                    "pair": p,
                    "run_id": sh["run_id"],
                    "num_kv_blocks": int(sh["num_kv_blocks"]),
                })

    arrays = {k: np.concatenate(v) for k, v in recs.items()}
    np.savez_compressed(os.path.join(args.out, "pair_counts.npz"),
                        k_nominal=np.int16(k_nominal),
                        runs=np.array(run_ids),
                        **arrays)
    torch.save(samples, os.path.join(args.out, "sample_pairs.pt"))
    meta = {
        "n_records": int(arrays["inter"].size),
        "n_shards": len(shards),
        "runs": run_ids,
        "k_nominal": k_nominal,
        "n_sample_pairs": len(samples),
    }
    with open(os.path.join(args.out, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
