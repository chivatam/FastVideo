"""Phase-2 Part G: GPU metadata-construction microbench + optional MMA probe.

Benchmarks both representations at the real 720p workload scale:
    12 heads x 720 CTA pairs = 8640 pairs, K = 144 (int16 inputs, as captured)

Approaches:
    sorted_searchsorted : batched two-pointer equivalent (rows ARE sorted,
                          Part H) — shared/private and union+membership.
    bitmap              : scatter to [N, 1440] bool, AND/XOR, re-extract.

Reports latency (CUDA events, median of iters), output metadata bytes, and
temporary memory. Compares against a kernel-runtime budget measured from the
in-tree sm_100a kernel if importable, else the Triton VSA forward.

Optional probe: batched GEMM M=64 vs M=128 at fixed K-resident data — a
rough "what does the second Q block cost once KV is loaded" signal.

    python artifacts/vsa_local_reuse_phase2/bench_metadata.py \
        --sample-pairs /mnt/nvme/outputs/phase2_pairs/sample_pairs.pt \
        --out artifacts/vsa_local_reuse_phase2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metadata_gpu import (build_shared_private_batched, build_shared_private_bitmap, build_union_membership_batched)

N_HEADS = 12
N_PAIRS_PER_HEAD = 720
NK = 1440
K = 144


def cuda_time(fn, iters: int = 50, warmup: int = 10) -> float:
    """Median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1) * 1e3)
    times.sort()
    return times[len(times) // 2]


def real_pair_batch(sample_pairs: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] int16 batch built by tiling real captured rows to workload scale."""
    samples = torch.load(sample_pairs, weights_only=False)
    q0 = torch.stack([s["q0_idx"] for s in samples]).to(device)
    q1 = torch.stack([s["q1_idx"] for s in samples]).to(device)
    n_target = N_HEADS * N_PAIRS_PER_HEAD
    reps = (n_target + q0.shape[0] - 1) // q0.shape[0]
    return q0.repeat(reps, 1)[:n_target], q1.repeat(reps, 1)[:n_target]


def measure_kernel_budget(device: str) -> dict:
    """Per-(layer, head-batch) sparse attention runtime at the 720p shape."""
    out: dict = {}
    S = NK * 64
    q, k, v = (torch.randn(1, N_HEADS, S, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
    scores = torch.randn(1, N_HEADS, NK, NK, device=device)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, scores.topk(K, dim=-1).indices, True)
    vbs = torch.full((NK, ), 64, dtype=torch.int32, device=device)
    try:
        from fastvideo_kernel.triton_kernels.index import map_to_index
        idx, num = map_to_index(keep)
        idx = idx.to(torch.int32).contiguous()
        num = num.to(torch.int32).contiguous()
        # the existing per-call index build the pair metadata would fuse into
        out["map_to_index_us"] = cuda_time(lambda: map_to_index(keep), iters=20)
    except Exception as e:  # pragma: no cover
        out["error"] = f"map_to_index unavailable: {e}"
        return out

    try:
        from fastvideo_kernel import block_sparse_attn_sm100a as sm100a
        if sm100a.is_supported(q, vbs):
            out["sm100a_us"] = cuda_time(lambda: sm100a.block_sparse_attn_sm100a(q, k, v, idx, num, vbs), iters=20)
    except Exception as e:
        out["sm100a_error"] = str(e)
    try:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
        out["triton_us"] = cuda_time(lambda: block_sparse_attn_triton(q, k, v, idx, num, vbs), iters=20)
    except Exception as e:
        out["triton_error"] = str(e)
    return out


def mma_probe(device: str) -> dict:
    """Batched GEMM M=64 vs M=128: rough cost of the second Q on resident KV."""
    B = 4096
    k_res = torch.randn(B, 128, 64, device=device, dtype=torch.bfloat16)
    q64 = torch.randn(B, 64, 128, device=device, dtype=torch.bfloat16)
    q128 = torch.randn(B, 128, 128, device=device, dtype=torch.bfloat16)
    t64 = cuda_time(lambda: torch.bmm(q64, k_res))
    t128 = cuda_time(lambda: torch.bmm(q128, k_res))
    return {"bmm_m64_us": t64, "bmm_m128_us": t128, "second_q_cost_ratio": t128 / t64}


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-pairs", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    q0, q1 = real_pair_batch(args.sample_pairs, args.device)
    n = q0.shape[0]
    rows = []

    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t = cuda_time(lambda: build_shared_private_batched(q0, q1))
    peak = torch.cuda.max_memory_allocated() - base_mem
    sh, c_sh, p0, c_p0, p1, c_p1 = build_shared_private_batched(q0, q1)
    meta_bytes = sum(x.numel() * x.element_size() for x in (sh, c_sh, p0, c_p0, p1, c_p1))
    rows.append({
        "approach": "shared_private_sorted_searchsorted",
        "latency_us": t,
        "ns_per_pair": 1e3 * t / n,
        "metadata_bytes": meta_bytes,
        "temp_peak_bytes": int(peak)
    })

    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t = cuda_time(lambda: build_union_membership_batched(q0, q1))
    peak = torch.cuda.max_memory_allocated() - base_mem
    un, c_un, mem = build_union_membership_batched(q0, q1)
    meta_bytes = sum(x.numel() * x.element_size() for x in (un, c_un, mem))
    rows.append({
        "approach": "union_membership_sorted_searchsorted",
        "latency_us": t,
        "ns_per_pair": 1e3 * t / n,
        "metadata_bytes": meta_bytes,
        "temp_peak_bytes": int(peak)
    })

    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t = cuda_time(lambda: build_shared_private_bitmap(q0, q1, NK))
    peak = torch.cuda.max_memory_allocated() - base_mem
    rows.append({
        "approach": "shared_private_bitmap",
        "latency_us": t,
        "ns_per_pair": 1e3 * t / n,
        "metadata_bytes": meta_bytes,
        "temp_peak_bytes": int(peak)
    })

    try:
        from metadata_triton import build_shared_private_triton
        # correctness cross-check vs the torch reference before timing
        ts = build_shared_private_triton(q0[:64], q1[:64])
        rs = build_shared_private_batched(q0[:64], q1[:64])
        for i_t, i_r in ((0, 0), (2, 2), (4, 4)):
            cnt = rs[i_t + 1]
            for row_i in range(64):
                c = int(cnt[row_i])
                assert torch.equal(ts[i_t][row_i, :c].long(), rs[i_r][row_i, :c].long())
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        t = cuda_time(lambda: build_shared_private_triton(q0, q1))
        peak = torch.cuda.max_memory_allocated() - base_mem
        sp_bytes = (3 * n * K + 3 * n) * 4
        rows.append({
            "approach": "shared_private_triton_onekernel",
            "latency_us": t,
            "ns_per_pair": 1e3 * t / n,
            "metadata_bytes": sp_bytes,
            "temp_peak_bytes": int(peak)
        })
    except Exception as e:  # pragma: no cover
        rows.append({
            "approach": f"shared_private_triton_onekernel FAILED: {e}",
            "latency_us": float("nan"),
            "ns_per_pair": float("nan"),
            "metadata_bytes": 0,
            "temp_peak_bytes": 0
        })

    budget = measure_kernel_budget(args.device)
    probe = mma_probe(args.device)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    df.to_csv(os.path.join(args.out, "results", "metadata_bench.csv"), index=False)
    payload = {
        "n_pairs": n,
        "K": K,
        "num_kv_blocks": NK,
        "kernel_budget_per_layer_call": budget,
        "mma_m64_vs_m128_probe": probe
    }
    with open(os.path.join(args.out, "results", "metadata_bench_context.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(df.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
