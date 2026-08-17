"""C3 native-proof: runtime + profiler + work-scaling receipts for D0.

Produces artifacts/sparsefp4_native/raw/performance/c3_native_proof.json with

1. runtime receipt — dtypes/shapes/strides of everything entering the sparse
   FP4 call (packed E2M1 Q/K, uint8 E4M3 scale tensors, BF16 V, int32
   count/index lists), GPU arch, kernel class;
2. profiler receipt — CUDA kernel symbols recorded by torch.profiler for
   (a) D0 sparse FP4, (b) B0 dense FP4, (c) C0 sparse BF16;
3. work scaling — CUDA-event median latency at retained fractions
   100/50/25/10% for both the FP4 and BF16 sparse kernels at the Wan
   480x832x81 VSA-tiled shape (B=1, S=39936, H=12, D=128), plus the true
   dense kernels as anchors.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

OUT = Path("artifacts/sparsefp4_native/raw/performance/c3_native_proof.json")
B, S, H, D = 1, 39936, 12, 128
Q_SPARSE, K_SPARSE = 256, 128
WARMUP, REPS = 10, 50


def make_sparse(retained, seed=0):
    num_m, num_n = S // Q_SPARSE, S // K_SPARSE
    k = max(1, round(retained * num_n))
    g = torch.Generator(device="cuda").manual_seed(seed)
    sc = torch.rand(B, H, num_m, num_n, generator=g, device="cuda")
    mask = torch.zeros(B, H, num_m, num_n, dtype=torch.bool, device="cuda")
    mask.scatter_(-1, sc.topk(k, dim=-1).indices, True)
    cnt = mask.sum(-1).to(torch.int32).contiguous()
    ar = torch.arange(num_n, device="cuda").expand_as(mask)
    key = torch.where(mask, ar, torch.full_like(ar, num_n))
    packed = torch.sort(key, dim=-1).values
    idx = torch.where(packed == num_n, torch.zeros_like(packed), packed)\
        .to(torch.int32).contiguous()
    return BlockSparseTensorsTorch(
        full_block_cnt=cnt, full_block_idx=idx,
        mask_block_cnt=torch.zeros_like(cnt),
        mask_block_idx=torch.zeros_like(idx)), mask.float().mean().item()


def cuda_time(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    n = len(times)
    return dict(median_ms=times[n // 2], p10_ms=times[n // 10],
                p90_ms=times[9 * n // 10], n=n)


def profile_kernels(fn, tag):
    from torch.profiler import profile, ProfilerActivity
    fn()  # ensure compiled
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        if evt.device_type is not None and evt.self_device_time_total > 0:
            rows.append(dict(name=evt.key, cuda_time_us=evt.self_device_time_total,
                             count=evt.count))
    rows.sort(key=lambda r: -r["cuda_time_us"])
    return {"tag": tag, "kernels": rows[:8]}


def main():
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    qf4, kf4 = qf4[:, :S].contiguous(), kf4[:, :S].contiguous()

    receipt = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "shape": dict(B=B, S=S, H=H, D=D),
        "q_fp4": dict(dtype=str(qf4.dtype), shape=list(qf4.shape),
                      stride=list(qf4.stride())),
        "k_fp4": dict(dtype=str(kf4.dtype), shape=list(kf4.shape),
                      stride=list(kf4.stride())),
        "sfq": dict(dtype=str(sfq.dtype), shape=list(sfq.shape),
                    stride=list(sfq.stride()),
                    semantics="E4M3 per-16-element scale factors, FA4 MMA layout"),
        "sfk": dict(dtype=str(sfk.dtype), shape=list(sfk.shape),
                    stride=list(sfk.stride())),
        "v": dict(dtype=str(v.dtype), shape=list(v.shape)),
        "note_no_bf16_materialization": (
            "The sparse call consumes qf4/kf4 (torch.float4_e2m1fn_x2) directly; "
            "interface.py builds cute pointers of type Float4E2M1FN from their "
            "data_ptr (fp4_qk branch). No dequantized Q/K tensor exists in the "
            "call path."),
        "sparse_semantics": (
            "full_block_idx/count lists per (batch, head, 256-token Q row) of "
            "retained 128-token K blocks; kernel iterates only listed blocks "
            "(produce_block_sparse_loads_sm100 -> load_block_list_sm100), "
            "loading K/V tiles AND their scale-factor tiles by block index."),
    }

    sparse_tensors = {}
    realized = {}
    for r in (1.0, 0.5, 0.25, 0.10):
        sparse_tensors[r], realized[r] = make_sparse(r)

    def d0(r):
        return lambda: _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False,
                                       block_sparse_tensors=sparse_tensors[r])

    def c0(r):
        return lambda: _flash_attn_fwd(q, k, v, causal=False,
                                       block_sparse_tensors=sparse_tensors[r])

    dense_fp4 = lambda: _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False)
    dense_bf16 = lambda: _flash_attn_fwd(q, k, v, causal=False)

    profiler = [
        profile_kernels(d0(0.10), "D0_sparse_fp4_retained10"),
        profile_kernels(dense_fp4, "B0_dense_fp4"),
        profile_kernels(c0(0.10), "C0_sparse_bf16_retained10"),
        profile_kernels(dense_bf16, "A0_dense_bf16"),
    ]

    scaling = []
    scaling.append(dict(arm="A0_dense_bf16_kernel", **cuda_time(dense_bf16)))
    scaling.append(dict(arm="B0_dense_fp4_kernel", **cuda_time(dense_fp4)))
    for r in (1.0, 0.5, 0.25, 0.10):
        scaling.append(dict(arm="D0_sparse_fp4", retained=r,
                            realized=realized[r], **cuda_time(d0(r))))
        scaling.append(dict(arm="C0_sparse_bf16", retained=r,
                            realized=realized[r], **cuda_time(c0(r))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runtime_receipt": receipt,
                               "profiler": profiler,
                               "work_scaling": scaling}, indent=2) + "\n")
    print(json.dumps({"work_scaling": scaling}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
