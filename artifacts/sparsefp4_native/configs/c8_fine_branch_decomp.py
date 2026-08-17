"""Decompose the P3-vs-P2G gap: wall-clock per fine-branch call at Wan shape.

Measures, with all Python/FFI/quantize overheads included:
  - fine BF16 (P2G's branch): _flash_attn_fwd BF16 + sparse lists + mask_mod
  - fine FP4 (P3's branch): quantize Q,K -> _flash_attn_fwd FP4 + same lists
  - fine FP4 pre-quantized (kernel-side only, no per-call quantize)
plus CUDA-event kernel time for each, so wall - kernel = host overhead.
"""
import time
import json

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo_kernel.block_sparse_attn_cute_fwd import _build_vbs_mask_mod

B, S, H, D = 1, 39936, 12, 128
TILE, QS, KS = 64, 256, 128
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
n64 = S // TILE
vbs = torch.full((n64,), TILE, device="cuda", dtype=torch.int32)
vbs[-8:] = 32
num_m, num_n = S // QS, S // KS
g = torch.Generator(device="cuda").manual_seed(0)
sc = torch.rand(B, H, num_m, num_n, generator=g, device="cuda")
mask = torch.zeros(B, H, num_m, num_n, dtype=torch.bool, device="cuda")
mask.scatter_(-1, sc.topk(int(0.24 * num_n), dim=-1).indices, True)
per128 = (vbs.view(-1, 2) < TILE).any(1).view(1, 1, 1, -1)


def pack(m):
    cnt = m.sum(-1).to(torch.int32).contiguous()
    ar = torch.arange(num_n, device="cuda").expand_as(m)
    key = torch.where(m, ar, torch.full_like(ar, num_n))
    p = torch.sort(key, -1).values
    return cnt, torch.where(p == num_n, torch.zeros_like(p), p).to(torch.int32).contiguous()


fc, fi = pack(mask & ~per128)
mc, mi = pack(mask & per128)
sparse = BlockSparseTensorsTorch(full_block_cnt=fc, full_block_idx=fi,
                                 mask_block_cnt=mc, mask_block_idx=mi)
mm = _build_vbs_mask_mod(TILE)
aux = [vbs]

qf4_pre, sfq_pre = _nvfp4_quantize_for_fa4(q)
kf4_pre, sfk_pre = _nvfp4_quantize_for_fa4(k)
qf4_pre, kf4_pre = qf4_pre[:, :S].contiguous(), kf4_pre[:, :S].contiguous()


def fine_bf16():
    return _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sparse,
                           mask_mod=mm, aux_tensors=aux)


def fine_fp4():
    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    return _flash_attn_fwd(qf4[:, :S], kf4[:, :S], v, mSFQ=sfq, mSFK=sfk,
                           causal=False, block_sparse_tensors=sparse,
                           mask_mod=mm, aux_tensors=aux)


def fine_fp4_prequant():
    return _flash_attn_fwd(qf4_pre, kf4_pre, v, mSFQ=sfq_pre, mSFK=sfk_pre,
                           causal=False, block_sparse_tensors=sparse,
                           mask_mod=mm, aux_tensors=aux)


def wall_ms(fn, warm=5, reps=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def kernel_ms(fn, warm=5, reps=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


res = {}
for name, fn in (("fine_bf16", fine_bf16), ("fine_fp4", fine_fp4),
                 ("fine_fp4_prequant", fine_fp4_prequant)):
    res[name] = dict(wall_ms=wall_ms(fn), event_ms=kernel_ms(fn))
    print(name, json.dumps(res[name]), flush=True)

gap_ms = res["fine_fp4"]["wall_ms"] - res["fine_bf16"]["wall_ms"]
print(f"per-call fp4-vs-bf16 wall gap: {gap_ms:.3f} ms -> x3000 = {gap_ms*3:.1f} s/video")
json.dump(res, open("artifacts/sparsefp4_native/raw/performance/fine_branch_wall.json", "w"), indent=2)
print("DECOMP_DONE")
