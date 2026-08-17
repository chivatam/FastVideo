"""C0 probe: does the FA4 fork's FP4 SM100 kernel trace with block_sparse_tensors?

Tiny shapes; the goal is only to see whether CuTe JIT tracing of the
use_block_sparsity=True branch of flash_fwd_sm100_fp4.py succeeds or which
signature mismatch it hits. Not a benchmark, not evidence for any arm.
"""
import os
import sys
import traceback

import torch

sys.stderr.write("probe start\n")

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

torch.manual_seed(0)
device = "cuda:0"

B, S, H, D = 1, 512, 2, 128
M_BLK, N_BLK = 128, 128

q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

qf4, sfq = _nvfp4_quantize_for_fa4(q)
kf4, sfk = _nvfp4_quantize_for_fa4(k)
qf4 = qf4[:, :S]
kf4 = kf4[:, :S]

from flash_attn.cute.block_sparsity import get_block_sparse_expected_shapes

# SM100 kernels use q_stage=2 → sparse Q granularity is 2*M_BLK tokens.
expected_cnt_shape, expected_idx_shape = get_block_sparse_expected_shapes(
    B, H, S, S, M_BLK, N_BLK, 2)
print("expected shapes:", expected_cnt_shape, expected_idx_shape, flush=True)
num_m = expected_cnt_shape[2]
num_n = expected_idx_shape[3]

# keep every other KV block (50% retained), as "full" blocks
full_cnt = torch.full((B, H, num_m), max(1, num_n // 2), device=device, dtype=torch.int32)
full_idx = torch.zeros((B, H, num_m, num_n), device=device, dtype=torch.int32)
keep = torch.arange(0, num_n, 2, device=device, dtype=torch.int32)
full_idx[:, :, :, : keep.numel()] = keep
mask_cnt = torch.zeros((B, H, num_m), device=device, dtype=torch.int32)
mask_idx = torch.zeros((B, H, num_m, num_n), device=device, dtype=torch.int32)

sparse = BlockSparseTensorsTorch(
    full_block_cnt=full_cnt,
    full_block_idx=full_idx,
    mask_block_cnt=mask_cnt,
    mask_block_idx=mask_idx,
)

print("=== step 1: dense FP4 (sanity) ===", flush=True)
try:
    out_dense, _ = _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False)
    print("dense FP4 OK", out_dense.shape, out_dense.dtype,
          torch.isfinite(out_dense.float()).all().item(), flush=True)
except Exception:
    traceback.print_exc()

print("=== step 2: FP4 + block_sparse_tensors ===", flush=True)
try:
    out_sp, _ = _flash_attn_fwd(
        qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False,
        block_sparse_tensors=sparse,
    )
    print("sparse FP4 TRACED+RAN", out_sp.shape,
          torch.isfinite(out_sp.float()).all().item(), flush=True)
except Exception:
    traceback.print_exc()

print("=== step 3: BF16 + block_sparse_tensors (control) ===", flush=True)
try:
    out_bf, _ = _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sparse)
    print("sparse BF16 OK", out_bf.shape,
          torch.isfinite(out_bf.float()).all().item(), flush=True)
except Exception:
    traceback.print_exc()
