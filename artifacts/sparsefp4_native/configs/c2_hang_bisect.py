"""Minimal repro: does the sparse-FP4 SM100 kernel hang when the persistent
scheduler has more CTAs than SMs (multi-wave)?

S=1024 (single wave on B200: num_m=4 x H) passed; S=4096 (num_m=16 x 12 heads
= 192 CTAs > 148 SMs) appeared to hang. Bisect over H at S=4096 and over S.
Each case runs in a subprocess with a timeout so a hang is a data point, not a
stall.
"""
import subprocess
import sys
import os

CASE_SRC = r'''
import sys, torch
S, H, retained = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
kernel = sys.argv[4]  # fp4 | bf16
B, D = 1, 128
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
num_m, num_n = S // 256, S // 128
kkeep = max(1, round(retained * num_n))
g = torch.Generator(device="cuda").manual_seed(0)
scores = torch.rand(B, H, num_m, num_n, generator=g, device="cuda")
mask = torch.zeros(B, H, num_m, num_n, dtype=torch.bool, device="cuda")
mask.scatter_(-1, scores.topk(kkeep, dim=-1).indices, True)
cnt = mask.sum(-1).to(torch.int32).contiguous()
ar = torch.arange(num_n, device="cuda").expand_as(mask)
key = torch.where(mask, ar, torch.full_like(ar, num_n))
packed = torch.sort(key, dim=-1).values
idx = torch.where(packed == num_n, torch.zeros_like(packed), packed).to(torch.int32).contiguous()
sparse = BlockSparseTensorsTorch(
    full_block_cnt=cnt, full_block_idx=idx,
    mask_block_cnt=torch.zeros_like(cnt), mask_block_idx=torch.zeros_like(idx))
if kernel == "fp4":
    qf4, sfq = _nvfp4_quantize_for_fa4(q); kf4, sfk = _nvfp4_quantize_for_fa4(k)
    out, _ = _flash_attn_fwd(qf4[:, :S], kf4[:, :S], v, mSFQ=sfq, mSFK=sfk,
                             causal=False, block_sparse_tensors=sparse)
else:
    out, _ = _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sparse)
torch.cuda.synchronize()
print("FINISHED", torch.isfinite(out.float()).all().item())
'''

def run_case(S, H, retained, kernel, timeout=240):
    cmd = [os.environ["FV_PYTHON"], "-c", CASE_SRC, str(S), str(H), str(retained), kernel]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = "FINISHED True" in r.stdout
        tag = "OK" if ok else f"FAIL rc={r.returncode}"
        if not ok:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            tag += " | " + " / ".join(tail)
    except subprocess.TimeoutExpired:
        tag = "HANG(timeout)"
    print(f"S={S:6d} H={H:2d} retained={retained:.2f} kernel={kernel}: {tag}", flush=True)

if __name__ == "__main__":
    for kernel in ("fp4", "bf16"):
        for (S, H, r) in [
            (1024, 4, 0.5),
            (2048, 12, 0.5),
            (4096, 4, 0.5),
            (4096, 12, 1.0),   # dense-as-sparse, multi-wave
            (4096, 12, 0.5),   # the suspected hang cell
            (4096, 12, 0.25),
        ]:
            run_case(S, H, r, kernel)
