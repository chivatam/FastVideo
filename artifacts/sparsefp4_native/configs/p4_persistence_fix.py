"""A/B the FA4_FP4_DISABLE_PERSISTENT fix at 480p and 720p sparse geometries.

Runs each config in a subprocess (env is compile-time) and reports CUDA-event
median kernel latency for the FP4 and BF16 sparse fine branches.
"""
import json
import os
import subprocess
import sys

CASE = r'''
import os, time, torch, json
S_TILES = int(os.sys.argv[1])       # 156 (480p) | 360 (720p)
keep = float(os.sys.argv[2])
B, H, D, TILE, QS, KS = 1, 12, 128, 64, 256, 128
S = S_TILES * 256
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
num_m, num_n = S // QS, S // KS
g = torch.Generator(device="cuda").manual_seed(0)
sc = torch.rand(B, H, num_m, num_n, generator=g, device="cuda")
mask = torch.zeros(B, H, num_m, num_n, dtype=torch.bool, device="cuda")
mask.scatter_(-1, sc.topk(max(1, round(keep * num_n)), dim=-1).indices, True)
cnt = mask.sum(-1).to(torch.int32).contiguous()
ar = torch.arange(num_n, device="cuda").expand_as(mask)
key = torch.where(mask, ar, torch.full_like(ar, num_n))
p = torch.sort(key, -1).values
idx = torch.where(p == num_n, torch.zeros_like(p), p).to(torch.int32).contiguous()
sp = BlockSparseTensorsTorch(full_block_cnt=cnt, full_block_idx=idx,
                             mask_block_cnt=torch.zeros_like(cnt),
                             mask_block_idx=torch.zeros_like(idx))
qf4, sfq = _nvfp4_quantize_for_fa4(q); kf4, sfk = _nvfp4_quantize_for_fa4(k)
qf4, kf4 = qf4[:, :S].contiguous(), kf4[:, :S].contiguous()

def ev(fn, warm=5, reps=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts)//2]

fp4 = ev(lambda: _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False,
                                 block_sparse_tensors=sp))
bf16 = ev(lambda: _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sp))
print(json.dumps(dict(tiles=S_TILES, keep=keep,
                      persistent=os.environ.get("FA4_FP4_DISABLE_PERSISTENT","0")=="0",
                      fp4_ms=fp4, bf16_ms=bf16)))
'''

results = []
for tiles in (156, 360):
    for disable in ("0", "1"):
        env = dict(os.environ, FA4_FP4_DISABLE_PERSISTENT=disable)
        r = subprocess.run([os.environ["FV_PYTHON"], "-c", CASE, str(tiles), "0.10"],
                           capture_output=True, text=True, env=env, timeout=1800)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if line:
            rec = json.loads(line[-1])
            results.append(rec)
            print(rec, flush=True)
        else:
            print(f"FAILED tiles={tiles} disable={disable}: {r.stderr[-300:]}", flush=True)

out = "artifacts/sparsefp4_native/raw/performance/p4_persistence_fix.json"
json.dump(results, open(out, "w"), indent=2)
print("PFIX_DONE")
