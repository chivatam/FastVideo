"""Priority-2: exact-10% VSA256/FA4-aligned controlled matrix.

Arms on identical captured Q/K/V per cell (VSA256 geometry, byte-identical
frozen mask for C0_256/D0_256, exact 1:1 FA4 mapping — no coarsening):

  A0      all-retained BF16 (same machinery)     — reference
  B0      all-retained native NVFP4              — quant-only
  C0_256  sparse BF16, frozen exact-10% mask     — sparse-only
  D0_256  native sparse NVFP4, same mask         — joint

Metrics vs A0 (+ D0 vs C0 conditional), valid tokens only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c5_operator_matrix import metrics  # noqa: E402

from flash_attn.cute.interface import _flash_attn_fwd  # noqa: E402
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch  # noqa: E402
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4  # noqa: E402
from fastvideo_kernel.block_sparse_attn_cute_fwd import _build_vbs_mask_mod  # noqa: E402

K_SPARSE = 128


def pack(m):
    num_n = m.shape[-1]
    cnt = m.sum(-1).to(torch.int32).contiguous()
    ar = torch.arange(num_n, device=m.device).expand_as(m)
    key = torch.where(m, ar, torch.full_like(ar, num_n))
    p = torch.sort(key, -1).values
    idx = torch.where(p == num_n, torch.zeros_like(p), p).to(torch.int32).contiguous()
    return cnt, idx


def lists_from_mask256(mask256, per128_valid):
    mask128 = mask256.repeat_interleave(2, dim=-1)
    nonempty = (per128_valid > 0).view(1, 1, 1, -1)
    isfull = (per128_valid == K_SPARSE).view(1, 1, 1, -1)
    fc, fi = pack(mask128 & isfull)
    mc, mi = pack(mask128 & nonempty & ~isfull)
    return BlockSparseTensorsTorch(full_block_cnt=fc, full_block_idx=fi,
                                   mask_block_cnt=mc, mask_block_idx=mi)


def process(path: Path, records: list):
    cell = torch.load(path, map_location="cuda", weights_only=False)
    q = cell["q_bshd"].cuda()
    k = cell["k_bshd"].cuda()
    v = cell["v_bshd"].cuda()
    mask256 = cell["mask256_bhqk"].cuda()
    vbs = cell["variable_block_sizes"].cuda()
    B, S, H, D = q.shape
    v0 = torch.clamp(vbs, max=K_SPARSE)
    v1 = torch.clamp(vbs - K_SPARSE, min=0)
    per128 = torch.stack([v0, v1], dim=1).reshape(-1).to(torch.int32)
    tok_valid = (torch.arange(256, device="cuda").view(1, -1) < vbs.view(-1, 1)).reshape(S)

    sp_dense = lists_from_mask256(torch.ones_like(mask256), per128)
    sp_frozen = lists_from_mask256(mask256, per128)
    mm = _build_vbs_mask_mod(K_SPARSE)
    aux = [per128.contiguous()]
    base = dict(cell=path.name, layer=cell["layer"], timestep=cell["timestep"],
                keep256=mask256.float().mean().item(), S=S)

    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    qf4, kf4 = qf4[:, :S], kf4[:, :S]

    def call(qq, kk, sf=None, lists=sp_frozen):
        kw = dict(causal=False, block_sparse_tensors=lists, mask_mod=mm, aux_tensors=aux)
        if sf is not None:
            return _flash_attn_fwd(qq, kk, v, mSFQ=sf[0], mSFK=sf[1], **kw)[0]
        return _flash_attn_fwd(qq, kk, v, **kw)[0]

    a0 = call(q, k, lists=sp_dense)
    b0 = call(qf4, kf4, sf=(sfq, sfk), lists=sp_dense)
    c0 = call(q, k)
    d0 = call(qf4, kf4, sf=(sfq, sfk))

    records.append({**base, "arm": "B0_vs_A0", **metrics(b0, a0, tok_valid)})
    records.append({**base, "arm": "C0_256_vs_A0", **metrics(c0, a0, tok_valid)})
    records.append({**base, "arm": "D0_256_vs_A0", **metrics(d0, a0, tok_valid)})
    records.append({**base, "arm": "D0_256_vs_C0_256", **metrics(d0, c0, tok_valid)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    records: list = []
    for d in args.cells_dirs:
        for i, p in enumerate(sorted(d.glob("cell_*.pt"))):
            process(p, records)
            print(f"[{d.name} {i+1}] {p.name}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    import collections
    import statistics
    by = collections.defaultdict(list)
    for r in records:
        by[(r["arm"], r["S"])].append(r["rel_l2"])
    print("\n| Arm | S | n | rel-L2 med | p90 |")
    print("|---|---|---|---|---|")
    for (arm, S), vals in sorted(by.items()):
        vals.sort()
        print(f"| {arm} | {S} | {len(vals)} | {statistics.median(vals):.4f} "
              f"| {vals[int(len(vals)*0.9)]:.4f} |")
    print("EXACT10_DONE")


if __name__ == "__main__":
    main()
