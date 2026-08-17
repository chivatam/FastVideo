"""B1/B2 precision ladder on captured cells (sparse arms, frozen masks).

Arms (all block-sparse with the C5 frozen FA4 mask + vbs mask_mod):
  SP-BF16            BF16 QK + BF16 PV          (C0 control)
  SP-NVFP4           NVFP4 QK + BF16 PV         (D0 baseline)
  SP-NVFP4-PV8       NVFP4 QK + FP8-E4M3 PV     (B1: skill's sanctioned next PV point)
  SP-MXFP8           MXFP8 QK (per-32 E8M0 SF) + BF16 PV   (B2 ladder point)

Metrics vs A0 (all-retained BF16 on identical machinery) per cell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c5_operator_matrix import (Q_SPARSE, K_SPARSE, TILE, coarsen_mask, mask_to_lists, metrics)

from flash_attn.cute.interface import _flash_attn_fwd
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo_kernel.block_sparse_attn_cute_fwd import _build_vbs_mask_mod


def mxfp8_quantize_fa4(x: torch.Tensor):
    """(B,S,H,D) bf16 -> MXFP8 e4m3 tensor + per-32 E8M0 SF in FA4 MMA layout."""
    from flashinfer.quantization import mxfp8_quantize, SfLayout
    b, s, h, d = x.shape
    assert s % 128 == 0
    t2d = x.reshape(b * s, h * d)
    fp8_data, sf_data = mxfp8_quantize(t2d, sf_swizzle_layout=SfLayout.layout_128x4)
    fp8 = fp8_data.reshape(b, s, h, d)
    sf_vec = 32
    rest_m = s // 128
    sf_k = d // sf_vec
    rest_k = sf_k // 4
    sf = sf_data.reshape(b * rest_m, (h * sf_k) // 4, 32, 4, 4)
    sf = sf.reshape(b, rest_m, h, rest_k, 32, 4, 4)
    sf = sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return fp8, sf


def run_cell(path: Path, records: list):
    cell = torch.load(path, map_location="cuda", weights_only=False)
    q = cell["q_bshd"].cuda()
    k = cell["k_bshd"].cuda()
    v = cell["v_bshd"].cuda()
    mask64 = cell["mask_bool_bhqk"].cuda()
    vbs = cell["variable_block_sizes"].cuda()
    B, S, H, D = q.shape
    tok_valid = (torch.arange(TILE, device="cuda").view(1, -1) < vbs.view(-1, 1)).reshape(S)
    per128 = (vbs.view(-1, 2) < TILE).any(dim=1)
    maskC = coarsen_mask(mask64)
    sparse_dense = mask_to_lists(torch.ones_like(maskC), per128)
    sparse_frozen = mask_to_lists(maskC, per128)
    mm = _build_vbs_mask_mod(TILE)
    aux = [vbs.to(torch.int32).contiguous()]
    base = dict(cell=path.name, layer=cell["layer"], timestep=cell["timestep"],
                keep_fa4=maskC.float().mean().item())

    def sparse_call(qq, kk, vv, sfq=None, sfk=None, lists=sparse_frozen):
        kwargs = dict(causal=False, block_sparse_tensors=lists, mask_mod=mm, aux_tensors=aux)
        if sfq is not None:
            return _flash_attn_fwd(qq, kk, vv, mSFQ=sfq, mSFK=sfk, **kwargs)[0]
        return _flash_attn_fwd(qq, kk, vv, **kwargs)[0]

    a0 = sparse_call(q, k, v, lists=sparse_dense)

    qf4, sfq4 = _nvfp4_quantize_for_fa4(q)
    kf4, sfk4 = _nvfp4_quantize_for_fa4(k)
    qf4, kf4 = qf4[:, :S], kf4[:, :S]
    q8, sfq8 = mxfp8_quantize_fa4(q)
    k8, sfk8 = mxfp8_quantize_fa4(k)
    v8 = v.to(torch.float8_e4m3fn)

    arms = {
        "SP-BF16": lambda: sparse_call(q, k, v),
        "SP-NVFP4": lambda: sparse_call(qf4, kf4, v, sfq4, sfk4),
        "SP-NVFP4-PV8": lambda: sparse_call(qf4, kf4, v8, sfq4, sfk4),
        "SP-MXFP8": lambda: sparse_call(q8, k8, v, sfq8, sfk8),
    }
    for name, fn in arms.items():
        try:
            out = fn()
            records.append({**base, "arm": name, **metrics(out, a0, tok_valid)})
        except Exception as exc:  # noqa: BLE001
            records.append({**base, "arm": name, "error": repr(exc)[:300]})
        print(f"  {name}: {records[-1].get('rel_l2', records[-1].get('error'))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", type=Path,
                    default=Path("/mnt/nvme/scratch/sparsefp4_native/c5-capture-s090/cells"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/sparsefp4_native/raw/operator/b_ladder.jsonl"))
    ap.add_argument("--limit", type=int, default=9)
    args = ap.parse_args()
    cells = sorted(args.cells_dir.glob("cell_*.pt"))[:: max(1, 25 // args.limit)][: args.limit]
    records: list = []
    for i, p in enumerate(cells):
        print(f"[{i+1}/{len(cells)}] {p.name}", flush=True)
        run_cell(p, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    import collections
    import statistics
    by = collections.defaultdict(list)
    for r in records:
        if "rel_l2" in r:
            by[r["arm"]].append(r["rel_l2"])
    print("\n| Arm | n | rel-L2 med | max |")
    print("|---|---|---|---|")
    for arm, vals in by.items():
        print(f"| {arm} | {len(vals)} | {statistics.median(vals):.4f} | {max(vals):.4f} |")
    errs = [r for r in records if "error" in r]
    for e in errs[:3]:
        print("ERROR", e["arm"], e["error"][:150])
    print("B_LADDER_DONE")


if __name__ == "__main__":
    main()
