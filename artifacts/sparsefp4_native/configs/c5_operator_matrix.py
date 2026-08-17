"""C5 controlled 2x2 operator matrix on captured Wan2.1 QKV + frozen VSA masks.

Arms (identical captured Q/K/V per cell; C0/D0 share the identical physical
mask and identical variable-block-size trimming):

  A0 DENSE-BF16   BF16 SM100 kernel, all KV blocks retained
  B0 DENSE-NVFP4  native NVFP4 QK (packed E2M1 + E4M3 SF), BF16 PV, all retained
  C0 SPARSE-BF16  BF16 kernel, frozen VSA mask coarsened to FA4 256x128 geometry
  D0 SPARSE-NVFP4 native NVFP4 QK, BF16 PV, byte-identical mask to C0

All four arms run through the same FA4 block-sparse machinery with the same
per-64-tile validity mask_mod (VSA zero-pads boundary tiles), so the only
differences are (a) QK compute precision and (b) which KV blocks are visited.

Also computed per cell:
  C0_TRITON_64    deployed-geometry control: fastvideo_kernel block-sparse
                  Triton kernel on the original 64x64-tile mask (quantifies
                  the mask-coarsening component of C0/D0)
  ORACLE_*        fp32 token-level SDPA references for A0 (validates the
                  harness) and D0 (dequantized NVFP4 oracle, C4-style)

Metrics are computed on valid (non-pad) query tokens only.

Usage:
  CUDA_VISIBLE_DEVICES=2 "$FV_PYTHON" c5_operator_matrix.py \
      --cells-dir /mnt/nvme/scratch/sparsefp4_native/c5-capture-s090/cells \
      --out artifacts/sparsefp4_native/raw/operator/c5_matrix_s090.jsonl
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
from nvfp4_dequant_oracle import dequantize_fa4

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo_kernel.block_sparse_attn_cute_fwd import _build_vbs_mask_mod

Q_SPARSE, K_SPARSE = 256, 128
TILE = 64


def coarsen_mask(mask64: torch.Tensor) -> torch.Tensor:
    """[B,H,nq64,nk64] -> [B,H,nq64*64/256, nk64*64/128] via any-pooling."""
    B, H, nq, nk = mask64.shape
    fq, fk = Q_SPARSE // TILE, K_SPARSE // TILE
    assert nq % fq == 0 and nk % fk == 0
    m = mask64.view(B, H, nq // fq, fq, nk // fk, fk)
    return m.any(dim=3).any(dim=-1)


def mask_to_lists(mask: torch.Tensor, kv_partial_cols: torch.Tensor):
    """Split retained blocks into full/partial lists (partial = vbs-trimmed)."""
    B, H, num_m, num_n = mask.shape
    is_partial = kv_partial_cols.view(1, 1, 1, num_n)
    full_map = mask & ~is_partial
    part_map = mask & is_partial

    def pack(m):
        cnt = m.sum(-1).to(torch.int32).contiguous()
        ar = torch.arange(num_n, device=m.device).expand_as(m)
        key = torch.where(m, ar, torch.full_like(ar, num_n))
        packed = torch.sort(key, dim=-1).values
        idx = torch.where(packed == num_n, torch.zeros_like(packed),
                          packed).to(torch.int32).contiguous()
        return cnt, idx

    full_cnt, full_idx = pack(full_map)
    mask_cnt, mask_idx = pack(part_map)
    return BlockSparseTensorsTorch(full_block_cnt=full_cnt, full_block_idx=full_idx,
                                   mask_block_cnt=mask_cnt, mask_block_idx=mask_idx)


def fa4_arm(q, k, v, sparse, vbs, *, fp4: bool):
    mask_mod = _build_vbs_mask_mod(TILE)
    aux = [vbs.to(torch.int32).contiguous()]
    if fp4:
        S = q.shape[1]
        qf4, sfq = _nvfp4_quantize_for_fa4(q)
        kf4, sfk = _nvfp4_quantize_for_fa4(k)
        out, _ = _flash_attn_fwd(qf4[:, :S], kf4[:, :S], v, mSFQ=sfq, mSFK=sfk,
                                 causal=False, block_sparse_tensors=sparse,
                                 mask_mod=mask_mod, aux_tensors=aux)
    else:
        out, _ = _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sparse,
                                 mask_mod=mask_mod, aux_tensors=aux)
    return out


def oracle(q32, k32, v32, mask_tok_cols: torch.Tensor, blockmask=None):
    """fp32 SDPA; mask_tok_cols[j] False -> column j never attended.
    blockmask: optional [B,H,num_m,num_n] at 256x128 granularity."""
    B, S, H, D = q32.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q32)
    for h in range(H):
        s = torch.matmul(q32[:, :, h].float(), k32[:, :, h].float().transpose(-1, -2)) * scale
        s = s.masked_fill(~mask_tok_cols.view(1, 1, S), float("-inf"))
        if blockmask is not None:
            tok = blockmask[:, h].repeat_interleave(Q_SPARSE, dim=-2)\
                                 .repeat_interleave(K_SPARSE, dim=-1)
            s = s.masked_fill(~tok, float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        out[:, :, h] = torch.matmul(p, v32[:, :, h].float())
    return out


def metrics(x, ref, valid_rows):
    """valid_rows: bool [S] — metric over valid query tokens only."""
    a = x.float()[:, valid_rows].flatten()
    b = ref.float()[:, valid_rows].flatten()
    diff = a - b
    mse = diff.pow(2).mean().item()
    rel_l2 = (diff.norm() / b.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    snr = 10 * math.log10(max(b.pow(2).mean().item(), 1e-30) / max(mse, 1e-30))
    return dict(mse=mse, rel_l2=rel_l2, cosine=cos, snr_db=snr,
                max_abs=diff.abs().max().item(),
                finite=bool(torch.isfinite(a).all()))


def triton_c0(q, k, v, mask64, vbs):
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    qb = q.transpose(1, 2).contiguous()
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()
    out = block_sparse_attn(qb, kb, vb, mask64, vbs)
    if isinstance(out, tuple):
        out = out[0]
    return out.transpose(1, 2).contiguous()


def process_cell(path: Path, records: list):
    cell = torch.load(path, map_location="cuda", weights_only=False)
    q = cell["q_bshd"].cuda()
    k = cell["k_bshd"].cuda()
    v = cell["v_bshd"].cuda()
    mask64 = cell["mask_bool_bhqk"].cuda()
    vbs = cell["variable_block_sizes"].cuda()
    B, S, H, D = q.shape
    n_tiles = vbs.numel()

    # token/column validity from vbs (each 64-tile zero-pads its tail)
    tok_valid = (torch.arange(TILE, device="cuda").view(1, -1) <
                 vbs.view(-1, 1)).reshape(S)
    # per-128-col partial flag: partial iff any of its two 64-tiles is partial
    assert n_tiles % 2 == 0
    per128 = vbs.view(-1, 2)
    kv_partial_cols = (per128 < TILE).any(dim=1)

    maskC = coarsen_mask(mask64)                       # frozen C0/D0 mask, 256x128
    dense = torch.ones_like(maskC)
    sparse_dense = mask_to_lists(dense, kv_partial_cols)
    sparse_frozen = mask_to_lists(maskC, kv_partial_cols)

    base = dict(cell=path.name, layer=cell["layer"], timestep=cell["timestep"],
                cfg_branch=cell["cfg_branch"], S=S, H=H, D=D,
                vsa_sparsity=cell["vsa_sparsity"], topk=cell["topk"],
                keep64=mask64.float().mean().item(),
                keep_fa4=maskC.float().mean().item())

    a0 = fa4_arm(q, k, v, sparse_dense, vbs, fp4=False)
    b0 = fa4_arm(q, k, v, sparse_dense, vbs, fp4=True)
    c0 = fa4_arm(q, k, v, sparse_frozen, vbs, fp4=False)
    d0 = fa4_arm(q, k, v, sparse_frozen, vbs, fp4=True)

    records.append({**base, "arm": "B0_vs_A0", **metrics(b0, a0, tok_valid)})
    records.append({**base, "arm": "C0_vs_A0", **metrics(c0, a0, tok_valid)})
    records.append({**base, "arm": "D0_vs_A0", **metrics(d0, a0, tok_valid)})
    records.append({**base, "arm": "D0_vs_C0", **metrics(d0, c0, tok_valid)})

    # deployed-geometry control (64x64 Triton, exact VSA mask)
    try:
        c0t = triton_c0(q, k, v, mask64, vbs)
        records.append({**base, "arm": "C0_TRITON64_vs_A0", **metrics(c0t, a0, tok_valid)})
    except Exception as exc:  # noqa: BLE001
        records.append({**base, "arm": "C0_TRITON64_vs_A0", "error": repr(exc)})

    # harness validation oracles
    a0_or = oracle(q, k, v, tok_valid)
    records.append({**base, "arm": "A0_vs_fp32_oracle", **metrics(a0, a0_or, tok_valid)})

    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    q_deq = dequantize_fa4(qf4[:, :S].contiguous(), sfq, S)
    k_deq = dequantize_fa4(kf4[:, :S].contiguous(), sfk, S)
    d0_or = oracle(q_deq, k_deq, v, tok_valid, blockmask=maskC)
    records.append({**base, "arm": "D0_vs_dequant_oracle", **metrics(d0, d0_or, tok_valid)})
    del a0_or, d0_or, q_deq, k_deq
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cells = sorted(args.cells_dir.glob("cell_*.pt"))
    if args.limit:
        cells = cells[: args.limit]
    records: list = []
    for i, path in enumerate(cells):
        process_cell(path, records)
        print(f"[{i+1}/{len(cells)}] {path.name} done", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
