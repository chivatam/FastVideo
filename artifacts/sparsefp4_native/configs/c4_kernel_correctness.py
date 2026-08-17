"""C4 kernel correctness: D0 (native sparse NVFP4) vs trusted dequantized oracle.

For each (shape, sparsity, seed) cell:
  - draw Q/K/V bf16, quantize Q/K with the production quantizer,
  - draw a random block mask at FA4 sparse granularity (256 Q x 128 K),
    always retaining >=1 block per row except one deliberately empty row
    at sparsity<1 to exercise the empty-tile correction,
  - D0     = FP4 kernel + block_sparse_tensors (native path under test)
  - C0ctl  = BF16 dense-SM100 kernel + same block_sparse_tensors
  - oracle = fp32 masked SDPA on dequantize_fa4(Q),dequantize_fa4(K),V
  - B0     = FP4 kernel dense; oracle_dense = fp32 SDPA on dequantized inputs
    (calibrates the tolerance: D0-vs-oracle should be comparable to
     B0-vs-oracle_dense).

Writes JSONL records to artifacts/sparsefp4_native/raw/operator/c4_correctness.jsonl
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_dequant_oracle import dequantize_fa4

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

OUT = "artifacts/sparsefp4_native/raw/operator/c4_correctness.jsonl"
Q_SPARSE, K_SPARSE = 256, 128  # SM100 sparse granularity (q_stage 2 x 128, n_block 128)


def make_mask(B, H, S, retained_frac, seed, empty_row=False):
    """Random block mask. empty_row=True is OUTSIDE the supported envelope:
    fully-empty Q rows deadlock the FP4 kernel's empty-tile correction in
    multi-wave persistent grids (see logs/c2_emptyrow_repro.log). Deployed
    VSA guarantees topk >= 1 per row, so the envelope matches production.
    Kept as an option only for single-wave shapes where it verifies the
    empty-row correction output is exactly zero."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_m, num_n = S // Q_SPARSE, S // K_SPARSE
    k = max(1, round(retained_frac * num_n))
    scores = torch.rand(B, H, num_m, num_n, generator=g, device="cuda")
    idx_top = scores.topk(k, dim=-1).indices
    mask = torch.zeros(B, H, num_m, num_n, dtype=torch.bool, device="cuda")
    mask.scatter_(-1, idx_top, True)
    if empty_row and retained_frac < 1.0 and num_m > 2:
        mask[:, :, num_m // 2] = False
    return mask


def mask_to_sparse_tensors(mask):
    B, H, num_m, num_n = mask.shape
    cnt = mask.sum(-1).to(torch.int32).contiguous()
    ar = torch.arange(num_n, device=mask.device).expand_as(mask)
    key = torch.where(mask, ar, torch.full_like(ar, num_n))
    packed = torch.sort(key, dim=-1).values  # ascending true indices first
    idx = torch.where(packed == num_n, torch.zeros_like(packed), packed)\
        .to(torch.int32).contiguous()
    empty_cnt = torch.zeros_like(cnt)
    empty_idx = torch.zeros_like(idx)
    return BlockSparseTensorsTorch(
        full_block_cnt=cnt, full_block_idx=idx,
        mask_block_cnt=empty_cnt, mask_block_idx=empty_idx)


def oracle_attention(q32, k32, v32, mask=None):
    """fp32 masked SDPA. mask: [B,H,num_m,num_n] block mask or None."""
    B, S, H, D = q32.shape
    scale = 1.0 / math.sqrt(D)
    qh = q32.permute(0, 2, 1, 3)
    kh = k32.permute(0, 2, 1, 3)
    vh = v32.permute(0, 2, 1, 3)
    out = torch.empty_like(qh)
    for h in range(H):  # head loop to bound memory
        s = torch.matmul(qh[:, h], kh[:, h].transpose(-1, -2)) * scale
        if mask is not None:
            tok = mask[:, h].repeat_interleave(Q_SPARSE, dim=-2)\
                            .repeat_interleave(K_SPARSE, dim=-1)
            s = s.masked_fill(~tok, float("-inf"))
        p = torch.softmax(s, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # fully-masked rows -> zero output
        out[:, h] = torch.matmul(p, vh[:, h])
    return out.permute(0, 2, 1, 3)


def metrics(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    diff = a - b
    mse = diff.pow(2).mean().item()
    rel_l2 = (diff.norm() / b.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    maxabs = diff.abs().max().item()
    snr = 10 * math.log10(max(b.pow(2).mean().item(), 1e-30) / max(mse, 1e-30))
    return dict(mse=mse, rel_l2=rel_l2, cosine=cos, max_abs=maxabs, snr_db=snr,
                finite=bool(torch.isfinite(a).all()))


def run_cell(B, S, H, D, retained, seed, rec):
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    qf4, kf4 = qf4[:, :S], kf4[:, :S]
    q_deq = dequantize_fa4(qf4, sfq, S)
    k_deq = dequantize_fa4(kf4, sfk, S)

    base = dict(B=B, S=S, H=H, D=D, retained=retained, seed=seed)

    if retained >= 1.0:
        out_d0, _ = _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False)
        ref = oracle_attention(q_deq, k_deq, v.float(), None)
        rec.append({**base, "arm": "B0_dense_fp4_vs_dequant_oracle",
                    **metrics(out_d0, ref)})
        out_bf, _ = _flash_attn_fwd(q, k, v, causal=False)
        ref_bf = oracle_attention(q.float(), k.float(), v.float(), None)
        rec.append({**base, "arm": "A0_dense_bf16_vs_fp32_oracle",
                    **metrics(out_bf, ref_bf)})
        return

    mask = make_mask(B, H, S, retained, seed,
                     empty_row=(S // Q_SPARSE * H <= 148))  # single-wave only
    sparse = mask_to_sparse_tensors(mask)
    realized = mask.float().mean().item()
    base["realized_retained"] = realized

    out_d0, _ = _flash_attn_fwd(qf4, kf4, v, mSFQ=sfq, mSFK=sfk, causal=False,
                                block_sparse_tensors=sparse)
    ref = oracle_attention(q_deq, k_deq, v.float(), mask)
    rec.append({**base, "arm": "D0_sparse_fp4_vs_dequant_oracle",
                **metrics(out_d0, ref)})

    out_c0, _ = _flash_attn_fwd(q, k, v, causal=False, block_sparse_tensors=sparse)
    ref_bf = oracle_attention(q.float(), k.float(), v.float(), mask)
    rec.append({**base, "arm": "C0_sparse_bf16_vs_fp32_oracle",
                **metrics(out_c0, ref_bf)})


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rec = []
    shapes = [
        (1, 1024, 4, 128),
        (1, 4096, 12, 128),
        (2, 2048, 8, 128),
        (1, 39936, 12, 128),  # Wan 480x832x81 VSA-tiled seqlen
    ]
    for (B, S, H, D) in shapes:
        for retained in (1.0, 0.5, 0.25, 0.10):
            for seed in (0, 1):
                run_cell(B, S, H, D, retained, seed, rec)
                print(f"done S={S} retained={retained} seed={seed}", flush=True)
    with open(OUT, "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    # summary
    import collections
    by_arm = collections.defaultdict(list)
    for r in rec:
        by_arm[r["arm"]].append(r)
    print("\n=== summary (median / worst) ===")
    for arm, rows in by_arm.items():
        cos = sorted(x["cosine"] for x in rows)
        rl2 = sorted(x["rel_l2"] for x in rows)
        fin = all(x["finite"] for x in rows)
        print(f"{arm:45s} n={len(rows):3d} cos_med={cos[len(cos)//2]:.6f} "
              f"cos_min={cos[0]:.6f} relL2_med={rl2[len(rl2)//2]:.3e} "
              f"relL2_max={rl2[-1]:.3e} finite={fin}")


if __name__ == "__main__":
    main()
