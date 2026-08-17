"""B3 receipt: NVFP4-quantized selector inputs leave the deployed VSA mask
essentially unchanged on captured cells.

Study 2 (F2) established this at scale on genuine VSA trajectories (NVFP4
routing damage = 0.11% of VSA's own sparsification error, matched-random
225x more damaging). This receipt re-verifies the mechanism on this study's
captured cells with the exact production quantizer round-trip.
Routing metrics here are a RECEIPT for keeping the selector in BF16 —
not headline evidence (skill: secondary diagnostics).
"""
import json
import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_dequant_oracle import dequantize_fa4

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean, fused_topk_mask)

CELLS = Path("/mnt/nvme/scratch/sparsefp4_native/c5-capture-s090/cells")
OUT = Path("artifacts/sparsefp4_native/raw/operator/b3_selector_receipt.json")
TILE = 64

rows = []
for path in sorted(CELLS.glob("cell_*.pt"))[::5]:
    cell = torch.load(path, map_location="cuda", weights_only=False)
    q = cell["q_bshd"].cuda()
    k = cell["k_bshd"].cuda()
    vbs = cell["variable_block_sizes"].cuda()
    topk = cell["topk"]
    S = q.shape[1]

    def deployed_mask(qx, kx):
        qb = qx.transpose(1, 2).contiguous()
        kb = kx.transpose(1, 2).contiguous()
        q_c = fused_block_mean(qb, vbs, TILE)
        k_c = fused_block_mean(kb, vbs, TILE)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (q.shape[-1]**0.5)
        return fused_topk_mask(scores, topk)

    m_bf16 = deployed_mask(q, k)
    qf4, sfq = _nvfp4_quantize_for_fa4(q)
    kf4, sfk = _nvfp4_quantize_for_fa4(k)
    q_deq = dequantize_fa4(qf4[:, :S].contiguous(), sfq, S).to(q.dtype)
    k_deq = dequantize_fa4(kf4[:, :S].contiguous(), sfk, S).to(k.dtype)
    m_fp4 = deployed_mask(q_deq, k_deq)

    agree = (m_bf16 == m_fp4).float().mean().item()
    kept_flip = ((m_bf16 != m_fp4) & m_bf16).sum().item() / max(1, m_bf16.sum().item())
    rows.append(dict(cell=path.name, mask_agreement=agree,
                     kept_block_flip_rate=kept_flip))
    print(rows[-1], flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2) + "\n")
import statistics
print(f"median mask agreement: {statistics.median(r['mask_agreement'] for r in rows):.6f}")
print(f"median kept-block flip rate: {statistics.median(r['kept_block_flip_rate'] for r in rows):.6f}")
print("B3_DONE")
