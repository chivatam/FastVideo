"""QAT training backend: genuine VSA sparse attention with fake-quant NVFP4 QK.

Implements the SLA2/Attn-QAT-style recovery recipe for the SparseFP4
composition (arXiv:2602.12675 SS low-bit QAT; arXiv:2603.00040; NVIDIA QAD
2601.20088): the *forward* pass sees NVFP4-quantized Q/K in the sparse fine
branch (exact production round-trip: flashinfer packing + per-16 E4M3 scales,
decoded by the validated oracle pair), while gradients flow through a
straight-through estimator in BF16. The selector and coarse branch stay
exactly deployed-VSA (BF16) — study 2 (F2) proved routing precision is not
binding, and the deployment target (P3/P4) also routes in BF16.

The model therefore *trains against the same attention operator it will be
served with*, up to tile geometry (training uses deployed (4,4,4) VSA; the
FA4-geometry P4 arm shares the same quantizer and sparsity family).

Enable with ``FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA_ATTN`` in a
training run. Inference-time use is valid but pointless (use P3/P4 backends).
"""

from __future__ import annotations

import math

import torch

from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionBackend, VideoSparseAttentionImpl,
                                                            VideoSparseAttentionMetadata)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _unpack_fp4(fp4_tensor: torch.Tensor) -> torch.Tensor:
    raw = fp4_tensor.view(torch.uint8)
    lo = raw & 0xF
    hi = raw >> 4
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)
    return _E2M1.to(raw.device)[codes.long()]


def _sf_to_canonical(sf_mma: torch.Tensor, batch: int, seqlen_padded: int, nheads: int,
                     headdim: int) -> torch.Tensor:
    sf_canonical = sf_mma.permute(6, 5, 2, 4, 0, 1, 3).contiguous()
    scales = sf_canonical.view(torch.float8_e4m3fn).float()
    scales = scales.permute(0, 2, 5, 4, 1, 3, 6)
    return scales.reshape(batch, seqlen_padded, nheads, headdim // 16)


def nvfp4_fake_quant_ste(x: torch.Tensor) -> torch.Tensor:
    """Exact production NVFP4 round-trip on BSHD ``x``, STE gradient."""
    from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
    b, s, h, d = x.shape
    with torch.no_grad():
        fp4, sf = _nvfp4_quantize_for_fa4(x)
        s_pad = fp4.shape[1]
        codes = _unpack_fp4(fp4)
        scales = _sf_to_canonical(sf, b, s_pad, h, d)
        deq = (codes * scales.repeat_interleave(16, dim=-1))[:, :s].to(x.dtype)
    return x + (deq - x).detach()


class SparseFP4QATVSABackend(VideoSparseAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_QAT_VSA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseFP4QATVSAImpl"]:
        return SparseFP4QATVSAImpl


class SparseFP4QATVSAImpl(VideoSparseAttentionImpl):
    """Deployed-VSA structure with fake-quant NVFP4 QK on the fine branch only.

    Matches P3/P4 serving semantics: selector + coarse branch in BF16
    (deployed arithmetic), fine sparse branch sees NVFP4 Q/K. The fine kernel
    is the autograd-enabled Triton block-sparse kernel; gradients reach Q/K
    through the STE wrapper.
    """

    _logged = False

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean, fused_topk_mask)
        from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, compute_topk)

        tile = math.prod(VSA_TILE_SIZE)
        block_sizes = attn_metadata.variable_block_sizes
        n_tiles = int(block_sizes.numel())
        topk = compute_topk(float(attn_metadata.VSA_sparsity), n_tiles)
        b, s, h, d = query.shape

        # fake-quant fine-branch inputs in BSHD (quantizer contract), STE grads
        q_fq = nvfp4_fake_quant_ste(query)
        k_fq = nvfp4_fake_quant_ste(key)

        q_bhsd = query.transpose(1, 2).contiguous()
        k_bhsd = key.transpose(1, 2).contiguous()
        v_bhsd = value.transpose(1, 2).contiguous()
        gate_bhsd = gate_compress.transpose(1, 2).contiguous()

        # selector + coarse branch: deployed BF16 arithmetic (differentiable)
        q_c = fused_block_mean(q_bhsd, block_sizes, tile)
        k_c = fused_block_mean(k_bhsd, block_sizes, tile)
        v_c = fused_block_mean(v_bhsd, block_sizes, tile)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        out_c = torch.matmul(attn, v_c)
        out_c = out_c.view(b, h, n_tiles, 1, d).expand(b, h, n_tiles, tile, d)\
            .reshape(b, h, s, d)
        with torch.no_grad():
            mask = fused_topk_mask(scores.detach(), topk)

        # fine branch: autograd Triton block-sparse kernel on fake-quant Q/K
        out_s = block_sparse_attn(q_fq.transpose(1, 2).contiguous(),
                                  k_fq.transpose(1, 2).contiguous(),
                                  v_bhsd, mask, block_sizes)
        if isinstance(out_s, tuple):
            out_s = out_s[0]

        if not SparseFP4QATVSAImpl._logged:
            logger.info("sparsefp4 QAT-VSA: NVFP4 fake-quant (STE) fine branch + BF16 "
                        "selector/coarse; topk=%d/%d tiles", topk, n_tiles)
            SparseFP4QATVSAImpl._logged = True

        return (out_c * gate_bhsd + out_s).transpose(1, 2)
