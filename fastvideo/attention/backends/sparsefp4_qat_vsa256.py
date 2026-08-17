"""DQ-VSA training backend: VSA256/FA4-aligned sparse attention, fake-quant NVFP4 QK.

Stage-2 recovery operator for the SparseFP4 composition (see
``artifacts/sparsefp4_native/TRAINING_RECOVERY_PLAN.md``): the *training-time*
twin of the P4 serving backend (``SPARSEFP4_VSA256_FA4_ATTN``).

Identical to P4 semantics wherever it matters for QAT:
  - selector at (4,8,8)=256-token tile geometry (same top-k policy, BF16),
  - coarse/compression branch BF16 (differentiable),
  - fine branch consumes NVFP4 fake-quantized Q/K produced by the *production*
    quantizer round-trip (flashinfer packing + per-16 E4M3 scales, decoded by
    the validated oracle pair) with a straight-through estimator,
  - PV stays BF16.

Differences from serving, by necessity:
  - the fine kernel is the autograd Triton block-sparse kernel (route A of
    ``fastvideo_kernel.block_sparse_attn_256``) rather than the forward-only
    FA4 CuTe kernel. Backward recomputes attention probabilities from the
    *saved fake-quantized* Q/K — the low-precision-consistent backward that
    Attn-QAT-style recipes require — and uses the saved O/logsumexp pair.

``fine_qat`` (per-impl attribute, default True) switches the fake-quant off;
a frozen teacher instance with ``fine_qat=False`` is exactly the P4G operator
up to fine-kernel numerics. Toggle with :func:`set_fine_qat`.

``FASTVIDEO_DQVSA_NAIVE_BWD=1`` selects the *naive* QAT baseline (T2 arm of
the recovery matrix): forward still sees the fake-quantized output, but
gradients flow through the BF16 fine path via an output-level STE —
i.e. the backward attention probabilities are computed from UNquantized
Q/K, the approximation Attn-QAT (arXiv:2603.00040 §3.2) warns against.
Default (T3 semantics): backward recomputes from the saved fake-quantized
Q/K.

Enable with ``FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA256_ATTN``.
"""

from __future__ import annotations

import os

import torch

from fastvideo.attention.backends.sparsefp4_qat_vsa import nvfp4_fake_quant_ste
from fastvideo.attention.backends.sparsefp4_vsa256_fa4 import (TILE_ELEMENTS, SparseFP4VSA256FA4Impl,
                                                               SparseFP4VSA256FA4MetadataBuilder)
from fastvideo.attention.backends.video_sparse_attn import (VideoSparseAttentionBackend, VideoSparseAttentionMetadata,
                                                            compute_topk)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SparseFP4QATVSA256Backend(VideoSparseAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_QAT_VSA256_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SparseFP4QATVSA256Impl]:
        return SparseFP4QATVSA256Impl

    @staticmethod
    def get_builder_cls() -> type[SparseFP4VSA256FA4MetadataBuilder]:
        return SparseFP4VSA256FA4MetadataBuilder


def set_fine_qat(model: torch.nn.Module, enabled: bool) -> int:
    """Flip fake-quant on every DQ-VSA impl under ``model``; returns count."""
    n = 0
    for module in model.modules():
        for attr in ("attn_impl", "impl"):
            impl = getattr(module, attr, None)
            if isinstance(impl, SparseFP4QATVSA256Impl):
                impl.fine_qat = enabled
                n += 1
    return n


class SparseFP4QATVSA256Impl(SparseFP4VSA256FA4Impl):
    """Training twin of the P4 arm: VSA256 selector + fake-quant NVFP4 fine QK."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fine_qat = True
        self.naive_bwd = os.environ.get("FASTVIDEO_DQVSA_NAIVE_BWD", "0") == "1"
        self._logged: bool = False

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean, fused_topk_mask)

        block_sizes = attn_metadata.variable_block_sizes
        n_tiles = int(block_sizes.numel())
        topk = compute_topk(float(attn_metadata.VSA_sparsity), n_tiles)
        b, s, h, d = query.shape

        # ---- selector + coarse branch: BF16, differentiable (same as P4/P4G) ----
        q_bhsd = query.transpose(1, 2).contiguous()
        k_bhsd = key.transpose(1, 2).contiguous()
        v_bhsd = value.transpose(1, 2).contiguous()
        q_c = fused_block_mean(q_bhsd, block_sizes, TILE_ELEMENTS)
        k_c = fused_block_mean(k_bhsd, block_sizes, TILE_ELEMENTS)
        v_c = fused_block_mean(v_bhsd, block_sizes, TILE_ELEMENTS)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        out_c = torch.matmul(attn, v_c)
        out_c = out_c.view(b, h, n_tiles, 1, d).expand(b, h, n_tiles, TILE_ELEMENTS, d)\
            .reshape(b, h, s, d).transpose(1, 2)
        with torch.no_grad():
            mask256 = fused_topk_mask(scores.detach(), topk)  # [B,H,n_tiles,n_tiles]

        # ---- fine branch: production fake-quant NVFP4 QK (STE), BF16 PV ----
        if self.fine_qat and self.naive_bwd:
            # T2 semantics: fake-quant forward VALUE, BF16 backward PATH.
            out_bf16, _ = block_sparse_attn_256(q_bhsd, k_bhsd, v_bhsd, mask256, block_sizes)
            with torch.no_grad():
                q_fq = nvfp4_fake_quant_ste(query).transpose(1, 2).contiguous()
                k_fq = nvfp4_fake_quant_ste(key).transpose(1, 2).contiguous()
                out_fq, _ = block_sparse_attn_256(q_fq, k_fq, v_bhsd, mask256, block_sizes)
            out_s = out_bf16 + (out_fq - out_bf16).detach()
        else:
            if self.fine_qat:
                q_f = nvfp4_fake_quant_ste(query)
                k_f = nvfp4_fake_quant_ste(key)
            else:  # teacher mode: P4G operator (BF16 fine QK, same mask policy)
                q_f, k_f = query, key
            out_s, _ = block_sparse_attn_256(
                q_f.transpose(1, 2).contiguous(),
                k_f.transpose(1, 2).contiguous(), v_bhsd, mask256, block_sizes)
        out_s = out_s.transpose(1, 2)

        if not self._logged:
            logger.info(
                "sparsefp4 DQ-VSA256: fine_qat=%s, naive_bwd=%s, tiles=%d x 256 tokens, "
                "keep256=%.4f (autograd Triton route-A fine kernel)", self.fine_qat, self.naive_bwd, n_tiles,
                mask256.float().mean().item())
            self._logged = True

        if gate_compress is not None:
            return out_c * gate_compress + out_s
        return out_c + out_s
