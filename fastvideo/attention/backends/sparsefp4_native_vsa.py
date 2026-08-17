"""P3 production backend: deployed VSA selector + coarse branch, native
sparse-NVFP4 fine branch (SparseFP4 native-composition study).

Identical to ``VIDEO_SPARSE_ATTN`` in everything but the fine sparse-branch
compute:

- selector: the kernel's own ``fused_block_mean`` -> bf16 matmul / sqrt(d)
  -> ``fused_topk_mask`` (byte-identical mask to deployed VSA),
- coarse/compression branch: identical pooled softmax, weighted by the
  model's ``gate_compress`` projection,
- fine branch: FA4 SM100 block-sparse attention consuming the frozen VSA
  mask coarsened to the kernel's 256x128 sparse geometry (any-pooled — a
  superset of the 64x64 mask), with per-64-tile validity trimming via
  ``mask_mod``. QK compute precision is switchable:

    FASTVIDEO_SPARSEFP4_FINE=nvfp4  (default) packed E2M1 Q/K + E4M3 SFs,
                                    native Blackwell block-scaled MMA, BF16 PV
    FASTVIDEO_SPARSEFP4_FINE=bf16   same kernel/geometry in BF16 — the
                                    geometry-control arm (P2g)

Enable with ``FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_NATIVE_VSA_ATTN``.
"""

from __future__ import annotations

import math
import os

import torch

from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, VideoSparseAttentionBackend,
                                                            VideoSparseAttentionImpl, VideoSparseAttentionMetadata,
                                                            compute_topk)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

TILE = math.prod(VSA_TILE_SIZE)
Q_SPARSE, K_SPARSE = 256, 128
FINE_ENV_VAR = "FASTVIDEO_SPARSEFP4_FINE"


def _fine_precision() -> str:
    value = os.environ.get(FINE_ENV_VAR, "nvfp4").strip().lower()
    if value not in ("nvfp4", "bf16"):
        raise ValueError(f"{FINE_ENV_VAR} must be 'nvfp4' or 'bf16', got {value!r}")
    return value


def _coarsen_mask(mask64: torch.Tensor) -> torch.Tensor:
    b, h, nq, nk = mask64.shape
    fq, fk = Q_SPARSE // TILE, K_SPARSE // TILE
    if nq % fq or nk % fk:
        raise ValueError(f"mask {mask64.shape} not divisible by coarsening {fq}x{fk}")
    return mask64.view(b, h, nq // fq, fq, nk // fk, fk).any(3).any(-1)


def _pack_lists(mask: torch.Tensor):
    num_n = mask.shape[-1]
    cnt = mask.sum(-1).to(torch.int32).contiguous()
    ar = torch.arange(num_n, device=mask.device).expand_as(mask)
    key = torch.where(mask, ar, torch.full_like(ar, num_n))
    packed = torch.sort(key, dim=-1).values
    idx = torch.where(packed == num_n, torch.zeros_like(packed), packed)\
        .to(torch.int32).contiguous()
    return cnt, idx


class SparseFP4NativeVSABackend(VideoSparseAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_NATIVE_VSA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseFP4NativeVSAImpl"]:
        return SparseFP4NativeVSAImpl


class SparseFP4NativeVSAImpl(VideoSparseAttentionImpl):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fine_precision = _fine_precision()
        self._logged = False

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean, fused_topk_mask)
        from flash_attn.cute.interface import _flash_attn_fwd
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from fastvideo_kernel.block_sparse_attn_cute_fwd import _build_vbs_mask_mod
        from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

        block_sizes = attn_metadata.variable_block_sizes
        n_blocks = int(block_sizes.numel())
        topk = compute_topk(float(attn_metadata.VSA_sparsity), n_blocks)
        b, s, h, d = query.shape

        # ---- selector + coarse branch: byte-identical to deployed VSA ----
        q_bhsd = query.transpose(1, 2).contiguous()
        k_bhsd = key.transpose(1, 2).contiguous()
        v_bhsd = value.transpose(1, 2).contiguous()
        q_c = fused_block_mean(q_bhsd, block_sizes, TILE)
        k_c = fused_block_mean(k_bhsd, block_sizes, TILE)
        v_c = fused_block_mean(v_bhsd, block_sizes, TILE)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        out_c = torch.matmul(attn, v_c)                       # [B,H,nq,D]
        out_c = out_c.view(b, h, n_blocks, 1, d).expand(b, h, n_blocks, TILE, d)\
            .reshape(b, h, s, d).transpose(1, 2)              # BSHD
        mask64 = fused_topk_mask(scores, topk)                # [B,H,nq,nk]

        # ---- fine branch: FA4 block-sparse (nvfp4 or bf16 QK) ----
        mask_fa4 = _coarsen_mask(mask64)
        per128_partial = (block_sizes.view(-1, K_SPARSE // TILE) < TILE).any(dim=1)
        is_partial = per128_partial.view(1, 1, 1, -1)
        full_cnt, full_idx = _pack_lists(mask_fa4 & ~is_partial)
        part_cnt, part_idx = _pack_lists(mask_fa4 & is_partial)
        sparse = BlockSparseTensorsTorch(full_block_cnt=full_cnt, full_block_idx=full_idx,
                                         mask_block_cnt=part_cnt, mask_block_idx=part_idx)
        mask_mod = _build_vbs_mask_mod(TILE)
        aux = [block_sizes.to(torch.int32).contiguous()]

        if self.fine_precision == "nvfp4":
            qf4, sfq = _nvfp4_quantize_for_fa4(query)
            kf4, sfk = _nvfp4_quantize_for_fa4(key)
            out_s, _ = _flash_attn_fwd(qf4[:, :s], kf4[:, :s], value, mSFQ=sfq, mSFK=sfk,
                                       causal=False, block_sparse_tensors=sparse,
                                       mask_mod=mask_mod, aux_tensors=aux)
        else:
            out_s, _ = _flash_attn_fwd(query, key, value, causal=False,
                                       block_sparse_tensors=sparse,
                                       mask_mod=mask_mod, aux_tensors=aux)

        if not self._logged:
            logger.info(
                "sparsefp4 native VSA: fine=%s, tile=%d, fa4 sparse geometry %dx%d, "
                "keep64=%.4f keep_fa4=%.4f", self.fine_precision, TILE, Q_SPARSE,
                K_SPARSE, mask64.float().mean().item(), mask_fa4.float().mean().item())
            self._logged = True

        if gate_compress is not None:
            return out_c * gate_compress + out_s
        return out_c + out_s
