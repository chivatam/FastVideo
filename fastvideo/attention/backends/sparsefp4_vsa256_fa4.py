"""VSA-on-FA4 experiment backend: 256-token tiles, FA4-native sparse geometry.

The deployed VSA uses (4,4,4)=64-token tiles, which forces its fine branch
onto a Triton kernel on sm_100 and forces any FA4 mapping through a lossy
any-pool coarsening (64x64 -> 256x128 raises retention ~2.4x). This backend
"turns VSA into FA4 type": the *selector geometry itself* is changed to
(4,8,8)=256-token tiles so the VSA top-k mask maps 1:1 onto the FA4 SM100
block-sparse granularity (Q: one 256-token tile == one q_stage*128 sparse
row; KV: one tile == exactly two 128-token blocks) — zero mask inflation,
and the fine branch runs on the FlashAttention-4 Blackwell kernel
(2-CTA-era pipeline, TMEM, block-scaled MMA) instead of Triton.

Same algorithm family as deployed VSA (pooled-mean selector, fused_topk_mask,
compression branch weighted by gate_compress); only the tile geometry and the
fine-branch kernel/precision differ.

Fine-branch QK precision via ``FASTVIDEO_SPARSEFP4_FINE``:
  nvfp4 (default) — packed E2M1 Q/K + per-16 E4M3 SFs, native block-scaled
                    MMA, BF16 PV (arm P4)
  bf16            — same kernel/geometry in BF16 (arm P4G)

Enable with ``FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_VSA256_FA4_ATTN``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionBackend, VideoSparseAttentionImpl, VideoSparseAttentionMetadata,
    VideoSparseAttentionMetadataBuilder, compute_topk, construct_variable_block_sizes,
    get_non_pad_index, get_reverse_tile_partition_indices, get_tile_partition_indices,
    scatter_into_tile_buf)
from fastvideo.attention.backends.sparsefp4_native_vsa import _fine_precision, _pack_lists
from fastvideo.logger import init_logger

logger = init_logger(__name__)

TILE256 = (4, 8, 8)
TILE_ELEMENTS = math.prod(TILE256)  # 256
K_SPARSE = 128  # FA4 KV block; one 256-tile == two K blocks


class SparseFP4VSA256FA4Backend(VideoSparseAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_VSA256_FA4_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseFP4VSA256FA4Impl"]:
        return SparseFP4VSA256FA4Impl

    @staticmethod
    def get_builder_cls() -> type["SparseFP4VSA256FA4MetadataBuilder"]:
        return SparseFP4VSA256FA4MetadataBuilder


class SparseFP4VSA256FA4MetadataBuilder(VideoSparseAttentionMetadataBuilder):
    """VSA metadata at (4,8,8)=256-token tile geometry."""

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        cache_tile_buf: bool = True,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])
        num_tiles = (math.ceil(dit_seq_shape[0] / TILE256[0]), math.ceil(dit_seq_shape[1] / TILE256[1]),
                     math.ceil(dit_seq_shape[2] / TILE256[2]))
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(dit_seq_shape, TILE256, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, TILE256, device)
        variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_size=TILE256)
        non_pad_index = get_non_pad_index(variable_block_sizes, TILE_ELEMENTS)
        untile_combined_index = non_pad_index[reverse_tile_partition_indices]

        return VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
            untile_combined_index=untile_combined_index,
            cache_tile_buf=cache_tile_buf)


class SparseFP4VSA256FA4Impl(VideoSparseAttentionImpl):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fine_precision = _fine_precision()
        self._logged = False
        from fastvideo.attention.backends.abstract import layer_idx_from_prefix
        self.layer_idx = layer_idx_from_prefix(kwargs.get("prefix", args[5] if len(args) > 5 else ""),
                                               default=-1)

    def _maybe_capture(self, cfg_path: str, query, key, value, mask256, block_sizes, topk,
                       attn_metadata) -> None:
        """Env-gated QKV+mask dump for the exact-10% controlled matrix."""
        import json
        from pathlib import Path

        from fastvideo.forward_context import get_forward_context

        cfg = json.loads(Path(cfg_path).read_text())
        ctx = get_forward_context()
        timestep = int(getattr(ctx, "current_timestep", -1) or 0)
        batch = getattr(ctx, "forward_batch", None)
        branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"
        if (self.layer_idx not in cfg["layers"] or timestep not in cfg["timesteps"]
                or branch not in cfg.get("cfg_branches", ["positive"])):
            return
        out_dir = Path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"cell_step{timestep:03d}_layer{self.layer_idx:03d}_{branch}.pt"
        torch.save({
            "q_bshd": query.detach().to(torch.bfloat16).cpu(),
            "k_bshd": key.detach().to(torch.bfloat16).cpu(),
            "v_bshd": value.detach().to(torch.bfloat16).cpu(),
            "mask256_bhqk": mask256.detach().to(torch.bool).cpu(),
            "variable_block_sizes": block_sizes.detach().cpu(),
            "topk": int(topk),
            "vsa_sparsity": float(attn_metadata.VSA_sparsity),
            "tile_size": list(TILE256),
            "layer": self.layer_idx,
            "timestep": timestep,
            "cfg_branch": branch,
        }, fname)
        logger.info("sparsefp4 capture256: wrote %s (keep %.4f)", fname,
                    mask256.float().mean().item())

    def tile(self, x: torch.Tensor, attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        """Base ``tile()`` hardcodes the 64-token tile; redo it at 256."""
        num_tiles = attn_metadata.num_tiles
        padded = (num_tiles[0] * TILE256[0]) * (num_tiles[1] * TILE256[1]) * (num_tiles[2] * TILE256[2])
        target_shape = (x.shape[0], padded, x.shape[-2], x.shape[-1])
        if not attn_metadata.cache_tile_buf:
            return scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, None,
                                         attn_metadata.tile_partition_indices)
        buf = scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, attn_metadata.tile_buf,
                                    attn_metadata.tile_partition_indices)
        attn_metadata.tile_buf = buf
        return buf

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
        n_tiles = int(block_sizes.numel())
        topk = compute_topk(float(attn_metadata.VSA_sparsity), n_tiles)
        b, s, h, d = query.shape
        self._timing_pre()

        # ---- selector + coarse branch (VSA algorithm at 256-tile geometry) ----
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
        mask256 = fused_topk_mask(scores, topk)  # [B,H,n_tiles,n_tiles], bool

        capture_cfg = os.environ.get("FASTVIDEO_SPARSEFP4_CAPTURE256", "")
        if capture_cfg:
            self._maybe_capture(capture_cfg, query, key, value, mask256, block_sizes, topk,
                                attn_metadata)

        # ---- exact 1:1 mapping onto FA4 sparse geometry (no coarsening) ----
        # Q: one 256-tile == one sparse row. KV: one tile == two 128-blocks.
        mask128 = mask256.repeat_interleave(2, dim=-1)  # [B,H,n_tiles,2*n_tiles]
        v0 = torch.clamp(block_sizes, max=K_SPARSE)          # valid in first 128-col
        v1 = torch.clamp(block_sizes - K_SPARSE, min=0)      # valid in second
        per128_valid = torch.stack([v0, v1], dim=1).reshape(-1).to(torch.int32)
        nonempty = (per128_valid > 0).view(1, 1, 1, -1)
        is_full = (per128_valid == K_SPARSE).view(1, 1, 1, -1)
        full_cnt, full_idx = _pack_lists(mask128 & is_full)
        part_cnt, part_idx = _pack_lists(mask128 & nonempty & ~is_full)
        sparse = BlockSparseTensorsTorch(full_block_cnt=full_cnt, full_block_idx=full_idx,
                                         mask_block_cnt=part_cnt, mask_block_idx=part_idx)
        mask_mod = _build_vbs_mask_mod(K_SPARSE)
        aux = [per128_valid.contiguous()]

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
                "sparsefp4 VSA256-FA4: fine=%s, tiles=%d x 256 tokens, keep256=%.4f "
                "(exact FA4 mapping, no coarsening)", self.fine_precision, n_tiles,
                mask256.float().mean().item())
            self._logged = True

        if gate_compress is not None:
            out = out_c * gate_compress + out_s
        else:
            out = out_c + out_s
        if os.environ.get("FASTVIDEO_SPARSEFP4_TIMING", "0") == "1":
            torch.cuda.synchronize()
            now = time.perf_counter()
            prev = getattr(self, "_t_last", None)
            if prev is not None:
                logger.info("sparsefp4 timing: fine=%s layer=%d fwd_wall=%.2f ms",
                            self.fine_precision, self.layer_idx, (now - prev) * 1000)
            self._t_last = now
        return out

    def _timing_pre(self) -> None:
        if os.environ.get("FASTVIDEO_SPARSEFP4_TIMING", "0") == "1":
            torch.cuda.synchronize()
            self._t_last = time.perf_counter()
