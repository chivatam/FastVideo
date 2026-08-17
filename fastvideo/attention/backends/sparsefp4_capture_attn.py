"""SparseFP4 native-composition study: QKV + frozen-VSA-mask capture backend.

Runs the model on **genuine** ``VIDEO_SPARSE_ATTN`` (this impl only adds a
side-channel dump, then delegates to the real VSA forward), and for a
configured set of ``(layer, timestep, cfg_branch)`` cells saves:

- the exact post-RoPE, VSA-tiled/padded BSHD ``q/k/v`` the kernel consumes,
- the **deployed** VSA block mask for those tensors, recomputed with the
  kernel's own functions (``fused_block_mean`` -> bf16 matmul / sqrt(d) ->
  ``fused_topk_mask``) and verified deterministic by double evaluation,
- ``variable_block_sizes``, topk, sparsity and provenance.

These captures feed the controlled A0/B0/C0/D0 operator matrix
(C0 and D0 must consume this mask byte-for-byte).

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_CAPTURE_ATTN
    FASTVIDEO_SPARSEFP4_CAPTURE=/path/to/capture_config.json
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from fastvideo.attention.backends.abstract import layer_idx_from_prefix
from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, VideoSparseAttentionBackend,
                                                            VideoSparseAttentionImpl, VideoSparseAttentionMetadata,
                                                            compute_topk)
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

CAPTURE_ENV_VAR = "FASTVIDEO_SPARSEFP4_CAPTURE"
TILE_ELEMENTS = math.prod(VSA_TILE_SIZE)


@dataclass(frozen=True)
class CaptureConfig:
    out_dir: str
    layers: tuple[int, ...]
    timesteps: tuple[int, ...]
    cfg_branches: tuple[str, ...]

    @classmethod
    def load(cls) -> "CaptureConfig":
        path = os.environ.get(CAPTURE_ENV_VAR)
        if not path:
            raise RuntimeError(f"{CAPTURE_ENV_VAR} must point to a capture config json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(out_dir=data["out_dir"],
                   layers=tuple(data["layers"]),
                   timesteps=tuple(data["timesteps"]),
                   cfg_branches=tuple(data.get("cfg_branches", ["positive"])))


class SparseFP4CaptureAttentionBackend(VideoSparseAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_CAPTURE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseFP4CaptureAttentionImpl"]:
        return SparseFP4CaptureAttentionImpl


class SparseFP4CaptureAttentionImpl(VideoSparseAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__(num_heads, head_size, causal, softmax_scale, num_kv_heads, prefix, **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.layer_idx = layer_idx_from_prefix(prefix, default=-1)
        self.config = CaptureConfig.load()

    def _deployed_mask(self, bhsd_q: torch.Tensor, bhsd_k: torch.Tensor,
                       block_sizes: torch.Tensor, topk: int) -> torch.Tensor:
        """VSA's own mask: kernel pooling, bf16 matmul, kernel top-k."""
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean, fused_topk_mask)
        q_c = fused_block_mean(bhsd_q, block_sizes, TILE_ELEMENTS)
        k_c = fused_block_mean(bhsd_k, block_sizes, TILE_ELEMENTS)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (self.head_size**0.5)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("non-finite VSA block scores during capture")
        return fused_topk_mask(scores, topk)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        config = self.config
        context = get_forward_context()
        timestep = int(getattr(context, "current_timestep", -1) or 0)
        batch = getattr(context, "forward_batch", None)
        cfg_branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"

        if (self.layer_idx in config.layers and timestep in config.timesteps
                and cfg_branch in config.cfg_branches):
            with torch.no_grad():
                block_sizes = attn_metadata.variable_block_sizes
                n_blocks = int(block_sizes.numel())
                sparsity = float(attn_metadata.VSA_sparsity)
                topk = compute_topk(sparsity, n_blocks)
                bhsd_q = query.transpose(1, 2).contiguous()
                bhsd_k = key.transpose(1, 2).contiguous()
                mask = self._deployed_mask(bhsd_q, bhsd_k, block_sizes, topk)
                mask2 = self._deployed_mask(bhsd_q, bhsd_k, block_sizes, topk)
                if not bool((mask == mask2).all()):
                    raise RuntimeError("captured VSA mask is not deterministic across recomputation")
                out_dir = Path(config.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = out_dir / (f"cell_step{timestep:03d}_layer{self.layer_idx:03d}_{cfg_branch}.pt")
                torch.save(
                    {
                        "q_bshd": query.detach().to(torch.bfloat16).cpu(),
                        "k_bshd": key.detach().to(torch.bfloat16).cpu(),
                        "v_bshd": value.detach().to(torch.bfloat16).cpu(),
                        "mask_bool_bhqk": mask.detach().to(torch.bool).cpu(),
                        "variable_block_sizes": block_sizes.detach().cpu(),
                        "topk": topk,
                        "vsa_sparsity": sparsity,
                        "tile_size": list(VSA_TILE_SIZE),
                        "layer": self.layer_idx,
                        "timestep": timestep,
                        "cfg_branch": cfg_branch,
                        "seq_len_padded": int(query.shape[1]),
                        "num_heads": int(query.shape[2]),
                        "head_dim": int(query.shape[3]),
                    }, fname)
                logger.info("sparsefp4 capture: wrote %s (mask keep frac %.4f)", fname,
                            mask.float().mean().item())
        return super().forward(query, key, value, gate_compress, attn_metadata)
