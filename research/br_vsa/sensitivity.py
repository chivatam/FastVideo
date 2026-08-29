from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.compressed_halo_vsa.compressed_support import (
    rank_normalized_topk_mask, )

DEFAULT_CANDIDATE_K = (32, 64, 96, 125, 192, 250, 375, 624)


@dataclass(frozen=True)
class SensitivityReplayResult:
    rows: list[dict[str, Any]]
    num_blocks: int
    num_heads: int
    total_seq_length: int


def _dense_unpadded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    non_pad_index: torch.Tensor,
) -> torch.Tensor:
    from flash_attn.cute.interface import flash_attn_func

    query_valid = query.index_select(1, non_pad_index).contiguous()
    key_valid = key.index_select(1, non_pad_index).contiguous()
    value_valid = value.index_select(1, non_pad_index).contiguous()
    output = flash_attn_func(
        query_valid,
        key_valid,
        value_valid,
        softmax_scale=query.shape[-1]**-0.5,
        causal=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2).contiguous()


def _per_head_metrics(
    output: torch.Tensor,
    dense_output: torch.Tensor,
) -> torch.Tensor:
    output_float = output.float()
    dense_float = dense_output.float()
    difference = output_float - dense_float
    reduction_dims = (0, 2, 3)
    difference_norm = difference.square().sum(dim=reduction_dims).sqrt()
    dense_norm = dense_float.square().sum(dim=reduction_dims).sqrt()
    output_norm = output_float.square().sum(dim=reduction_dims).sqrt()
    relative_l2 = difference_norm / dense_norm.clamp_min(1e-12)
    cosine = ((output_float * dense_float).sum(dim=reduction_dims) / (output_norm * dense_norm).clamp_min(1e-12)).clamp(
        -1.0, 1.0)
    cosine_error = 1.0 - cosine
    max_absolute_error = difference.abs().amax(dim=reduction_dims)
    return torch.stack(
        (
            relative_l2,
            cosine_error,
            max_absolute_error,
            output_norm,
            dense_norm,
        ),
        dim=0,
    )


def replay_vsa_sensitivity(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    candidate_k: tuple[int, ...] = DEFAULT_CANDIDATE_K,
    block_elements: int = 64,
) -> SensitivityReplayResult:
    """Replay one captured VSA call at fixed K values against dense attention.

    Inputs use the tiled BSHD layout consumed by ``VideoSparseAttentionImpl``.
    Metrics are reduced per head over batch, valid query tokens, and head
    channels. Sparse outputs include the checkpoint's native coarse residual,
    matching the output actually consumed by the transformer.
    """
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    num_blocks = int(variable_block_sizes.numel())
    if sequence != num_blocks * block_elements:
        raise ValueError("BR-VSA sensitivity replay requires the Wan 64-token tiled path: "
                         f"sequence={sequence}, blocks={num_blocks}, block_elements={block_elements}")
    if non_pad_index.numel() != int(variable_block_sizes.sum().item()):
        raise ValueError("non_pad_index and variable_block_sizes disagree")
    normalized_candidates = tuple(sorted({int(value) for value in candidate_k}))
    if not normalized_candidates:
        raise ValueError("At least one candidate K is required")
    if normalized_candidates[0] < 1 or normalized_candidates[-1] > num_blocks:
        raise ValueError(f"Candidate K must stay in [1, {num_blocks}], got {normalized_candidates}")

    query_coarse = fused_block_mean(
        query_bhsd,
        variable_block_sizes,
        block_elements,
    )
    key_coarse = fused_block_mean(
        key_bhsd,
        variable_block_sizes,
        block_elements,
    )
    value_coarse = fused_block_mean(
        value_bhsd,
        variable_block_sizes,
        block_elements,
    )
    coarse_scores = torch.matmul(
        query_coarse,
        key_coarse.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    coarse_attention = torch.softmax(coarse_scores, dim=-1)
    coarse_output = torch.matmul(coarse_attention, value_coarse)
    coarse_output = (coarse_output.view(
        batch,
        heads,
        num_blocks,
        1,
        head_dim,
    ).expand(-1, -1, -1, block_elements, -1).reshape(batch, heads, sequence, head_dim))
    dense_output = _dense_unpadded_attention(
        query,
        key,
        value,
        non_pad_index,
    )

    rows: list[dict[str, Any]] = []
    for exact_k in normalized_candidates:
        if exact_k == num_blocks:
            sparse_mask = torch.ones(
                coarse_scores.shape,
                dtype=torch.bool,
                device=coarse_scores.device,
            )
        else:
            sparse_mask = rank_normalized_topk_mask(
                coarse_scores,
                exact_k,
            )
        selected_counts = sparse_mask.sum(dim=-1)
        selected_min = int(selected_counts.min().item())
        selected_max = int(selected_counts.max().item())
        if selected_min != exact_k or selected_max != exact_k:
            raise RuntimeError("BR-VSA replay violated exact K: "
                               f"K={exact_k}, selected_min={selected_min}, selected_max={selected_max}")

        exact_output, _ = block_sparse_attn(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            sparse_mask,
            variable_block_sizes,
        )
        sparse_output = exact_output + coarse_output * gate_bhsd
        sparse_valid = sparse_output.index_select(2, non_pad_index)
        metrics = _per_head_metrics(
            sparse_valid,
            dense_output,
        ).detach().cpu()
        for head in range(heads):
            rows.append({
                "head": head,
                "K": exact_k,
                "relative_L2_error": float(metrics[0, head]),
                "cosine_error": float(metrics[1, head]),
                "max_absolute_error": float(metrics[2, head]),
                "output_norm": float(metrics[3, head]),
                "dense_output_norm": float(metrics[4, head]),
                "selected_count_min": selected_min,
                "selected_count_max": selected_max,
                "num_blocks": num_blocks,
                "total_seq_length": int(non_pad_index.numel()),
                "metric_scope": "native_fine_plus_coarse_residual_vs_dense",
            })
        del exact_output, sparse_output, sparse_valid, sparse_mask

    return SensitivityReplayResult(
        rows=rows,
        num_blocks=num_blocks,
        num_heads=heads,
        total_seq_length=int(non_pad_index.numel()),
    )
