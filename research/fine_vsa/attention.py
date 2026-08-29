from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.fine_vsa.fine_attention import (
    child_block_mean,
    fine_sparse_attention,
)
from research.fine_vsa.replay import (
    NATIVE_PARENT_K,
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
    select_children_fixed_tokens,
)

CHILD_WIDTH = 8
SELECTED_CHILD_BLOCKS = NOMINAL_KV_TOKENS // CHILD_WIDTH


@dataclass
class FineVSADecision:
    parent_blocks: int
    query_blocks: int
    child_width: int
    selected_child_blocks: int
    active_child_count: torch.Tensor
    native_actual_kv_tokens: torch.Tensor
    selected_actual_kv_tokens: torch.Tensor


def fine_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> tuple[torch.Tensor, FineVSADecision]:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    if sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError("Fine-VSA requires the native 64-token tiled Wan path")

    query_coarse = fused_block_mean(
        query_bhsd,
        parent_sizes,
        PARENT_WIDTH,
    )
    key_parent = fused_block_mean(
        key_bhsd,
        parent_sizes,
        PARENT_WIDTH,
    )
    value_parent = fused_block_mean(
        value_bhsd,
        parent_sizes,
        PARENT_WIDTH,
    )
    parent_scores = torch.matmul(
        query_coarse,
        key_parent.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    parent_attention = torch.softmax(parent_scores, dim=-1)
    coarse_output = torch.matmul(parent_attention, value_parent)
    coarse_output = (coarse_output.view(
        batch,
        heads,
        parent_blocks,
        1,
        head_dim,
    ).expand(-1, -1, -1, PARENT_WIDTH, -1).reshape(batch, heads, sequence, head_dim))

    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
    ).indices
    native_actual_kv_tokens = parent_sizes[native_indices].sum(dim=-1)

    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    selected_indices = select_children_fixed_tokens(
        child_scores,
        child_sizes,
        selected_blocks=SELECTED_CHILD_BLOCKS,
        factor=PARENT_WIDTH // CHILD_WIDTH,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=native_actual_kv_tokens,
        child_width=CHILD_WIDTH,
    )
    selected_actual_kv_tokens = child_sizes[selected_indices.long()].sum(dim=-1)
    exact_output, _ = fine_sparse_attention(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        selected_indices,
        child_sizes,
        child_width=CHILD_WIDTH,
        selected_counts=(selected_actual_kv_tokens // CHILD_WIDTH),
    )
    output = exact_output + coarse_output * gate_bhsd
    decision = FineVSADecision(
        parent_blocks=parent_blocks,
        query_blocks=int(query_coarse.shape[-2]),
        child_width=CHILD_WIDTH,
        selected_child_blocks=SELECTED_CHILD_BLOCKS,
        active_child_count=selected_actual_kv_tokens // CHILD_WIDTH,
        native_actual_kv_tokens=native_actual_kv_tokens,
        selected_actual_kv_tokens=selected_actual_kv_tokens,
    )
    return output.transpose(1, 2), decision


def summarize_fine_vsa_decision(decision: FineVSADecision, ) -> dict[str, Any]:
    token_error = (decision.selected_actual_kv_tokens - decision.native_actual_kv_tokens).abs()
    nominal_sparsity = 1.0 - (decision.selected_child_blocks * decision.child_width /
                              (decision.parent_blocks * PARENT_WIDTH))
    return {
        "parent_blocks":
        decision.parent_blocks,
        "query_blocks":
        decision.query_blocks,
        "parent_width":
        PARENT_WIDTH,
        "native_parent_k":
        NATIVE_PARENT_K,
        "child_width":
        decision.child_width,
        "selected_child_blocks":
        decision.selected_child_blocks,
        "nominal_kv_tokens": (decision.selected_child_blocks * decision.child_width),
        "nominal_pair_budget_ratio": (decision.selected_child_blocks * decision.child_width / NOMINAL_KV_TOKENS),
        "nominal_effective_sparsity":
        nominal_sparsity,
        "active_child_count_min":
        int(decision.active_child_count.min().item()),
        "active_child_count_max":
        int(decision.active_child_count.max().item()),
        "native_actual_kv_tokens_min":
        int(decision.native_actual_kv_tokens.min().item()),
        "native_actual_kv_tokens_max":
        int(decision.native_actual_kv_tokens.max().item()),
        "selected_actual_kv_tokens_min":
        int(decision.selected_actual_kv_tokens.min().item()),
        "selected_actual_kv_tokens_max":
        int(decision.selected_actual_kv_tokens.max().item()),
        "actual_kv_token_error_abs_max":
        int(token_error.max().item()),
        "budget_definition": ("1000 fixed 8-token descriptors; valid-token support matched "
                              "exactly to native VSA80 per query block"),
    }
