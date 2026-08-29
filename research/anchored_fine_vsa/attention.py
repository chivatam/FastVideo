from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.anchored_fine_vsa.selection import (
    CHILD_WIDTH,
    CHILDREN_PER_PARENT,
    SELECTED_CHILD_BLOCKS,
    select_anchored_support,
)
from research.fine_vsa.fine_attention import (
    child_block_mean,
    fine_sparse_attention,
)
from research.fine_vsa.replay import (
    NATIVE_PARENT_K,
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
)


@dataclass(frozen=True)
class AnchoredFineVSADecision:
    parent_blocks: int
    query_blocks: int
    anchor_parent_blocks: int
    anchor_actual_kv_tokens: torch.Tensor
    fine_actual_kv_tokens: torch.Tensor
    native_actual_kv_tokens: torch.Tensor
    selected_actual_kv_tokens: torch.Tensor
    fine_tail_anchor_overlap_tokens: torch.Tensor


def anchored_fine_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    *,
    anchor_parent_blocks: int,
) -> tuple[torch.Tensor, AnchoredFineVSADecision]:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    if sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError(
            "Anchored Fine-VSA requires the native 64-token tiled Wan path"
        )

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
    coarse_output = (
        coarse_output.view(
            batch,
            heads,
            parent_blocks,
            1,
            head_dim,
        )
        .expand(-1, -1, -1, PARENT_WIDTH, -1)
        .reshape(batch, heads, sequence, head_dim)
    )

    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    selection = select_anchored_support(
        child_scores,
        child_sizes,
        parent_scores,
        parent_sizes,
        anchor_parent_blocks=anchor_parent_blocks,
    )
    exact_output, _ = fine_sparse_attention(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        selection.selected_indices,
        child_sizes,
        child_width=CHILD_WIDTH,
    )
    output = exact_output + coarse_output * gate_bhsd
    decision = AnchoredFineVSADecision(
        parent_blocks=parent_blocks,
        query_blocks=int(query_coarse.shape[-2]),
        anchor_parent_blocks=anchor_parent_blocks,
        anchor_actual_kv_tokens=selection.anchor_actual_kv_tokens,
        fine_actual_kv_tokens=selection.fine_actual_kv_tokens,
        native_actual_kv_tokens=selection.native_actual_kv_tokens,
        selected_actual_kv_tokens=selection.selected_actual_kv_tokens,
        fine_tail_anchor_overlap_tokens=(
            selection.fine_tail_anchor_overlap_tokens
        ),
    )
    return output.transpose(1, 2), decision


def summarize_anchored_decision(
    decision: AnchoredFineVSADecision,
) -> dict[str, Any]:
    anchor_descriptors = (
        decision.anchor_parent_blocks * CHILDREN_PER_PARENT
    )
    fine_descriptors = SELECTED_CHILD_BLOCKS - anchor_descriptors
    token_error = (
        decision.selected_actual_kv_tokens
        - decision.native_actual_kv_tokens
    ).abs()
    nominal_sparsity = 1.0 - (
        SELECTED_CHILD_BLOCKS
        * CHILD_WIDTH
        / (decision.parent_blocks * PARENT_WIDTH)
    )
    return {
        "parent_blocks": decision.parent_blocks,
        "query_blocks": decision.query_blocks,
        "parent_width": PARENT_WIDTH,
        "native_parent_k": NATIVE_PARENT_K,
        "child_width": CHILD_WIDTH,
        "anchor_parent_blocks": decision.anchor_parent_blocks,
        "anchor_child_descriptors": anchor_descriptors,
        "fine_tail_child_descriptors": fine_descriptors,
        "selected_child_blocks": SELECTED_CHILD_BLOCKS,
        "nominal_anchor_kv_tokens": (
            decision.anchor_parent_blocks * PARENT_WIDTH
        ),
        "nominal_fine_kv_tokens": fine_descriptors * CHILD_WIDTH,
        "nominal_kv_tokens": NOMINAL_KV_TOKENS,
        "nominal_pair_budget_ratio": 1.0,
        "nominal_effective_sparsity": nominal_sparsity,
        "anchor_valid_tokens_min": int(
            decision.anchor_actual_kv_tokens.min().item()
        ),
        "anchor_valid_tokens_max": int(
            decision.anchor_actual_kv_tokens.max().item()
        ),
        "fine_valid_tokens_min": int(
            decision.fine_actual_kv_tokens.min().item()
        ),
        "fine_valid_tokens_max": int(
            decision.fine_actual_kv_tokens.max().item()
        ),
        "total_valid_tokens_min": int(
            decision.selected_actual_kv_tokens.min().item()
        ),
        "total_valid_tokens_max": int(
            decision.selected_actual_kv_tokens.max().item()
        ),
        "native_valid_tokens_min": int(
            decision.native_actual_kv_tokens.min().item()
        ),
        "native_valid_tokens_max": int(
            decision.native_actual_kv_tokens.max().item()
        ),
        "actual_kv_token_error_abs_max": int(token_error.max().item()),
        "fine_tail_anchor_overlap_tokens_max": int(
            decision.fine_tail_anchor_overlap_tokens.max().item()
        ),
        "budget_definition": (
            "fixed 1000 eight-token descriptors; top native parent anchors "
            "plus a disjoint fine tail; valid-token support exactly matched "
            "to native VSA80 per query block"
        ),
    }
