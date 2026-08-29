from __future__ import annotations

from dataclasses import dataclass

import torch

from research.fine_vsa.replay import (
    NATIVE_PARENT_K,
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
    select_children_fixed_tokens,
)

CHILD_WIDTH = 8
CHILDREN_PER_PARENT = PARENT_WIDTH // CHILD_WIDTH
SELECTED_CHILD_BLOCKS = NOMINAL_KV_TOKENS // CHILD_WIDTH
ANCHOR_PARENT_BLOCKS = {
    "anchor25": 31,
    "anchor50": 62,
}


@dataclass(frozen=True)
class AnchoredSelection:
    anchor_parent_blocks: int
    native_parent_indices: torch.Tensor
    anchor_parent_indices: torch.Tensor
    anchor_child_indices: torch.Tensor
    fine_tail_indices: torch.Tensor
    selected_indices: torch.Tensor
    native_actual_kv_tokens: torch.Tensor
    anchor_actual_kv_tokens: torch.Tensor
    fine_actual_kv_tokens: torch.Tensor
    selected_actual_kv_tokens: torch.Tensor
    fine_tail_anchor_overlap_tokens: torch.Tensor


def parent_indices_to_children(
    parent_indices: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.arange(
        CHILDREN_PER_PARENT,
        device=parent_indices.device,
        dtype=parent_indices.dtype,
    )
    return (
        parent_indices[..., None] * CHILDREN_PER_PARENT
        + offsets
    ).flatten(-2)


def selected_support_mask(
    selected_indices: torch.Tensor,
    child_sizes: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros(
        (*selected_indices.shape[:-1], child_sizes.numel()),
        dtype=torch.bool,
        device=selected_indices.device,
    )
    valid = child_sizes[selected_indices.long()].gt(0)
    mask.scatter_(-1, selected_indices.long(), valid)
    return mask


def support_token_count(
    support_mask: torch.Tensor,
    child_sizes: torch.Tensor,
) -> torch.Tensor:
    return (
        support_mask.to(child_sizes.dtype)
        * child_sizes.view(
            *((1,) * (support_mask.ndim - 1)),
            -1,
        )
    ).sum(dim=-1)


def select_pure_fine_support(
    child_scores: torch.Tensor,
    child_sizes: torch.Tensor,
    parent_scores: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
        sorted=True,
    ).indices
    native_actual = parent_sizes[native_indices].sum(dim=-1)
    selected = select_children_fixed_tokens(
        child_scores,
        child_sizes,
        selected_blocks=SELECTED_CHILD_BLOCKS,
        factor=CHILDREN_PER_PARENT,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=native_actual,
        child_width=CHILD_WIDTH,
    )
    return selected, native_indices, native_actual


def select_anchored_support(
    child_scores: torch.Tensor,
    child_sizes: torch.Tensor,
    parent_scores: torch.Tensor,
    parent_sizes: torch.Tensor,
    *,
    anchor_parent_blocks: int,
) -> AnchoredSelection:
    if anchor_parent_blocks not in ANCHOR_PARENT_BLOCKS.values():
        raise ValueError(
            "Only the frozen Anchor-25 and Anchor-50 ratios are allowed"
        )
    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
        sorted=True,
    ).indices
    anchor_indices = native_indices[..., :anchor_parent_blocks]
    anchor_children = parent_indices_to_children(anchor_indices)
    native_actual = parent_sizes[native_indices].sum(dim=-1)
    anchor_actual = parent_sizes[anchor_indices].sum(dim=-1)
    fine_target = native_actual - anchor_actual
    fine_tail_blocks = (
        SELECTED_CHILD_BLOCKS
        - anchor_parent_blocks * CHILDREN_PER_PARENT
    )

    tail_candidate = torch.ones_like(
        child_scores,
        dtype=torch.bool,
    )
    tail_candidate.scatter_(-1, anchor_children.long(), False)
    fine_tail = select_children_fixed_tokens(
        child_scores,
        child_sizes,
        selected_blocks=fine_tail_blocks,
        factor=CHILDREN_PER_PARENT,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=fine_target,
        child_width=CHILD_WIDTH,
        candidate_mask=tail_candidate,
    )
    selected = torch.cat(
        [anchor_children.to(torch.int32), fine_tail],
        dim=-1,
    )
    fine_actual = child_sizes[fine_tail.long()].sum(dim=-1)
    selected_actual = child_sizes[selected.long()].sum(dim=-1)
    anchor_mask = selected_support_mask(anchor_children, child_sizes)
    tail_mask = selected_support_mask(fine_tail, child_sizes)
    overlap = support_token_count(
        anchor_mask & tail_mask,
        child_sizes,
    )
    if overlap.any():
        raise RuntimeError(
            "Anchored Fine-VSA fine tail overlaps native anchor tokens"
        )
    if not torch.equal(selected_actual, native_actual):
        raise RuntimeError(
            "Anchored Fine-VSA failed exact valid-token matching"
        )
    if selected.shape[-1] != SELECTED_CHILD_BLOCKS:
        raise RuntimeError(
            "Anchored Fine-VSA violated its fixed descriptor budget"
        )
    return AnchoredSelection(
        anchor_parent_blocks=anchor_parent_blocks,
        native_parent_indices=native_indices,
        anchor_parent_indices=anchor_indices,
        anchor_child_indices=anchor_children.to(torch.int32),
        fine_tail_indices=fine_tail,
        selected_indices=selected,
        native_actual_kv_tokens=native_actual,
        anchor_actual_kv_tokens=anchor_actual,
        fine_actual_kv_tokens=fine_actual,
        selected_actual_kv_tokens=selected_actual,
        fine_tail_anchor_overlap_tokens=overlap,
    )
