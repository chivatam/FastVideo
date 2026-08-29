from __future__ import annotations

import torch

from research.anchored_fine_vsa.selection import (
    CHILDREN_PER_PARENT,
    SELECTED_CHILD_BLOCKS,
    select_anchored_support,
    selected_support_mask,
)
from research.fine_vsa.fine_attention import child_block_sizes


def _synthetic_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    parent_sizes = torch.tensor(
        [64] * 120 + [32] * 5 + [8] * 5,
        dtype=torch.int32,
    )
    parent_scores = torch.arange(
        parent_sizes.numel(),
        0,
        -1,
        dtype=torch.float32,
    ).view(1, 1, 1, -1)
    child_sizes = child_block_sizes(parent_sizes, 8)
    child_scores = torch.arange(
        child_sizes.numel(),
        0,
        -1,
        dtype=torch.float32,
    ).view(1, 1, 1, -1)
    return child_scores, child_sizes, parent_scores, parent_sizes


def test_anchored_support_matches_native_valid_tokens() -> None:
    inputs = _synthetic_inputs()
    for anchor_count in (31, 62):
        selection = select_anchored_support(
            *inputs,
            anchor_parent_blocks=anchor_count,
        )
        assert selection.selected_indices.shape[-1] == (
            SELECTED_CHILD_BLOCKS
        )
        assert torch.equal(
            selection.selected_actual_kv_tokens,
            selection.native_actual_kv_tokens,
        )
        assert not selection.fine_tail_anchor_overlap_tokens.any()


def test_fine_tail_excludes_anchor_regions() -> None:
    inputs = _synthetic_inputs()
    selection = select_anchored_support(
        *inputs,
        anchor_parent_blocks=31,
    )
    child_sizes = inputs[1]
    anchor_mask = selected_support_mask(
        selection.anchor_child_indices,
        child_sizes,
    )
    tail_mask = selected_support_mask(
        selection.fine_tail_indices,
        child_sizes,
    )
    assert not (anchor_mask & tail_mask).any()
    assert selection.anchor_child_indices.shape[-1] == (
        31 * CHILDREN_PER_PARENT
    )
    assert selection.fine_tail_indices.shape[-1] == (
        SELECTED_CHILD_BLOCKS - 31 * CHILDREN_PER_PARENT
    )
