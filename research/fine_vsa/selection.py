from __future__ import annotations

from dataclasses import dataclass

import torch

from research.fine_vsa.fine_attention import child_block_sizes


@dataclass(frozen=True)
class Fine8Metadata:
    source_tensor: torch.Tensor
    child_sizes: torch.Tensor
    valid_child_mask: torch.Tensor


_METADATA_BY_DEVICE: dict[tuple[str, int | None], Fine8Metadata] = {}


def fine8_metadata(
    parent_sizes: torch.Tensor,
    *,
    child_width: int = 8,
    parent_width: int = 64,
) -> Fine8Metadata:
    """Cache fixed child geometry for the current per-step VSA metadata."""
    key = (parent_sizes.device.type, parent_sizes.device.index)
    cached = _METADATA_BY_DEVICE.get(key)
    if cached is not None and cached.source_tensor is parent_sizes:
        return cached
    child_sizes = child_block_sizes(
        parent_sizes,
        child_width,
        parent_width=parent_width,
    ).to(torch.int32).contiguous()
    invalid_sizes = child_sizes.ne(0) & child_sizes.ne(child_width)
    if invalid_sizes.any().item():
        raise ValueError("Fine8 requires child geometry containing only full or padded "
                         f"{child_width}-token blocks")
    metadata = Fine8Metadata(
        source_tensor=parent_sizes,
        child_sizes=child_sizes,
        valid_child_mask=child_sizes.eq(child_width),
    )
    _METADATA_BY_DEVICE[key] = metadata
    return metadata


def select_fine8_fixed_tokens(
    child_scores: torch.Tensor,
    valid_child_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    selected_blocks: int,
    child_width: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact fixed-token Fine8 selection for 8-or-0 child geometry.

    The returned tensor contains the maximum active descriptor count needed by
    any row. Only the leading ``selected_counts`` entries are active per row.
    Matching the generic selector's top-k width preserves its exact tie
    behavior while avoiding its filler and scatter bookkeeping.
    """
    if valid_child_mask.ndim != 1:
        raise ValueError("valid_child_mask must be one-dimensional")
    if child_scores.shape[-1] != valid_child_mask.numel():
        raise ValueError("Fine8 score and child-mask geometry disagree")
    if selected_blocks > int(valid_child_mask.numel()):
        raise ValueError("Fine8 selected-block count exceeds child capacity")
    selected_counts = torch.div(
        target_tokens,
        child_width,
        rounding_mode="floor",
    ).to(torch.int32)
    if target_tokens.remainder(child_width).any().item():
        raise ValueError("Fine8 token targets must follow child-width tiling")
    maximum = int(selected_counts.max().item())
    if maximum > selected_blocks:
        raise ValueError("Fine8 target exceeds the fixed descriptor budget")
    masked_scores = child_scores.masked_fill(
        ~valid_child_mask.view(
            *((1, ) * (child_scores.ndim - 1)),
            -1,
        ),
        -float("inf"),
    )
    selected_indices = torch.topk(
        masked_scores,
        maximum,
        dim=-1,
    ).indices.to(torch.int32)
    return selected_indices, selected_counts
