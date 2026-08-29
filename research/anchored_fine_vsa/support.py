from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.anchored_fine_vsa.selection import (
    ANCHOR_PARENT_BLOCKS,
    CHILD_WIDTH,
    parent_indices_to_children,
    select_anchored_support,
    select_pure_fine_support,
    selected_support_mask,
    support_token_count,
)
from research.fine_vsa.fine_attention import child_block_mean
from research.fine_vsa.replay import PARENT_WIDTH


@dataclass(frozen=True)
class SupportOverlapResult:
    rows: list[dict[str, Any]]
    geometry: dict[str, Any]


def _intersection_tokens(
    left: torch.Tensor,
    right: torch.Tensor,
    child_sizes: torch.Tensor,
) -> torch.Tensor:
    return support_token_count(left & right, child_sizes)


def _rank_band_mask(
    native_parent_indices: torch.Tensor,
    child_sizes: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    children = parent_indices_to_children(
        native_parent_indices[..., start:stop]
    )
    return selected_support_mask(children, child_sizes)


def _query_mean(values: torch.Tensor) -> torch.Tensor:
    return values.float().mean(dim=(0, 1))


def _query_max(values: torch.Tensor) -> torch.Tensor:
    return values.float().amax(dim=(0, 1))


def analyze_support_overlap(
    query: torch.Tensor,
    key: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> SupportOverlapResult:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    if sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError(
            "Anchored support analysis requires native 64-token tiling"
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
    parent_scores = torch.matmul(
        query_coarse,
        key_parent.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    pure_indices, native_indices, native_tokens = (
        select_pure_fine_support(
            child_scores,
            child_sizes,
            parent_scores,
            parent_sizes,
        )
    )
    pure_mask = selected_support_mask(pure_indices, child_sizes)
    native_mask = selected_support_mask(
        parent_indices_to_children(native_indices),
        child_sizes,
    )
    pure_tokens = support_token_count(pure_mask, child_sizes)
    if not torch.equal(pure_tokens, native_tokens):
        raise RuntimeError(
            "Pure Fine-VSA support failed native valid-token matching"
        )

    top31_mask = _rank_band_mask(
        native_indices,
        child_sizes,
        0,
        31,
    )
    rank32_62_mask = _rank_band_mask(
        native_indices,
        child_sizes,
        31,
        62,
    )
    rank63_125_mask = _rank_band_mask(
        native_indices,
        child_sizes,
        62,
        125,
    )
    columns: dict[str, torch.Tensor] = {
        "native_valid_tokens_mean": _query_mean(native_tokens),
        "fine8_native_overlap_tokens_mean": _query_mean(
            _intersection_tokens(
                pure_mask,
                native_mask,
                child_sizes,
            )
        ),
        "fine8_native_overlap_fraction_mean": _query_mean(
            _intersection_tokens(
                pure_mask,
                native_mask,
                child_sizes,
            )
            / native_tokens.clamp_min(1)
        ),
        "fine8_omitted_native_top31_tokens_mean": _query_mean(
            support_token_count(
                top31_mask & ~pure_mask,
                child_sizes,
            )
        ),
        "fine8_omitted_native_rank32_62_tokens_mean": _query_mean(
            support_token_count(
                rank32_62_mask & ~pure_mask,
                child_sizes,
            )
        ),
        "fine8_omitted_native_rank63_125_tokens_mean": _query_mean(
            support_token_count(
                rank63_125_mask & ~pure_mask,
                child_sizes,
            )
        ),
    }

    for name, anchor_count in ANCHOR_PARENT_BLOCKS.items():
        selection = select_anchored_support(
            child_scores,
            child_sizes,
            parent_scores,
            parent_sizes,
            anchor_parent_blocks=anchor_count,
        )
        anchor_mask = selected_support_mask(
            selection.anchor_child_indices,
            child_sizes,
        )
        tail_mask = selected_support_mask(
            selection.fine_tail_indices,
            child_sizes,
        )
        anchored_mask = anchor_mask | tail_mask
        anchored_tokens = support_token_count(
            anchored_mask,
            child_sizes,
        )
        restored_native = support_token_count(
            anchored_mask & native_mask & ~pure_mask,
            child_sizes,
        )
        columns.update(
            {
                f"{name}_anchor_valid_tokens_mean": _query_mean(
                    selection.anchor_actual_kv_tokens
                ),
                f"{name}_fine_valid_tokens_mean": _query_mean(
                    selection.fine_actual_kv_tokens
                ),
                f"{name}_total_valid_tokens_mean": _query_mean(
                    anchored_tokens
                ),
                f"{name}_anchor_fraction_mean": _query_mean(
                    selection.anchor_actual_kv_tokens
                    / native_tokens.clamp_min(1)
                ),
                f"{name}_fine_tail_fraction_mean": _query_mean(
                    selection.fine_actual_kv_tokens
                    / native_tokens.clamp_min(1)
                ),
                f"{name}_pure_fine_overlap_fraction_mean": _query_mean(
                    _intersection_tokens(
                        anchored_mask,
                        pure_mask,
                        child_sizes,
                    )
                    / native_tokens.clamp_min(1)
                ),
                f"{name}_native_overlap_fraction_mean": _query_mean(
                    _intersection_tokens(
                        anchored_mask,
                        native_mask,
                        child_sizes,
                    )
                    / native_tokens.clamp_min(1)
                ),
                f"{name}_restored_native_tokens_mean": _query_mean(
                    restored_native
                ),
                f"{name}_token_error_abs_max": _query_max(
                    (
                        selection.selected_actual_kv_tokens
                        - native_tokens
                    ).abs()
                ),
                f"{name}_fine_tail_anchor_overlap_tokens_max": (
                    _query_max(
                        selection.fine_tail_anchor_overlap_tokens
                    )
                ),
            }
        )

    cpu_columns = {
        name: values.detach().cpu().tolist()
        for name, values in columns.items()
    }
    query_blocks = int(query_coarse.shape[-2])
    rows = [
        {
            "event_type": "anchored_support_overlap",
            "query_block": query_block,
            **{
                name: values[query_block]
                for name, values in cpu_columns.items()
            },
        }
        for query_block in range(query_blocks)
    ]
    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "child_width": CHILD_WIDTH,
        "query_blocks": query_blocks,
        "heads": heads,
        "padded_tokens": sequence,
        "valid_tokens": int(parent_sizes.sum().item()),
        "support_rows_per_call": query_blocks,
        "support_scope": (
            "one row per query block; token counts and overlap fractions "
            "averaged over attention heads"
        ),
    }
    return SupportOverlapResult(rows=rows, geometry=geometry)
