from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.fine_vsa.fine_attention import (
    child_block_mean,
    child_block_sizes,
    fine_sparse_attention,
)


PARENT_WIDTH = 64
NATIVE_PARENT_K = 125
NOMINAL_KV_TOKENS = PARENT_WIDTH * NATIVE_PARENT_K
VARIANTS = (
    ("native64_global", 64, 125, None),
    ("kv32_global", 32, 250, None),
    ("kv16_global", 16, 500, None),
    ("kv8_global", 8, 1000, None),
    ("kv16_parent300", 16, 500, 300),
)


@dataclass(frozen=True)
class FineReplayResult:
    error_rows: list[dict[str, Any]]
    mass_rows: list[dict[str, Any]]
    kernel_rows: list[dict[str, Any]]
    geometry: dict[str, Any]


def _dense_padded_attention(
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
    output_bhsd = output.transpose(1, 2).contiguous()
    padded = torch.zeros(
        (
            query.shape[0],
            query.shape[2],
            query.shape[1],
            query.shape[3],
        ),
        device=query.device,
        dtype=query.dtype,
    )
    padded[:, :, non_pad_index] = output_bhsd
    return padded


def _query_block_metrics(
    output: torch.Tensor,
    dense: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, sequence, head_dim = output.shape
    blocks = parent_sizes.numel()
    output_blocks = output.float().view(
        batch,
        heads,
        blocks,
        PARENT_WIDTH,
        head_dim,
    )
    dense_blocks = dense.float().view(
        batch,
        heads,
        blocks,
        PARENT_WIDTH,
        head_dim,
    )
    valid = (
        torch.arange(
            PARENT_WIDTH,
            device=output.device,
        )[None, None, None, :, None]
        < parent_sizes[None, None, :, None, None]
    )
    output_blocks = output_blocks * valid
    dense_blocks = dense_blocks * valid
    difference = output_blocks - dense_blocks
    reduction_dims = (3, 4)
    difference_norm = difference.square().sum(dim=reduction_dims).sqrt()
    dense_norm = dense_blocks.square().sum(dim=reduction_dims).sqrt()
    output_norm = output_blocks.square().sum(dim=reduction_dims).sqrt()
    relative_l2 = difference_norm / dense_norm.clamp_min(1e-12)
    cosine = (
        (output_blocks * dense_blocks).sum(dim=reduction_dims)
        / (output_norm * dense_norm).clamp_min(1e-12)
    ).clamp(-1.0, 1.0)
    return relative_l2, 1.0 - cosine


def _summary_rows(
    relative_l2: torch.Tensor,
    cosine_error: torch.Tensor,
    *,
    variant: str,
    child_width: int,
    selected_blocks: int,
    parent_pool: int | None,
    actual_kv_tokens: torch.Tensor,
    native_actual_kv_tokens: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantile_points = torch.tensor(
        [0.5, 0.9, 0.99],
        device=relative_l2.device,
    )
    query_pair_weight = parent_sizes.float().view(1, 1, -1)
    for scope, head, l2_values, cos_values, actual_values, native_values in [
        (
            "all_heads_query_blocks",
            None,
            relative_l2.flatten(),
            cosine_error.flatten(),
            actual_kv_tokens.flatten(),
            native_actual_kv_tokens.flatten(),
        ),
        *[
            (
                "head_query_blocks",
                head_index,
                relative_l2[:, head_index].flatten(),
                cosine_error[:, head_index].flatten(),
                actual_kv_tokens[:, head_index].flatten(),
                native_actual_kv_tokens[:, head_index].flatten(),
            )
            for head_index in range(relative_l2.shape[1])
        ],
    ]:
        l2_quantiles = torch.quantile(l2_values, quantile_points)
        cos_quantiles = torch.quantile(cos_values, quantile_points)
        if head is None:
            actual_pairs = (
                actual_kv_tokens.float() * query_pair_weight
            ).sum()
            native_pairs = (
                native_actual_kv_tokens.float() * query_pair_weight
            ).sum()
        else:
            actual_pairs = (
                actual_kv_tokens[:, head].float()
                * query_pair_weight[:, 0]
            ).sum()
            native_pairs = (
                native_actual_kv_tokens[:, head].float()
                * query_pair_weight[:, 0]
            ).sum()
        rows.append(
            {
                "event_type": "fine_vsa_error",
                "scope": scope,
                "head": head,
                "variant": variant,
                "child_width": child_width,
                "selected_blocks": selected_blocks,
                "parent_pool": parent_pool,
                "nominal_kv_tokens": selected_blocks * child_width,
                "nominal_pair_budget_ratio": (
                    selected_blocks * child_width / NOMINAL_KV_TOKENS
                ),
                "budget_definition": (
                    "fixed descriptors plus per-query valid-token support "
                    "matched exactly to native VSA80"
                ),
                "actual_kv_tokens_mean": float(
                    actual_values.float().mean().item()
                ),
                "actual_kv_tokens_min": int(actual_values.min().item()),
                "actual_kv_tokens_max": int(actual_values.max().item()),
                "actual_pair_budget_ratio": float(
                    (actual_pairs / native_pairs.clamp_min(1)).item()
                ),
                "relative_L2_mean": float(l2_values.mean().item()),
                "relative_L2_median": float(l2_quantiles[0].item()),
                "relative_L2_p90": float(l2_quantiles[1].item()),
                "relative_L2_p99": float(l2_quantiles[2].item()),
                "cosine_error_mean": float(cos_values.mean().item()),
                "cosine_error_median": float(cos_quantiles[0].item()),
                "cosine_error_p90": float(cos_quantiles[1].item()),
                "cosine_error_p99": float(cos_quantiles[2].item()),
                "query_blocks": int(l2_values.numel()),
            }
        )
    return rows


def select_children_fixed_tokens(
    child_scores: torch.Tensor,
    child_sizes: torch.Tensor,
    *,
    selected_blocks: int,
    factor: int,
    parent_scores: torch.Tensor,
    parent_pool: int | None,
    target_tokens: torch.Tensor,
    child_width: int,
) -> torch.Tensor:
    if target_tokens.max().item() > selected_blocks * child_width:
        raise RuntimeError("Fine-VSA target exceeds the fixed descriptor budget")
    if target_tokens.remainder(8).any():
        raise RuntimeError("Fine-VSA token targets must follow 8-token tiling")

    candidate = torch.ones_like(child_scores, dtype=torch.bool)
    if parent_pool is not None:
        parent_indices = torch.topk(
            parent_scores,
            parent_pool,
            dim=-1,
        ).indices
        parent_candidate = torch.zeros_like(parent_scores, dtype=torch.bool)
        parent_candidate.scatter_(-1, parent_indices, True)
        candidate = parent_candidate.repeat_interleave(factor, dim=-1)

    flat_scores = child_scores.flatten(0, -2)
    flat_candidate = candidate.flatten(0, -2)
    flat_target = target_tokens.flatten()
    rows = flat_scores.shape[0]
    zero_indices = torch.nonzero(
        child_sizes.eq(0),
        as_tuple=False,
    ).flatten()
    if not zero_indices.numel():
        raise RuntimeError(
            "Fine-VSA valid-token matching requires padded child descriptors"
        )
    filler = zero_indices[
        torch.arange(
            selected_blocks,
            device=child_scores.device,
        ).remainder(zero_indices.numel())
    ]
    selected = filler.view(1, -1).expand(rows, -1).clone()

    category_counts: list[tuple[int, torch.Tensor]] = []
    remaining = flat_target
    categories = sorted(
        (
            int(value)
            for value in torch.unique(child_sizes).tolist()
            if int(value) > 0
        ),
        reverse=True,
    )
    for size in categories:
        count = torch.div(
            remaining,
            size,
            rounding_mode="floor",
        )
        category_counts.append((size, count))
        remaining = remaining - count * size
    if remaining.any():
        raise RuntimeError("Fine-VSA could not exactly factor the token target")
    total_active = sum(
        (count for _, count in category_counts),
        start=torch.zeros_like(flat_target),
    )
    if (total_active > selected_blocks).any():
        raise RuntimeError("Fine-VSA active descriptors exceed fixed K")

    write_offset = torch.zeros(
        rows,
        device=child_scores.device,
        dtype=torch.long,
    )
    row_grid = torch.arange(
        rows,
        device=child_scores.device,
    )[:, None]
    for size, count in category_counts:
        maximum = int(count.max().item())
        if maximum == 0:
            continue
        category_mask = (
            child_sizes.view(1, -1).eq(size)
            & flat_candidate
        )
        category_scores = flat_scores.masked_fill(
            ~category_mask,
            -float("inf"),
        )
        top = torch.topk(
            category_scores,
            maximum,
            dim=-1,
        )
        positions = torch.arange(
            maximum,
            device=child_scores.device,
        )[None, :]
        take = positions < count[:, None]
        if not torch.isfinite(top.values[take]).all():
            raise RuntimeError(
                "Fine-VSA hierarchy cannot supply an exact token-matched "
                "child budget"
            )
        destination = write_offset[:, None] + positions
        selected[
            row_grid.expand_as(take)[take],
            destination[take],
        ] = top.indices[take]
        write_offset = write_offset + count
    return selected.view(*child_scores.shape[:-1], selected_blocks).to(
        torch.int32
    )


def _native_internal_mass(
    query_coarse: torch.Tensor,
    key_bhsd: torch.Tensor,
    parent_scores: torch.Tensor,
    parent_sizes: torch.Tensor,
) -> list[dict[str, Any]]:
    batch, heads, query_blocks, head_dim = query_coarse.shape
    parent_blocks = parent_sizes.numel()
    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
    ).indices
    token_scores = torch.matmul(
        query_coarse,
        key_bhsd.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    token_scores = token_scores.view(
        batch,
        heads,
        query_blocks,
        parent_blocks,
        PARENT_WIDTH,
    )
    selected_scores = torch.gather(
        token_scores,
        -2,
        native_indices[..., None].expand(
            *native_indices.shape,
            PARENT_WIDTH,
        ),
    )
    selected_sizes = parent_sizes[native_indices]
    valid = (
        torch.arange(
            PARENT_WIDTH,
            device=query_coarse.device,
        )[None, None, None, None, :]
        < selected_sizes[..., None]
    )
    probabilities = torch.softmax(
        selected_scores.float().masked_fill(~valid, -float("inf")),
        dim=-1,
    )
    top_values = torch.topk(
        probabilities,
        32,
        dim=-1,
    ).values
    fractions = {
        8: top_values[..., :8].sum(dim=-1),
        16: top_values[..., :16].sum(dim=-1),
        32: top_values.sum(dim=-1),
    }
    rows: list[dict[str, Any]] = []
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9, 0.99],
        device=query_coarse.device,
    )
    for head in range(heads):
        for scope, scope_mask in (
            ("all_selected_parents", selected_sizes[:, head].gt(0)),
            ("full_64_token_parents", selected_sizes[:, head].eq(PARENT_WIDTH)),
        ):
            row: dict[str, Any] = {
                "event_type": "native_block_internal_mass",
                "scope": scope,
                "head": head,
                "selected_parent_blocks": NATIVE_PARENT_K,
                "mass_proxy": (
                    "coarse-query softmax over exact token logits within each "
                    "selected native parent"
                ),
            }
            for top_tokens, values in fractions.items():
                flat = values[:, head][scope_mask]
                summary = torch.quantile(flat, quantiles)
                row[f"top{top_tokens}_mean"] = float(flat.mean().item())
                row[f"top{top_tokens}_p10"] = float(summary[0].item())
                row[f"top{top_tokens}_median"] = float(summary[1].item())
                row[f"top{top_tokens}_p90"] = float(summary[2].item())
                row[f"top{top_tokens}_p99"] = float(summary[3].item())
            row["selected_block_samples"] = int(scope_mask.sum().item())
            rows.append(row)
    return rows


def replay_fine_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    validate_native_kernel: bool = False,
) -> FineReplayResult:
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
        raise ValueError("Fine-VSA replay requires the native 64-token tiled path")
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
    dense_output = _dense_padded_attention(
        query,
        key,
        value,
        non_pad_index,
    )
    mass_rows = _native_internal_mass(
        query_coarse,
        key_bhsd,
        parent_scores,
        parent_sizes,
    )

    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
    ).indices
    native_actual_kv_tokens = parent_sizes[native_indices].sum(dim=-1)
    rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    for variant, child_width, selected_blocks, parent_pool in VARIANTS:
        if child_width == PARENT_WIDTH:
            child_key = None
            child_scores = None
            child_sizes = parent_sizes
            selected_indices = native_indices.to(torch.int32)
        else:
            child_key, child_sizes = child_block_mean(
                key_bhsd,
                parent_sizes,
                child_width,
            )
            child_scores = torch.matmul(
                query_coarse,
                child_key.transpose(-2, -1),
            ) / math.sqrt(head_dim)
            factor = PARENT_WIDTH // child_width
            selected_indices = select_children_fixed_tokens(
                child_scores,
                child_sizes,
                selected_blocks=selected_blocks,
                factor=factor,
                parent_scores=parent_scores,
                parent_pool=parent_pool,
                target_tokens=native_actual_kv_tokens,
                child_width=child_width,
            )
        exact_output, _ = fine_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            selected_indices,
            child_sizes,
            child_width=child_width,
        )
        if validate_native_kernel and child_width == PARENT_WIDTH:
            from fastvideo_kernel.block_sparse_attn import block_sparse_attn

            native_mask = torch.zeros_like(parent_scores, dtype=torch.bool)
            native_mask.scatter_(-1, selected_indices.long(), True)
            reference_output, _ = block_sparse_attn(
                query_bhsd,
                key_bhsd,
                value_bhsd,
                native_mask,
                parent_sizes,
            )
            difference = exact_output.float() - reference_output.float()
            kernel_rows.append(
                {
                    "event_type": "fine_vsa_kernel_validation",
                    "variant": variant,
                    "relative_L2": float(
                        (
                            difference.norm()
                            / reference_output.float().norm().clamp_min(1e-12)
                        ).item()
                    ),
                    "max_absolute_error": float(
                        difference.abs().max().item()
                    ),
                    "finite": bool(torch.isfinite(exact_output).all().item()),
                }
            )
            del reference_output, native_mask, difference
        sparse_output = exact_output + coarse_output * gate_bhsd
        relative_l2, cosine_error = _query_block_metrics(
            sparse_output,
            dense_output,
            parent_sizes,
        )
        actual_kv_tokens = child_sizes[selected_indices.long()].sum(dim=-1)
        rows.extend(
            _summary_rows(
                relative_l2,
                cosine_error,
                variant=variant,
                child_width=child_width,
                selected_blocks=selected_blocks,
                parent_pool=parent_pool,
                actual_kv_tokens=actual_kv_tokens,
                native_actual_kv_tokens=native_actual_kv_tokens,
                parent_sizes=parent_sizes,
            )
        )
        del (
            child_key,
            child_scores,
            selected_indices,
            exact_output,
            sparse_output,
            relative_l2,
            cosine_error,
            actual_kv_tokens,
        )

    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "padded_tokens": sequence,
        "valid_tokens": int(non_pad_index.numel()),
        "native_selected_parent_blocks": NATIVE_PARENT_K,
        "native_nominal_kv_tokens_per_query_block": NOMINAL_KV_TOKENS,
        "native_nominal_pairs_per_full_query_block": (
            NOMINAL_KV_TOKENS * PARENT_WIDTH
        ),
        "heads": heads,
        "head_dim": head_dim,
        "kv16_valid_child_blocks": int(
            child_block_sizes(parent_sizes, 16).gt(0).sum().item()
        ),
        "kv16_parent_pool_200_status": (
            "infeasible: ragged boundary parents can provide fewer than "
            "500 nonempty children"
        ),
        "kv16_parent_pool_300_status": (
            "feasible: every 300-parent pool contains at least 500 "
            "nonempty children for this geometry"
        ),
    }
    return FineReplayResult(
        error_rows=rows,
        mass_rows=mass_rows,
        kernel_rows=kernel_rows,
        geometry=geometry,
    )
