from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from research.fine_vsa.fine_attention import (
    child_block_mean,
    child_block_sizes,
    fine_sparse_attention,
)
from research.fine_vsa.replay import (
    NATIVE_PARENT_K,
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
    _dense_padded_attention,
    _query_block_metrics,
    _summary_rows,
    select_children_fixed_tokens,
)
from research.vector_vsa.token_attention import token_sparse_attention

AGGREGATIONS = ("max", "top2_mean", "logsumexp")
VECTOR_WIDTHS = (8, 16)
BASELINE_WIDTHS = (64, 32, 16, 8)
RAW_VARIANT = "raw_k_token"
EXPECTED_VARIANTS = (
    "native64_pooled",
    "fine32_pooled",
    "fine16_pooled",
    "fine8_pooled",
    RAW_VARIANT,
    "raw_vec8_max",
    "raw_vec8_top2_mean",
    "raw_vec8_logsumexp",
    "raw_vec16_max",
    "raw_vec16_top2_mean",
    "raw_vec16_logsumexp",
)


@dataclass(frozen=True)
class VectorReplayResult:
    error_rows: list[dict[str, Any]]
    alignment_rows: list[dict[str, Any]]
    structure_rows: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    geometry: dict[str, Any]


def _elapsed_ms(fn):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def aggregate_raw_scores(
    raw_scores: torch.Tensor,
    token_sizes: torch.Tensor,
    *,
    width: int,
    aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if width not in VECTOR_WIDTHS:
        raise ValueError(f"Unsupported vector width: {width}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if raw_scores.shape[-1] % width:
        raise ValueError("Raw score sequence must be divisible by width")
    group_sizes = token_sizes.view(-1, width).sum(dim=-1)
    values = raw_scores.view(*raw_scores.shape[:-1], -1, width)
    valid = (
        token_sizes.view(-1, width)
        .gt(0)
        .view(
            *((1,) * (values.ndim - 2)),
            -1,
            width,
        )
    )
    masked = values.masked_fill(~valid, -float("inf"))
    if aggregation == "max":
        aggregated = masked.max(dim=-1).values
    elif aggregation == "logsumexp":
        aggregated = torch.logsumexp(masked.float(), dim=-1).to(raw_scores.dtype)
    else:
        top = torch.topk(masked, 2, dim=-1).values
        finite = torch.isfinite(top)
        aggregated = top.masked_fill(~finite, 0).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)
    return aggregated, group_sizes


def select_raw_tokens(
    raw_scores: torch.Tensor,
    token_sizes: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    max_tokens: int = NOMINAL_KV_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(target_tokens.max().item()) > max_tokens:
        raise RuntimeError("Raw-K target exceeds the fixed descriptor budget")
    if target_tokens.remainder(8).any():
        raise RuntimeError("Raw-K targets must match native 8-token tiling")
    valid = token_sizes.view(
        *((1,) * (raw_scores.ndim - 1)),
        -1,
    ).gt(0)
    ranked = torch.topk(
        raw_scores.masked_fill(~valid, -float("inf")),
        max_tokens,
        dim=-1,
    )
    positions = torch.arange(
        max_tokens,
        device=raw_scores.device,
    ).view(*((1,) * target_tokens.ndim), max_tokens)
    active = positions < target_tokens[..., None]
    if not torch.isfinite(ranked.values[active]).all():
        raise RuntimeError("Raw-K could not supply exact valid-token support")
    return ranked.indices.to(torch.int32), target_tokens.to(torch.int32)


def _rank_correlation(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted[..., valid].float()
    target = target[..., valid].float()
    predicted_rank = torch.argsort(
        torch.argsort(predicted, dim=-1, stable=True),
        dim=-1,
        stable=True,
    ).float()
    target_rank = torch.argsort(
        torch.argsort(target, dim=-1, stable=True),
        dim=-1,
        stable=True,
    ).float()
    predicted_rank -= predicted_rank.mean(dim=-1, keepdim=True)
    target_rank -= target_rank.mean(dim=-1, keepdim=True)
    return (predicted_rank * target_rank).sum(dim=-1) / (
        predicted_rank.square().sum(dim=-1).sqrt() * target_rank.square().sum(dim=-1).sqrt()
    ).clamp_min(1e-12)


def _alignment_rows(
    predicted_scores: torch.Tensor,
    true_token_mass: torch.Tensor,
    selected_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    width: int,
    variant: str,
) -> list[dict[str, Any]]:
    true_group_mass = true_token_mass.view(
        *true_token_mass.shape[:-1],
        -1,
        width,
    ).sum(dim=-1)
    valid = group_sizes.gt(0)
    correlation = _rank_correlation(
        predicted_scores,
        true_group_mass,
        valid,
    )
    selected_mass = torch.gather(
        true_group_mass,
        -1,
        selected_indices.long(),
    ).sum(dim=-1)
    selected_count = selected_indices.shape[-1]
    true_top = torch.topk(
        true_group_mass.masked_fill(
            ~valid.view(*((1,) * (true_group_mass.ndim - 1)), -1),
            -float("inf"),
        ),
        selected_count,
        dim=-1,
    ).indices
    selected_mask = torch.zeros_like(true_group_mass, dtype=torch.bool)
    selected_mask.scatter_(-1, selected_indices.long(), True)
    overlap = (
        torch.gather(
            selected_mask,
            -1,
            true_top,
        )
        .float()
        .mean(dim=-1)
    )
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=predicted_scores.device,
    )
    rows: list[dict[str, Any]] = []
    for scope, head, corr_values, mass_values, overlap_values in [
        (
            "all_heads_query_blocks",
            None,
            correlation.flatten(),
            selected_mass.flatten(),
            overlap.flatten(),
        ),
        *[
            (
                "head_query_blocks",
                head_index,
                correlation[:, head_index].flatten(),
                selected_mass[:, head_index].flatten(),
                overlap[:, head_index].flatten(),
            )
            for head_index in range(correlation.shape[1])
        ],
    ]:
        corr_q = torch.quantile(corr_values, quantiles)
        mass_q = torch.quantile(mass_values, quantiles)
        overlap_q = torch.quantile(overlap_values, quantiles)
        rows.append(
            {
                "event_type": "vector_vsa_alignment",
                "scope": scope,
                "head": head,
                "variant": variant,
                "routing_width": width,
                "spearman_mean": float(corr_values.mean().item()),
                "spearman_p10": float(corr_q[0].item()),
                "spearman_median": float(corr_q[1].item()),
                "spearman_p90": float(corr_q[2].item()),
                "retained_dense_mass_mean": float(mass_values.mean().item()),
                "retained_dense_mass_p10": float(mass_q[0].item()),
                "retained_dense_mass_median": float(mass_q[1].item()),
                "retained_dense_mass_p90": float(mass_q[2].item()),
                "top_support_overlap_mean": float(overlap_values.mean().item()),
                "top_support_overlap_p10": float(overlap_q[0].item()),
                "top_support_overlap_median": float(overlap_q[1].item()),
                "top_support_overlap_p90": float(overlap_q[2].item()),
                "mass_definition": (
                    "dense softmax over every valid original K token using "
                    "the unchanged native pooled query representation"
                ),
            }
        )
    return rows


def _raw_alignment_rows(
    raw_scores: torch.Tensor,
    true_token_mass: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    token_sizes: torch.Tensor,
) -> list[dict[str, Any]]:
    valid = token_sizes.gt(0)
    correlation = _rank_correlation(raw_scores, true_token_mass, valid)
    positions = torch.arange(
        selected_indices.shape[-1],
        device=selected_indices.device,
    ).view(*((1,) * selected_counts.ndim), -1)
    active = positions < selected_counts[..., None]
    gathered_mass = torch.gather(
        true_token_mass,
        -1,
        selected_indices.long(),
    )
    selected_mass = gathered_mass.masked_fill(~active, 0).sum(dim=-1)
    rows: list[dict[str, Any]] = []
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=raw_scores.device,
    )
    for scope, head, corr_values, mass_values in [
        (
            "all_heads_query_blocks",
            None,
            correlation.flatten(),
            selected_mass.flatten(),
        ),
        *[
            (
                "head_query_blocks",
                head_index,
                correlation[:, head_index].flatten(),
                selected_mass[:, head_index].flatten(),
            )
            for head_index in range(correlation.shape[1])
        ],
    ]:
        corr_q = torch.quantile(corr_values, quantiles)
        mass_q = torch.quantile(mass_values, quantiles)
        rows.append(
            {
                "event_type": "vector_vsa_alignment",
                "scope": scope,
                "head": head,
                "variant": RAW_VARIANT,
                "routing_width": 1,
                "spearman_mean": float(corr_values.mean().item()),
                "spearman_p10": float(corr_q[0].item()),
                "spearman_median": float(corr_q[1].item()),
                "spearman_p90": float(corr_q[2].item()),
                "retained_dense_mass_mean": float(mass_values.mean().item()),
                "retained_dense_mass_p10": float(mass_q[0].item()),
                "retained_dense_mass_median": float(mass_q[1].item()),
                "retained_dense_mass_p90": float(mass_q[2].item()),
                "top_support_overlap_mean": 1.0,
                "top_support_overlap_p10": 1.0,
                "top_support_overlap_median": 1.0,
                "top_support_overlap_p90": 1.0,
                "mass_definition": (
                    "dense softmax over every valid original K token using "
                    "the unchanged native pooled query representation; raw "
                    "score ranking is therefore the exact mass ranking"
                ),
            }
        )
    return rows


def _support_structure_rows(
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    *,
    key_sequence: int,
) -> list[dict[str, Any]]:
    flat_indices = selected_indices.flatten(0, -2).long()
    flat_counts = selected_counts.flatten().long()
    max_selected = flat_indices.shape[-1]
    positions = torch.arange(
        max_selected,
        device=selected_indices.device,
    )[None, :]
    valid = positions < flat_counts[:, None]
    sortable = torch.where(
        valid,
        flat_indices,
        key_sequence + positions,
    )
    ordered = torch.sort(sortable, dim=-1).values
    contiguous_from_previous = valid & torch.roll(valid, 1, dims=-1) & ordered.eq(torch.roll(ordered, 1, dims=-1) + 1)
    contiguous_from_previous[:, 0] = False
    contiguous_to_next = valid & torch.roll(valid, -1, dims=-1) & torch.roll(ordered, -1, dims=-1).eq(ordered + 1)
    contiguous_to_next[:, -1] = False
    starts = valid & ~contiguous_from_previous
    ends = valid & ~contiguous_to_next
    start_positions = torch.where(starts, positions, -1)
    start_for_token = torch.cummax(start_positions, dim=-1).values
    end_positions = torch.where(ends, positions, max_selected)
    end_for_token = torch.flip(
        torch.cummin(
            torch.flip(end_positions, dims=[-1]),
            dim=-1,
        ).values,
        dims=[-1],
    )
    run_length_for_token = end_for_token - start_for_token + 1
    run_lengths = run_length_for_token[ends].long()
    valid_tokens = valid.sum()
    histogram = torch.bincount(
        run_lengths,
        minlength=max_selected + 1,
    )
    cumulative = histogram.cumsum(dim=0)
    run_count = ends.sum()

    def histogram_quantile(q: float) -> int:
        target = torch.ceil(run_count.float() * q).long().clamp_min(1)
        return int(torch.searchsorted(cumulative, target).item())

    return [
        {
            "event_type": "vector_vsa_support_structure",
            "scope": "all_heads_query_blocks",
            "variant": RAW_VARIANT,
            "query_rows": int(flat_indices.shape[0]),
            "selected_tokens": int(valid_tokens.item()),
            "runs": int(run_count.item()),
            "run_length_mean": float(run_lengths.float().mean().item()),
            "run_length_median": histogram_quantile(0.50),
            "run_length_p90": histogram_quantile(0.90),
            "run_length_p99": histogram_quantile(0.99),
            "run_length_max": int(run_lengths.max().item()),
            "fraction_tokens_in_runs_ge8": float(((valid & run_length_for_token.ge(8)).sum() / valid_tokens).item()),
            "fraction_tokens_in_runs_ge16": float(((valid & run_length_for_token.ge(16)).sum() / valid_tokens).item()),
            "fraction_tokens_in_runs_ge32": float(((valid & run_length_for_token.ge(32)).sum() / valid_tokens).item()),
            "contiguity_definition": (
                "adjacent original padded-sequence K positions after "
                "sorting each query block's selected raw token indices"
            ),
        }
    ]


def replay_vector_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
) -> VectorReplayResult:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    if batch != 1:
        raise ValueError("Vector-VSA census currently requires batch=1")
    if sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError("Vector-VSA requires native padded KV64 geometry")

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
    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
    ).indices
    native_actual = parent_sizes[native_indices].sum(dim=-1)

    raw_scores, raw_score_ms = _elapsed_ms(
        lambda: (
            torch.matmul(
                query_coarse,
                key_bhsd.transpose(-2, -1),
            )
            / math.sqrt(head_dim)
        )
    )
    token_sizes = child_block_sizes(parent_sizes, 1)
    valid_tokens = token_sizes.view(1, 1, 1, -1).gt(0)
    true_token_mass = torch.softmax(
        raw_scores.float().masked_fill(~valid_tokens, -float("inf")),
        dim=-1,
    )

    error_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    def record_error(
        *,
        variant: str,
        exact_output: torch.Tensor,
        actual_tokens: torch.Tensor,
        child_width: int,
        selected_blocks: int,
        execution_ms: float,
        metadata: dict[str, Any],
    ) -> None:
        sparse_output = exact_output + coarse_output * gate_bhsd
        relative_l2, cosine_error = _query_block_metrics(
            sparse_output,
            dense_output,
            parent_sizes,
        )
        rows = _summary_rows(
            relative_l2,
            cosine_error,
            variant=variant,
            child_width=child_width,
            selected_blocks=selected_blocks,
            parent_pool=None,
            actual_kv_tokens=actual_tokens,
            native_actual_kv_tokens=native_actual,
            parent_sizes=parent_sizes,
        )
        for row in rows:
            row["event_type"] = "vector_vsa_error"
            row["execution_ms"] = execution_ms
            row.update(metadata)
        error_rows.extend(rows)

    pooled_variants = [
        ("native64_pooled", 64, 125),
        ("fine32_pooled", 32, 250),
        ("fine16_pooled", 16, 500),
        ("fine8_pooled", 8, 1000),
    ]
    for variant, width, selected_blocks in pooled_variants:
        if width == PARENT_WIDTH:
            scores = parent_scores
            sizes = parent_sizes
            selected = native_indices.to(torch.int32)
            scoring_ms = 0.0
            selection_ms = 0.0
        else:
            child_key, sizes = child_block_mean(
                key_bhsd,
                parent_sizes,
                width,
            )
            scores, scoring_ms = _elapsed_ms(
                lambda child_key=child_key: (
                    torch.matmul(
                        query_coarse,
                        child_key.transpose(-2, -1),
                    )
                    / math.sqrt(head_dim)
                )
            )
            selected, selection_ms = _elapsed_ms(
                lambda scores=scores, sizes=sizes, selected_blocks=selected_blocks, width=width: (
                    select_children_fixed_tokens(
                        scores,
                        sizes,
                        selected_blocks=selected_blocks,
                        factor=PARENT_WIDTH // width,
                        parent_scores=parent_scores,
                        parent_pool=None,
                        target_tokens=native_actual,
                        child_width=width,
                    )
                )
            )
        exact_output, execution_ms = _elapsed_ms(
            lambda selected=selected, sizes=sizes, width=width: fine_sparse_attention(
                query_bhsd,
                key_bhsd,
                value_bhsd,
                selected,
                sizes,
                child_width=width,
            )[0]
        )
        actual = sizes[selected.long()].sum(dim=-1)
        metadata = {
            "candidate_kind": "pooled_baseline",
            "routing_width": width,
            "execution_width": width,
            "aggregation": "mean_k_before_dot",
            "raw_score_ms": raw_score_ms,
            "aggregation_ms": 0.0,
            "scoring_ms": scoring_ms,
            "selection_ms": selection_ms,
        }
        record_error(
            variant=variant,
            exact_output=exact_output,
            actual_tokens=actual,
            child_width=width,
            selected_blocks=selected_blocks,
            execution_ms=execution_ms,
            metadata=metadata,
        )
        alignment_rows.extend(
            _alignment_rows(
                scores,
                true_token_mass,
                selected,
                sizes,
                width=width,
                variant=variant,
            )
        )
        benchmark_rows.append(
            {
                "event_type": "vector_vsa_benchmark",
                "variant": variant,
                **metadata,
                "execution_ms": execution_ms,
            }
        )
        del exact_output, actual
        if width != PARENT_WIDTH:
            del child_key

    (raw_selected, raw_counts), raw_selection_ms = _elapsed_ms(
        lambda: select_raw_tokens(
            raw_scores,
            token_sizes,
            native_actual,
        )
    )
    raw_output, raw_execution_ms = _elapsed_ms(
        lambda: token_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            raw_selected,
            raw_counts,
        )[0]
    )
    raw_metadata = {
        "candidate_kind": "raw_token",
        "routing_width": 1,
        "execution_width": 1,
        "aggregation": "none",
        "raw_score_ms": raw_score_ms,
        "aggregation_ms": 0.0,
        "scoring_ms": raw_score_ms,
        "selection_ms": raw_selection_ms,
    }
    record_error(
        variant=RAW_VARIANT,
        exact_output=raw_output,
        actual_tokens=raw_counts,
        child_width=1,
        selected_blocks=NOMINAL_KV_TOKENS,
        execution_ms=raw_execution_ms,
        metadata=raw_metadata,
    )
    alignment_rows.extend(
        _raw_alignment_rows(
            raw_scores,
            true_token_mass,
            raw_selected,
            raw_counts,
            token_sizes,
        )
    )
    structure_rows.extend(
        _support_structure_rows(
            raw_selected,
            raw_counts,
            key_sequence=sequence,
        )
    )
    benchmark_rows.append(
        {
            "event_type": "vector_vsa_benchmark",
            "variant": RAW_VARIANT,
            **raw_metadata,
            "execution_ms": raw_execution_ms,
        }
    )
    del raw_output

    for width in VECTOR_WIDTHS:
        for aggregation in AGGREGATIONS:
            (scores, sizes), aggregation_ms = _elapsed_ms(
                lambda width=width, aggregation=aggregation: aggregate_raw_scores(
                    raw_scores,
                    token_sizes,
                    width=width,
                    aggregation=aggregation,
                )
            )
            selected_blocks = NOMINAL_KV_TOKENS // width
            selected, selection_ms = _elapsed_ms(
                lambda scores=scores, sizes=sizes, width=width, selected_blocks=selected_blocks: (
                    select_children_fixed_tokens(
                        scores,
                        sizes,
                        selected_blocks=selected_blocks,
                        factor=PARENT_WIDTH // width,
                        parent_scores=parent_scores,
                        parent_pool=None,
                        target_tokens=native_actual,
                        child_width=width,
                    )
                )
            )
            exact_output, execution_ms = _elapsed_ms(
                lambda selected=selected, sizes=sizes, width=width: fine_sparse_attention(
                    query_bhsd,
                    key_bhsd,
                    value_bhsd,
                    selected,
                    sizes,
                    child_width=width,
                )[0]
            )
            actual = sizes[selected.long()].sum(dim=-1)
            variant = f"raw_vec{width}_{aggregation}"
            metadata = {
                "candidate_kind": "raw_score_vector",
                "routing_width": 1,
                "execution_width": width,
                "aggregation": aggregation,
                "raw_score_ms": raw_score_ms,
                "aggregation_ms": aggregation_ms,
                "scoring_ms": raw_score_ms + aggregation_ms,
                "selection_ms": selection_ms,
            }
            record_error(
                variant=variant,
                exact_output=exact_output,
                actual_tokens=actual,
                child_width=width,
                selected_blocks=selected_blocks,
                execution_ms=execution_ms,
                metadata=metadata,
            )
            alignment_rows.extend(
                _alignment_rows(
                    scores,
                    true_token_mass,
                    selected,
                    sizes,
                    width=width,
                    variant=variant,
                )
            )
            benchmark_rows.append(
                {
                    "event_type": "vector_vsa_benchmark",
                    "variant": variant,
                    **metadata,
                    "execution_ms": execution_ms,
                }
            )
            del scores, sizes, selected, exact_output, actual

    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "padded_tokens": sequence,
        "valid_tokens": int(non_pad_index.numel()),
        "native_selected_parent_blocks": NATIVE_PARENT_K,
        "native_nominal_kv_tokens": NOMINAL_KV_TOKENS,
        "raw_token_descriptor_capacity": NOMINAL_KV_TOKENS,
        "raw_score_semantics": (
            "unchanged pooled Qc dot unpooled original K token divided by sqrt(head_dim); raw logits ranked directly"
        ),
        "coarse_residual_policy": (
            "unchanged native pooled-Q/K/V coarse output and gate residual for every baseline and candidate"
        ),
    }
    return VectorReplayResult(
        error_rows=error_rows,
        alignment_rows=alignment_rows,
        structure_rows=structure_rows,
        benchmark_rows=benchmark_rows,
        geometry=geometry,
    )
