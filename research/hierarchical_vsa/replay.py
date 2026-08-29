from __future__ import annotations

import math
import time
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
    _dense_padded_attention,
    _query_block_metrics,
    _summary_rows,
    select_children_fixed_tokens,
)

AGGREGATORS = ("mass", "logsumexp", "top2_mean", "max")
SCORE_WIDTHS = (8, 16)
SOFT_PRIORS = (0.0, 0.25, 0.5)
PARENT_POOLS = ("all", "top300")
EXEC_WIDTHS = (64, 128)


@dataclass(frozen=True)
class HierarchicalReplayResult:
    error_rows: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    geometry: dict[str, Any]


def aggregate_execution_scores(
    child_scores: torch.Tensor,
    child_sizes: torch.Tensor,
    *,
    score_width: int,
    execution_width: int,
    aggregation: str,
) -> torch.Tensor:
    if aggregation not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    if execution_width % score_width:
        raise ValueError("Execution width must be divisible by score width")
    factor = execution_width // score_width
    groups = child_scores.shape[-1] // factor
    scores = child_scores.view(
        *child_scores.shape[:-1],
        groups,
        factor,
    )
    sizes = child_sizes.view(groups, factor)
    valid = sizes.gt(0).view(
        *((1,) * (scores.ndim - 2)),
        groups,
        factor,
    )
    masked = scores.masked_fill(~valid, -float("inf"))
    if aggregation == "mass":
        log_weight = sizes.clamp_min(1).float().log().view(
            *((1,) * (scores.ndim - 2)),
            groups,
            factor,
        )
        return torch.logsumexp(
            (scores.float() + log_weight).masked_fill(
                ~valid,
                -float("inf"),
            ),
            dim=-1,
        ).to(scores.dtype)
    if aggregation == "logsumexp":
        return torch.logsumexp(masked.float(), dim=-1).to(scores.dtype)
    if aggregation == "max":
        return masked.max(dim=-1).values
    top = torch.topk(masked, min(2, factor), dim=-1).values
    finite = torch.isfinite(top)
    return (
        top.masked_fill(~finite, 0).sum(dim=-1)
        / finite.sum(dim=-1).clamp_min(1)
    )


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    centered = scores - scores.mean(dim=-1, keepdim=True)
    return centered / centered.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)


def _soft_prior(
    hierarchical: torch.Tensor,
    native: torch.Tensor,
    value: float,
) -> torch.Tensor:
    if value == 0:
        return hierarchical
    return (
        (1.0 - value) * _normalize_scores(hierarchical)
        + value * _normalize_scores(native)
    )


def select_exec64_fixed_histogram(
    scores: torch.Tensor,
    parent_sizes: torch.Tensor,
    native_indices: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    candidate = (
        torch.ones_like(scores, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.to(torch.bool)
    )
    flat_scores = scores.flatten(0, -2)
    flat_candidate = candidate.flatten(0, -2)
    flat_native = native_indices.flatten(0, -2)
    rows = flat_scores.shape[0]
    selected = torch.empty(
        (rows, NATIVE_PARENT_K),
        dtype=torch.long,
        device=scores.device,
    )
    write_offset = torch.zeros(
        rows,
        dtype=torch.long,
        device=scores.device,
    )
    row_grid = torch.arange(rows, device=scores.device)[:, None]
    native_sizes = parent_sizes[flat_native]
    for size in sorted(
        (int(value) for value in torch.unique(parent_sizes).tolist()),
        reverse=True,
    ):
        count = native_sizes.eq(size).sum(dim=-1)
        maximum = int(count.max().item())
        if maximum == 0:
            continue
        eligible = (
            parent_sizes.view(1, -1).eq(size)
            & flat_candidate
        )
        ranked = torch.topk(
            flat_scores.masked_fill(~eligible, -float("inf")),
            maximum,
            dim=-1,
        )
        positions = torch.arange(
            maximum,
            device=scores.device,
        )[None, :]
        take = positions < count[:, None]
        if not torch.isfinite(ranked.values[take]).all():
            raise RuntimeError(
                "Hierarchical top-300 pool cannot reproduce the native "
                "valid-size histogram"
            )
        destination = write_offset[:, None] + positions
        selected[
            row_grid.expand_as(take)[take],
            destination[take],
        ] = ranked.indices[take]
        write_offset += count
    if not write_offset.eq(NATIVE_PARENT_K).all():
        raise RuntimeError("KV64 fixed histogram did not fill K=125")
    return selected.view(*scores.shape[:-1], NATIVE_PARENT_K).to(
        torch.int32
    )


def select_exec128_under_budget(
    scores: torch.Tensor,
    group_sizes: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    selected_groups: int = 62,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_scores = scores.flatten(0, -2).float()
    flat_target = target_tokens.flatten()
    normalized_sizes = group_sizes.float() / 128.0
    minimum = torch.topk(
        -group_sizes.float(),
        selected_groups,
    ).indices
    minimum_tokens = group_sizes[minimum].sum()
    if (flat_target < minimum_tokens).any():
        raise RuntimeError(
            "KV128 fixed group count cannot fit the native valid budget"
        )
    raw = torch.topk(flat_scores, selected_groups, dim=-1).indices
    raw_tokens = group_sizes[raw].sum(dim=-1)
    low = torch.zeros_like(flat_target, dtype=torch.float32)
    high = torch.ones_like(low)
    minimum = minimum.expand(flat_scores.shape[0], -1)
    feasible_indices = minimum
    feasible_tokens = minimum_tokens.expand_as(flat_target)

    # QK logits are not guaranteed to have a fixed scale. Establish a
    # feasible per-row upper bracket before binary search instead of relying
    # on a fixed penalty that can be too small for high-magnitude states.
    for _ in range(32):
        adjusted = (
            flat_scores
            - high[:, None] * normalized_sizes[None, :]
        )
        indices = torch.topk(
            adjusted,
            selected_groups,
            dim=-1,
        ).indices
        tokens = group_sizes[indices].sum(dim=-1)
        feasible = tokens <= flat_target
        feasible_indices = torch.where(
            feasible[:, None],
            indices,
            feasible_indices,
        )
        feasible_tokens = torch.where(
            feasible,
            tokens,
            feasible_tokens,
        )
        if feasible.all():
            break
        high = torch.where(feasible, high, high * 2.0)

    for _ in range(28):
        middle = (low + high) * 0.5
        adjusted = (
            flat_scores
            - middle[:, None] * normalized_sizes[None, :]
        )
        indices = torch.topk(
            adjusted,
            selected_groups,
            dim=-1,
        ).indices
        tokens = group_sizes[indices].sum(dim=-1)
        feasible = tokens <= flat_target
        feasible_indices = torch.where(
            feasible[:, None],
            indices,
            feasible_indices,
        )
        feasible_tokens = torch.where(
            feasible,
            tokens,
            feasible_tokens,
        )
        low = torch.where(feasible, low, middle)
        high = torch.where(feasible, middle, high)

    selected = feasible_indices
    selected_tokens = feasible_tokens
    use_raw = raw_tokens <= flat_target
    selected = torch.where(use_raw[:, None], raw, selected)
    selected_tokens = torch.where(
        use_raw,
        raw_tokens,
        selected_tokens,
    )
    if (selected_tokens > flat_target).any():
        raise RuntimeError("KV128 execution exceeded native valid support")
    shape = (*scores.shape[:-1], selected_groups)
    return (
        selected.view(shape).to(torch.int32),
        selected_tokens.view(scores.shape[:-1]),
    )


def _groups_to_parent_indices(
    group_indices: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.tensor(
        [0, 1],
        device=group_indices.device,
        dtype=group_indices.dtype,
    )
    return (group_indices[..., None] * 2 + offsets).flatten(-2)


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


def replay_hierarchical_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
) -> HierarchicalReplayResult:
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
        raise ValueError("HF-VSA replay requires native KV64 tiling")

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
    parent_scores, native_score_ms = _elapsed_ms(
        lambda: torch.matmul(
            query_coarse,
            key_parent.transpose(-2, -1),
        )
        / math.sqrt(head_dim)
    )
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
        sorted=True,
    ).indices
    native_actual = parent_sizes[native_indices].sum(dim=-1)
    top300 = torch.topk(
        parent_scores,
        300,
        dim=-1,
    ).indices
    top300_mask = torch.zeros_like(parent_scores, dtype=torch.bool)
    top300_mask.scatter_(-1, top300, True)

    error_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    def evaluate(
        *,
        variant: str,
        selected_indices: torch.Tensor,
        nominal_width: int,
        nominal_blocks: int,
        actual_tokens: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        exact_output, execution_ms = _elapsed_ms(
            lambda: fine_sparse_attention(
                query_bhsd,
                key_bhsd,
                value_bhsd,
                selected_indices,
                parent_sizes,
                child_width=PARENT_WIDTH,
            )[0]
        )
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
            child_width=nominal_width,
            selected_blocks=nominal_blocks,
            parent_pool=(
                300 if metadata.get("parent_pool") == "top300" else None
            ),
            actual_kv_tokens=actual_tokens,
            native_actual_kv_tokens=native_actual,
            parent_sizes=parent_sizes,
        )
        for row in rows:
            row["event_type"] = "hierarchical_vsa_error"
            row.update(metadata)
            row["execution_ms"] = execution_ms
        error_rows.extend(rows)

    evaluate(
        variant="native64_global",
        selected_indices=native_indices.to(torch.int32),
        nominal_width=64,
        nominal_blocks=125,
        actual_tokens=native_actual,
        metadata={
            "candidate_kind": "baseline",
            "score_width": 64,
            "execution_width": 64,
            "aggregation": "native_mean",
            "soft_native_prior": 1.0,
            "parent_pool": "all",
        },
    )

    child_data: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for score_width in SCORE_WIDTHS:
        child_key, child_sizes = child_block_mean(
            key_bhsd,
            parent_sizes,
            score_width,
        )
        child_scores, score_ms = _elapsed_ms(
            lambda child_key=child_key: torch.matmul(
                query_coarse,
                child_key.transpose(-2, -1),
            )
            / math.sqrt(head_dim)
        )
        child_data[score_width] = (
            child_scores,
            child_sizes,
            score_ms,
        )
    fine_scores, fine_sizes, _ = child_data[8]
    fine_indices = select_children_fixed_tokens(
        fine_scores,
        fine_sizes,
        selected_blocks=1000,
        factor=8,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=native_actual,
        child_width=8,
    )
    fine_output, fine_execution_ms = _elapsed_ms(
        lambda: fine_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            fine_indices,
            fine_sizes,
            child_width=8,
        )[0]
    )
    fine_sparse = fine_output + coarse_output * gate_bhsd
    fine_l2, fine_cosine = _query_block_metrics(
        fine_sparse,
        dense_output,
        parent_sizes,
    )
    fine_actual = fine_sizes[fine_indices.long()].sum(dim=-1)
    rows = _summary_rows(
        fine_l2,
        fine_cosine,
        variant="kv8_global",
        child_width=8,
        selected_blocks=1000,
        parent_pool=None,
        actual_kv_tokens=fine_actual,
        native_actual_kv_tokens=native_actual,
        parent_sizes=parent_sizes,
    )
    for row in rows:
        row["event_type"] = "hierarchical_vsa_error"
        row.update(
            {
                "candidate_kind": "baseline",
                "score_width": 8,
                "execution_width": 8,
                "aggregation": "fine_global",
                "soft_native_prior": 0.0,
                "parent_pool": "all",
                "execution_ms": fine_execution_ms,
            }
        )
    error_rows.extend(rows)

    for score_width in SCORE_WIDTHS:
        child_scores, child_sizes, score_ms = child_data[score_width]
        for execution_width in EXEC_WIDTHS:
            group_sizes = child_sizes.view(
                sequence // execution_width,
                execution_width // score_width,
            ).sum(dim=-1)
            if execution_width == 64:
                native_prior = parent_scores
                pools = PARENT_POOLS
                nominal_blocks = 125
            else:
                parent_pair_sizes = parent_sizes.view(-1, 2)
                native_prior = (
                    parent_scores.view(
                        *parent_scores.shape[:-1],
                        -1,
                        2,
                    )
                    * parent_pair_sizes.float().view(
                        *((1,) * (parent_scores.ndim - 1)),
                        -1,
                        2,
                    )
                ).sum(dim=-1) / parent_pair_sizes.sum(
                    dim=-1
                ).clamp_min(1).view(
                    *((1,) * (parent_scores.ndim - 1)),
                    -1,
                )
                pools = ("all",)
                nominal_blocks = 62
            for aggregation in AGGREGATORS:
                hierarchical, aggregation_ms = _elapsed_ms(
                    lambda aggregation=aggregation: (
                        aggregate_execution_scores(
                            child_scores,
                            child_sizes,
                            score_width=score_width,
                            execution_width=execution_width,
                            aggregation=aggregation,
                        )
                    )
                )
                for parent_pool in pools:
                    candidate_mask = (
                        top300_mask if parent_pool == "top300" else None
                    )
                    for soft_prior in SOFT_PRIORS:
                        combined = _soft_prior(
                            hierarchical,
                            native_prior,
                            soft_prior,
                        )
                        selection_start = torch.cuda.Event(
                            enable_timing=True
                        )
                        selection_end = torch.cuda.Event(
                            enable_timing=True
                        )
                        selection_start.record()
                        if execution_width == 64:
                            selected = select_exec64_fixed_histogram(
                                combined,
                                parent_sizes,
                                native_indices,
                                candidate_mask=candidate_mask,
                            )
                            actual = parent_sizes[
                                selected.long()
                            ].sum(dim=-1)
                            selected_parent = selected
                        else:
                            selected_groups, actual = (
                                select_exec128_under_budget(
                                    combined,
                                    group_sizes,
                                    native_actual,
                                )
                            )
                            selected_parent = _groups_to_parent_indices(
                                selected_groups
                            ).to(torch.int32)
                        selection_end.record()
                        selection_end.synchronize()
                        selection_ms = float(
                            selection_start.elapsed_time(selection_end)
                        )
                        variant = (
                            f"s{score_width}_e{execution_width}_"
                            f"{aggregation}_{parent_pool}_"
                            f"l{soft_prior:.2f}"
                        )
                        evaluate(
                            variant=variant,
                            selected_indices=selected_parent,
                            nominal_width=execution_width,
                            nominal_blocks=nominal_blocks,
                            actual_tokens=actual,
                            metadata={
                                "candidate_kind": "hierarchical",
                                "score_width": score_width,
                                "execution_width": execution_width,
                                "aggregation": aggregation,
                                "soft_native_prior": soft_prior,
                                "parent_pool": parent_pool,
                                "fine_score_ms": score_ms,
                                "aggregation_ms": aggregation_ms,
                                "selection_ms": selection_ms,
                            },
                        )

    benchmark_rows.append(
        {
            "event_type": "hierarchical_scoring_benchmark",
            "native_score_ms": native_score_ms,
            "score8_ms": child_data[8][2],
            "score16_ms": child_data[16][2],
            "native_execution_ms": next(
                row["execution_ms"]
                for row in error_rows
                if row["variant"] == "native64_global"
            ),
            "fine8_execution_ms": fine_execution_ms,
        }
    )
    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "padded_tokens": sequence,
        "valid_tokens": int(non_pad_index.numel()),
        "native_selected_parent_blocks": NATIVE_PARENT_K,
        "native_nominal_kv_tokens": NOMINAL_KV_TOKENS,
        "exec128_selected_groups": 62,
        "exec128_nominal_kv_tokens": 62 * 128,
        "candidate_count": 72,
        "aggregation_derivation": (
            "Fine child scores are logits. Hierarchical mass uses "
            "logsumexp(child_logit + log(valid_child_tokens)); plain "
            "logsumexp, top-2 mean, and max are diagnostics."
        ),
    }
    return HierarchicalReplayResult(
        error_rows=error_rows,
        benchmark_rows=benchmark_rows,
        geometry=geometry,
    )
