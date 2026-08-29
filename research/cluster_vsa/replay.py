from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from research.cluster_vsa.clustering import (
    balanced_recursive_order,
    build_slot_permutation,
    cluster_labels_from_order,
    permute_bhsd,
    slot_valid_mask,
)
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

CLUSTER_VARIANTS = (
    "k_head_pca64",
    "k_shared_pca64",
)


@dataclass(frozen=True)
class ClusterReplayResult:
    error_rows: list[dict[str, Any]]
    analysis_rows: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    assignment_rows: list[dict[str, Any]]
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


def _block_mean(
    tensor: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    batch, heads, sequence, head_dim = tensor.shape
    blocks = int(block_sizes.numel())
    if sequence != blocks * PARENT_WIDTH:
        raise ValueError("Tensor does not follow KV64 slot geometry")
    values = tensor.view(
        batch,
        heads,
        blocks,
        PARENT_WIDTH,
        head_dim,
    )
    valid = (
        torch.arange(
            PARENT_WIDTH,
            device=tensor.device,
        )[None, None, None, :, None]
        < block_sizes[None, None, :, None, None]
    )
    return (values * valid).sum(dim=-2) / block_sizes.clamp_min(1).view(
        1,
        1,
        blocks,
        1,
    )


def select_cluster_fixed_histogram(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    native_sizes: torch.Tensor,
    native_indices: torch.Tensor,
) -> torch.Tensor:
    """Rank clustered slots while matching native selected size counts."""
    flat_scores = scores.flatten(0, -2)
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
    selected_native_sizes = native_sizes[flat_native]
    for size in sorted(
        (int(value) for value in torch.unique(native_sizes).tolist()),
        reverse=True,
    ):
        count = selected_native_sizes.eq(size).sum(dim=-1)
        maximum = int(count.max().item())
        if maximum == 0:
            continue
        eligible = cluster_sizes.view(1, -1).eq(size)
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
            raise RuntimeError("Cluster capacity multiset cannot reproduce native support")
        destination = write_offset[:, None] + positions
        selected[
            row_grid.expand_as(take)[take],
            destination[take],
        ] = ranked.indices[take]
        write_offset += count
    if not write_offset.eq(NATIVE_PARENT_K).all():
        raise RuntimeError("Cluster histogram selector did not fill K=125")
    return selected.view(*scores.shape[:-1], NATIVE_PARENT_K).to(torch.int32)


def _coherence_rows(
    normalized_key: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    variant: str,
) -> list[dict[str, Any]]:
    centroids = _block_mean(normalized_key, block_sizes).float()
    coherence = centroids.norm(dim=-1).clamp(0, 1)
    valid_blocks = block_sizes.gt(0)
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=normalized_key.device,
    )
    rows: list[dict[str, Any]] = []
    for scope, head, values in [
        (
            "all_heads_blocks",
            None,
            coherence[..., valid_blocks].flatten(),
        ),
        *[
            (
                "head_blocks",
                head_index,
                coherence[:, head_index, valid_blocks].flatten(),
            )
            for head_index in range(coherence.shape[1])
        ],
    ]:
        summary = torch.quantile(values, quantiles)
        rows.append(
            {
                "event_type": "cluster_block_coherence",
                "variant": variant,
                "scope": scope,
                "head": head,
                "mean": float(values.mean().item()),
                "p10": float(summary[0].item()),
                "median": float(summary[1].item()),
                "p90": float(summary[2].item()),
                "blocks": int(values.numel()),
                "definition": (
                    "mean cosine of normalized K tokens to the normalized "
                    "cluster centroid; equal to norm(mean(normalized K))"
                ),
            }
        )
    return rows


def _rank_correlation(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_blocks: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted[..., valid_blocks].float()
    target = target[..., valid_blocks].float()
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


def _true_block_mass(
    token_scores: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    valid_slots = slot_valid_mask(block_sizes)
    probabilities = torch.softmax(
        token_scores.float().masked_fill(
            ~valid_slots.view(1, 1, 1, -1),
            -float("inf"),
        ),
        dim=-1,
    )
    return probabilities.view(
        *probabilities.shape[:-1],
        block_sizes.numel(),
        PARENT_WIDTH,
    ).sum(dim=-1)


def _alignment_rows(
    predicted_scores: torch.Tensor,
    token_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    variant: str,
) -> list[dict[str, Any]]:
    true_mass = _true_block_mass(token_scores, block_sizes)
    valid_blocks = block_sizes.gt(0)
    spearman = _rank_correlation(
        predicted_scores,
        true_mass,
        valid_blocks,
    )
    selected_mass = torch.gather(
        true_mass,
        -1,
        selected_indices.long(),
    ).sum(dim=-1)
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=predicted_scores.device,
    )
    rows: list[dict[str, Any]] = []
    for scope, head, correlation, recall in [
        (
            "all_heads_query_blocks",
            None,
            spearman.flatten(),
            selected_mass.flatten(),
        ),
        *[
            (
                "head_query_blocks",
                head_index,
                spearman[:, head_index].flatten(),
                selected_mass[:, head_index].flatten(),
            )
            for head_index in range(spearman.shape[1])
        ],
    ]:
        correlation_q = torch.quantile(correlation, quantiles)
        recall_q = torch.quantile(recall, quantiles)
        rows.append(
            {
                "event_type": "cluster_coarse_true_mass_alignment",
                "variant": variant,
                "scope": scope,
                "head": head,
                "spearman_mean": float(correlation.mean().item()),
                "spearman_p10": float(correlation_q[0].item()),
                "spearman_median": float(correlation_q[1].item()),
                "spearman_p90": float(correlation_q[2].item()),
                "top125_mass_recall_mean": float(recall.mean().item()),
                "top125_mass_recall_p10": float(recall_q[0].item()),
                "top125_mass_recall_median": float(recall_q[1].item()),
                "top125_mass_recall_p90": float(recall_q[2].item()),
                "query_blocks": int(correlation.numel()),
                "true_mass_definition": (
                    "dense softmax mass from each native 64-query block mean to every valid original K token"
                ),
            }
        )
    return rows


def _internal_mass_rows(
    token_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    variant: str,
) -> list[dict[str, Any]]:
    block_scores = token_scores.view(
        *token_scores.shape[:-1],
        block_sizes.numel(),
        PARENT_WIDTH,
    )
    selected_scores = torch.gather(
        block_scores,
        -2,
        selected_indices[..., None].expand(
            *selected_indices.shape,
            PARENT_WIDTH,
        ),
    )
    selected_sizes = block_sizes[selected_indices.long()]
    valid = (
        torch.arange(
            PARENT_WIDTH,
            device=token_scores.device,
        )[None, None, None, None, :]
        < selected_sizes[..., None]
    )
    probabilities = torch.softmax(
        selected_scores.float().masked_fill(
            ~valid,
            -float("inf"),
        ),
        dim=-1,
    )
    top_values = torch.topk(probabilities, 32, dim=-1).values
    fractions = {
        8: top_values[..., :8].sum(dim=-1),
        16: top_values[..., :16].sum(dim=-1),
        32: top_values.sum(dim=-1),
    }
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=token_scores.device,
    )
    rows: list[dict[str, Any]] = []
    for scope, head, mask in [
        ("all_heads_selected_blocks", None, selected_sizes.gt(0)),
        *[
            (
                "head_selected_blocks",
                head_index,
                selected_sizes[:, head_index].gt(0),
            )
            for head_index in range(selected_sizes.shape[1])
        ],
    ]:
        row: dict[str, Any] = {
            "event_type": "cluster_internal_mass",
            "variant": variant,
            "scope": scope,
            "head": head,
            "selected_blocks": NATIVE_PARENT_K,
            "mass_proxy": ("softmax over exact token logits inside each selected block"),
        }
        for top_tokens, values in fractions.items():
            selected_values = values[mask] if head is None else values[:, head][mask]
            summary = torch.quantile(selected_values, quantiles)
            row[f"top{top_tokens}_mean"] = float(selected_values.mean().item())
            row[f"top{top_tokens}_p10"] = float(summary[0].item())
            row[f"top{top_tokens}_median"] = float(summary[1].item())
            row[f"top{top_tokens}_p90"] = float(summary[2].item())
        row["selected_block_samples"] = int(mask.sum().item())
        rows.append(row)
    return rows


def _grouping_trace_rows(
    valid_order: torch.Tensor,
    block_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    variant: str,
    shared_across_heads: bool,
) -> list[dict[str, Any]]:
    labels = cluster_labels_from_order(valid_order, block_sizes)
    sample_count = min(512, valid_order.shape[-1])
    sample_positions = (
        torch.linspace(
            0,
            valid_order.shape[-1] - 1,
            sample_count,
            device=valid_order.device,
        )
        .round()
        .long()
    )
    original_order = non_pad_index[valid_order]
    heads = [0] if shared_across_heads else list(range(valid_order.shape[1]))
    rows = []
    for head in heads:
        order = original_order[0, head]
        jump = order[1:].sub(order[:-1]).abs().float()
        sampled = labels[0, head, sample_positions]
        rows.append(
            {
                "event_type": "cluster_grouping_trace",
                "variant": variant,
                "head": -1 if shared_across_heads else head,
                "shared_across_heads": shared_across_heads,
                "sample_token_positions": sample_positions.cpu().tolist(),
                "sample_cluster_labels": sampled.cpu().tolist(),
                "mean_adjacent_original_index_jump": float(jump.mean().item()),
                "median_adjacent_original_index_jump": float(jump.median().item()),
                "permutation_checksum": int(
                    (
                        order.long()
                        * torch.arange(
                            1,
                            order.numel() + 1,
                            device=order.device,
                            dtype=torch.long,
                        )
                    )
                    .sum()
                    .item()
                ),
            }
        )
    return rows


def _assignment_rows(
    valid_order: torch.Tensor,
    block_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    variant: str,
    shared_across_heads: bool,
) -> list[dict[str, Any]]:
    original_order = non_pad_index[valid_order]
    heads = [0] if shared_across_heads else list(range(valid_order.shape[1]))
    sizes = block_sizes.cpu().tolist()
    return [
        {
            "event_type": "cluster_assignment",
            "variant": variant,
            "head": -1 if shared_across_heads else head,
            "shared_across_heads": shared_across_heads,
            "valid_token_permutation": original_order[
                0,
                head,
            ]
            .cpu()
            .tolist(),
            "cluster_sizes": sizes,
        }
        for head in heads
    ]


def replay_cluster_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    capture_assignments: bool = False,
) -> ClusterReplayResult:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    valid_tokens = int(non_pad_index.numel())
    if batch != 1:
        raise ValueError("Cluster-VSA census currently requires batch=1")
    if sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError("Cluster-VSA requires native padded KV64 geometry")
    if int(parent_sizes.sum().item()) != valid_tokens:
        raise ValueError("Parent sizes and valid-token index disagree")

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
    native_token_scores = torch.matmul(
        query_coarse,
        key_bhsd.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    normalized_key = F.normalize(key_bhsd.float(), dim=-1).to(key_bhsd.dtype)

    error_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    def evaluate(
        *,
        variant: str,
        selected_indices: torch.Tensor,
        exact_key: torch.Tensor,
        exact_value: torch.Tensor,
        key_block_sizes: torch.Tensor,
        token_scores: torch.Tensor,
        predicted_scores: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        exact_output, execution_ms = _elapsed_ms(
            lambda: fine_sparse_attention(
                query_bhsd,
                exact_key,
                exact_value,
                selected_indices,
                key_block_sizes,
                child_width=PARENT_WIDTH,
            )[0]
        )
        sparse_output = exact_output + coarse_output * gate_bhsd
        relative_l2, cosine_error = _query_block_metrics(
            sparse_output,
            dense_output,
            parent_sizes,
        )
        actual = key_block_sizes[selected_indices.long()].sum(dim=-1)
        rows = _summary_rows(
            relative_l2,
            cosine_error,
            variant=variant,
            child_width=PARENT_WIDTH,
            selected_blocks=NATIVE_PARENT_K,
            parent_pool=None,
            actual_kv_tokens=actual,
            native_actual_kv_tokens=native_actual,
            parent_sizes=parent_sizes,
        )
        for row in rows:
            row["event_type"] = "cluster_vsa_error"
            row["execution_ms"] = execution_ms
            row.update(metadata)
        error_rows.extend(rows)
        analysis_rows.extend(
            _alignment_rows(
                predicted_scores,
                token_scores,
                selected_indices,
                key_block_sizes,
                variant=variant,
            )
        )
        analysis_rows.extend(
            _internal_mass_rows(
                token_scores,
                selected_indices,
                key_block_sizes,
                variant=variant,
            )
        )

    evaluate(
        variant="native64_spatial",
        selected_indices=native_indices.to(torch.int32),
        exact_key=key_bhsd,
        exact_value=value_bhsd,
        key_block_sizes=parent_sizes,
        token_scores=native_token_scores,
        predicted_scores=parent_scores,
        metadata={
            "candidate_kind": "baseline",
            "grouping": "native_spatial",
            "shared_across_heads": False,
            "fixed_slot_width": 64,
        },
    )
    analysis_rows.extend(
        _coherence_rows(
            normalized_key,
            parent_sizes,
            variant="native64_spatial",
        )
    )

    fine_key, fine_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        8,
    )
    fine_scores = torch.matmul(
        query_coarse,
        fine_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)
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
        variant="fine8_spatial",
        child_width=8,
        selected_blocks=1000,
        parent_pool=None,
        actual_kv_tokens=fine_actual,
        native_actual_kv_tokens=native_actual,
        parent_sizes=parent_sizes,
    )
    for row in rows:
        row["event_type"] = "cluster_vsa_error"
        row.update(
            {
                "candidate_kind": "baseline",
                "grouping": "spatial_fine8",
                "shared_across_heads": False,
                "fixed_slot_width": 8,
                "execution_ms": fine_execution_ms,
            }
        )
    error_rows.extend(rows)

    key_valid = key_bhsd.index_select(2, non_pad_index)
    shared_source = (
        F.normalize(
            key_valid.float(),
            dim=-1,
        )
        .mean(dim=1, keepdim=True)
        .to(key_valid.dtype)
    )
    grouping_inputs = {
        "k_head_pca64": (key_valid, False),
        "k_shared_pca64": (shared_source, True),
    }
    cluster_sizes = torch.sort(
        parent_sizes,
        descending=True,
    ).values
    for variant, (grouping_input, shared) in grouping_inputs.items():
        grouping, grouping_ms = _elapsed_ms(
            lambda grouping_input=grouping_input: balanced_recursive_order(
                grouping_input,
                leaf_size=PARENT_WIDTH,
                principal_iterations=2,
            )
        )
        valid_order = grouping.valid_order
        if shared:
            valid_order = valid_order.expand(-1, heads, -1)
        permutation, permutation_build_ms = _elapsed_ms(
            lambda valid_order=valid_order: build_slot_permutation(
                valid_order,
                non_pad_index,
                cluster_sizes,
                block_width=PARENT_WIDTH,
            )
        )
        (
            (
                permuted_key,
                permuted_value,
                permuted_normalized_key,
                permuted_token_scores,
            ),
            permutation_ms,
        ) = _elapsed_ms(
            lambda permutation=permutation: (
                permute_bhsd(key_bhsd, permutation),
                permute_bhsd(value_bhsd, permutation),
                permute_bhsd(normalized_key, permutation),
                torch.gather(
                    native_token_scores,
                    -1,
                    permutation[:, :, None, :].expand(
                        -1,
                        -1,
                        parent_blocks,
                        -1,
                    ),
                ),
            )
        )
        cluster_key, centroid_ms = _elapsed_ms(
            lambda normalized=permuted_normalized_key: _block_mean(
                normalized,
                cluster_sizes,
            )
        )
        cluster_scores, scoring_ms = _elapsed_ms(
            lambda centroid=cluster_key: (
                torch.matmul(
                    query_coarse,
                    centroid.transpose(-2, -1),
                )
                / math.sqrt(head_dim)
            )
        )
        selected, selection_ms = _elapsed_ms(
            lambda scores=cluster_scores: select_cluster_fixed_histogram(
                scores,
                cluster_sizes,
                parent_sizes,
                native_indices,
            )
        )
        metadata = {
            "candidate_kind": "cluster",
            "grouping": ("balanced_recursive_approx_pca_median"),
            "shared_across_heads": shared,
            "fixed_slot_width": 64,
            "grouping_ms": grouping_ms,
            "permutation_build_ms": permutation_build_ms,
            "permutation_ms": permutation_ms,
            "centroid_ms": centroid_ms,
            "scoring_ms": scoring_ms,
            "selection_ms": selection_ms,
            "split_depth": grouping.split_depth,
            "principal_iterations": 2,
            "power_of_two_working_tokens": grouping.padded_tokens,
        }
        evaluate(
            variant=variant,
            selected_indices=selected,
            exact_key=permuted_key,
            exact_value=permuted_value,
            key_block_sizes=cluster_sizes,
            token_scores=permuted_token_scores,
            predicted_scores=cluster_scores,
            metadata=metadata,
        )
        analysis_rows.extend(
            _coherence_rows(
                permuted_normalized_key,
                cluster_sizes,
                variant=variant,
            )
        )
        trace_rows = _grouping_trace_rows(
            valid_order,
            cluster_sizes,
            non_pad_index,
            variant=variant,
            shared_across_heads=shared,
        )
        for row in trace_rows:
            row.update(metadata)
        analysis_rows.extend(trace_rows)
        if capture_assignments:
            assignment_rows.extend(
                _assignment_rows(
                    valid_order,
                    cluster_sizes,
                    non_pad_index,
                    variant=variant,
                    shared_across_heads=shared,
                )
            )
        benchmark_rows.append(
            {
                "event_type": "cluster_vsa_benchmark",
                "variant": variant,
                **metadata,
                "execution_ms": next(row["execution_ms"] for row in error_rows if row["variant"] == variant),
            }
        )
        del (
            grouping,
            valid_order,
            permutation,
            permuted_key,
            permuted_value,
            permuted_normalized_key,
            permuted_token_scores,
            cluster_key,
            cluster_scores,
            selected,
        )

    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "padded_tokens": sequence,
        "valid_tokens": valid_tokens,
        "native_selected_parent_blocks": NATIVE_PARENT_K,
        "native_nominal_kv_tokens": NOMINAL_KV_TOKENS,
        "cluster_slot_size_policy": (
            "same 624 fixed-width slots and same ragged-size multiset as "
            "native, sorted by capacity to preserve full recursive leaves; "
            "selected size histogram exactly matched per query row"
        ),
        "cluster_variants": len(CLUSTER_VARIANTS),
        "centroid_policy": (
            "mean of normalized K tokens for routing only; exact output uses "
            "original permuted K/V and unchanged native coarse residual"
        ),
    }
    return ClusterReplayResult(
        error_rows=error_rows,
        analysis_rows=analysis_rows,
        benchmark_rows=benchmark_rows,
        assignment_rows=assignment_rows,
        geometry=geometry,
    )
