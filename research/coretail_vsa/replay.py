from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeVar
from collections.abc import Callable

import torch

from research.anchored_fine_vsa.selection import (
    ANCHOR_PARENT_BLOCKS,
    CHILD_WIDTH,
    select_anchored_support,
    select_pure_fine_support,
)
from research.coretail_vsa.selection import (
    CORE_PARENT_COUNTS,
    CoreMaskTable,
    select_coretail_support,
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
)

EXPECTED_VARIANTS = (
    "dense_bf16",
    "native64",
    "fine8",
    "native_anchor25",
    "native_anchor50",
    "calib_core25_tail",
    "calib_core50_tail",
    "calib_core25_only",
    "calib_core50_only",
)
T = TypeVar("T")


@dataclass(frozen=True)
class CoreTailReplayResult:
    error_rows: list[dict[str, Any]]
    accounting_rows: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    geometry: dict[str, Any]


def _elapsed_ms(fn: Callable[[], T]) -> tuple[T, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _error_rows(
    relative_l2: torch.Tensor,
    cosine_error: torch.Tensor,
    *,
    variant: str,
    nominal_kv_tokens: int,
    actual_kv_tokens: torch.Tensor,
    native_actual_kv_tokens: torch.Tensor,
    parent_sizes: torch.Tensor,
    execution_ms: float,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    quantiles = torch.tensor(
        [0.5, 0.9, 0.99],
        device=relative_l2.device,
    )
    query_pair_weight = parent_sizes.float().view(1, 1, -1)
    for scope, head, l2_values, cos_values in [
        (
            "all_heads_query_blocks",
            None,
            relative_l2.flatten(),
            cosine_error.flatten(),
        ),
            *[(
                "head_query_blocks",
                index,
                relative_l2[:, index].flatten(),
                cosine_error[:, index].flatten(),
            ) for index in range(relative_l2.shape[1])],
    ]:
        l2_summary = torch.quantile(l2_values, quantiles)
        cosine_summary = torch.quantile(cos_values, quantiles)
        if head is None:
            actual_pairs = (actual_kv_tokens.float() * query_pair_weight).sum()
            native_pairs = (native_actual_kv_tokens.float() * query_pair_weight).sum()
            actual_values = actual_kv_tokens.flatten()
        else:
            actual_pairs = (actual_kv_tokens[:, head].float() * query_pair_weight[:, 0]).sum()
            native_pairs = (native_actual_kv_tokens[:, head].float() * query_pair_weight[:, 0]).sum()
            actual_values = actual_kv_tokens[:, head].flatten()
        rows.append({
            "event_type": "coretail_vsa_error",
            "scope": scope,
            "head": head,
            "variant": variant,
            "nominal_kv_tokens": nominal_kv_tokens,
            "nominal_pair_budget_ratio": (nominal_kv_tokens / NOMINAL_KV_TOKENS),
            "actual_kv_tokens_mean": float(actual_values.float().mean().item()),
            "actual_kv_tokens_min": int(actual_values.min().item()),
            "actual_kv_tokens_max": int(actual_values.max().item()),
            "actual_pair_budget_ratio": float((actual_pairs / native_pairs.clamp_min(1)).item()),
            "unused_pair_capacity": float((native_pairs - actual_pairs).item()),
            "relative_L2_mean": float(l2_values.mean().item()),
            "relative_L2_median": float(l2_summary[0].item()),
            "relative_L2_p90": float(l2_summary[1].item()),
            "relative_L2_p99": float(l2_summary[2].item()),
            "cosine_error_mean": float(cos_values.mean().item()),
            "cosine_error_median": float(cosine_summary[0].item()),
            "cosine_error_p90": float(cosine_summary[1].item()),
            "cosine_error_p99": float(cosine_summary[2].item()),
            "query_blocks": int(l2_values.numel()),
            "execution_ms": execution_ms,
            **metadata,
        })
    return rows


def replay_coretail_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    non_pad_index: torch.Tensor,
    *,
    timestep: int,
    layer: int,
    core_masks: CoreMaskTable,
) -> CoreTailReplayResult:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    if batch != 1 or sequence != parent_blocks * PARENT_WIDTH:
        raise ValueError("CoreTail replay requires native Wan tiled geometry")

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
    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)

    errors: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []

    zero = torch.zeros(
        (batch, heads, parent_blocks),
        device=query.device,
    )
    errors.extend(
        _error_rows(
            zero,
            zero,
            variant="dense_bf16",
            nominal_kv_tokens=parent_blocks * PARENT_WIDTH,
            actual_kv_tokens=torch.full(
                (batch, heads, parent_blocks),
                int(parent_sizes.sum().item()),
                device=query.device,
                dtype=parent_sizes.dtype,
            ),
            native_actual_kv_tokens=native_actual,
            parent_sizes=parent_sizes,
            execution_ms=0.0,
            metadata={
                "candidate_kind": "dense_reference",
                "static_parent_blocks": 0,
                "dynamic_child_width": 0,
            },
        ))

    def record(
        *,
        variant: str,
        selected: torch.Tensor,
        sizes: torch.Tensor,
        child_width: int,
        nominal_tokens: int,
        metadata: dict[str, Any],
    ) -> None:
        exact_output, execution_ms = _elapsed_ms(lambda: fine_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            selected,
            sizes,
            child_width=child_width,
        )[0])
        sparse_output = exact_output + coarse_output * gate_bhsd
        relative_l2, cosine = _query_block_metrics(
            sparse_output,
            dense_output,
            parent_sizes,
        )
        actual = sizes[selected.long()].sum(dim=-1)
        errors.extend(
            _error_rows(
                relative_l2,
                cosine,
                variant=variant,
                nominal_kv_tokens=nominal_tokens,
                actual_kv_tokens=actual,
                native_actual_kv_tokens=native_actual,
                parent_sizes=parent_sizes,
                execution_ms=execution_ms,
                metadata=metadata,
            ))
        benchmarks.append({
            "event_type": "coretail_vsa_benchmark",
            "variant": variant,
            "execution_ms": execution_ms,
            **metadata,
        })

    record(
        variant="native64",
        selected=native_indices.to(torch.int32),
        sizes=parent_sizes,
        child_width=PARENT_WIDTH,
        nominal_tokens=NOMINAL_KV_TOKENS,
        metadata={
            "candidate_kind": "baseline",
            "static_parent_blocks": 0,
            "dynamic_child_width": PARENT_WIDTH,
        },
    )
    fine_indices, _, _ = select_pure_fine_support(
        child_scores,
        child_sizes,
        parent_scores,
        parent_sizes,
    )
    record(
        variant="fine8",
        selected=fine_indices,
        sizes=child_sizes,
        child_width=CHILD_WIDTH,
        nominal_tokens=NOMINAL_KV_TOKENS,
        metadata={
            "candidate_kind": "baseline",
            "static_parent_blocks": 0,
            "dynamic_child_width": CHILD_WIDTH,
        },
    )

    for name, parent_count in ANCHOR_PARENT_BLOCKS.items():
        selection = select_anchored_support(
            child_scores,
            child_sizes,
            parent_scores,
            parent_sizes,
            anchor_parent_blocks=parent_count,
        )
        record(
            variant=f"native_{name}",
            selected=selection.selected_indices,
            sizes=child_sizes,
            child_width=CHILD_WIDTH,
            nominal_tokens=NOMINAL_KV_TOKENS,
            metadata={
                "candidate_kind": "native_anchor_ablation",
                "static_parent_blocks": parent_count,
                "dynamic_child_width": CHILD_WIDTH,
            },
        )

    for parent_count in CORE_PARENT_COUNTS:
        core_indices = core_masks.indices(
            timestep=timestep,
            layer=layer,
            core_parent_blocks=parent_count,
            device=query.device,
        )
        selection = select_coretail_support(
            child_scores,
            child_sizes,
            parent_scores,
            parent_sizes,
            core_indices,
        )
        label = "25" if parent_count == 31 else "50"
        accounting.append({
            "event_type":
            "coretail_pair_accounting",
            "variant":
            f"calib_core{label}_tail",
            "static_parent_blocks":
            parent_count,
            "static_active_parent_blocks_mean":
            float(selection.core_active_parent_blocks.float().mean().item()),
            "static_active_parent_blocks_min":
            int(selection.core_active_parent_blocks.min().item()),
            "static_active_parent_blocks_max":
            int(selection.core_active_parent_blocks.max().item()),
            "static_budget_projection_fraction":
            float(selection.core_active_parent_blocks.lt(parent_count).float().mean().item()),
            "static_valid_tokens_mean":
            float(selection.core_actual_kv_tokens.float().mean().item()),
            "dynamic_valid_tokens_mean":
            float(selection.fine_tail_actual_kv_tokens.float().mean().item()),
            "total_valid_tokens_mean":
            float(selection.selected_actual_kv_tokens.float().mean().item()),
            "native_valid_tokens_mean":
            float(native_actual.float().mean().item()),
            "tail_active_descriptors_mean":
            float(selection.tail_active_descriptors.float().mean().item()),
            "tail_active_descriptors_max":
            int(selection.tail_active_descriptors.max().item()),
            "duplicate_valid_tokens_max":
            int(selection.duplicate_valid_tokens.max().item()),
            "valid_token_error_abs_max":
            int((selection.selected_actual_kv_tokens - native_actual).abs().max().item()),
        })
        record(
            variant=f"calib_core{label}_tail",
            selected=selection.selected_indices,
            sizes=child_sizes,
            child_width=CHILD_WIDTH,
            nominal_tokens=NOMINAL_KV_TOKENS,
            metadata={
                "candidate_kind":
                "calibrated_coretail",
                "static_parent_blocks":
                parent_count,
                "static_active_parent_blocks_mean":
                float(selection.core_active_parent_blocks.float().mean().item()),
                "static_budget_projection_fraction":
                float(selection.core_active_parent_blocks.lt(parent_count).float().mean().item()),
                "dynamic_child_width":
                CHILD_WIDTH,
                "tail_active_descriptors_mean":
                float(selection.tail_active_descriptors.float().mean().item()),
            },
        )
        record(
            variant=f"calib_core{label}_only",
            selected=selection.core_child_indices,
            sizes=child_sizes,
            child_width=CHILD_WIDTH,
            nominal_tokens=parent_count * PARENT_WIDTH,
            metadata={
                "candidate_kind": "static_only_ablation",
                "static_parent_blocks": parent_count,
                "dynamic_child_width": 0,
            },
        )

    geometry = {
        "parent_blocks": parent_blocks,
        "parent_width": PARENT_WIDTH,
        "padded_tokens": sequence,
        "valid_tokens": int(non_pad_index.numel()),
        "native_selected_parent_blocks": NATIVE_PARENT_K,
        "native_nominal_kv_tokens": NOMINAL_KV_TOKENS,
        "core_mask_calibration_prompt_hash": (core_masks.calibration_prompt_hash),
        "core_mask_quantile_semantics": core_masks.quantile_semantics,
    }
    return CoreTailReplayResult(
        error_rows=errors,
        accounting_rows=accounting,
        benchmark_rows=benchmarks,
        geometry=geometry,
    )
