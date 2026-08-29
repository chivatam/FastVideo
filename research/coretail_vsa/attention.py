from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeVar
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from research.coretail_vsa.selection import (
    CoreMaskTable,
    CoreTailSelection,
    select_coretail_support,
)
from research.fine_vsa.fine_attention import (
    child_block_mean,
    fine_sparse_attention,
)
from research.fine_vsa.replay import (
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
)

CHILD_WIDTH = 8
CORE_PARENT_BLOCKS = 31
NOMINAL_SPARSITY = 1.0 - (NOMINAL_KV_TOKENS / (624 * PARENT_WIDTH))
T = TypeVar("T")

_MERGE_VALIDATION: dict[str, Any] | None = None
_STATIC_GEOMETRY_VALIDATED = False


@triton.jit
def _online_softmax_merge_kernel(
    output_a_ptr,
    lse_a_ptr,
    output_b_ptr,
    lse_b_ptr,
    output_ptr,
    lse_ptr,
    rows: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, head_dim)
    valid_rows = row_offsets < rows
    lse_a = tl.load(
        lse_a_ptr + row_offsets,
        mask=valid_rows,
        other=-float("inf"),
    ).to(tl.float32)
    lse_b = tl.load(
        lse_b_ptr + row_offsets,
        mask=valid_rows,
        other=-float("inf"),
    ).to(tl.float32)
    merged_max = tl.maximum(lse_a, lse_b)
    weight_a = tl.exp2(lse_a - merged_max)
    weight_b = tl.exp2(lse_b - merged_max)
    denominator = weight_a + weight_b
    offsets = row_offsets[:, None] * head_dim + dim_offsets[None, :]
    valid = valid_rows[:, None]
    output_a = tl.load(
        output_a_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    output_b = tl.load(
        output_b_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    merged = (output_a * weight_a[:, None] + output_b * weight_b[:, None]) / denominator[:, None]
    tl.store(output_ptr + offsets, merged, mask=valid)
    tl.store(
        lse_ptr + row_offsets,
        merged_max + tl.log2(denominator),
        mask=valid_rows,
    )


def online_softmax_merge(
    output_a: torch.Tensor,
    lse_a: torch.Tensor,
    output_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly merge disjoint partial softmaxes whose LSE uses log base two."""
    if output_a.shape != output_b.shape:
        raise ValueError("Partial attention outputs must have equal shapes")
    if output_a.ndim != 4:
        raise ValueError("Partial attention outputs must use [B,H,S,D]")
    if lse_a.shape != output_a.shape[:-1] or lse_b.shape != lse_a.shape:
        raise ValueError("Partial LSE tensors must use [B,H,S]")
    if not output_a.is_cuda:
        return online_softmax_merge_reference(
            output_a,
            lse_a,
            output_b,
            lse_b,
        )
    output_a = output_a.contiguous()
    output_b = output_b.contiguous()
    lse_a = lse_a.contiguous()
    lse_b = lse_b.contiguous()
    output = torch.empty_like(output_a)
    lse = torch.empty_like(lse_a, dtype=torch.float32)
    rows = lse.numel()
    block_m = 64
    _online_softmax_merge_kernel[(triton.cdiv(rows, block_m), )](
        output_a,
        lse_a,
        output_b,
        lse_b,
        output,
        lse,
        rows=rows,
        head_dim=output.shape[-1],
        BLOCK_M=block_m,
        num_warps=8,
        num_stages=2,
    )
    return output, lse


def online_softmax_merge_reference(
    output_a: torch.Tensor,
    lse_a: torch.Tensor,
    output_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_max = torch.maximum(lse_a.float(), lse_b.float())
    weight_a = torch.exp2(lse_a.float() - merged_max)
    weight_b = torch.exp2(lse_b.float() - merged_max)
    denominator = weight_a + weight_b
    output = (output_a.float() * weight_a[..., None] + output_b.float() * weight_b[..., None]) / denominator[..., None]
    merged_lse = merged_max + torch.log2(denominator)
    return output.to(output_a.dtype), merged_lse


@dataclass(frozen=True)
class CoreTailSystemsDecision:
    component_events: dict[
        str,
        tuple[torch.cuda.Event, torch.cuda.Event],
    ]
    metrics: dict[str, torch.Tensor]
    static_metadata_precomputed: bool
    static_selection_online_topk: bool
    static_parent_blocks: int


def _evented(
    name: str,
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
    fn: Callable[[], T],
) -> T:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    events[name] = (start, end)
    return result


def _selector(
    query_bhsd: torch.Tensor,
    key_bhsd: torch.Tensor,
    value_bhsd: torch.Tensor,
    parent_sizes: torch.Tensor,
    *,
    timestep: int,
    layer: int,
    core_masks: CoreMaskTable,
) -> tuple[torch.Tensor, CoreTailSelection, torch.Tensor]:
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    _, _, _, head_dim = query_bhsd.shape
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
    coarse_output = coarse_output.view(
        *coarse_output.shape[:-2],
        coarse_output.shape[-2],
        1,
        head_dim,
    ).expand(
        -1,
        -1,
        -1,
        PARENT_WIDTH,
        -1,
    ).reshape_as(query_bhsd)
    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    core_indices = core_masks.indices(
        timestep=timestep,
        layer=layer,
        core_parent_blocks=CORE_PARENT_BLOCKS,
        device=query_bhsd.device,
    )
    selection = select_coretail_support(
        child_scores,
        child_sizes,
        parent_scores,
        parent_sizes,
        core_indices,
    )
    return coarse_output, selection, child_sizes


def _validate_merge_once(
    query_bhsd: torch.Tensor,
    key_bhsd: torch.Tensor,
    value_bhsd: torch.Tensor,
    child_sizes: torch.Tensor,
    selection: CoreTailSelection,
    merged_output: torch.Tensor,
    merged_lse: torch.Tensor,
) -> None:
    global _MERGE_VALIDATION
    if _MERGE_VALIDATION is not None:
        return
    union = torch.cat(
        [
            selection.core_child_indices,
            selection.fine_tail_indices,
        ],
        dim=-1,
    )
    union_counts = (selection.core_child_indices.shape[-1] + selection.tail_active_descriptors)
    reference_output, reference_lse = fine_sparse_attention(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        union,
        child_sizes,
        child_width=CHILD_WIDTH,
        selected_counts=union_counts,
    )
    difference = merged_output.float() - reference_output.float()
    relative_l2 = (difference.norm() / reference_output.float().norm().clamp_min(1e-12))
    lse_abs_max = (merged_lse.float() - reference_lse.float()).abs().max()
    finite = (torch.isfinite(merged_output).all() & torch.isfinite(merged_lse).all())
    torch.cuda.synchronize()
    _MERGE_VALIDATION = {
        "event_type": "coretail_merge_validation",
        "reference": "single Fine8 kernel over static-union-dynamic support",
        "relative_L2": float(relative_l2.item()),
        "lse_abs_max_log2": float(lse_abs_max.item()),
        "finite": bool(finite.item()),
        "tolerance_relative_L2": 0.01,
        "passes": bool(finite.item() and relative_l2.item() <= 0.01 and lse_abs_max.item() <= 0.05),
    }
    if not _MERGE_VALIDATION["passes"]:
        raise RuntimeError("CoreTail online-softmax merge failed union validation: "
                           f"{_MERGE_VALIDATION!r}")


def get_merge_validation() -> dict[str, Any] | None:
    return None if _MERGE_VALIDATION is None else dict(_MERGE_VALIDATION)


def coretail_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    parent_sizes: torch.Tensor,
    *,
    timestep: int,
    layer: int,
    core_masks: CoreMaskTable,
) -> tuple[torch.Tensor, CoreTailSystemsDecision]:
    """Core25 static KV64 plus disjoint dynamic Fine8 with exact LSE merge."""
    global _STATIC_GEOMETRY_VALIDATED
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    def prepare() -> tuple[torch.Tensor, ...]:
        return (
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            gate_compress.transpose(1, 2).contiguous(),
        )

    query_bhsd, key_bhsd, value_bhsd, gate_bhsd = _evented(
        "metadata",
        events,
        prepare,
    )
    batch, heads, sequence, _ = query_bhsd.shape
    if batch != 1 or sequence != parent_sizes.numel() * PARENT_WIDTH:
        raise ValueError("CoreTail-VSA requires native Wan 64-token tiled geometry")
    coarse_output, selection, child_sizes = _evented(
        "selector",
        events,
        lambda: _selector(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            parent_sizes,
            timestep=timestep,
            layer=layer,
            core_masks=core_masks,
        ),
    )
    if not _STATIC_GEOMETRY_VALIDATED:
        if not bool(selection.core_active_parent_blocks.eq(CORE_PARENT_BLOCKS).all().item()):
            raise RuntimeError("Core25 does not fit the fixed native valid-token lower bound")
        _STATIC_GEOMETRY_VALIDATED = True

    static_indices = selection.core_parent_indices.contiguous()
    static_counts = torch.full(
        static_indices.shape[:-1],
        CORE_PARENT_BLOCKS,
        device=static_indices.device,
        dtype=torch.int32,
    )

    def static_attention() -> tuple[torch.Tensor, torch.Tensor]:
        from fastvideo_kernel.block_sparse_attn_sm100a import (
            block_sparse_attn_sm100a,
            is_supported,
        )

        if not is_supported(query_bhsd, parent_sizes):
            raise RuntimeError("CoreTail static KV64 requires the B200 sm100a kernel")
        output, lse = block_sparse_attn_sm100a(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            static_indices,
            static_counts,
            parent_sizes,
            need_lse=True,
        )
        if lse is None:
            raise RuntimeError("Static KV64 kernel did not return LSE")
        return output, lse

    static_output, static_lse = _evented(
        "static_kernel",
        events,
        static_attention,
    )
    dynamic_output, dynamic_lse = _evented(
        "dynamic_kernel",
        events,
        lambda: fine_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            selection.fine_tail_indices,
            child_sizes,
            child_width=CHILD_WIDTH,
            selected_counts=selection.tail_active_descriptors,
        ),
    )
    merged_output, merged_lse = _evented(
        "merge",
        events,
        lambda: online_softmax_merge(
            static_output,
            static_lse,
            dynamic_output,
            dynamic_lse,
        ),
    )
    _validate_merge_once(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        child_sizes,
        selection,
        merged_output,
        merged_lse,
    )
    output = _evented(
        "residual",
        events,
        lambda: (merged_output + coarse_output * gate_bhsd).transpose(1, 2),
    )
    return output, CoreTailSystemsDecision(
        component_events=events,
        metrics={
            "static_active_parent_blocks_min": (selection.core_active_parent_blocks.min()),
            "static_active_parent_blocks_max": (selection.core_active_parent_blocks.max()),
            "static_valid_tokens_mean": (selection.core_actual_kv_tokens.float().mean()),
            "dynamic_active_descriptors_mean": (selection.tail_active_descriptors.float().mean()),
            "dynamic_active_descriptors_min": (selection.tail_active_descriptors.min()),
            "dynamic_active_descriptors_max": (selection.tail_active_descriptors.max()),
            "dynamic_valid_tokens_mean": (selection.fine_tail_actual_kv_tokens.float().mean()),
            "total_valid_tokens_mean": (selection.selected_actual_kv_tokens.float().mean()),
            "native_valid_tokens_mean": (selection.native_actual_kv_tokens.float().mean()),
            "actual_kv_token_error_abs_max":
            (selection.selected_actual_kv_tokens - selection.native_actual_kv_tokens).abs().max(),
            "duplicate_valid_tokens_max": (selection.duplicate_valid_tokens.max()),
        },
        static_metadata_precomputed=True,
        static_selection_online_topk=False,
        static_parent_blocks=CORE_PARENT_BLOCKS,
    )


def summarize_systems_decision(decision: CoreTailSystemsDecision, ) -> dict[str, Any]:
    metrics = decision.metrics
    timings = {f"{name}_ms": start.elapsed_time(end) for name, (start, end) in decision.component_events.items()}
    internal_total = sum(timings.values())
    return {
        "static_parent_blocks": decision.static_parent_blocks,
        "static_nominal_kv_tokens": (decision.static_parent_blocks * PARENT_WIDTH),
        "dynamic_nominal_kv_tokens": (NOMINAL_KV_TOKENS - decision.static_parent_blocks * PARENT_WIDTH),
        "nominal_pair_budget_ratio": 1.0,
        "nominal_effective_sparsity": NOMINAL_SPARSITY,
        "static_metadata_precomputed": (decision.static_metadata_precomputed),
        "static_selection_online_topk": (decision.static_selection_online_topk),
        "static_active_parent_blocks_min": int(metrics["static_active_parent_blocks_min"].item()),
        "static_active_parent_blocks_max": int(metrics["static_active_parent_blocks_max"].item()),
        "static_valid_tokens_mean": float(metrics["static_valid_tokens_mean"].item()),
        "dynamic_active_descriptors_mean": float(metrics["dynamic_active_descriptors_mean"].item()),
        "dynamic_active_descriptors_min": int(metrics["dynamic_active_descriptors_min"].item()),
        "dynamic_active_descriptors_max": int(metrics["dynamic_active_descriptors_max"].item()),
        "dynamic_valid_tokens_mean": float(metrics["dynamic_valid_tokens_mean"].item()),
        "total_valid_tokens_mean": float(metrics["total_valid_tokens_mean"].item()),
        "native_valid_tokens_mean": float(metrics["native_valid_tokens_mean"].item()),
        "actual_kv_token_error_abs_max": int(metrics["actual_kv_token_error_abs_max"].item()),
        "duplicate_valid_tokens_max": int(metrics["duplicate_valid_tokens_max"].item()),
        "internal_component_total_ms": internal_total,
        **timings,
    }
