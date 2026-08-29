from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import triton
import triton.language as tl


CompressedSupportMode = Literal["rectified", "compressed_halo"]


@dataclass
class CompressedSupportDecision:
    mode: CompressedSupportMode
    num_blocks: int
    exact_k: int
    omitted_blocks: int
    block_elements: int
    num_query_rows: int
    selected_count_min: torch.Tensor
    selected_count_max: torch.Tensor
    retained_mass_mean: torch.Tensor | None = None
    retained_mass_min: torch.Tensor | None = None
    retained_mass_p10: torch.Tensor | None = None
    retained_mass_p50: torch.Tensor | None = None
    retained_mass_p90: torch.Tensor | None = None
    omitted_mass_mean: torch.Tensor | None = None
    gate_abs_mean: torch.Tensor | None = None
    gate_rms: torch.Tensor | None = None
    coarse_score_abs_max: torch.Tensor | None = None
    coarse_score_range_max: torch.Tensor | None = None
    coarse_score_nonfinite_count: torch.Tensor | None = None
    correction_abs_mean: torch.Tensor | None = None
    correction_rms: torch.Tensor | None = None
    correction_relative_l2: torch.Tensor | None = None
    halo_abs_mean: torch.Tensor | None = None
    halo_rms: torch.Tensor | None = None
    halo_weight_mean: torch.Tensor | None = None
    halo_weight_p50: torch.Tensor | None = None
    halo_weight_p90: torch.Tensor | None = None
    coarse_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    selector_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    fine_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    rectification_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    halo_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    merge_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None


def _sample_quantiles(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = values.float().flatten()
    if flat.numel() > 16384:
        stride = math.ceil(flat.numel() / 16384)
        flat = flat[::stride]
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.1, 0.5, 0.9], device=flat.device),
    )
    return quantiles[0], quantiles[1], quantiles[2]


def rank_normalized_topk_mask(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Run the native fused selector after a rank-preserving affine map.

    The fused selector uses fp32 bisection. Large row ranges can make the
    bisection threshold land just below the K-th bf16 value, selecting K+1
    entries. Mapping each row to [-1, 0] preserves ordering and exact ties
    while keeping the numerical search range bounded.
    """
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_topk_mask,
    )

    scores_float = scores.float()
    row_max = scores_float.amax(dim=-1, keepdim=True)
    row_min = scores_float.amin(dim=-1, keepdim=True)
    row_span = (row_max - row_min).clamp_min(torch.finfo(torch.float32).tiny)
    normalized_scores = (scores_float - row_max) / row_span
    return fused_topk_mask(normalized_scores, topk)


def rectified_output(
    exact_output: torch.Tensor,
    coarse_attention: torch.Tensor,
    value_coarse: torch.Tensor,
    native_mask: torch.Tensor,
    gate: torch.Tensor,
    *,
    block_elements: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply parameter-free coarse-mass rectification.

    Inputs use BHSD/BHQK layouts. The omitted coarse contribution is left
    unnormalized, so its magnitude already contains the omitted probability
    mass. The selected coarse contribution is excluded.
    """
    batch, heads, sequence, head_dim = exact_output.shape
    num_query_blocks = coarse_attention.shape[-2]
    retained_mass = coarse_attention.masked_fill(
        ~native_mask,
        0.0,
    ).sum(dim=-1)
    omitted_attention = coarse_attention.masked_fill(native_mask, 0.0)
    omitted_output = torch.matmul(omitted_attention, value_coarse)
    omitted_output = (
        omitted_output.view(
            batch,
            heads,
            num_query_blocks,
            1,
            head_dim,
        )
        .expand(-1, -1, -1, block_elements, -1)
        .reshape(batch, heads, sequence, head_dim)
    )
    retained_fine = (
        retained_mass.view(batch, heads, num_query_blocks, 1, 1)
        .expand(-1, -1, -1, block_elements, 1)
        .reshape(batch, heads, sequence, 1)
    )
    output = retained_fine.to(exact_output.dtype) * exact_output
    output = output + omitted_output * gate
    return output, retained_mass, omitted_output


def merge_online_outputs(
    exact_output: torch.Tensor,
    exact_lse_log2: torch.Tensor,
    halo_output: torch.Tensor,
    halo_lse_log2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two independently normalized outputs under one softmax."""
    maximum = torch.maximum(exact_lse_log2, halo_lse_log2)
    exact_weight = torch.exp2(exact_lse_log2 - maximum)
    halo_weight = torch.exp2(halo_lse_log2 - maximum)
    denominator = exact_weight + halo_weight
    output = (
        exact_output.float() * exact_weight.unsqueeze(-1) + halo_output.float() * halo_weight.unsqueeze(-1)
    ) / denominator.unsqueeze(-1)
    halo_fraction = halo_weight / denominator
    return output.to(exact_output.dtype), halo_fraction


@triton.jit
def _merge_core_halo_coarse_kernel(
    exact_ptr,
    exact_lse_ptr,
    halo_ptr,
    halo_lse_ptr,
    coarse_ptr,
    gate_ptr,
    output_ptr,
    halo_fraction_ptr,
    stride_eb,
    stride_eh,
    stride_es,
    stride_ed,
    stride_elb,
    stride_elh,
    stride_els,
    stride_hb,
    stride_hh,
    stride_hs,
    stride_hd,
    stride_hlb,
    stride_hlh,
    stride_hls,
    stride_cb,
    stride_ch,
    stride_cs,
    stride_cd,
    stride_gb,
    stride_gh,
    stride_gs,
    stride_gd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_fb,
    stride_fh,
    stride_fs,
    num_heads: tl.constexpr,
    sequence: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    store_fraction: tl.constexpr,
):
    sequence_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    offsets_m = sequence_block * block_m + tl.arange(0, block_m)
    offsets_d = tl.arange(0, head_dim)
    valid_m = offsets_m < sequence

    exact_lse = tl.load(
        exact_lse_ptr
        + batch * stride_elb
        + head * stride_elh
        + offsets_m * stride_els,
        mask=valid_m,
        other=-float("inf"),
    )
    halo_lse = tl.load(
        halo_lse_ptr
        + batch * stride_hlb
        + head * stride_hlh
        + offsets_m * stride_hls,
        mask=valid_m,
        other=-float("inf"),
    )
    maximum = tl.maximum(exact_lse, halo_lse)
    exact_weight = tl.exp2(exact_lse - maximum)
    halo_weight = tl.exp2(halo_lse - maximum)
    denominator = exact_weight + halo_weight

    tensor_mask = valid_m[:, None]
    exact = tl.load(
        exact_ptr
        + batch * stride_eb
        + head * stride_eh
        + offsets_m[:, None] * stride_es
        + offsets_d[None, :] * stride_ed,
        mask=tensor_mask,
        other=0.0,
    )
    halo = tl.load(
        halo_ptr
        + batch * stride_hb
        + head * stride_hh
        + offsets_m[:, None] * stride_hs
        + offsets_d[None, :] * stride_hd,
        mask=tensor_mask,
        other=0.0,
    )
    coarse = tl.load(
        coarse_ptr
        + batch * stride_cb
        + head * stride_ch
        + offsets_m[:, None] * stride_cs
        + offsets_d[None, :] * stride_cd,
        mask=tensor_mask,
        other=0.0,
    )
    gate = tl.load(
        gate_ptr
        + batch * stride_gb
        + head * stride_gh
        + offsets_m[:, None] * stride_gs
        + offsets_d[None, :] * stride_gd,
        mask=tensor_mask,
        other=0.0,
    )
    merged = (
        exact.to(tl.float32) * exact_weight[:, None]
        + halo.to(tl.float32) * halo_weight[:, None]
    ) / denominator[:, None]
    output = merged + coarse.to(tl.float32) * gate.to(tl.float32)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od,
        output,
        mask=tensor_mask,
    )
    if store_fraction:
        tl.store(
            halo_fraction_ptr
            + batch * stride_fb
            + head * stride_fh
            + offsets_m * stride_fs,
            halo_weight / denominator,
            mask=valid_m,
        )


def merge_core_halo_with_coarse(
    exact_output: torch.Tensor,
    exact_lse_log2: torch.Tensor,
    halo_output: torch.Tensor,
    halo_lse_log2: torch.Tensor,
    coarse_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    return_halo_fraction: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse exact/halo normalization with the checkpoint's coarse residual."""
    if not exact_output.is_cuda:
        raise ValueError("merge_core_halo_with_coarse requires CUDA tensors")
    if exact_output.shape != halo_output.shape:
        raise ValueError("Exact and halo output shapes must match")
    if exact_output.shape != coarse_output.shape or exact_output.shape != gate.shape:
        raise ValueError("Exact, coarse, and gate shapes must match")
    batch, heads, sequence, head_dim = exact_output.shape
    if head_dim != 128:
        raise ValueError(f"The optimized merge requires D=128, got {head_dim}")

    output = torch.empty_like(exact_output)
    halo_fraction = (
        torch.empty_like(exact_lse_log2)
        if return_halo_fraction
        else None
    )
    fraction_storage = (
        halo_fraction
        if halo_fraction is not None
        else exact_lse_log2
    )
    grid = (triton.cdiv(sequence, 64), batch * heads)
    _merge_core_halo_coarse_kernel[grid](
        exact_output,
        exact_lse_log2,
        halo_output,
        halo_lse_log2,
        coarse_output,
        gate,
        output,
        fraction_storage,
        *exact_output.stride(),
        *exact_lse_log2.stride(),
        *halo_output.stride(),
        *halo_lse_log2.stride(),
        *coarse_output.stride(),
        *gate.stride(),
        *output.stride(),
        *fraction_storage.stride(),
        num_heads=heads,
        sequence=sequence,
        head_dim=head_dim,
        block_m=64,
        store_fraction=return_halo_fraction,
        num_warps=8,
        num_stages=2,
    )
    return output, halo_fraction


@triton.jit
def _compressed_halo_forward_kernel(
    query_ptr,
    key_coarse_ptr,
    value_coarse_ptr,
    native_mask_ptr,
    block_sizes_ptr,
    output_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mn,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    num_heads: tl.constexpr,
    num_blocks: tl.constexpr,
    head_dim: tl.constexpr,
    block_elements: tl.constexpr,
    block_n: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    offsets_m = query_block * block_elements + tl.arange(0, block_elements)
    offsets_d = tl.arange(0, head_dim)
    query_pointers = (
        query_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd
    )
    query = tl.load(query_pointers)

    row_max = tl.full([block_elements], -float("inf"), tl.float32)
    row_sum = tl.zeros([block_elements], tl.float32)
    accumulator = tl.zeros([block_elements, head_dim], tl.float32)
    scale_log2 = (1.0 / tl.sqrt(float(head_dim))) * 1.4426950408889634

    for start_n in range(0, num_blocks, block_n):
        offsets_n = start_n + tl.arange(0, block_n)
        valid_n = offsets_n < num_blocks
        key_pointers = (
            key_coarse_ptr
            + batch * stride_kb
            + head * stride_kh
            + offsets_n[:, None] * stride_kn
            + offsets_d[None, :] * stride_kd
        )
        value_pointers = (
            value_coarse_ptr
            + batch * stride_vb
            + head * stride_vh
            + offsets_n[:, None] * stride_vn
            + offsets_d[None, :] * stride_vd
        )
        key = tl.load(
            key_pointers,
            mask=valid_n[:, None],
            other=0.0,
        )
        value = tl.load(
            value_pointers,
            mask=valid_n[:, None],
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(key)) * scale_log2
        block_sizes = tl.load(
            block_sizes_ptr + offsets_n,
            mask=valid_n,
            other=1,
        ).to(tl.float32)
        logits += tl.log2(block_sizes)[None, :]
        selected = tl.load(
            native_mask_ptr + batch * stride_mb + head * stride_mh + query_block * stride_mq + offsets_n * stride_mn,
            mask=valid_n,
            other=1,
        ).to(tl.int1)
        halo_valid = valid_n & ~selected
        logits = tl.where(
            halo_valid[None, :],
            logits,
            -float("inf"),
        )

        next_max = tl.maximum(row_max, tl.max(logits, axis=1))
        has_support = next_max != -float("inf")
        alpha = tl.where(
            has_support,
            tl.exp2(row_max - next_max),
            0.0,
        )
        probabilities = tl.where(
            has_support[:, None],
            tl.exp2(logits - next_max[:, None]),
            0.0,
        )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator *= alpha[:, None]
        accumulator = tl.dot(
            probabilities.to(tl.bfloat16),
            value,
            accumulator,
        )
        row_max = next_max

    output = accumulator / row_sum[:, None]
    output_pointers = (
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od
    )
    lse_pointers = lse_ptr + batch * stride_lb + head * stride_lh + offsets_m * stride_ls
    tl.store(output_pointers, output)
    tl.store(lse_pointers, row_max + tl.log2(row_sum))


def compressed_halo_attention(
    query_bhsd: torch.Tensor,
    key_coarse: torch.Tensor,
    value_coarse: torch.Tensor,
    native_mask: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    block_elements: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query_bhsd.is_cuda:
        raise ValueError("compressed_halo_attention requires CUDA tensors")
    if query_bhsd.ndim != 4:
        raise ValueError(f"query_bhsd must be [B,H,S,D], got {query_bhsd.shape}")
    batch, heads, sequence, head_dim = query_bhsd.shape
    num_blocks = key_coarse.shape[-2]
    if sequence != native_mask.shape[-2] * block_elements:
        raise ValueError(
            f"Query sequence does not match coarse query blocks: sequence={sequence}, q_blocks={native_mask.shape[-2]}"
        )
    if head_dim != 128 or block_elements != 64:
        raise ValueError(
            f"The optimized CH-VSA kernel currently requires D=128 and B=64, got D={head_dim}, B={block_elements}"
        )
    if key_coarse.shape != value_coarse.shape:
        raise ValueError("key_coarse and value_coarse shapes must match")
    if native_mask.shape != (
        batch,
        heads,
        sequence // block_elements,
        num_blocks,
    ):
        raise ValueError(f"Unexpected native mask shape: {native_mask.shape}")

    output = torch.empty_like(query_bhsd)
    lse = torch.empty(
        (batch, heads, sequence),
        device=query_bhsd.device,
        dtype=torch.float32,
    )
    grid = (sequence // block_elements, batch * heads)
    _compressed_halo_forward_kernel[grid](
        query_bhsd,
        key_coarse,
        value_coarse,
        native_mask,
        variable_block_sizes,
        output,
        lse,
        *query_bhsd.stride(),
        *key_coarse.stride(),
        *value_coarse.stride(),
        *native_mask.stride(),
        *output.stride(),
        *lse.stride(),
        num_heads=heads,
        num_blocks=num_blocks,
        head_dim=head_dim,
        block_elements=block_elements,
        block_n=64,
        num_warps=8,
        num_stages=3,
    )
    return output, lse


def _make_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def compressed_support_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    mode: CompressedSupportMode,
    sparsity: float = 0.8,
    block_elements: int = 64,
    detailed_trace: bool = True,
) -> tuple[torch.Tensor, CompressedSupportDecision]:
    from fastvideo.attention.backends.video_sparse_attn import compute_topk
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    num_blocks = variable_block_sizes.numel()
    if sequence != num_blocks * block_elements:
        raise ValueError(
            "Compressed-support VSA requires the Wan 64-token tiled path: "
            f"sequence={sequence}, blocks={num_blocks}, B={block_elements}"
        )
    exact_k = compute_topk(sparsity, num_blocks)

    coarse_events = _make_event_pair()
    selector_events = _make_event_pair()
    fine_events = _make_event_pair()
    rectification_events = _make_event_pair() if mode == "rectified" else None
    halo_events = _make_event_pair() if mode == "compressed_halo" else None
    merge_events = _make_event_pair() if mode == "compressed_halo" else None

    coarse_events[0].record()
    query_coarse = fused_block_mean(
        query_bhsd,
        variable_block_sizes,
        block_elements,
    )
    key_coarse = fused_block_mean(
        key_bhsd,
        variable_block_sizes,
        block_elements,
    )
    value_coarse = fused_block_mean(
        value_bhsd,
        variable_block_sizes,
        block_elements,
    )
    coarse_scores = torch.matmul(
        query_coarse,
        key_coarse.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    coarse_attention = torch.softmax(coarse_scores, dim=-1)
    coarse_output = torch.matmul(coarse_attention, value_coarse)
    coarse_output = (
        coarse_output.view(
            batch,
            heads,
            num_blocks,
            1,
            head_dim,
        )
        .expand(-1, -1, -1, block_elements, -1)
        .reshape(batch, heads, sequence, head_dim)
    )
    coarse_events[1].record()

    selector_events[0].record()
    native_mask = rank_normalized_topk_mask(coarse_scores, exact_k)
    selected_counts = native_mask.sum(dim=-1)
    selector_events[1].record()

    fine_events[0].record()
    exact_output, exact_lse = block_sparse_attn(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        native_mask,
        variable_block_sizes,
    )
    fine_events[1].record()
    if mode == "compressed_halo" and exact_lse is None:
        raise RuntimeError("Compressed-support VSA requires sparse-kernel LSE")

    retained_mass = None
    native_output = None
    correction = None
    contribution = None
    halo_output = None
    halo_fraction = None

    if mode == "rectified":
        assert rectification_events is not None
        rectification_events[0].record()
        output, retained_mass, omitted_output = rectified_output(
            exact_output,
            coarse_attention,
            value_coarse,
            native_mask,
            gate_bhsd,
            block_elements=block_elements,
        )
        rectification_events[1].record()
        if detailed_trace:
            native_output = exact_output + coarse_output * gate_bhsd
            correction = output - native_output
            contribution = omitted_output
    elif mode == "compressed_halo":
        assert halo_events is not None
        assert merge_events is not None
        assert exact_lse is not None
        halo_events[0].record()
        halo_output, halo_lse = compressed_halo_attention(
            query_bhsd,
            key_coarse,
            value_coarse,
            native_mask,
            variable_block_sizes,
            block_elements=block_elements,
        )
        halo_events[1].record()
        merge_events[0].record()
        output, halo_fraction = merge_core_halo_with_coarse(
            exact_output,
            exact_lse,
            halo_output,
            halo_lse,
            coarse_output,
            gate_bhsd,
            return_halo_fraction=detailed_trace,
        )
        merge_events[1].record()
        if detailed_trace:
            retained_mass = coarse_attention.masked_fill(
                ~native_mask,
                0.0,
            ).sum(dim=-1)
            native_output = exact_output + coarse_output * gate_bhsd
            correction = output - native_output
            contribution = correction
    else:
        raise ValueError(f"Unsupported compressed-support mode: {mode}")

    decision = CompressedSupportDecision(
        mode=mode,
        num_blocks=num_blocks,
        exact_k=exact_k,
        omitted_blocks=num_blocks - exact_k,
        block_elements=block_elements,
        num_query_rows=selected_counts.numel(),
        selected_count_min=selected_counts.min(),
        selected_count_max=selected_counts.max(),
        coarse_events=coarse_events,
        selector_events=selector_events,
        fine_events=fine_events,
        rectification_events=rectification_events,
        halo_events=halo_events,
        merge_events=merge_events,
    )
    if detailed_trace:
        assert retained_mass is not None
        assert correction is not None
        assert native_output is not None
        assert contribution is not None
        retained_float = retained_mass.float()
        retained_p10, retained_p50, retained_p90 = _sample_quantiles(retained_float)
        retained_mean = retained_float.mean()
        decision.retained_mass_mean = retained_mean
        decision.retained_mass_min = retained_float.min()
        decision.retained_mass_p10 = retained_p10
        decision.retained_mass_p50 = retained_p50
        decision.retained_mass_p90 = retained_p90
        decision.omitted_mass_mean = 1.0 - retained_mean
        decision.gate_abs_mean = gate_bhsd.float().abs().mean()
        decision.gate_rms = gate_bhsd.float().square().mean().sqrt()
        coarse_scores_float = coarse_scores.float()
        decision.coarse_score_abs_max = coarse_scores_float.abs().max()
        decision.coarse_score_range_max = (
            coarse_scores_float.max(dim=-1).values
            - coarse_scores_float.min(dim=-1).values
        ).max()
        decision.coarse_score_nonfinite_count = (
            ~torch.isfinite(coarse_scores_float)
        ).sum()
        correction_float = correction.float()
        native_float = native_output.float()
        contribution_float = contribution.float()
        decision.correction_abs_mean = correction_float.abs().mean()
        decision.correction_rms = correction_float.square().mean().sqrt()
        decision.correction_relative_l2 = correction_float.norm().div(native_float.norm().clamp_min(1e-12))
        decision.halo_abs_mean = contribution_float.abs().mean()
        decision.halo_rms = contribution_float.square().mean().sqrt()
        if halo_fraction is not None:
            halo_p10, halo_p50, halo_p90 = _sample_quantiles(halo_fraction)
            del halo_p10
            decision.halo_weight_mean = halo_fraction.float().mean()
            decision.halo_weight_p50 = halo_p50
            decision.halo_weight_p90 = halo_p90
    return output.transpose(1, 2), decision


def summarize_compressed_support_decision(
    decision: CompressedSupportDecision,
) -> dict[str, Any]:
    def scalar(value: torch.Tensor | None) -> float | None:
        return None if value is None else float(value.item())

    result: dict[str, Any] = {
        "mode": decision.mode,
        "num_blocks": decision.num_blocks,
        "exact_k": decision.exact_k,
        "omitted_blocks": decision.omitted_blocks,
        "block_elements": decision.block_elements,
        "num_query_rows": decision.num_query_rows,
        "selected_count_min": int(decision.selected_count_min.item()),
        "selected_count_max": int(decision.selected_count_max.item()),
        "retained_mass_mean": scalar(decision.retained_mass_mean),
        "retained_mass_min": scalar(decision.retained_mass_min),
        "retained_mass_p10": scalar(decision.retained_mass_p10),
        "retained_mass_p50": scalar(decision.retained_mass_p50),
        "retained_mass_p90": scalar(decision.retained_mass_p90),
        "omitted_mass_mean": scalar(decision.omitted_mass_mean),
        "gate_abs_mean": scalar(decision.gate_abs_mean),
        "gate_rms": scalar(decision.gate_rms),
        "coarse_score_abs_max": scalar(decision.coarse_score_abs_max),
        "coarse_score_range_max": scalar(decision.coarse_score_range_max),
        "coarse_score_nonfinite_count": (
            None
            if decision.coarse_score_nonfinite_count is None
            else int(decision.coarse_score_nonfinite_count.item())
        ),
        "correction_abs_mean": scalar(decision.correction_abs_mean),
        "correction_rms": scalar(decision.correction_rms),
        "correction_relative_l2": scalar(decision.correction_relative_l2),
        "halo_abs_mean": scalar(decision.halo_abs_mean),
        "halo_rms": scalar(decision.halo_rms),
        "halo_weight_mean": scalar(decision.halo_weight_mean),
        "halo_weight_p50": scalar(decision.halo_weight_p50),
        "halo_weight_p90": scalar(decision.halo_weight_p90),
        "nominal_exact_tokens": (decision.exact_k * decision.block_elements),
        "compressed_support_tokens": decision.omitted_blocks,
        "nominal_dense_tokens": (decision.num_blocks * decision.block_elements),
        "dense_equivalent_support_ratio": (decision.exact_k * decision.block_elements + decision.omitted_blocks)
        / (decision.num_blocks * decision.block_elements),
    }
    for label, events in (
        ("native_coarse_ms", decision.coarse_events),
        ("native_topk_ms", decision.selector_events),
        ("native_exact_fine_ms", decision.fine_events),
        ("rectification_ms", decision.rectification_events),
        ("compressed_halo_fused_ms", decision.halo_events),
        ("merge_ms", decision.merge_events),
    ):
        if events is not None:
            result[label] = events[0].elapsed_time(events[1])
    return result
