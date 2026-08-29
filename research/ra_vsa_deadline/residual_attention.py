from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class ResidualAwareVSAPolicy:
    native_fraction: float
    native_sparsity: float = 0.8
    risk_formula: str = "coarse_mass_x_key_heterogeneity"
    instrument_splits: tuple[float, ...] = ()
    detailed_trace: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.native_fraction <= 1.0:
            raise ValueError(
                f"native_fraction must be in [0, 1], got {self.native_fraction}"
            )
        if not 0.0 <= self.native_sparsity < 1.0:
            raise ValueError(
                f"native_sparsity must be in [0, 1), got {self.native_sparsity}"
            )
        if self.risk_formula != "coarse_mass_x_key_heterogeneity":
            raise ValueError(f"Unsupported risk formula: {self.risk_formula}")
        if any(not 0.0 <= value <= 1.0 for value in self.instrument_splits):
            raise ValueError(
                f"instrument_splits must be in [0, 1], got {self.instrument_splits}"
            )

    def slots(self, num_blocks: int) -> tuple[int, int, int]:
        total = max(
            1,
            min(
                math.ceil((1.0 - self.native_sparsity) * num_blocks),
                num_blocks,
            ),
        )
        rescue = min(total, max(0, int(round(total * (1.0 - self.native_fraction)))))
        return total, total - rescue, rescue

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_fraction": self.native_fraction,
            "rescue_fraction": 1.0 - self.native_fraction,
            "native_sparsity": self.native_sparsity,
            "risk_formula": self.risk_formula,
            "instrument_splits": list(self.instrument_splits),
            "detailed_trace": self.detailed_trace,
        }


@dataclass
class ResidualDecision:
    num_blocks: int
    total_slots: int
    native_slots: int
    rescue_slots: int
    num_query_rows: int
    selected_count_min: torch.Tensor
    selected_count_max: torch.Tensor
    replacement_fraction_mean: torch.Tensor
    replacement_fraction_min: torch.Tensor
    replacement_fraction_max: torch.Tensor
    key_heterogeneity_mean: torch.Tensor
    key_heterogeneity_p50: torch.Tensor
    key_heterogeneity_p90: torch.Tensor
    risk_mean: torch.Tensor
    risk_p50: torch.Tensor
    risk_p90: torch.Tensor
    gate_abs_mean: torch.Tensor | None = None
    gate_rms: torch.Tensor | None = None
    rescue_key_heterogeneity_mean: torch.Tensor | None = None
    removed_key_heterogeneity_mean: torch.Tensor | None = None
    hypothetical_replacement_means: dict[str, torch.Tensor] = field(
        default_factory=dict
    )
    example_native_mask: torch.Tensor | None = None
    example_final_mask: torch.Tensor | None = None
    example_scores: torch.Tensor | None = None
    example_risk: torch.Tensor | None = None
    example_key_heterogeneity: torch.Tensor | None = None
    coarse_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    selector_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    fine_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None


def _topk_mask(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    # RA-VSA must preserve exactly K even when many BF16 scores tie. The
    # production fused threshold mask can return K+1 on such rows, while
    # index-based torch.topk always returns exactly K indices.
    indices = torch.topk(scores, topk, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1,
        indices,
        True,
    )


@triton.jit
def _block_mean_square_energy_kernel(
    key_ptr,
    block_sizes_ptr,
    output_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    num_heads: tl.constexpr,
    num_blocks: tl.constexpr,
    head_dim: tl.constexpr,
    block_elements: tl.constexpr,
    tile: tl.constexpr,
):
    program = tl.program_id(0)
    block = program % num_blocks
    head = (program // num_blocks) % num_heads
    batch = program // (num_blocks * num_heads)
    valid_tokens = tl.load(block_sizes_ptr + block)
    accumulator = 0.0
    elements = block_elements * head_dim
    for start in range(0, elements, tile):
        offsets = start + tl.arange(0, tile)
        token = offsets // head_dim
        feature = offsets % head_dim
        valid = (offsets < elements) & (token < valid_tokens)
        pointers = (
            key_ptr
            + batch * stride_b
            + head * stride_h
            + (block * block_elements + token) * stride_s
            + feature * stride_d
        )
        values = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
        accumulator += tl.sum(values * values, axis=0)
    tl.store(output_ptr + program, accumulator / valid_tokens)


def _block_mean_square_energy(
    key_bhsd: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    batch, heads, sequence, head_dim = key_bhsd.shape
    num_blocks = variable_block_sizes.numel()
    if sequence != num_blocks * block_elements:
        raise ValueError(
            f"Expected sequence={num_blocks * block_elements}, got {sequence}"
        )
    output = torch.empty(
        (batch, heads, num_blocks),
        device=key_bhsd.device,
        dtype=torch.float32,
    )
    grid = (batch * heads * num_blocks,)
    _block_mean_square_energy_kernel[grid](
        key_bhsd,
        variable_block_sizes,
        output,
        key_bhsd.stride(0),
        key_bhsd.stride(1),
        key_bhsd.stride(2),
        key_bhsd.stride(3),
        num_heads=heads,
        num_blocks=num_blocks,
        head_dim=head_dim,
        block_elements=block_elements,
        tile=1024,
    )
    return output


def key_heterogeneity_from_pooled(
    key_coarse: torch.Tensor,
    *,
    key_bhsd: torch.Tensor | None = None,
    variable_block_sizes: torch.Tensor | None = None,
    block_elements: int = 64,
) -> torch.Tensor:
    """Normalized within-block K variance.

    With token-level K available, this evaluates the exact variance identity:

        U_K = 1 - ||mean(K_block)||^2 / mean(||K_token||^2)

    using a fused block reduction that does not materialize K squared. The
    pooled-only fallback is retained for CPU unit tests.
    """
    if key_coarse.ndim != 4:
        raise ValueError(
            "key_coarse must be [B,H,KV,D], got "
            f"shape={tuple(key_coarse.shape)}"
        )
    heads, head_dim = key_coarse.shape[1], key_coarse.shape[-1]
    pooled_energy = key_coarse.float().square().sum(dim=(1, 3))
    if key_bhsd is not None:
        if variable_block_sizes is None:
            raise ValueError(
                "variable_block_sizes is required with token-level K"
            )
        token_energy = _block_mean_square_energy(
            key_bhsd,
            variable_block_sizes,
            block_elements,
        ).sum(dim=1)
        denominator = token_energy.clamp_min(torch.finfo(torch.float32).tiny)
    else:
        denominator = torch.full_like(
            pooled_energy,
            float(heads * head_dim),
        )
    return (1.0 - pooled_energy / denominator).clamp(0.0, 1.0)


def _sample_quantiles(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.float().flatten()
    if flat.numel() > 8192:
        stride = math.ceil(flat.numel() / 8192)
        flat = flat[::stride]
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.5, 0.9], device=flat.device),
    )
    return quantiles[0], quantiles[1]


def _select_once(
    scores: torch.Tensor,
    risk: torch.Tensor,
    total_slots: int,
    native_fraction: float,
) -> torch.Tensor:
    rescue_slots = min(
        total_slots,
        max(0, int(round(total_slots * (1.0 - native_fraction)))),
    )
    native_slots = total_slots - rescue_slots
    native_mask = _topk_mask(scores, native_slots)
    if rescue_slots == 0:
        return native_mask
    rescue_scores = risk.masked_fill(native_mask, float("-inf"))
    rescue_mask = _topk_mask(rescue_scores, rescue_slots)
    return native_mask | rescue_mask


def select_residual_mask(
    scores: torch.Tensor,
    coarse_attention: torch.Tensor,
    key_coarse: torch.Tensor,
    policy: ResidualAwareVSAPolicy,
    *,
    gate_compress: torch.Tensor | None = None,
    key_bhsd: torch.Tensor | None = None,
    variable_block_sizes: torch.Tensor | None = None,
    block_elements: int = 64,
) -> tuple[torch.Tensor, ResidualDecision]:
    if scores.ndim != 4:
        raise ValueError(
            f"scores must be [B,H,Q,KV], got shape={tuple(scores.shape)}"
        )
    num_blocks = scores.shape[-1]
    total_slots, native_slots, rescue_slots = policy.slots(num_blocks)
    key_heterogeneity = key_heterogeneity_from_pooled(
        key_coarse,
        key_bhsd=key_bhsd,
        variable_block_sizes=variable_block_sizes,
        block_elements=block_elements,
    )
    risk = coarse_attention * key_heterogeneity[:, None, None, :].to(
        coarse_attention.dtype
    )

    final_mask = _select_once(
        scores,
        risk,
        total_slots,
        policy.native_fraction,
    )
    selected_counts = final_mask.sum(dim=-1)
    needs_comparison = policy.detailed_trace or bool(
        policy.instrument_splits
    )
    if needs_comparison:
        full_native_mask = (
            final_mask
            if policy.native_fraction == 1.0
            else _topk_mask(scores, total_slots)
        )
        overlap = (final_mask & full_native_mask).sum(dim=-1)
        replacement_fraction = (
            (total_slots - overlap).float() / float(total_slots)
        )
        uk_p50, uk_p90 = _sample_quantiles(key_heterogeneity)
        risk_p50, risk_p90 = _sample_quantiles(risk)
    else:
        full_native_mask = None
        nan = torch.full(
            (),
            float("nan"),
            device=scores.device,
            dtype=torch.float32,
        )
        replacement_fraction = nan
        uk_p50 = uk_p90 = risk_p50 = risk_p90 = nan
    decision = ResidualDecision(
        num_blocks=num_blocks,
        total_slots=total_slots,
        native_slots=native_slots,
        rescue_slots=rescue_slots,
        num_query_rows=selected_counts.numel(),
        selected_count_min=selected_counts.min(),
        selected_count_max=selected_counts.max(),
        replacement_fraction_mean=replacement_fraction.mean(),
        replacement_fraction_min=replacement_fraction.min(),
        replacement_fraction_max=replacement_fraction.max(),
        key_heterogeneity_mean=key_heterogeneity.mean(),
        key_heterogeneity_p50=uk_p50,
        key_heterogeneity_p90=uk_p90,
        risk_mean=risk.float().mean(),
        risk_p50=risk_p50,
        risk_p90=risk_p90,
    )

    if gate_compress is not None and policy.detailed_trace:
        gate = gate_compress.float()
        decision.gate_abs_mean = gate.abs().mean()
        decision.gate_rms = gate.square().mean().sqrt()

    if policy.detailed_trace:
        assert full_native_mask is not None
        added = final_mask & ~full_native_mask
        removed = full_native_mask & ~final_mask
        uk = key_heterogeneity[:, None, None, :]
        decision.rescue_key_heterogeneity_mean = (
            (added * uk).float().sum()
            / added.sum().clamp_min(1)
        )
        decision.removed_key_heterogeneity_mean = (
            (removed * uk).float().sum()
            / removed.sum().clamp_min(1)
        )

        replacement_flat = (total_slots - overlap).reshape(-1)
        example_index = replacement_flat.argmax()
        native_rows = full_native_mask.reshape(-1, num_blocks)
        final_rows = final_mask.reshape(-1, num_blocks)
        score_rows = scores.reshape(-1, num_blocks)
        risk_rows = risk.expand_as(scores).reshape(-1, num_blocks)
        batch_rows = scores.shape[1] * scores.shape[2]
        batch_index = torch.div(example_index, batch_rows, rounding_mode="floor")
        decision.example_native_mask = native_rows[example_index].clone()
        decision.example_final_mask = final_rows[example_index].clone()
        decision.example_scores = score_rows[example_index].clone()
        decision.example_risk = risk_rows[example_index].clone()
        decision.example_key_heterogeneity = key_heterogeneity[
            batch_index
        ].clone()

    for split in policy.instrument_splits:
        assert full_native_mask is not None
        hypothetical = _select_once(scores, risk, total_slots, split)
        hypothetical_overlap = (
            hypothetical & full_native_mask
        ).sum(dim=-1)
        label = f"native_{int(round(split * 1000)):04d}"
        decision.hypothetical_replacement_means[label] = (
            (total_slots - hypothetical_overlap).float().mean()
            / float(total_slots)
        )

    return final_mask, decision


def residual_aware_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor | None,
    variable_block_sizes: torch.Tensor,
    policy: ResidualAwareVSAPolicy,
    *,
    block_elements: int = 64,
) -> tuple[torch.Tensor, ResidualDecision]:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = (
        None
        if gate_compress is None
        else gate_compress.transpose(1, 2).contiguous()
    )
    batch, heads, query_length, head_dim = query_bhsd.shape
    num_blocks = variable_block_sizes.numel()
    if query_length != num_blocks * block_elements:
        raise ValueError(
            "RA-VSA requires the Wan 64-token tiled path: "
            f"query_length={query_length}, num_blocks={num_blocks}"
        )

    coarse_start = torch.cuda.Event(enable_timing=True)
    coarse_end = torch.cuda.Event(enable_timing=True)
    selector_start = torch.cuda.Event(enable_timing=True)
    selector_end = torch.cuda.Event(enable_timing=True)
    fine_start = torch.cuda.Event(enable_timing=True)
    fine_end = torch.cuda.Event(enable_timing=True)

    coarse_start.record()
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
    scores = torch.matmul(
        query_coarse,
        key_coarse.transpose(-2, -1),
    ) / math.sqrt(head_dim)
    coarse_attention = torch.softmax(scores, dim=-1)
    coarse_output = torch.matmul(coarse_attention, value_coarse)
    coarse_output = (
        coarse_output.view(batch, heads, num_blocks, 1, head_dim)
        .expand(-1, -1, -1, block_elements, -1)
        .reshape(batch, heads, query_length, head_dim)
    )
    coarse_end.record()

    selector_start.record()
    mask, decision = select_residual_mask(
        scores,
        coarse_attention,
        key_coarse,
        policy,
        gate_compress=gate_compress,
        key_bhsd=key_bhsd,
        variable_block_sizes=variable_block_sizes,
        block_elements=block_elements,
    )
    selector_end.record()

    fine_start.record()
    sparse_output = block_sparse_attn(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        mask,
        variable_block_sizes,
    )[0]
    fine_end.record()

    output = (
        sparse_output + coarse_output
        if gate_bhsd is None
        else sparse_output + coarse_output * gate_bhsd
    )
    decision.coarse_events = (coarse_start, coarse_end)
    decision.selector_events = (selector_start, selector_end)
    decision.fine_events = (fine_start, fine_end)
    return output.transpose(1, 2), decision


def summarize_residual_decision(
    decision: ResidualDecision,
) -> dict[str, Any]:
    def scalar(value: torch.Tensor | None) -> float | None:
        return None if value is None else float(value.item())

    result: dict[str, Any] = {
        "num_query_rows": decision.num_query_rows,
        "num_blocks": decision.num_blocks,
        "total_slots": decision.total_slots,
        "native_slots": decision.native_slots,
        "rescue_slots": decision.rescue_slots,
        "selected_count_min": int(decision.selected_count_min.item()),
        "selected_count_max": int(decision.selected_count_max.item()),
        "replacement_fraction_mean": scalar(
            decision.replacement_fraction_mean
        ),
        "replacement_fraction_min": scalar(
            decision.replacement_fraction_min
        ),
        "replacement_fraction_max": scalar(
            decision.replacement_fraction_max
        ),
        "key_heterogeneity_mean": scalar(
            decision.key_heterogeneity_mean
        ),
        "key_heterogeneity_p50": scalar(
            decision.key_heterogeneity_p50
        ),
        "key_heterogeneity_p90": scalar(
            decision.key_heterogeneity_p90
        ),
        "risk_mean": scalar(decision.risk_mean),
        "risk_p50": scalar(decision.risk_p50),
        "risk_p90": scalar(decision.risk_p90),
        "gate_abs_mean": scalar(decision.gate_abs_mean),
        "gate_rms": scalar(decision.gate_rms),
        "rescue_key_heterogeneity_mean": scalar(
            decision.rescue_key_heterogeneity_mean
        ),
        "removed_key_heterogeneity_mean": scalar(
            decision.removed_key_heterogeneity_mean
        ),
    }
    for label, value in decision.hypothetical_replacement_means.items():
        result[f"{label}_replacement_fraction_mean"] = scalar(value)
    if decision.coarse_events is not None:
        result["coarse_selector_ms"] = decision.coarse_events[
            0
        ].elapsed_time(decision.coarse_events[1])
    if decision.selector_events is not None:
        result["metadata_selector_ms"] = decision.selector_events[
            0
        ].elapsed_time(decision.selector_events[1])
    if decision.fine_events is not None:
        result["fine_attention_ms"] = decision.fine_events[
            0
        ].elapsed_time(decision.fine_events[1])

    if decision.example_native_mask is not None:
        native = decision.example_native_mask.cpu()
        final = decision.example_final_mask.cpu()
        scores = decision.example_scores.float().cpu()
        risk = decision.example_risk.float().cpu()
        uk = decision.example_key_heterogeneity.float().cpu()
        removed = torch.where(native & ~final)[0]
        added = torch.where(final & ~native)[0]
        result.update(
            {
                "example_removed_blocks": removed.tolist(),
                "example_added_blocks": added.tolist(),
                "example_removed_native_scores": scores[removed].tolist(),
                "example_added_native_scores": scores[added].tolist(),
                "example_removed_risk_scores": risk[removed].tolist(),
                "example_added_risk_scores": risk[added].tolist(),
                "example_removed_key_heterogeneity": uk[removed].tolist(),
                "example_added_key_heterogeneity": uk[added].tolist(),
            }
        )
    return result
