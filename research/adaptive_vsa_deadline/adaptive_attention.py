from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AdaptiveVSAPolicy:
    retained_mass_threshold: float
    maximum_sparsity: float
    candidate_sparsities: tuple[float, ...] = (0.8, 0.7, 0.6, 0.4, 0.0)
    native_sparsity: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.retained_mass_threshold <= 1.0:
            raise ValueError(
                "retained_mass_threshold must be in (0, 1], got "
                f"{self.retained_mass_threshold}"
            )
        if not 0.0 <= self.maximum_sparsity < 1.0:
            raise ValueError(
                f"maximum_sparsity must be in [0, 1), got {self.maximum_sparsity}"
            )
        if 0.0 not in self.candidate_sparsities:
            raise ValueError("candidate_sparsities must include dense fallback 0.0")
        if any(not 0.0 <= value < 1.0 for value in self.candidate_sparsities):
            raise ValueError(
                f"candidate_sparsities must be in [0, 1), got {self.candidate_sparsities}"
            )
        if not 0.0 <= self.native_sparsity < 1.0:
            raise ValueError(
                f"native_sparsity must be in [0, 1), got {self.native_sparsity}"
            )

    def budget_options(self, num_blocks: int) -> tuple[tuple[int, float], ...]:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        options: dict[int, float] = {}
        for sparsity in self.candidate_sparsities:
            if sparsity <= self.maximum_sparsity + 1e-12:
                topk = max(
                    1,
                    min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks),
                )
                options[topk] = max(options.get(topk, -1.0), sparsity)
        options[num_blocks] = 0.0
        return tuple(sorted(options.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "retained_mass_threshold": self.retained_mass_threshold,
            "maximum_sparsity": self.maximum_sparsity,
            "minimum_density_floor": 1.0 - self.maximum_sparsity,
            "candidate_sparsities": list(self.candidate_sparsities),
            "native_sparsity": self.native_sparsity,
        }


@dataclass
class AdaptiveDecision:
    selected_topk: torch.Tensor
    selected_retained_mass: torch.Tensor
    native_retained_mass: torch.Tensor
    budget_options: tuple[tuple[int, float], ...]
    num_blocks: int


def select_adaptive_mask(
    scores: torch.Tensor,
    policy: AdaptiveVSAPolicy,
) -> tuple[torch.Tensor, AdaptiveDecision]:
    if scores.ndim != 4:
        raise ValueError(
            f"scores must be [B, H, Q, KV], got shape={tuple(scores.shape)}"
        )
    num_blocks = scores.shape[-1]
    options = policy.budget_options(num_blocks)

    probabilities = torch.softmax(scores.float(), dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(
        dim=-1,
        descending=True,
    )
    cumulative_mass = sorted_probabilities.cumsum(dim=-1)

    selected_topk = torch.full(
        scores.shape[:-1],
        num_blocks,
        device=scores.device,
        dtype=torch.long,
    )
    unresolved = torch.ones_like(selected_topk, dtype=torch.bool)
    for topk, _sparsity in options:
        retained_mass = cumulative_mass[..., topk - 1]
        choose = unresolved & (
            retained_mass >= policy.retained_mass_threshold
        )
        selected_topk = torch.where(
            choose,
            torch.full_like(selected_topk, topk),
            selected_topk,
        )
        unresolved &= ~choose

    ranks = torch.arange(
        num_blocks,
        device=scores.device,
        dtype=torch.long,
    )
    selected_sorted = ranks.view(
        *((1,) * (scores.ndim - 1)),
        num_blocks,
    ) < selected_topk.unsqueeze(-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, selected_sorted)

    selected_retained_mass = cumulative_mass.gather(
        -1,
        (selected_topk - 1).unsqueeze(-1),
    ).squeeze(-1)
    native_topk = max(
        1,
        min(
            math.ceil((1.0 - policy.native_sparsity) * num_blocks),
            num_blocks,
        ),
    )
    decision = AdaptiveDecision(
        selected_topk=selected_topk,
        selected_retained_mass=selected_retained_mass,
        native_retained_mass=cumulative_mass[..., native_topk - 1],
        budget_options=options,
        num_blocks=num_blocks,
    )
    return mask, decision


def adaptive_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor | None,
    variable_block_sizes: torch.Tensor,
    policy: AdaptiveVSAPolicy,
    *,
    block_elements: int = 64,
) -> tuple[torch.Tensor, AdaptiveDecision]:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
    )

    if query.ndim != 4:
        raise ValueError(
            f"query must be [B, S, H, D], got shape={tuple(query.shape)}"
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
            "Adaptive VSA currently requires the Wan 64-token tiled path: "
            f"query_length={query_length}, num_blocks={num_blocks}, "
            f"block_elements={block_elements}"
        )

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

    mask, decision = select_adaptive_mask(scores, policy)
    sparse_output = block_sparse_attn(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        mask,
        variable_block_sizes,
    )[0]
    output = (
        sparse_output + coarse_output
        if gate_bhsd is None
        else sparse_output + coarse_output * gate_bhsd
    )
    return output.transpose(1, 2), decision


def summarize_decision(decision: AdaptiveDecision) -> dict[str, Any]:
    selected_topk = decision.selected_topk
    num_rows = selected_topk.numel()
    mean_density = selected_topk.float().mean() / decision.num_blocks
    selected_mass = decision.selected_retained_mass.float().flatten()
    native_mass = decision.native_retained_mass.float().flatten()
    quantiles = torch.tensor(
        [0.1, 0.5, 0.9],
        device=selected_topk.device,
        dtype=torch.float32,
    )
    selected_mass_quantiles = torch.quantile(selected_mass, quantiles)
    native_mass_quantiles = torch.quantile(native_mass, quantiles)

    summary: dict[str, Any] = {
        "num_query_rows": int(num_rows),
        "num_blocks": int(decision.num_blocks),
        "effective_sparsity": float(1.0 - mean_density.item()),
        "selected_retained_mass_mean": float(selected_mass.mean().item()),
        "selected_retained_mass_min": float(selected_mass.min().item()),
        "selected_retained_mass_p10": float(selected_mass_quantiles[0].item()),
        "selected_retained_mass_p50": float(selected_mass_quantiles[1].item()),
        "selected_retained_mass_p90": float(selected_mass_quantiles[2].item()),
        "native_retained_mass_mean": float(native_mass.mean().item()),
        "native_retained_mass_min": float(native_mass.min().item()),
        "native_retained_mass_p10": float(native_mass_quantiles[0].item()),
        "native_retained_mass_p50": float(native_mass_quantiles[1].item()),
        "native_retained_mass_p90": float(native_mass_quantiles[2].item()),
    }
    for topk, sparsity in decision.budget_options:
        label = f"s{int(round(sparsity * 100)):02d}"
        count = int((selected_topk == topk).sum().item())
        summary[f"decision_{label}_count"] = count
        summary[f"decision_{label}_fraction"] = count / num_rows
    return summary
