from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from research.compressed_halo_vsa.compressed_support import (
    rank_normalized_topk_mask, )


@dataclass(frozen=True)
class BudgetRedistributedPolicy:
    k_table: tuple[tuple[tuple[int, ...], ...], ...]
    candidate_k: tuple[int, ...]
    native_budget: int
    allocated_budget: int
    num_blocks: int
    granularity: str

    @classmethod
    def from_path(cls, path: str | Path) -> BudgetRedistributedPolicy:
        payload = json.loads(Path(path).read_text())
        table = tuple(tuple(tuple(int(value) for value in heads) for heads in layers) for layers in payload["k_table"])
        policy = cls(
            k_table=table,
            candidate_k=tuple(int(value) for value in payload["candidate_K"]),
            native_budget=int(payload["native_budget"]),
            allocated_budget=int(payload["allocated_budget"]),
            num_blocks=int(payload["num_blocks"]),
            granularity=str(payload["granularity"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        values = [exact_k for step in self.k_table for layer in step for exact_k in layer]
        if not values:
            raise ValueError("BR-VSA K table is empty")
        if any(value not in self.candidate_k for value in values):
            raise ValueError("BR-VSA K table contains unsupported K values")
        if sum(values) != self.allocated_budget:
            raise ValueError("BR-VSA K table and allocated budget disagree")
        if self.allocated_budget > self.native_budget:
            raise ValueError("BR-VSA K table exceeds the native global budget")
        if max(values) > self.num_blocks:
            raise ValueError("BR-VSA K table exceeds available coarse blocks")

    def k_for(self, step: int, layer: int) -> tuple[int, ...]:
        return self.k_table[step][layer]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_k": list(self.candidate_k),
            "native_budget": self.native_budget,
            "allocated_budget": self.allocated_budget,
            "budget_ratio": self.allocated_budget / self.native_budget,
            "num_blocks": self.num_blocks,
            "granularity": self.granularity,
        }


@dataclass
class BudgetRedistributedDecision:
    requested_k: tuple[int, ...]
    num_blocks: int
    num_query_blocks: int
    selected_count_min: torch.Tensor
    selected_count_max: torch.Tensor
    selected_count_error_abs_max: torch.Tensor
    selected_total_per_query: torch.Tensor
    unique_k: tuple[int, ...]


def budget_redistributed_video_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    requested_k: tuple[int, ...],
    *,
    block_elements: int = 64,
) -> tuple[torch.Tensor, BudgetRedistributedDecision]:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()
    gate_bhsd = gate_compress.transpose(1, 2).contiguous()
    batch, heads, sequence, head_dim = query_bhsd.shape
    num_blocks = int(variable_block_sizes.numel())
    if len(requested_k) != heads:
        raise ValueError(f"BR-VSA expected {heads} per-head K values, got {len(requested_k)}")
    if sequence != num_blocks * block_elements:
        raise ValueError("BR-VSA requires the Wan 64-token tiled path")
    if min(requested_k) < 1 or max(requested_k) > num_blocks:
        raise ValueError("BR-VSA requested K outside the available block range")

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
    coarse_output = (coarse_output.view(
        batch,
        heads,
        num_blocks,
        1,
        head_dim,
    ).expand(-1, -1, -1, block_elements, -1).reshape(batch, heads, sequence, head_dim))

    sparse_mask = torch.empty_like(coarse_scores, dtype=torch.bool)
    unique_k = tuple(sorted(set(int(value) for value in requested_k)))
    for exact_k in unique_k:
        head_indices = [index for index, value in enumerate(requested_k) if int(value) == exact_k]
        if exact_k == num_blocks:
            group_mask = torch.ones(
                (
                    batch,
                    len(head_indices),
                    coarse_scores.shape[-2],
                    num_blocks,
                ),
                device=coarse_scores.device,
                dtype=torch.bool,
            )
        else:
            group_mask = rank_normalized_topk_mask(
                coarse_scores[:, head_indices],
                exact_k,
            )
        sparse_mask[:, head_indices] = group_mask

    selected_counts = sparse_mask.sum(dim=-1)
    requested_tensor = torch.tensor(
        requested_k,
        device=selected_counts.device,
        dtype=selected_counts.dtype,
    ).view(1, heads, 1)
    exact_output, _ = block_sparse_attn(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        sparse_mask,
        variable_block_sizes,
    )
    output = exact_output + coarse_output * gate_bhsd
    decision = BudgetRedistributedDecision(
        requested_k=tuple(int(value) for value in requested_k),
        num_blocks=num_blocks,
        num_query_blocks=int(coarse_scores.shape[-2]),
        selected_count_min=selected_counts.min(),
        selected_count_max=selected_counts.max(),
        selected_count_error_abs_max=(selected_counts - requested_tensor).abs().max(),
        selected_total_per_query=selected_counts.sum(dim=1),
        unique_k=unique_k,
    )
    return output.transpose(1, 2), decision


def summarize_budget_redistributed_decision(decision: BudgetRedistributedDecision, ) -> dict[str, Any]:
    return {
        "requested_k": list(decision.requested_k),
        "requested_total_k": sum(decision.requested_k),
        "requested_mean_k": sum(decision.requested_k) / len(decision.requested_k),
        "unique_k": list(decision.unique_k),
        "num_unique_k": len(decision.unique_k),
        "num_blocks": decision.num_blocks,
        "num_query_blocks": decision.num_query_blocks,
        "selected_count_min": int(decision.selected_count_min.item()),
        "selected_count_max": int(decision.selected_count_max.item()),
        "selected_count_error_abs_max": int(decision.selected_count_error_abs_max.item()),
        "selected_total_per_query_min": int(decision.selected_total_per_query.min().item()),
        "selected_total_per_query_max": int(decision.selected_total_per_query.max().item()),
    }
