from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BalancedGrouping:
    """A deterministic similarity ordering over valid token positions."""

    valid_order: torch.Tensor
    normalized_vectors: torch.Tensor
    padded_tokens: int
    split_depth: int


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("Token count must be positive")
    return 1 << (value - 1).bit_length()


def balanced_recursive_order(
    vectors: torch.Tensor,
    *,
    leaf_size: int = 64,
    principal_iterations: int = 2,
) -> BalancedGrouping:
    """Order [B,H,N,D] vectors with balanced recursive median partitions.

    Every split initializes an approximate principal direction from the
    maximum-variance feature, applies a fixed number of covariance power
    iterations, then performs a stable median partition by projection. The
    valid sequence is padded to a power of two with sentinels; sentinels are
    forced down the final branch and removed from the result.
    """
    if vectors.ndim != 4:
        raise ValueError("vectors must have shape [B,H,N,D]")
    if leaf_size <= 0 or leaf_size & (leaf_size - 1):
        raise ValueError("leaf_size must be a positive power of two")
    if principal_iterations < 0:
        raise ValueError("principal_iterations must be nonnegative")
    batch, heads, tokens, head_dim = vectors.shape
    padded_tokens = _next_power_of_two(tokens)
    if padded_tokens < leaf_size:
        padded_tokens = leaf_size
    if padded_tokens % leaf_size:
        raise ValueError("Power-of-two padding must be divisible by leaf size")
    split_depth = int(math.log2(padded_tokens // leaf_size))

    normalized = F.normalize(vectors.float(), dim=-1).to(vectors.dtype)
    padding = padded_tokens - tokens
    normalized_padded = F.pad(normalized, (0, 0, 0, padding)) if padding else normalized

    batch_heads = batch * heads
    current_vectors = normalized_padded.reshape(
        batch_heads,
        padded_tokens,
        head_dim,
    )
    current_indices = torch.arange(
        padded_tokens,
        device=vectors.device,
        dtype=torch.long,
    ).expand(batch_heads, -1)

    groups = 1
    group_width = padded_tokens
    for _ in range(split_depth):
        grouped_vectors = current_vectors.view(
            batch_heads,
            groups,
            group_width,
            head_dim,
        )
        grouped_indices = current_indices.view(
            batch_heads,
            groups,
            group_width,
        )
        valid = grouped_indices.lt(tokens)
        valid_float = valid.float()
        count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        values = grouped_vectors.float()
        mean = (values * valid_float[..., None]).sum(dim=-2) / count
        variance = ((values - mean[..., None, :]).square() * valid_float[..., None]).sum(dim=-2) / count
        split_dimension = variance.argmax(dim=-1)
        direction = F.one_hot(
            split_dimension,
            num_classes=head_dim,
        ).float()
        centered = values - mean[..., None, :]
        centered = centered * valid_float[..., None]
        for _ in range(principal_iterations):
            projection = (centered * direction[..., None, :]).sum(dim=-1)
            direction = (centered * projection[..., None]).sum(dim=-2)
            direction = F.normalize(direction, dim=-1)
        projections = (centered * direction[..., None, :]).sum(dim=-1)
        projections = projections.float().masked_fill(
            ~valid,
            float("inf"),
        )
        rank = torch.argsort(
            projections,
            dim=-1,
            stable=True,
        )
        grouped_indices = torch.gather(
            grouped_indices,
            -1,
            rank,
        )
        grouped_vectors = torch.gather(
            grouped_vectors,
            -2,
            rank[..., None].expand(-1, -1, -1, head_dim),
        )
        groups *= 2
        group_width //= 2
        current_indices = grouped_indices.reshape(
            batch_heads,
            padded_tokens,
        )
        current_vectors = grouped_vectors.reshape(
            batch_heads,
            padded_tokens,
            head_dim,
        )

    ordered = current_indices.view(
        batch,
        heads,
        padded_tokens,
    )
    valid_order = ordered[..., :tokens]
    if valid_order.ge(tokens).any():
        raise RuntimeError("Recursive grouping did not move all sentinel tokens to the end")
    expected = torch.arange(
        tokens,
        device=vectors.device,
    ).expand(batch, heads, -1)
    if not torch.equal(
        torch.sort(valid_order, dim=-1).values,
        expected,
    ):
        raise RuntimeError("Recursive grouping did not produce a permutation")
    return BalancedGrouping(
        valid_order=valid_order,
        normalized_vectors=normalized,
        padded_tokens=padded_tokens,
        split_depth=split_depth,
    )


def slot_valid_mask(
    block_sizes: torch.Tensor,
    *,
    block_width: int = 64,
) -> torch.Tensor:
    return (
        torch.arange(
            block_width,
            device=block_sizes.device,
        )[None, :]
        < block_sizes[:, None]
    ).reshape(-1)


def build_slot_permutation(
    valid_order: torch.Tensor,
    non_pad_index: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    block_width: int = 64,
) -> torch.Tensor:
    """Map clustered valid-token order into fixed-width padded KV slots."""
    if valid_order.ndim != 3:
        raise ValueError("valid_order must have shape [B,H,N]")
    if valid_order.shape[-1] != non_pad_index.numel():
        raise ValueError("valid_order and non_pad_index disagree")
    sequence = block_sizes.numel() * block_width
    valid_slots = slot_valid_mask(
        block_sizes,
        block_width=block_width,
    )
    if int(valid_slots.sum().item()) != non_pad_index.numel():
        raise ValueError("Block sizes do not cover all valid tokens")
    ordered_original = non_pad_index[valid_order]
    permutation = torch.zeros(
        (*valid_order.shape[:-1], sequence),
        dtype=torch.long,
        device=valid_order.device,
    )
    permutation[..., valid_slots] = ordered_original
    return permutation


def permute_bhsd(
    tensor: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim != 4 or permutation.ndim != 3:
        raise ValueError("Expected BHSD tensor and BHS permutation")
    return torch.gather(
        tensor,
        2,
        permutation[..., None].expand(
            -1,
            -1,
            -1,
            tensor.shape[-1],
        ),
    )


def restore_bhsd(
    permuted: torch.Tensor,
    permutation: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    block_width: int = 64,
) -> torch.Tensor:
    """Restore valid query outputs from clustered slots to padded order."""
    valid_slots = slot_valid_mask(
        block_sizes,
        block_width=block_width,
    )
    source = permuted[..., valid_slots, :]
    destination = permutation[..., valid_slots]
    restored = torch.zeros_like(permuted)
    restored.scatter_(
        2,
        destination[..., None].expand(
            -1,
            -1,
            -1,
            permuted.shape[-1],
        ),
        source,
    )
    return restored


def cluster_labels_from_order(
    valid_order: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Return a cluster label for every token in original valid-token order."""
    capacities = block_sizes.long()
    labels_in_cluster_order = torch.repeat_interleave(
        torch.arange(
            capacities.numel(),
            device=capacities.device,
        ),
        capacities,
    )
    if labels_in_cluster_order.numel() != valid_order.shape[-1]:
        raise ValueError("Cluster capacities and valid order disagree")
    labels = torch.empty_like(valid_order)
    labels.scatter_(
        -1,
        valid_order,
        labels_in_cluster_order.expand_as(valid_order),
    )
    return labels
