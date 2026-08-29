from __future__ import annotations

import torch

from research.cluster_vsa.clustering import (
    balanced_recursive_order,
    build_slot_permutation,
    cluster_labels_from_order,
    permute_bhsd,
    restore_bhsd,
)


def test_balanced_order_is_deterministic_permutation() -> None:
    vectors = torch.tensor(
        [
            [
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [1.0, 1.0],
                    [1.1, 1.0],
                    [-1.0, -1.0],
                ]
            ]
        ]
    )
    first = balanced_recursive_order(vectors, leaf_size=2)
    second = balanced_recursive_order(vectors, leaf_size=2)
    assert torch.equal(first.valid_order, second.valid_order)
    assert torch.equal(
        torch.sort(first.valid_order, dim=-1).values,
        torch.arange(5).view(1, 1, 5),
    )


def test_slot_permutation_and_restore_round_trip() -> None:
    block_sizes = torch.tensor([3, 2])
    non_pad = torch.tensor([0, 1, 2, 4, 5])
    order = torch.tensor([[[4, 3, 2, 1, 0]]])
    permutation = build_slot_permutation(
        order,
        non_pad,
        block_sizes,
        block_width=4,
    )
    tensor = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
    permuted = permute_bhsd(tensor, permutation)
    restored = restore_bhsd(
        permuted,
        permutation,
        block_sizes,
        block_width=4,
    )
    assert torch.equal(restored[..., non_pad, :], tensor[..., non_pad, :])


def test_cluster_labels_match_capacity_boundaries() -> None:
    order = torch.tensor([[[2, 0, 3, 1, 4]]])
    labels = cluster_labels_from_order(
        order,
        torch.tensor([2, 3]),
    )
    assert labels.tolist() == [[[0, 1, 0, 1, 1]]]
