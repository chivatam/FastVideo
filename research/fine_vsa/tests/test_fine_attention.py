from __future__ import annotations

import torch

from research.fine_vsa.fine_attention import child_block_sizes
from research.fine_vsa.replay import select_children_fixed_tokens


def test_child_block_sizes_preserve_valid_tokens() -> None:
    parents = torch.tensor([64, 32, 16, 8], dtype=torch.int32)
    for width in (32, 16, 8):
        children = child_block_sizes(parents, width)
        assert int(children.sum()) == int(parents.sum())
        assert int(children.max()) <= width


def test_nominal_pair_budget_is_fixed() -> None:
    layouts = ((64, 125), (32, 250), (16, 500), (8, 1000))
    assert {width * count for width, count in layouts} == {8000}


def test_fixed_token_selector_matches_native_support() -> None:
    scores = torch.tensor(
        [[[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]]]]
    )
    sizes = torch.tensor([16, 16, 16, 16, 8, 0, 0, 0])
    parent_scores = torch.tensor([[[[1.0, 0.5]]]])
    selected = select_children_fixed_tokens(
        scores,
        sizes,
        selected_blocks=4,
        factor=4,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=torch.tensor([[[40]]]),
        child_width=16,
    )
    assert selected.shape == (1, 1, 1, 4)
    assert sizes[selected.long()].sum().item() == 40
