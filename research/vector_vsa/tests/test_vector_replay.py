from __future__ import annotations

import torch

from research.vector_vsa.replay import (
    aggregate_raw_scores,
    select_raw_tokens,
)


def test_raw_score_aggregations() -> None:
    scores = torch.tensor([[[[1.0, 3.0, 2.0, 4.0, 9.0, 0.0, 0.0, 0.0]]]])
    sizes = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])
    maximum, group_sizes = aggregate_raw_scores(
        scores,
        sizes,
        width=8,
        aggregation="max",
    )
    top2, _ = aggregate_raw_scores(
        scores,
        sizes,
        width=8,
        aggregation="top2_mean",
    )
    lse, _ = aggregate_raw_scores(
        scores,
        sizes,
        width=8,
        aggregation="logsumexp",
    )
    assert group_sizes.tolist() == [5]
    assert maximum.item() == 9.0
    assert top2.item() == 6.5
    expected = torch.logsumexp(scores[..., :5].float(), dim=-1)
    assert torch.allclose(lse.squeeze(-1), expected)


def test_raw_selection_matches_variable_targets() -> None:
    scores = torch.tensor([[[[0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]])
    sizes = torch.ones(10, dtype=torch.int32)
    target = torch.tensor([[[8]]])
    selected, counts = select_raw_tokens(
        scores,
        sizes,
        target,
        max_tokens=8,
    )
    assert counts.item() == 8
    assert set(selected.flatten().tolist()) == set(range(1, 9))
