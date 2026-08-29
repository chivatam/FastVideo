from __future__ import annotations

import torch

from research.hierarchical_vsa.replay import (
    aggregate_execution_scores,
    select_exec128_under_budget,
    select_exec64_fixed_histogram,
)


def test_mass_aggregation_is_size_weighted_logsumexp() -> None:
    scores = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]])
    sizes = torch.tensor([8, 8, 8, 0])
    result = aggregate_execution_scores(
        scores,
        sizes,
        score_width=8,
        execution_width=16,
        aggregation="mass",
    )
    expected = torch.tensor(
        [[[[torch.log(torch.tensor(16.0)), 1.0 + torch.log(torch.tensor(8.0))]]]]
    ).reshape_as(result)
    assert torch.allclose(result, expected, atol=1e-6)


def test_exec64_reproduces_native_size_histogram() -> None:
    sizes = torch.tensor([64] * 130 + [32] * 20 + [16] * 20 + [8] * 20)
    scores = torch.arange(
        sizes.numel(),
        dtype=torch.float32,
    ).view(1, 1, 1, -1)
    native = torch.cat(
        [
            torch.arange(100),
            torch.arange(130, 145),
            torch.arange(150, 157),
            torch.arange(170, 173),
        ]
    ).view(1, 1, 1, -1)
    selected = select_exec64_fixed_histogram(
        scores,
        sizes,
        native,
    )
    assert selected.shape[-1] == 125
    assert torch.equal(
        torch.sort(sizes[selected.long()], dim=-1).values,
        torch.sort(sizes[native], dim=-1).values,
    )


def test_exec128_never_exceeds_valid_budget() -> None:
    scores = torch.arange(80, dtype=torch.float32).view(1, 1, 1, -1)
    sizes = torch.tensor([128] * 60 + [64] * 10 + [32] * 10)
    target = torch.tensor([[[6500]]])
    selected, tokens = select_exec128_under_budget(
        scores,
        sizes,
        target,
        selected_groups=62,
    )
    assert selected.shape[-1] == 62
    assert int(tokens.item()) <= int(target.item())


def test_exec128_brackets_high_magnitude_scores() -> None:
    scores = (
        torch.arange(80, dtype=torch.float32) * 10_000
    ).view(1, 1, 1, -1)
    sizes = torch.tensor([128] * 60 + [64] * 10 + [32] * 10)
    target = torch.tensor([[[6500]]])
    selected, tokens = select_exec128_under_budget(
        scores,
        sizes,
        target,
        selected_groups=62,
    )
    assert selected.shape[-1] == 62
    assert int(tokens.item()) <= int(target.item())
