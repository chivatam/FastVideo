from __future__ import annotations

import torch

from research.adaptive_vsa_deadline.adaptive_attention import (
    AdaptiveVSAPolicy,
    select_adaptive_mask,
)


def test_uniform_scores_fall_back_to_dense_for_high_mass_threshold() -> None:
    scores = torch.zeros(1, 1, 2, 10)
    policy = AdaptiveVSAPolicy(
        retained_mass_threshold=0.95,
        maximum_sparsity=0.8,
    )

    mask, decision = select_adaptive_mask(scores, policy)

    assert torch.equal(decision.selected_topk, torch.full((1, 1, 2), 10))
    assert torch.equal(mask.sum(dim=-1), decision.selected_topk)


def test_concentrated_scores_keep_vsa80_budget() -> None:
    scores = torch.full((1, 1, 1, 10), -20.0)
    scores[..., :2] = 20.0
    policy = AdaptiveVSAPolicy(
        retained_mass_threshold=0.95,
        maximum_sparsity=0.8,
    )

    mask, decision = select_adaptive_mask(scores, policy)

    assert decision.selected_topk.item() == 2
    assert mask.sum().item() == 2


def test_rows_can_choose_different_budgets() -> None:
    scores = torch.zeros(1, 1, 2, 10)
    scores[..., 0, :2] = 20.0
    policy = AdaptiveVSAPolicy(
        retained_mass_threshold=0.95,
        maximum_sparsity=0.8,
    )

    mask, decision = select_adaptive_mask(scores, policy)

    assert decision.selected_topk.flatten().tolist() == [2, 10]
    assert torch.equal(mask.sum(dim=-1), decision.selected_topk)


def test_conservative_floor_excludes_vsa80_budget() -> None:
    policy = AdaptiveVSAPolicy(
        retained_mass_threshold=0.95,
        maximum_sparsity=0.7,
    )

    assert policy.budget_options(10)[0] == (4, 0.7)
