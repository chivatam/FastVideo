from __future__ import annotations

import torch

from research.ra_vsa_deadline.residual_attention import (
    ResidualAwareVSAPolicy,
    key_heterogeneity_from_pooled,
    select_residual_mask,
)


def test_policy_preserves_exact_native_budget() -> None:
    scores = torch.randn(1, 2, 3, 40)
    probabilities = torch.softmax(scores, dim=-1)
    key_coarse = torch.randn(1, 2, 40, 8) * 0.25
    policy = ResidualAwareVSAPolicy(native_fraction=0.75)

    mask, decision = select_residual_mask(
        scores,
        probabilities,
        key_coarse,
        policy,
    )

    assert decision.total_slots == 8
    assert decision.native_slots == 6
    assert decision.rescue_slots == 2
    assert torch.equal(mask.sum(dim=-1), torch.full((1, 2, 3), 8))


def test_rescue_can_replace_native_tail() -> None:
    scores = torch.tensor([[[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]]]])
    probabilities = torch.softmax(scores, dim=-1)
    key_coarse = torch.ones(1, 1, 8, 4)
    key_coarse[..., 7, :] = 0.0
    policy = ResidualAwareVSAPolicy(
        native_fraction=0.5,
        native_sparsity=0.5,
    )

    mask, decision = select_residual_mask(
        scores,
        probabilities,
        key_coarse,
        policy,
    )

    assert mask.sum().item() == 4
    assert mask[..., 7].item()
    assert decision.replacement_fraction_mean.item() > 0.0


def test_pooled_key_heterogeneity_detects_cancellation() -> None:
    coherent = torch.ones(1, 2, 1, 4)
    cancelled = torch.zeros(1, 2, 1, 4)

    coherent_risk = key_heterogeneity_from_pooled(coherent)
    cancelled_risk = key_heterogeneity_from_pooled(cancelled)

    assert cancelled_risk.item() > coherent_risk.item()


def test_forced_replacement_is_disjoint_from_entire_native_topk() -> None:
    scores = torch.tensor(
        [[[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]]]]
    )
    probabilities = torch.softmax(scores, dim=-1)
    key_coarse = torch.ones(1, 1, 8, 4)
    key_coarse[..., 6:, :] = 0.0
    policy = ResidualAwareVSAPolicy(
        native_fraction=0.5,
        native_sparsity=0.5,
        force_outside_native=True,
    )

    mask, decision = select_residual_mask(
        scores,
        probabilities,
        key_coarse,
        policy,
    )

    native = torch.zeros_like(mask)
    native[..., :4] = True
    rescue = mask & ~native
    assert mask.sum().item() == 4
    assert rescue.sum().item() == 2
    assert not (rescue & native).any()
    assert decision.replacement_count_min.item() == 2
    assert decision.replacement_count_max.item() == 2
    assert decision.replacement_fraction_mean.item() == 0.5
