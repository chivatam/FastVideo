from __future__ import annotations

import torch

from research.br_vsa.sensitivity import _per_head_metrics


def test_per_head_metrics_zero_for_equal_outputs() -> None:
    dense = torch.randn(2, 3, 5, 4)
    metrics = _per_head_metrics(dense, dense)
    torch.testing.assert_close(metrics[0], torch.zeros(3))
    torch.testing.assert_close(metrics[1], torch.zeros(3))
    torch.testing.assert_close(metrics[2], torch.zeros(3))


def test_per_head_metrics_are_reduced_independently() -> None:
    dense = torch.ones(1, 2, 3, 4)
    output = dense.clone()
    output[:, 1] = 0
    metrics = _per_head_metrics(output, dense)
    torch.testing.assert_close(metrics[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(metrics[1], torch.tensor([0.0, 1.0]))
