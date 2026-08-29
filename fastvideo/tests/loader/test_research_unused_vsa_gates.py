from __future__ import annotations

import pytest
import torch

from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict


def _identity(name: str):
    return name, None, None


def _checkpoint():
    yield "weight", torch.tensor([[2.0]])
    yield "blocks.0.to_gate_compress.weight", torch.tensor([3.0])


def test_research_dense_baseline_skips_only_vsa_gate(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    monkeypatch.setenv("FASTVIDEO_RESEARCH_ALLOW_UNUSED_VSA_GATES", "1")
    load_model_from_full_model_state_dict(
        model,
        _checkpoint(),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        training_mode=False,
        param_names_mapping=_identity,
    )
    torch.testing.assert_close(model.weight, torch.tensor([[2.0]]))


def test_research_dense_baseline_keeps_other_unexpected_keys_strict(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    monkeypatch.setenv("FASTVIDEO_RESEARCH_ALLOW_UNUSED_VSA_GATES", "1")

    def checkpoint():
        yield "weight", torch.tensor([[2.0]])
        yield "blocks.0.unrelated.weight", torch.tensor([3.0])

    with pytest.raises(ValueError, match="blocks.0.unrelated.weight"):
        load_model_from_full_model_state_dict(
            model,
            checkpoint(),
            device=torch.device("cpu"),
            param_dtype=torch.float32,
            strict=True,
            training_mode=False,
            param_names_mapping=_identity,
        )
