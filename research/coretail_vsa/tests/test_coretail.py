from __future__ import annotations

import json
from pathlib import Path

import torch

from research.coretail_vsa.calibrate import _linear_p10
from research.coretail_vsa.prompts import select_external_prompts
from research.coretail_vsa.selection import select_coretail_support
from research.fine_vsa.fine_attention import child_block_sizes


def test_linear_p10_uses_frozen_interpolation() -> None:
    values = torch.arange(32, dtype=torch.float32).view(32, 1)
    assert torch.allclose(_linear_p10(values), torch.tensor([3.1]))


def test_external_prompt_split_is_disjoint_and_deterministic(
    tmp_path: Path,
) -> None:
    source = {
        f"prompt {index}": {
            "caption": f"prompt {index}",
            "dimension": ["test"],
        }
        for index in range(60)
    }
    evaluation = [{"prompt": "prompt 0"}]
    source_path = tmp_path / "source.json"
    evaluation_path = tmp_path / "evaluation.json"
    source_path.write_text(json.dumps(source))
    evaluation_path.write_text(json.dumps(evaluation))
    first = select_external_prompts(
        source_path,
        evaluation_path,
        calibration_count=32,
        heldout_count=8,
    )
    second = select_external_prompts(
        source_path,
        evaluation_path,
        calibration_count=32,
        heldout_count=8,
    )
    calibration, heldout = first
    assert calibration["prompt_id"].tolist() == second[0][
        "prompt_id"
    ].tolist()
    assert heldout["prompt_id"].tolist() == second[1][
        "prompt_id"
    ].tolist()
    assert not (
        set(calibration["prompt_sha256"])
        & set(heldout["prompt_sha256"])
    )
    assert "prompt 0" not in set(calibration["prompt"])
    assert "prompt 0" not in set(heldout["prompt"])


def test_coretail_selection_matches_native_valid_tokens() -> None:
    parent_sizes = torch.full((128,), 64, dtype=torch.int32)
    parent_sizes[-1] = 8
    child_sizes = child_block_sizes(parent_sizes, 8)
    query_shape = (1, 2, 3)
    parent_scores = torch.full((*query_shape, 128), -10.0)
    parent_scores[..., :125] = torch.arange(
        125,
        dtype=torch.float32,
    )
    child_scores = torch.randn(*query_shape, child_sizes.numel())
    core = torch.arange(31).view(1, 1, 1, 31).expand(
        *query_shape,
        31,
    )
    selection = select_coretail_support(
        child_scores,
        child_sizes,
        parent_scores,
        parent_sizes,
        core,
    )
    assert torch.equal(
        selection.selected_actual_kv_tokens,
        selection.native_actual_kv_tokens,
    )
    assert selection.duplicate_valid_tokens.max().item() == 0
    assert selection.selected_actual_kv_tokens.max().item() == 8000


def test_coretail_projects_static_core_into_ragged_native_budget() -> None:
    parent_sizes = torch.full((624,), 8, dtype=torch.int32)
    parent_sizes[:31] = 64
    child_sizes = child_block_sizes(parent_sizes, 8)
    query_shape = (1, 1, 1)
    parent_scores = torch.zeros((*query_shape, 624))
    parent_scores[..., 100:225] = 10.0
    child_scores = torch.randn(*query_shape, child_sizes.numel())
    core = torch.arange(31).view(1, 1, 1, 31)
    selection = select_coretail_support(
        child_scores,
        child_sizes,
        parent_scores,
        parent_sizes,
        core,
    )
    assert selection.native_actual_kv_tokens.item() == 1000
    assert selection.core_active_parent_blocks.item() == 15
    assert selection.core_actual_kv_tokens.item() == 960
    assert selection.fine_tail_actual_kv_tokens.item() == 40
    assert selection.selected_actual_kv_tokens.item() == 1000
    assert selection.duplicate_valid_tokens.item() == 0
