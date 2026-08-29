from __future__ import annotations

import json

from research.br_vsa.attention import BudgetRedistributedPolicy


def test_policy_validates_exact_budget(tmp_path) -> None:
    path = tmp_path / "table.json"
    path.write_text(
        json.dumps(
            {
                "candidate_K": [32, 125],
                "native_budget": 250,
                "allocated_budget": 157,
                "num_blocks": 624,
                "granularity": "step_layer_head",
                "k_table": [[[32, 125]]],
            }
        )
    )
    policy = BudgetRedistributedPolicy.from_path(path)
    assert policy.k_for(0, 0) == (32, 125)
