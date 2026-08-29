from __future__ import annotations

import numpy as np

from research.br_vsa.allocation import solve_exact_multiple_choice


def test_exact_multiple_choice_honors_budget_and_objective() -> None:
    candidate_k = np.array([1, 2, 4])
    errors = np.array(
        [
            [4.0, 2.0, 0.0],
            [3.0, 2.5, 0.0],
        ]
    )
    selected, objective = solve_exact_multiple_choice(
        errors,
        candidate_k,
        budget=5,
    )
    assert selected.tolist() == [4, 1]
    assert objective == 3.0
