from __future__ import annotations

import pandas as pd

from research.br_vsa.analyze_census import (
    _concentration,
    _spearman,
)


def test_spearman_detects_identical_and_reversed_rankings() -> None:
    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _spearman(left, left) == 1.0
    assert _spearman(left, left.iloc[::-1].reset_index(drop=True)) == -1.0


def test_concentration_reports_dominant_unit() -> None:
    frame = pd.DataFrame(
        {
            "relative_L2_error_mean": [90.0, *([10.0 / 9.0] * 9)],
        }
    )
    result = _concentration(frame)
    top20 = result.loc[
        result["top_percent"].eq(20),
        "error_fraction",
    ].iloc[0]
    assert top20 > 0.90
