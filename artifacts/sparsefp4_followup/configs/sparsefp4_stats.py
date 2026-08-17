"""Shared statistics for the follow-up study: cached column access and bootstrap.

Both ``f4_gates.py`` (which decides verdicts) and ``f5_figures.py`` (which draws them)
need the same medians, the same ratio-of-medians, and the same intervals. Keeping one
implementation here means a figure cannot silently disagree with the gate that
validated it.

The unit of replication throughout is the **prompt**: cells within one prompt share a
denoising trajectory, so resampling cells would treat 30 layers x 12 heads as
independent evidence and produce intervals far too narrow.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np

BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20260816


class Cache:
    """Column store over a ``build_stats_cache.py`` ``.npz``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = np.load(path, allow_pickle=False)

    def num(self, name: str) -> np.ndarray:
        return self.data[f"num_{name}"]

    def integer(self, name: str) -> np.ndarray:
        return self.data[f"int_{name}"]

    def codes(self, name: str) -> np.ndarray:
        return self.data[f"cat_{name}_codes"]

    def levels(self, name: str) -> list[str]:
        return [str(value) for value in self.data[f"cat_{name}_levels"]]

    def code_of(self, name: str, value: str) -> int:
        found = np.nonzero(self.data[f"cat_{name}_levels"] == value)[0]
        return int(found[0]) if len(found) else -1

    def select(self, arm: str, sparsity: float, seed: int | None = None) -> np.ndarray:
        code = self.code_of("arm", arm)
        if code < 0:
            return np.zeros(len(self.num("jaccard")), dtype=bool)
        mask = (self.codes("arm") == code) & (self.num("sparsity") == sparsity)
        if seed is not None:
            mask &= self.integer("seed") == seed
        return mask


def damage_shares(cache: Cache, mask: np.ndarray) -> np.ndarray:
    """Per-cell ``|wrong_mask_excess| / sparsification_error``, NaN where undefined."""
    wrong = cache.num("wrong_mask_excess")[mask]
    sparsification = cache.num("sparsification_error")[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.abs(wrong) / sparsification
    share[~np.isfinite(share)] = np.nan
    return share


def group_statistics(cache: Cache, mask: np.ndarray) -> dict[str, float | None]:
    """Median Jaccard, median damage share, and the ratio-of-medians isolation."""
    if not mask.any():
        return {"jaccard": None, "share": None, "isolation": None}
    wrong = cache.num("wrong_mask_excess")[mask]
    random_excess = cache.num("random_matched_excess")[mask]
    median_wrong = float(np.nanmedian(wrong))
    median_random = float(np.nanmedian(random_excess)) if np.isfinite(random_excess).any() else float("nan")
    return {
        "jaccard": float(np.nanmedian(cache.num("jaccard")[mask])),
        "share": float(np.nanmedian(damage_shares(cache, mask))),
        "isolation": (abs(median_random) / abs(median_wrong) if median_wrong and np.isfinite(median_random) else None),
    }


def bootstrap_over_prompts(
    cache: Cache,
    mask: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | None]:
    """Percentile bootstrap of ``statistic`` over resampled prompts.

    ``statistic`` receives positional indices into the selected cells, so it can
    compute a median, a ratio of medians, or anything else on the resampled pool.
    """
    if not mask.any():
        return {"point": None, "ci_low": None, "ci_high": None, "n_prompts": 0}
    prompt_codes = cache.codes("prompt_id")[mask]
    prompts = np.unique(prompt_codes)
    by_prompt = [np.nonzero(prompt_codes == prompt)[0] for prompt in prompts]
    point = statistic(np.arange(int(mask.sum())))
    if len(prompts) < 2:
        return {"point": point, "ci_low": None, "ci_high": None, "n_prompts": int(len(prompts))}

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picks = rng.integers(0, len(prompts), size=len(prompts))
        draws[index] = statistic(np.concatenate([by_prompt[pick] for pick in picks]))
    clean = draws[np.isfinite(draws)]
    if clean.size == 0:
        return {"point": point, "ci_low": None, "ci_high": None, "n_prompts": int(len(prompts))}
    return {
        "point": point,
        "ci_low": float(np.quantile(clean, 0.025)),
        "ci_high": float(np.quantile(clean, 0.975)),
        "n_prompts": int(len(prompts)),
        "n_resamples": int(clean.size),
    }


def share_and_isolation_statistics(
        cache: Cache, mask: np.ndarray) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], float]]:
    """Closures computing the damage share and isolation ratio over cell indices."""
    wrong = cache.num("wrong_mask_excess")[mask]
    random_excess = cache.num("random_matched_excess")[mask]
    share = damage_shares(cache, mask)

    def share_statistic(index: np.ndarray) -> float:
        return float(np.nanmedian(share[index]))

    def isolation_statistic(index: np.ndarray) -> float:
        median_wrong = np.nanmedian(wrong[index])
        median_random = np.nanmedian(random_excess[index])
        if not np.isfinite(median_wrong) or median_wrong == 0 or not np.isfinite(median_random):
            return float("nan")
        return float(abs(median_random) / abs(median_wrong))

    return share_statistic, isolation_statistic


def interval_side(low: float | None, high: float | None, threshold: float) -> str | None:
    """Which side of ``threshold`` an entire interval falls on, or ``"straddles"``."""
    if low is None or high is None:
        return None
    if low > threshold:
        return "above"
    if high < threshold:
        return "below"
    return "straddles"
