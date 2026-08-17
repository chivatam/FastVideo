"""Explain a dumped F2 budget violation by replaying VSA's bisection in fp32.

``_check_budget`` saves the offending score row. This reproduces
``_fused_topk_mask_kernel``'s threshold search on that row in plain PyTorch so the
exact mechanism is visible: where the 32-iteration bisection lands, how many
scores fall strictly above it, and whether the tie-fill can still hit the budget.

    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f2_explain_violation.py <dump.pt>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Bisection:
    """The kernel's threshold search, unpacked so each step can be inspected."""

    threshold: float
    n_above: int
    n_at: int
    n_needed_at_threshold: int
    kernel_selected: int
    trace: list[tuple[int, float, float, float, int]] = field(default_factory=list)


def replay_bisection(scores_f32: torch.Tensor, topk: int, iterations: int = 32) -> Bisection:
    """Mirror the Triton kernel's threshold search exactly (see fused_compress_topk.py)."""
    valid = torch.ones_like(scores_f32, dtype=torch.bool)
    finite = valid & (scores_f32 > float("-inf"))
    lo = torch.where(finite, scores_f32, torch.tensor(float("inf"))).min()
    hi = torch.where(valid, scores_f32, torch.tensor(float("-inf"))).max()
    lo = torch.minimum(lo, hi)
    trace: list[tuple[int, float, float, float, int]] = []
    for i in range(iterations):
        mid = (lo + hi) * 0.5
        count_ge = int((scores_f32 >= mid).sum())
        if i < 4 or i >= iterations - 3:
            trace.append((i, float(lo), float(hi), float(mid), count_ge))
        if count_ge >= topk:
            lo = mid
        else:
            hi = mid
    threshold = float(lo)
    n_above = int((scores_f32 > threshold).sum())
    n_at = int((scores_f32 == threshold).sum())
    return Bisection(
        threshold=threshold,
        n_above=n_above,
        n_at=n_at,
        n_needed_at_threshold=topk - n_above,
        kernel_selected=n_above + max(0, min(n_at, topk - n_above)),
        trace=trace,
    )


def main() -> int:
    dump = Path(sys.argv[1])
    payload = torch.load(dump, map_location="cpu")
    scores = payload["scores_row"]
    topk = int(payload["topk"])
    print(f"dump: {dump.name}")
    print(f"arm={payload['arm_id']} rule={payload['rule']} dtype={scores.dtype} index={payload['index']}")
    print(f"layer={payload['layer_idx']} timestep={payload['timestep']} "
          f"cfg={payload['cfg_branch']} sparsity={payload['sparsity']} topk={topk}")
    print(f"mask selected: {int(payload['mask_row'].sum())} (budget {topk})")
    scores_f32 = scores.float()
    ordered = scores_f32.sort(descending=True).values
    kth, next_val = float(ordered[topk - 1]), float(ordered[topk])
    print("\nscore row:")
    print(f"  n={scores.numel()} distinct={int(torch.unique(scores_f32).numel())} "
          f"min={float(scores_f32.min()):.6g} max={float(scores_f32.max()):.6g}")
    print(f"  k-th largest={kth:.9g}  (k+1)-th={next_val:.9g}  gap={kth - next_val:.3g}")
    print(f"  count(> kth)={int((scores_f32 > kth).sum())}  count(== kth)={int((scores_f32 == kth).sum())}")

    result = replay_bisection(scores_f32, topk)
    print("\nreplayed kernel bisection:")
    for i, lo, hi, mid, count_ge in result.trace:
        print(f"  iter {i:>2}: lo={lo:.9g} hi={hi:.9g} mid={mid:.9g} count_ge={count_ge}")
    print(f"  threshold={result.threshold:.9g}")
    print(f"  n_above={result.n_above}  n_at_threshold={result.n_at}  "
          f"n_needed_at_threshold={result.n_needed_at_threshold}")
    print(f"  => kernel would select {result.kernel_selected} (budget {topk})")

    if result.n_above > topk:
        print("\nDIAGNOSIS: more scores lie strictly above the converged threshold than the budget, so the\n"
              "tie-fill cannot compensate (n_needed is negative) and the kernel over-selects. The 32-iteration\n"
              "fp32 bisection did not resolve the k-th value — check the score range against bf16 ULP.")
    elif result.kernel_selected == topk:
        print("\nNOTE: replay hits the budget exactly, so the divergence is not in the threshold search itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
