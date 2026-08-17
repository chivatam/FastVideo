"""Mask-comparison and damage metrics for the SparseFP4 paper-validation study.

Split out of the backend so the same code is exercised by a GPU self-test with
hand-built masks, where the right answers are known by construction.

The metric definitions follow ``references/FOLLOWUP_SPEC.md`` exactly, including
two details that are easy to get wrong and that study 1 called out:

1. **Masks are equal-size**, so ``recall`` and ``precision`` are the same number
   and ``jaccard = recall / (2 - recall)``. Both are emitted for readers'
   convenience but they are *one* measurement; the analysis must never cite them
   as mutual corroboration.
2. **Wrong-mask excess is a paired per-cell difference**, aggregated as a
   distribution of differences — not a difference of independently pooled
   medians, which would discard the pairing that makes the comparison powerful.
"""

from __future__ import annotations

import hashlib

import torch


def mask_hash(mask: torch.Tensor) -> str:
    """Short stable digest of a bool mask, for pairing/provenance checks."""
    packed = mask.detach().to(torch.uint8).cpu().contiguous().numpy().tobytes()
    return hashlib.blake2b(packed, digest_size=8).hexdigest()


def mask_comparison(candidate: torch.Tensor,
                    reference: torch.Tensor,
                    allow_budget_mismatch: bool = False) -> dict[str, float | int]:
    """Per-head mask agreement between two ``[n_q_blocks, n_k_blocks]`` masks.

    Precision should change *which* blocks are selected, never *how many*
    (FOLLOWUP_SPEC rule 5), so by default an unequal per-query-block count raises:
    for the F1 scorer probe, where selection is a pure ``topk``, it can only mean a
    bug.

    ``allow_budget_mismatch=True`` is for F2, where the selector under study is
    VSA's own ``fused_topk_mask``. That kernel really does return ``topk + 1`` blocks
    on rows whose k-th and (k+1)-th scores tie exactly, because its 32-iteration fp32
    bisection converges toward the k-th value from below and never lands on it, so
    both tied scores test as strictly above the threshold and the tie-fill branch is
    skipped (reproduced in ``f2_kernel_topk_bug.py``). That is a property of the
    object being measured, not an error in measuring it, so F2 records the deviation
    per row instead of aborting — and ``f2_aggregate`` gates on the rate staying
    negligible.
    """
    if candidate.shape != reference.shape:
        raise ValueError(f"mask shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}")
    cand_counts = candidate.sum(dim=-1)
    ref_counts = reference.sum(dim=-1)
    budget_mismatch_rows = int((cand_counts != ref_counts).sum().item())
    if budget_mismatch_rows and not allow_budget_mismatch:
        disagreeing = (cand_counts != ref_counts).nonzero()
        first = tuple(int(v) for v in disagreeing[0].tolist())
        raise RuntimeError("candidate and reference masks retain different block counts; "
                           "the routing budget must be identical across precision arms "
                           f"(rows_disagreeing={budget_mismatch_rows}/{cand_counts.numel()}, first_row={first}, "
                           f"candidate_count={int(cand_counts[first])}, reference_count={int(ref_counts[first])}, "
                           f"candidate_count_range=({int(cand_counts.min())},{int(cand_counts.max())}), "
                           f"reference_count_range=({int(ref_counts.min())},{int(ref_counts.max())}))")
    intersection = int((candidate & reference).sum().item())
    union = int((candidate | reference).sum().item())
    reference_total = int(ref_counts.sum().item())
    swaps_per_query = (reference & ~candidate).sum(dim=-1)
    changed_query_blocks = int((swaps_per_query > 0).sum().item())
    n_q_blocks = int(candidate.shape[-2])
    return {
        "intersection": intersection,
        "union": union,
        "selected_reference": reference_total,
        "selected_candidate": int(cand_counts.sum().item()),
        "recall": intersection / max(1, reference_total),
        "jaccard": intersection / max(1, union),
        "num_swaps": int(swaps_per_query.sum().item()),
        "swaps_per_query_block": float(swaps_per_query.to(torch.float64).mean().item()),
        "max_swaps_in_a_query_block": int(swaps_per_query.max().item()),
        "frac_query_blocks_changed": changed_query_blocks / max(1, n_q_blocks),
        "frac_decisions_changed": int((candidate != reference).sum().item()) / max(1, candidate.numel()),
        "budget_mismatch_rows": budget_mismatch_rows,
        "budget_mismatch_frac": budget_mismatch_rows / max(1, cand_counts.numel()),
        "budget_excess_blocks": int((cand_counts - ref_counts).clamp(min=0).sum().item()),
    }


def spearman_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation over the last axis, averaged over leading axes.

    Ties get average ranks. That matters here rather than being a formality: a
    low-precision scorer produces *genuine* ties — they are part of what the
    phase measures — and plain ``argsort`` ranking would break them arbitrarily
    and report a spuriously low correlation for exactly the arms under study.

    Tie groups are resolved with a segmented scatter-add rather than a Python
    walk over runs. The walk was correct but cost ~200 ms per call at
    ``n_k=512``, which made Spearman the single most expensive metric in the
    phase and would have forced sampling it so thinly as to be uninformative.
    """

    def ranks(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        order = x.argsort(dim=-1)
        sorted_values = x.gather(-1, order)
        positions = torch.arange(n, device=x.device, dtype=torch.float64).expand_as(x)

        # Label maximal runs of equal values, then give every member of a run the
        # mean of that run's positions.
        starts_new_group = torch.ones_like(sorted_values, dtype=torch.bool)
        starts_new_group[..., 1:] = sorted_values[..., 1:] != sorted_values[..., :-1]
        group = starts_new_group.to(torch.int64).cumsum(dim=-1) - 1
        position_sum = torch.zeros_like(positions).scatter_add_(-1, group, positions)
        group_size = torch.zeros_like(positions).scatter_add_(-1, group, torch.ones_like(positions))
        mean_rank_sorted = (position_sum / group_size.clamp(min=1.0)).gather(-1, group)
        return torch.empty_like(positions).scatter_(-1, order, mean_rank_sorted)

    ra, rb = ranks(a.to(torch.float64)), ranks(b.to(torch.float64))
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    numerator = (ra * rb).sum(dim=-1)
    denominator = (ra.pow(2).sum(dim=-1).sqrt() * rb.pow(2).sum(dim=-1).sqrt()).clamp(min=1e-30)
    return float((numerator / denominator).mean().item())


def boundary_diagnostics(reference_scores: torch.Tensor, candidate_mask: torch.Tensor, reference_mask: torch.Tensor,
                         k: int) -> dict[str, float | int | None]:
    """FP64-shadow boundary metrics for one head.

    ``reference_scores`` must be the **fp64 shadow** scores, not the deployed
    lower-precision ones (FOLLOWUP_SPEC "Boundary diagnostics", and study 1's
    trap 8: fp32 scores land margins on a power-of-two grid and manufacture
    ~110 exact ties per cell, which would contaminate exactly this measurement).

    ``normalized_pair_gap`` is the score distance between the blocks a candidate
    dropped and those it added, in units of that query block's score spread. Near
    zero means the swap happened at a near-degenerate boundary — the mechanism
    study 1 proposed. This is the number that decides whether a *low-precision
    scorer* still only errs where erring is free.
    """
    ordered = torch.sort(reference_scores, dim=-1, descending=True, stable=True).values
    margin = ordered[..., k - 1] - ordered[..., k]
    spread = (ordered[..., 0] - ordered[..., -1]).clamp(min=1e-300)
    dropped = reference_mask & ~candidate_mask
    added = ~reference_mask & candidate_mask
    n_dropped = dropped.sum(dim=-1)
    n_added = added.sum(dim=-1)
    rows_with_swap = (n_dropped > 0) & (n_added > 0)

    # Mean dropped score minus mean added score, per query block, in units of that
    # block's score spread. Vectorized over query blocks: the equivalent Python
    # loop cost ~6 ms per head, which across 12 heads x 12 arms was the phase's
    # second-largest cost after the sparse kernel itself.
    drop_mean = (reference_scores * dropped).sum(dim=-1) / n_dropped.clamp(min=1)
    add_mean = (reference_scores * added).sum(dim=-1) / n_added.clamp(min=1)
    normalized_gap = ((drop_mean - add_mean).abs() / spread)[rows_with_swap]
    gap_tensor = normalized_gap if normalized_gap.numel() else None

    return {
        "reference_margin_fp64_median": float(margin.median().item()),
        "reference_margin_fp64_min": float(margin.min().item()),
        "reference_margin_norm_fp64_median": float((margin / spread).median().item()),
        "exact_ties_fp64": int((margin == 0).sum().item()),
        "tie_denominator_query_blocks": int(margin.numel()),
        "changed_pair_gap_fp64_median": (None if gap_tensor is None else float(gap_tensor.median().item())),
        "changed_pair_gap_fp64_p90": (None if gap_tensor is None else float(gap_tensor.quantile(0.90).item())),
        "changed_pair_gap_fp64_max": (None if gap_tensor is None else float(gap_tensor.max().item())),
        "n_query_blocks_with_swap": int(rows_with_swap.sum().item()),
    }


def deployed_tie_count(scores: torch.Tensor, k: int) -> int:
    """Exact ``s_(k) == s_(k+1)`` ties in the *deployed* (possibly low) precision.

    Reported alongside the fp64 shadow count so a low-precision scorer's own tie
    behaviour is visible — a low-precision scorer that manufactures thousands of
    ties is making its selection partly by tie-break rule rather than by score,
    and that is a finding, not a nuisance.
    """
    ordered = torch.sort(scores, dim=-1, descending=True, stable=True).values
    return int((ordered[..., k - 1] == ordered[..., k]).sum().item())
