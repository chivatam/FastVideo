"""Reproduce the F2 budget-mismatch failure on synthetic-but-realistic routing inputs.

The full-sweep worker raised "candidate and reference masks retain different block
counts". This isolates which selection rule disagrees on count and why, with no
model in the loop, so the fix can be verified in seconds rather than minutes.

    CUDA_VISIBLE_DEVICES=<free gpu> "$FV_PYTHON" \
        artifacts/sparsefp4_followup/configs/f2_budget_debug.py
"""

from __future__ import annotations

import sys

import torch

from fastvideo.attention.backends.vsa_precision_probe_attn import vsa_select


def report(name: str, mask: torch.Tensor, topk: int) -> tuple[int, int]:
    counts = mask.sum(dim=-1)
    lo, hi = int(counts.min()), int(counts.max())
    flag = "" if (lo == hi == topk) else "   <-- BUDGET VIOLATION"
    print(f"  {name:<46} counts=[{lo},{hi}] want={topk}{flag}")
    return lo, hi


def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    n_q, n_kv, topk = 8, 624, 62
    failures = []

    print("case 1: generic random bf16 scores (few ties)")
    scores = torch.randn(1, 2, n_q, n_kv, device=device, dtype=torch.bfloat16)
    for rule in ("kernel", "exact_index_order", "torch_topk"):
        mask, _ = vsa_select(scores if rule != "exact_index_order" else scores, topk, rule)
        lo, hi = report(rule, mask, topk)
        if not (lo == hi == topk):
            failures.append(("case1", rule, lo, hi))

    # bf16 has ~8 bits of mantissa, so real block scores tie constantly. Force
    # heavy ties to probe the threshold/tie-break path the kernel uses.
    print("\ncase 2: heavy ties (bf16 quantised to few distinct values)")
    tied = (torch.randn(1, 2, n_q, n_kv, device=device) * 2).round().to(torch.bfloat16)
    for rule in ("kernel", "exact_index_order", "torch_topk"):
        mask, _ = vsa_select(tied, topk, rule)
        lo, hi = report(rule, mask, topk)
        if not (lo == hi == topk):
            failures.append(("case2", rule, lo, hi))

    print("\ncase 3: all-equal scores (maximal tie degeneracy)")
    flat = torch.full((1, 2, n_q, n_kv), 0.5, device=device, dtype=torch.bfloat16)
    for rule in ("kernel", "exact_index_order", "torch_topk"):
        mask, _ = vsa_select(flat, topk, rule)
        lo, hi = report(rule, mask, topk)
        if not (lo == hi == topk):
            failures.append(("case3", rule, lo, hi))

    print("\ncase 4: fp64/fp32 scores through exact_index_order (intervention B path)")
    for dtype in (torch.float32, torch.float64):
        mask, _ = vsa_select(tied.to(dtype), topk, "exact_index_order")
        lo, hi = report(f"exact_index_order@{dtype}", mask, topk)
        if not (lo == hi == topk):
            failures.append(("case4", str(dtype), lo, hi))

    print("\ncase 5: topk > n_kv (kernel clamps, sort-based rules must too)")
    over = min(n_kv + 10, n_kv + 10)
    for rule in ("kernel", "exact_index_order", "torch_topk"):
        try:
            mask, _ = vsa_select(tied, over, rule)
            report(f"{rule} (topk={over})", mask, n_kv)
        except Exception as exc:  # noqa: BLE001 - diagnostic surface
            print(f"  {rule:<46} raised {type(exc).__name__}: {exc}")
            failures.append(("case5", rule, -1, -1))

    print("\ncase 6: non-finite scores (does the fp32 bisection converge?)")
    for label, filler in (("one +inf", float("inf")), ("one -inf", float("-inf")), ("one nan", float("nan"))):
        probe = tied.clone()
        probe[0, 0, 0, 3] = filler
        for rule in ("kernel", "exact_index_order"):
            mask, _ = vsa_select(probe, topk, rule)
            lo, hi = report(f"{rule} ({label})", mask, topk)
            if not (lo == hi == topk):
                failures.append(("case6", f"{rule}/{label}", lo, hi))

    print()
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("all selection rules honour the budget exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
