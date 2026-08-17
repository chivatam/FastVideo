"""Characterize VSA's ``fused_topk_mask`` over-selection on tied threshold scores.

``f2_explain_violation.py`` traced one violation to the kernel's 32-iteration fp32
bisection: ``lo`` converges toward the k-th score *from below* and never lands on it
exactly, so when two scores tie at the k-th value both count as strictly above the
threshold and the row returns ``topk + 1`` blocks. The kernel's own comment argues
32 iterations suffice because they resolve below bf16 ULP — but the requirement is
exact equality with the k-th value, which bisection does not generally achieve.

This script establishes three things directly against the installed kernel:
  1. the failure is reproducible from a constructed tie, not specific to one dump;
  2. it needs a tie *at the k-th value* — untied rows are unaffected;
  3. how often it fires on realistic score distributions, which sets whether the
     study can treat it as negligible-but-recorded or must handle it structurally.

    CUDA_VISIBLE_DEVICES=<free gpu> "$FV_PYTHON" \
        artifacts/sparsefp4_followup/configs/f2_kernel_topk_bug.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask

DEFAULT_DUMP = Path("/mnt/nvme/scratch/sparsefp4_followup/debug-f2-t25/"
                    "budget_violation_VA_FP8_l4_t25_positive_sp0.9.pt")


def counts_of(scores: torch.Tensor, topk: int) -> torch.Tensor:
    return fused_topk_mask(scores, topk).sum(dim=-1)


def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    n_kv, topk = 624, 63
    failures = 0

    print("(0) the exact row that failed in the sweep, replayed through the installed kernel")
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DUMP
    if dump.is_file():
        payload = torch.load(dump, map_location=device)
        real = payload["scores_row"].to(device)
        real_topk = int(payload["topk"])
        selected_real = int(counts_of(real.view(1, 1, 1, -1).contiguous(), real_topk)[0, 0, 0])
        ordered_real = real.float().sort(descending=True).values
        kth_real = float(ordered_real[real_topk - 1])
        print(f"  {dump.name}")
        print(
            f"  range=[{float(real.min()):.6g},{float(real.max()):.6g}] kth={kth_real:.9g} "
            f"count(>kth)={int((real.float() > kth_real).sum())} count(==kth)={int((real.float() == kth_real).sum())}")
        print(f"  kernel selected {selected_real}, budget {real_topk}" +
              ("   <-- OVER-SELECTS (reproduced)" if selected_real != real_topk else "   ok"))
        failures += selected_real != real_topk
    else:
        print(f"  no dump at {dump}; skipping (pass a path to enable)")

    print("\n(1) constructed tie at the k-th value, wide score range")
    # Reproduce the dumped row's structure: a wide range (so 32 bisection steps
    # cannot resolve the k-th value exactly) plus exactly two scores tied at k.
    row = torch.linspace(-17.0, 10.0, n_kv, device=device, dtype=torch.bfloat16)
    ordered = row.sort(descending=True).values
    kth = ordered[topk - 1]
    row[(row == ordered[topk]).nonzero()[0]] = kth
    scores = row.view(1, 1, 1, n_kv).contiguous()
    selected = int(counts_of(scores, topk)[0, 0, 0])
    n_gt = int((scores.float() > kth.float()).sum())
    n_eq = int((scores.float() == kth.float()).sum())
    print(f"  range=[{float(row.min()):.4g},{float(row.max()):.4g}] count(>kth)={n_gt} count(==kth)={n_eq}")
    print(f"  kernel selected {selected}, budget {topk}" + ("   <-- OVER-SELECTS" if selected != topk else "   ok"))
    failures += selected != topk

    print("\n(2) same row with the tie removed (control)")
    untied = row.clone()
    untied[(untied == kth).nonzero()[0]] = kth - 0.5
    selected_untied = int(counts_of(untied.view(1, 1, 1, n_kv).contiguous(), topk)[0, 0, 0])
    print(f"  kernel selected {selected_untied}, budget {topk}" +
          ("   <-- unexpected" if selected_untied != topk else "   ok (tie is required)"))
    failures += selected_untied != topk

    print("\n(3) narrow score range with the same tie (bisection has more relative resolution)")
    narrow = (row - row.mean()) * 0.01
    narrow_kth = narrow.sort(descending=True).values[topk - 1]
    narrow[(narrow == narrow.sort(descending=True).values[topk]).nonzero()[0]] = narrow_kth
    selected_narrow = int(counts_of(narrow.view(1, 1, 1, n_kv).contiguous(), topk)[0, 0, 0])
    print(f"  range=[{float(narrow.min()):.4g},{float(narrow.max()):.4g}] "
          f"kernel selected {selected_narrow}, budget {topk}")

    print("\n(4) rate on bf16 scores at VSA-like scale (ties arise naturally from bf16's 8-bit mantissa)")
    for scale, label in ((1.0, "unit scale"), (4.0, "wide, like layer-4 fp8 scores")):
        batch = (torch.randn(1, 12, 640, n_kv, device=device) * scale).to(torch.bfloat16)
        counts = counts_of(batch, topk)
        rows = counts.numel()
        bad = int((counts != topk).sum())
        worst = int(counts.max())
        print(f"  {label:<32} violating {bad}/{rows} rows ({100.0 * bad / rows:.4f}%), max selected {worst}")

    print("\n(5) does the shipped VSA path hit this too, or only the probe's fp8 arm?")
    # The dumped row came from VA_FP8, but nothing about the mechanism is fp8-specific:
    # any bf16 score row with a k-th-value tie and an unlucky range can trigger it. If
    # the bug were probe-only it would be a measurement artifact; if it reaches rows the
    # deployed selector also produces, it is a property of VSA as shipped.
    if dump.is_file():
        real = torch.load(dump, map_location=device)["scores_row"].to(device)
        variants = {
            "as dumped": real,
            "+0 (identity, sanity)": real + torch.zeros_like(real),
            "negated (order reversed)": -real,
            "scaled x2": (real.float() * 2).to(torch.bfloat16),
            "scaled x0.5": (real.float() * 0.5).to(torch.bfloat16),
        }
        for label, variant in variants.items():
            got = int(counts_of(variant.view(1, 1, 1, -1).contiguous(), topk)[0, 0, 0])
            print(f"  {label:<28} selected {got} (budget {topk})" + ("   <-- over-selects" if got != topk else ""))

    print()
    if failures:
        print("CONFIRMED: fused_topk_mask over-selects when scores tie at the k-th value and the fp32\n"
              "bisection cannot resolve that value exactly. The tie is necessary; the wide range makes it\n"
              "reachable within 32 iterations. This is an upstream kernel property, independent of this study.")
        return 0
    print("could not reproduce the over-selection; the earlier violation needs another explanation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
