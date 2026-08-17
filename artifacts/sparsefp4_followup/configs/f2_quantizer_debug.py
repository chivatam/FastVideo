"""Locate the +inf that broke F2's kernel top-k, by layout and precision.

Case 6 of ``f2_budget_debug.py`` showed a single ``+inf`` score makes
``fused_topk_mask``'s fp32 bisection diverge and over-select. This asks which
routing precision produces non-finite values, and whether tensor layout
(``[B,S,H,D]`` as ``quantize_router_input`` is defined vs ``[B,H,S,D]`` as VSA
hands it over) is what triggers it.

    CUDA_VISIBLE_DEVICES=<free gpu> "$FV_PYTHON" \
        artifacts/sparsefp4_followup/configs/f2_quantizer_debug.py
"""

from __future__ import annotations

import sys

import torch

from fastvideo.attention.backends.routing_probe_attn import quantize_router_input


def probe(tag: str, tensor: torch.Tensor, precision: str) -> bool:
    try:
        out, saturation = quantize_router_input(tensor, precision)
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        print(f"  {tag:<34} {precision:<10} raised {type(exc).__name__}: {exc}")
        return False
    finite = bool(torch.isfinite(out).all())
    n_inf = int((~torch.isfinite(out)).sum())
    amax = float(out[torch.isfinite(out)].abs().max()) if finite or n_inf < out.numel() else float("nan")
    status = "ok" if finite else f"NON-FINITE ({n_inf} elems)"
    print(f"  {tag:<34} {precision:<10} shape={tuple(out.shape)} amax={amax:.4g} sat={saturation:.4g} {status}")
    return finite


def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    batch, heads, seq, dim = 1, 12, 39936, 128
    ok = True

    # Wan's bf16 activations at these layers sit around unit scale; include a
    # near-zero head and a large-outlier head since those are what stress scaling.
    bshd = (torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16) * 0.5)
    bshd[:, :, 3] *= 1e-4
    bshd[:, :, 7] *= 40.0

    print("layout [B,S,H,D] (the layout quantize_router_input documents)")
    for precision in ("fp8_e4m3", "nvfp4", "nvfp4_sim"):
        ok &= probe("bshd", bshd, precision)

    print("\nlayout [B,H,S,D] (what VSA hands the probe; the old F2 bug)")
    bhsd = bshd.transpose(1, 2).contiguous()
    for precision in ("fp8_e4m3", "nvfp4", "nvfp4_sim"):
        ok &= probe("bhsd", bhsd, precision)

    print("\nedge case: an all-zero head (scale clamps to tiny)")
    zeroed = bshd.clone()
    zeroed[:, :, 5] = 0.0
    for precision in ("fp8_e4m3", "nvfp4", "nvfp4_sim"):
        ok &= probe("bshd, one zero head", zeroed, precision)

    print("\nedge case: values near bf16 max (scaling headroom)")
    extreme = bshd.clone()
    extreme[:, :, 9] = torch.full_like(extreme[:, :, 9], 3.0e38)
    for precision in ("fp8_e4m3", "nvfp4", "nvfp4_sim"):
        ok &= probe("bshd, near-bf16-max head", extreme, precision)

    print()
    print("all outputs finite" if ok else "at least one configuration produced non-finite routing values")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
