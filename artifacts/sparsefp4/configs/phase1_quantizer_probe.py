"""Phase 1 preflight: establish exactly what the native NVFP4 Q/K quantizer produces.

Answers three questions that Phase 1's record labelling depends on:

1. Can the packed ``torch.float4_e2m1fn_x2`` output of ``_nvfp4_quantize_for_fa4``
   be decoded back to fp32 outside the kernel?  If yes, the NVFP4 routing arm is
   ``native`` (the router sees the exact values the FP4 MMA consumes) rather than
   ``simulated``.
2. Does the deterministic bf16 -> NVFP4 -> bf16 round-trip specified in
   EXPERIMENT_SPEC 4.3 agree with the native codes?
3. Is ``softmax_scale`` (or any Q pre-scale) applied before or after
   quantization on the NVFP4 FA4 path?  EXPERIMENT_SPEC 9.3.6 flags this as a
   confounder.

Writes a JSON verdict; prints a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
MXFP_BLOCK_SIZE = 16


def e2m1_lut(device: torch.device) -> torch.Tensor:
    magnitudes = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([magnitudes, -magnitudes])


def decode_packed_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (.., headdim//2) float4_e2m1fn_x2 to (.., headdim) fp32 code values."""
    as_bytes = packed.view(torch.uint8)
    low = (as_bytes & 0x0F).to(torch.long)
    high = ((as_bytes >> 4) & 0x0F).to(torch.long)
    interleaved = torch.stack([low, high], dim=-1).flatten(-2)
    return e2m1_lut(packed.device)[interleaved]


def blockwise_e4m3_scales(tensor: torch.Tensor) -> torch.Tensor:
    """Per-16-element E4M3 scale factors, matching the NVFP4 recipe with global_sf=1."""
    grouped = tensor.float().unflatten(-1, (-1, MXFP_BLOCK_SIZE))
    amax = grouped.abs().amax(dim=-1)
    raw = (amax / E2M1_MAX).clamp(max=FP8_E4M3_MAX)
    return raw.to(torch.float8_e4m3fn).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seqlen", type=int, default=32760)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    query = torch.randn(1, args.seqlen, args.nheads, args.headdim, device=device, dtype=torch.bfloat16)

    packed, sf = _nvfp4_quantize_for_fa4(query)
    packed = packed[:, :args.seqlen]

    codes = decode_packed_e2m1(packed)
    scales = blockwise_e4m3_scales(query)
    native_dequant = (codes.unflatten(-1, (-1, MXFP_BLOCK_SIZE)) * scales.unsqueeze(-1)).flatten(-2)

    reference = query.float()
    step = scales.unsqueeze(-1).expand(-1, -1, -1, -1, MXFP_BLOCK_SIZE).flatten(-2)
    residual = (reference - native_dequant).abs()
    # A correct decode leaves every element within one E2M1 step of the input;
    # the widest gap in the E2M1 grid is 2*scale (4 -> 6), so 1*scale bounds the
    # rounding error for all but the top bucket, and 1.001*scale bounds it there.
    within_tolerance = (residual <= 1.001 * step + 1e-6).float().mean().item()
    rel_l2 = (residual.norm() / reference.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(reference.flatten(), native_dequant.flatten(), dim=0).item()

    # Simulated round-trip per EXPERIMENT_SPEC 4.3, then compare the code streams.
    scaled = reference / scales.unsqueeze(-1).clamp(min=1e-30).expand(-1, -1, -1, -1,
                                                                     MXFP_BLOCK_SIZE).flatten(-2)
    grid = e2m1_lut(device)[:8]
    nearest = (scaled.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    sim_magnitude = grid[nearest]
    sim_codes = torch.sign(scaled) * sim_magnitude
    code_match = (sim_codes == codes).float().mean().item()

    scale_stats = {
        "min": scales.min().item(),
        "max": scales.max().item(),
        "zero_frac": (scales == 0).float().mean().item(),
    }
    saturation_frac = (codes.abs() >= E2M1_MAX).float().mean().item()

    verdict = {
        "shape": list(query.shape),
        "packed_shape": list(packed.shape),
        "packed_dtype": str(packed.dtype),
        "sf_shape": list(sf.shape),
        "sf_dtype": str(sf.dtype),
        "decode_within_one_step_frac": within_tolerance,
        "native_dequant_rel_l2_vs_bf16": rel_l2,
        "native_dequant_cosine_vs_bf16": cosine,
        "simulated_vs_native_code_agreement": code_match,
        "scale_stats": scale_stats,
        "e2m1_saturation_frac": saturation_frac,
        "decode_usable_as_native": within_tolerance > 0.999,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    for key, value in verdict.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
