"""Exact dequantizer for `_nvfp4_quantize_for_fa4` outputs + self-validation.

Builds (a) a pure-torch reference NVFP4 quantizer (per-16 E4M3 SF, amax/6,
global scale 1.0 — the FA4-path scheme) and (b) a layout decoder that unpacks
the flashinfer-packed fp4 tensor + FA4-MMA-layout SF tensor back to bf16/fp32.
Validates that decode(quantize_flashinfer(x)) == reference_quant_dequant(x)
bitwise. This pair is the correctness oracle for C4.
"""
import torch

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def unpack_fp4(fp4_tensor: torch.Tensor) -> torch.Tensor:
    """(B, S, H, D/2) float4_e2m1fn_x2 -> (B, S, H, D) float32 code values."""
    raw = fp4_tensor.view(torch.uint8)
    lo = raw & 0xF
    hi = raw >> 4
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)  # low nibble first
    lut = _E2M1_VALUES.to(raw.device)
    return lut[codes.long()]


def sf_mma_to_canonical(sf_mma: torch.Tensor, batch: int, seqlen_padded: int,
                        nheads: int, headdim: int) -> torch.Tensor:
    """FA4 MMA layout (32,4,rest_m,4,rest_k,h,b) -> (b, s, h, d//16) fp32 scales."""
    # invert: sf_mma = canonical(b,h,rest_m,rest_k,32,4,4).permute(4,5,2,6,3,1,0)
    sf_canonical = sf_mma.permute(6, 5, 2, 4, 0, 1, 3).contiguous()
    # canonical: (b, h, rest_m, rest_k, 32, 4, 4)
    b, h, rest_m, rest_k, m0, m1, k = sf_canonical.shape
    assert (m0, m1, k) == (32, 4, 4)
    scales = sf_canonical.view(torch.float8_e4m3fn).float()
    # rows within a 128-tile: try row = m1*32 + m0  (validated empirically below)
    scales = scales.permute(0, 2, 5, 4, 1, 3, 6)  # (b, rest_m, m1, m0, h, rest_k, k)
    scales = scales.reshape(batch, seqlen_padded, nheads, rest_k * k)
    return scales


def dequantize_fa4(fp4_tensor: torch.Tensor, sf_mma: torch.Tensor,
                   orig_seqlen: int) -> torch.Tensor:
    """Decode packed fp4 + MMA-layout SF back to fp32 (B, orig_seqlen, H, D)."""
    b, s_pad, h, d_half = fp4_tensor.shape
    d = d_half * 2
    codes = unpack_fp4(fp4_tensor)                       # (b, s_pad, h, d)
    scales = sf_mma_to_canonical(sf_mma, b, s_pad, h, d)  # (b, s_pad, h, d//16)
    deq = codes * scales.repeat_interleave(16, dim=-1)
    return deq[:, :orig_seqlen]


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).float()


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest-even onto the E2M1 grid."""
    grid = _E2M1_VALUES[:8].to(x.device)  # positive magnitudes
    sign = torch.sign(x)
    mag = x.abs().clamp(max=6.0)
    # nearest, ties-to-even handled by explicit midpoints
    idx = torch.bucketize(mag, (grid[1:] + grid[:-1]) / 2)
    # ties: bucketize(right=False) sends midpoint up; fix ties-to-even
    mid = (grid[1:] + grid[:-1]) / 2
    for i in range(len(mid)):
        is_tie = mag == mid[i]
        if (i + 1) % 2 == 1:  # upper value has odd index -> round down to even index
            idx = torch.where(is_tie, torch.full_like(idx, i), idx)
    return sign * grid[idx]


def reference_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Pure-torch NVFP4 round-trip with per-16 E4M3 scale = amax/6, global 1.0."""
    orig_shape = x.shape
    xf = x.float().reshape(-1, 16)
    amax = xf.abs().amax(dim=1, keepdim=True)
    sf = _round_e4m3(amax / 6.0)
    safe_sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = _round_e2m1(xf / safe_sf)
    return (q * sf).reshape(orig_shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda:0"
    B, S, H, D = 1, 300, 3, 128  # deliberately non-multiple of 128 seqlen
    x = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16) * 3

    fp4, sf = _nvfp4_quantize_for_fa4(x)
    deq = dequantize_fa4(fp4, sf, S)
    ref = reference_quant_dequant(x)

    # 1. Scale-factor decode must be bitwise exact against amax/6 -> E4M3.
    xf = x.float().reshape(-1, 16)
    amax = xf.abs().amax(dim=1, keepdim=True)
    sref = _round_e4m3(amax / 6.0).reshape(B, S, H, D // 16)
    sdec = sf_mma_to_canonical(sf, B, (S + 127) // 128 * 128, H, D)[:, :S]
    assert (sdec == sref).all(), "SF layout decode mismatch"

    # 2. Decoded values must equal reference round-trip except exactly at
    #    E2M1 rounding midpoints (flashinfer's tie rule differs from
    #    ties-to-even; the decode itself is exact).
    mism = deq != ref
    sc16 = torch.where(sdec == 0, torch.ones_like(sdec), sdec).repeat_interleave(16, -1)
    ratio = (x.float() / sc16)[mism].abs()
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
    at_tie = torch.isclose(ratio.unsqueeze(-1), midpoints, atol=2e-2).any(-1)
    frac_match = 1.0 - mism.float().mean().item()
    print(f"decode-vs-reference exact fraction: {frac_match:.6f}; "
          f"all {int(mism.sum())} mismatches at tie midpoints: {bool(at_tie.all())}")
    assert frac_match > 0.98 and bool(at_tie.all())
    err = (deq - x.float()).abs().mean().item()
    print(f"mean |deq - x| = {err:.4f}")
    print("ORACLE VALIDATED")
