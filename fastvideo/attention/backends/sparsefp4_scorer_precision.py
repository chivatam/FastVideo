"""Scorer arithmetic precision primitives for the SparseFP4 paper-validation study.

Study 1 quantized the **routing inputs** (Q/K representation) and then pooled and
scored those values in fp64. That leaves a gap its own report named: the cheap
block scorer's *arithmetic* was never run at low precision, so "NVFP4 routing is
safe" was really "NVFP4-*represented* routing scored exactly is safe". This module
supplies the two axes separately, because collapsing them is the single easiest way
to overstate the claim (``references/FOLLOWUP_SPEC.md``, "Scorer-precision
experiment semantics"):

``representation precision``
    how the Q/K vectors handed to the scorer are represented — reuses study 1's
    ``quantize_router_input`` so the NVFP4 arm still decodes values a real
    flashinfer ``fp4_quantize_sm100`` call produced.
``arithmetic precision``
    how the block mean-pool and the block dot product are *computed*.

Every arm therefore carries three labels — ``repr``, ``pool``, ``score`` — and a
recorded accumulation semantics string, never a single word like "nvfp4 router".

**Why the accumulation semantics are recorded rather than assumed.** A bf16
``torch.matmul`` on a Blackwell tensor core accumulates in fp32; a bf16
``Tensor.sum`` accumulates in fp32 too (ATen picks an ``acc_type``). Calling
either "bf16 arithmetic" would be wrong in exactly the direction that flatters the
null result, since fp32 accumulation hides most of the low-precision damage. So
this module implements both variants explicitly:

``*_acc_fp32``
    the honest description of what the hardware/library does by default.
``*_acc_low``
    a genuinely low-precision reduction: an explicit sequential loop that casts
    back to the working dtype after every partial sum, which is the worst case a
    real fused low-precision scorer kernel could exhibit.

Reporting both brackets the truth instead of guessing which one a hypothetical
kernel would implement.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

import torch

# e2m1 / e4m3 constants, duplicated from routing_probe_attn so this module can be
# imported by the GPU-free self-test without pulling in the attention stack.
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
MXFP_BLOCK_SIZE = 16

DTYPES: dict[str, torch.dtype] = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# --------------------------------------------------------------------------- #
# Pooling: the block mean, at a specified arithmetic precision.
# --------------------------------------------------------------------------- #
def pool_blocks_precision(
    padded: torch.Tensor,
    block_sizes: torch.Tensor,
    slot: int,
    arithmetic: str,
    accumulate: str,
) -> tuple[torch.Tensor, str]:
    """Masked-mean pool a padded ``[B, S, H, D]`` layout into ``[H, n_blocks, D]``.

    Numerically identical to ``sparsefp4_numerics.pool_geometry_blocks`` when
    ``arithmetic="fp32"`` and ``accumulate="native"``, which is what study 1 ran;
    the extra arguments only widen it. Pad slots hold zeros and the divisor is
    each block's **true** token count, so a short boundary tile is not diluted.

    Returns ``(pooled, accumulation_semantics)``. The second element is written
    into every raw record so no aggregate can silently mislabel the arm.
    """
    if arithmetic not in DTYPES:
        raise ValueError(f"unknown pooling arithmetic {arithmetic!r}")
    dtype = DTYPES[arithmetic]
    batch, _, heads, dim = padded.shape
    n_blocks = int(block_sizes.numel())
    grouped = padded.view(batch, n_blocks, slot, heads, dim)
    divisor = block_sizes.to(dtype).view(1, -1, 1, 1)

    if accumulate == "native":
        # ATen promotes the accumulator for reduced-precision reductions, so the
        # honest label for a bf16 input here is "bf16 values, fp32 accumulation".
        summed = grouped.to(dtype).sum(dim=2)
        semantics = (f"pool={arithmetic}_values_torch_sum" +
                     ("_acc_fp32" if dtype in (torch.bfloat16, torch.float16) else f"_acc_{arithmetic}"))
    elif accumulate == "low":
        # Genuinely low-precision reduction: cast back to ``dtype`` after every
        # partial sum. Sequential over the token axis, vectorized over everything
        # else, so it is a handful of elementwise kernels rather than a Python
        # loop over blocks.
        acc = grouped[:, :, 0].to(dtype)
        for index in range(1, slot):
            acc = (acc + grouped[:, :, index].to(dtype)).to(dtype)
        summed = acc
        semantics = f"pool={arithmetic}_values_sequential_acc_{arithmetic}"
    else:
        raise ValueError(f"unknown pooling accumulation {accumulate!r}")

    pooled = (summed / divisor).to(dtype)
    return pooled.permute(0, 2, 1, 3)[0], semantics


# --------------------------------------------------------------------------- #
# Post-pool quantization: what an FP8 / NVFP4 scorer would actually *store*.
# --------------------------------------------------------------------------- #
def quantize_pooled_fp8_e4m3(pooled: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Round pooled block vectors to FP8-E4M3 with a per-head amax scale.

    A real FP8 block scorer holds its pooled vectors in FP8, so the pooled
    tensor — not just the token-level input — has to be rounded. Scaling is
    per-head (dim 0 of ``[H, n_blocks, D]``), matching study 1's FP8 router
    scaling granularity so the two are comparable.
    """
    amax = pooled.float().abs().amax(dim=(1, 2), keepdim=True)
    scale = (amax / FP8_E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    scaled = (pooled.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = scaled.to(torch.float8_e4m3fn).float()
    saturated = float((quantized.abs() >= FP8_E4M3_MAX).float().mean().item())
    return quantized * scale, saturated


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=torch.float32)
    return torch.cat([magnitudes, -magnitudes])


def quantize_pooled_nvfp4(pooled: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Round pooled block vectors to NVFP4 (e2m1 + per-16 e4m3 scales).

    Deterministic round-to-nearest against the exact e2m1 code set, with the same
    per-16-element E4M3 scale-factor recipe the flashinfer quantizer uses. This is
    a **simulated** NVFP4 representation of the pooled vectors: it reproduces the
    format's value set exactly, but no native NVFP4 GEMM is invoked for the block
    dot product, so arms built on it may make numerical claims only.
    """
    grouped = pooled.float().unflatten(-1, (-1, MXFP_BLOCK_SIZE))
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / E2M1_MAX).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    safe = scale.clamp(min=2.0**-9)
    scaled = (grouped / safe).clamp(-E2M1_MAX, E2M1_MAX)
    codes = _e2m1_lut(pooled.device)[:8]
    # Round to nearest representable magnitude, keeping the sign.
    magnitude = scaled.abs().unsqueeze(-1)
    nearest = (magnitude - codes.view(1, 1, 1, 1, -1)).abs().argmin(dim=-1)
    rounded = codes[nearest] * scaled.sign()
    saturated = float((rounded.abs() >= E2M1_MAX).float().mean().item())
    return (rounded * safe).flatten(-2).to(pooled.dtype), saturated


# --------------------------------------------------------------------------- #
# Block scores: the dot product, at a specified arithmetic precision.
# --------------------------------------------------------------------------- #
def score_blocks_precision(
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    softmax_scale: float,
    arithmetic: str,
    accumulate: str,
) -> tuple[torch.Tensor, str]:
    """``[H, n_q, D] x [H, D, n_k] -> [H, n_q, n_k]`` at a stated precision.

    ``accumulate="native"`` uses ``torch.matmul``, whose accumulator is fp32 for
    bf16/fp16 inputs on this hardware — recorded in the returned semantics string
    rather than described as pure low precision. ``accumulate="low"`` runs an
    explicit rank-1 loop over the head dimension, casting the accumulator back to
    the working dtype after each update, which is genuine low-precision
    accumulation.

    The fp32 arm runs inside :func:`exact_fp32_matmul`, which disables TF32 for
    the duration of the scorer matmul only. Scoping it this way is deliberate:
    the ambient TF32 state belongs to the model's own trajectory, which must stay
    byte-identical to study 1's, so it is read and recorded but never mutated
    globally. Without the scope an "fp32" scorer on Blackwell would silently be a
    19-bit-mantissa scorer, which is a different experiment.
    """
    if arithmetic not in DTYPES:
        raise ValueError(f"unknown score arithmetic {arithmetic!r}")
    dtype = DTYPES[arithmetic]
    query = pooled_query.to(dtype)
    key = pooled_key.to(dtype)

    if accumulate == "native":
        if dtype is torch.float32:
            with exact_fp32_matmul():
                scores = query @ key.transpose(-1, -2)
            semantics = "score=fp32_values_torch_matmul_acc_fp32_tf32_disabled"
        else:
            scores = query @ key.transpose(-1, -2)
            semantics = (f"score={arithmetic}_values_torch_matmul" +
                         ("_acc_fp32" if dtype in (torch.bfloat16, torch.float16) else f"_acc_{arithmetic}"))
    elif accumulate == "low":
        acc = torch.zeros((query.shape[0], query.shape[1], key.shape[1]), dtype=dtype, device=query.device)
        key_t = key.transpose(-1, -2)
        for index in range(query.shape[-1]):
            acc = (acc + query[..., index:index + 1] * key_t[..., index:index + 1, :]).to(dtype)
        scores = acc
        semantics = f"score={arithmetic}_values_sequential_rank1_acc_{arithmetic}"
    else:
        raise ValueError(f"unknown score accumulation {accumulate!r}")

    return (scores.to(torch.float64) * softmax_scale), semantics


def score_blocks_fp8_native(pooled_query: torch.Tensor, pooled_key: torch.Tensor,
                            softmax_scale: float) -> tuple[torch.Tensor, str] | None:
    """Native FP8-E4M3 block dot product via ``torch._scaled_mm``, or ``None``.

    Blackwell exposes a real FP8 GEMM, so the FP8 scorer arm can be *native*
    rather than simulated — worth the extra code, because a simulated arm cannot
    support any statement about what an FP8 scorer kernel would compute. Returns
    ``None`` when the op is unavailable or rejects these shapes, and the caller
    then falls back to the faithful simulated arm and relabels it.

    Accumulation is fp32 inside the tensor core; that is stated in the semantics
    string and is a property of the hardware, not a choice.
    """
    heads, n_q, dim = pooled_query.shape
    n_k = pooled_key.shape[1]
    if dim % 16 != 0 or n_k % 16 != 0:
        return None
    try:
        out = torch.empty((heads, n_q, n_k), dtype=torch.float32, device=pooled_query.device)
        with exact_fp32_matmul():
            for head in range(heads):
                q_amax = pooled_query[head].float().abs().amax().clamp(min=torch.finfo(torch.float32).tiny)
                k_amax = pooled_key[head].float().abs().amax().clamp(min=torch.finfo(torch.float32).tiny)
                q_scale = (q_amax / FP8_E4M3_MAX).to(torch.float32)
                k_scale = (k_amax / FP8_E4M3_MAX).to(torch.float32)
                q_fp8 = (pooled_query[head].float() / q_scale).clamp(-FP8_E4M3_MAX,
                                                                     FP8_E4M3_MAX).to(torch.float8_e4m3fn)
                k_fp8 = (pooled_key[head].float() / k_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
                out[head] = torch._scaled_mm(
                    q_fp8,
                    k_fp8.t(),
                    scale_a=q_scale.view(1, 1),
                    scale_b=k_scale.view(1, 1),
                    out_dtype=torch.float32,
                )
    except Exception:
        return None
    return (out.to(torch.float64) * softmax_scale), "score=fp8_e4m3_torch_scaled_mm_acc_fp32_native"


def assert_exact_fp32_matmul() -> None:
    """Deprecated in favour of :func:`exact_fp32_matmul`; kept for the self-test."""
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("allow_tf32 is enabled; the fp32 scorer arm would not be fp32")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError(f"float32_matmul_precision is {torch.get_float32_matmul_precision()!r}, expected 'highest'")


@contextmanager
def exact_fp32_matmul() -> Iterator[None]:
    """Force true fp32 matmul for the enclosed block, then restore ambient state.

    Two separate mechanisms silently downgrade an "fp32" matmul inside FastVideo's
    denoising loop, and both must be neutralized or the fp32 scorer arm is not fp32:

    1.  **TF32.** FastVideo runs the pipeline in a *worker subprocess*, so a
        driver-side ``allow_tf32 = False`` never reaches it; on this
        torch/Blackwell build the worker reads ``allow_tf32 == True``, giving a
        19-bit mantissa.
    2.  **Autocast.** The denoising loop wraps the transformer in
        ``torch.autocast(device_type="cuda", dtype=bf16)``, and autocast casts
        ``torch.matmul`` inputs **down to bf16** — including fp32 inputs. Verified
        directly: inside autocast an fp32 @ fp32 matmul returns a bf16 tensor.
        (fp64 inputs are left alone, which is why study 1's fp64 scorer was
        unaffected and this went unnoticed.)

    Both are restored on exit so the model's own matmuls keep whatever settings its
    trajectory uses — the study must not change the trajectory it measures.
    """
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


@contextmanager
def declared_precision_arithmetic() -> Iterator[None]:
    """Disable autocast for a whole side-channel measurement block.

    Every arm in a precision ablation must compute in the dtype its label claims.
    Under the denoising loop's ``autocast(bf16)`` that is false by default: any
    ``matmul`` with fp32 inputs silently returns bf16, so an "fp32" arm and a
    "bf16" arm collapse onto the same numbers and a precision study reports a
    spurious null.

    This wrapper is deliberately coarse — it covers pooling, scoring, selection and
    the sparse-kernel calls for all arms at once — because a per-operation opt-out
    is exactly the kind of thing that gets forgotten when an arm is added later.
    TF32 is left alone here; only the fp32 paths need it, and they take
    :func:`exact_fp32_matmul` for that.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        yield


def autocast_state() -> dict[str, object]:
    """Ambient autocast state, recorded per record so the fix is auditable."""
    return {
        "worker_autocast_enabled": bool(torch.is_autocast_enabled()),
        "worker_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
    }


def ambient_fp32_state() -> dict[str, object]:
    """The process's TF32 settings, recorded in every raw record.

    Study 1's report lists ``allow_tf32 = False`` under determinism flags, but
    that was measured in the **driver** process; the worker that actually runs the
    DiT is a separate process and reads a different value. Emitting the worker's
    own state per record makes that distinction auditable instead of inherited.
    """
    return {
        "worker_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "worker_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "worker_float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


# --------------------------------------------------------------------------- #
# The arm matrix.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScorerArm:
    """One cell of the ``representation x arithmetic`` scorer matrix.

    ``arm_id`` is the short label used in tables; ``label`` is the full
    ``repr=…, pool=…, score=…`` string FOLLOWUP_SPEC requires so a single word can
    never stand in for the pair of axes.
    """

    arm_id: str
    representation: str
    pool_arithmetic: str
    pool_accumulate: str
    score_arithmetic: str
    score_accumulate: str
    quantize_pooled: str | None
    native_or_simulated: str
    purpose: str

    @property
    def label(self) -> str:
        pooled = f"{self.pool_arithmetic}" + (f"+{self.quantize_pooled}" if self.quantize_pooled else "")
        return (f"repr={self.representation}, pool={pooled}/{self.pool_accumulate}, "
                f"score={self.score_arithmetic}/{self.score_accumulate}")


# A 2 x 6 factorial: representation in {bf16, nvfp4} crossed with scorer
# arithmetic in {fp64, fp32, bf16(acc fp32), bf16(acc bf16), fp8, nvfp4-like}.
#
# The factorial matters. A one-sided ladder (only ever lowering both axes
# together) cannot distinguish "representation is what breaks routing" from
# "arithmetic is what breaks routing", and the whole point of this phase is that
# study 1 only measured the representation axis. Crossing them makes the
# high-precision-scorer rescue at fixed representation directly measurable, which
# is what the F1.6 decision rule is stated in terms of.
#
# R0 and R1 reproduce study 1's reference and NVFP4 conditions exactly, so this
# phase re-derives its own baseline instead of trusting a remembered number.
SCORER_ARMS: tuple[ScorerArm, ...] = (
    ScorerArm("R0", "bf16", "fp64", "native", "fp64", "native", None, "native",
              "scientific reference: exact arithmetic on unquantized Q/K"),
    ScorerArm("R1", "nvfp4", "fp64", "native", "fp64", "native", None, "native",
              "study 1's condition: NVFP4 representation, exact arithmetic"),
    ScorerArm("R2", "bf16", "fp32", "native", "fp32", "native", None, "native",
              "fp32 arithmetic, exact representation"),
    ScorerArm("R3", "nvfp4", "fp32", "native", "fp32", "native", None, "native",
              "fp32 arithmetic, NVFP4 representation"),
    ScorerArm("R4", "bf16", "bf16", "native", "bf16", "native", None, "native",
              "bf16 values with library fp32 accumulation, exact representation"),
    ScorerArm("R5", "nvfp4", "bf16", "native", "bf16", "native", None, "native",
              "bf16 values with library fp32 accumulation, NVFP4 representation"),
    ScorerArm("R4L", "bf16", "bf16", "low", "bf16", "low", None, "native",
              "genuine bf16 accumulation (worst case), exact representation"),
    ScorerArm("R5L", "nvfp4", "bf16", "low", "bf16", "low", None, "native",
              "genuine bf16 accumulation (worst case), NVFP4 representation"),
    ScorerArm("R6", "bf16", "fp32", "native", "fp8_e4m3", "native", "fp8_e4m3", "native",
              "FP8 pooled vectors + native FP8 dot, exact representation"),
    ScorerArm("R7", "nvfp4", "fp32", "native", "fp8_e4m3", "native", "fp8_e4m3", "native",
              "FP8 pooled vectors + native FP8 dot, NVFP4 representation"),
    ScorerArm("R8", "bf16", "fp32", "native", "fp32", "native", "nvfp4", "simulated",
              "NVFP4-like pooled block vectors, exact representation"),
    ScorerArm("R9", "nvfp4", "fp32", "native", "fp32", "native", "nvfp4", "simulated",
              "NVFP4-like pooled block vectors, NVFP4 representation"),
)

ARMS_BY_ID: dict[str, ScorerArm] = {arm.arm_id: arm for arm in SCORER_ARMS}

REFERENCE_ARM = "R0"
STUDY1_ARM = "R1"

# Arms grouped by representation, so "does higher-precision *arithmetic* rescue
# anything at a fixed representation" is a lookup rather than a re-derivation.
ARITHMETIC_LADDER: tuple[str, ...] = ("fp64", "fp32", "bf16_acc_fp32", "bf16_acc_bf16", "fp8", "nvfp4_like")
LADDER_POSITION: dict[str, str] = {
    "R0": "fp64",
    "R1": "fp64",
    "R2": "fp32",
    "R3": "fp32",
    "R4": "bf16_acc_fp32",
    "R5": "bf16_acc_fp32",
    "R4L": "bf16_acc_bf16",
    "R5L": "bf16_acc_bf16",
    "R6": "fp8",
    "R7": "fp8",
    "R8": "nvfp4_like",
    "R9": "nvfp4_like",
}
