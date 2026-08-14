"""Execution backend for Phase 5 of the SparseFP4 study: end-to-end video.

Phase 2's :mod:`precision_sparse_attn` deliberately feeds the model **dense
BF16** and measures configurations B-F on the side, so that every arm shares one
denoising trajectory. Phase 5 asks the opposite question -- *does the arm change
the video a user sees* -- so here the model consumes the arm's attention for
real: the returned tensor is what flows into the residual stream, every layer,
every step, both CFG branches.

Six arms, matching ``SKILL.md`` Phase 3.3:

===================  ========  ==========================  =========  ==========
arm                  sparsity  attention compute           router     provenance
===================  ========  ==========================  =========  ==========
DENSE-BF16           none      BF16 (FA4 dense)            n/a        native
DENSE-FP4            none      NVFP4 Q/K + BF16 PV         n/a        native
SPARSE-BF16          set       BF16                        bf16       native
SPARSE-FP4-NAIVE     set       NVFP4 Q/K + BF16 PV         nvfp4      simulated
SPARSE-FP4-ROUTE8    set       NVFP4 Q/K + BF16 PV         fp8_e4m3   simulated
SPARSE-FP4-ROUTE16   set       NVFP4 Q/K + BF16 PV         bf16       simulated
===================  ========  ==========================  =========  ==========

"NVFP4" always means **NVFP4 Q/K with BF16 PV** -- that is what the FA4 kernel
implements. There is no native sparse-NVFP4 kernel in this repository, so the
three ``SPARSE-FP4-*`` arms are **numerical-only: no latency claim may be made
for them** (``GO_NO_GO.md`` scoping item 3). Only ``DENSE-BF16`` and
``DENSE-FP4`` are native end-to-end paths and may carry timing numbers.

Router masks are scored in **fp64** (``STATUS.md`` trap 8) at Phase 1/2's raster
``128x64`` geometry and executed on the block-sparse kernel's ``64x64`` grid by
splitting each 128-token query block into its two constituent kernel rows, so
the executed mask is the mask Phase 1 and Phase 2 measured. This is *not* VSA's
``(4,4,4)`` cube geometry (``STATUS.md`` trap 3); no claim about the deployed VSA
path follows from it.

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_EXEC_ATTN
    FASTVIDEO_SPARSEFP4_PHASE5=/path/to/phase5_config.json
"""

from __future__ import annotations

import atexit
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.routing_probe_attn import pool_blocks_1d, quantize_router_input
from fastvideo.attention.backends.sparsefp4_numerics import (KERNEL_BLOCK, assert_kernel_scale_matches,
                                                             assert_query_grid_alignment, block_scores, dense_bf16,
                                                             dense_nvfp4_native, expand_query_axis, pad_to_kernel_block,
                                                             sparse_bf16, topk_block_mask, variable_block_sizes_for)
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

PHASE5_ENV_VAR = "FASTVIDEO_SPARSEFP4_PHASE5"


@dataclass(frozen=True)
class Arm:
    """One row of the Phase 3.3 table.

    ``compute`` is the precision the *attention* runs at; ``router`` is the
    precision the block scores are derived from. The two are independent axes
    (SKILL 3.2) and must never be conflated in a label.
    """

    arm_id: str
    sparse: bool
    compute: str
    router: str | None
    compute_label: str
    native_or_simulated: str
    latency_claim_allowed: bool


ARMS: dict[str, Arm] = {
    "DENSE-BF16":
    Arm("DENSE-BF16", False, "bf16", None, "dense_bf16_fa4", "native", True),
    "DENSE-FP4":
    Arm("DENSE-FP4", False, "nvfp4_native", None, "dense_nvfp4_qk_bf16_pv", "native", True),
    "SPARSE-BF16":
    Arm("SPARSE-BF16", True, "bf16", "bf16", "sparse_bf16_triton", "native", True),
    "SPARSE-FP4-NAIVE":
    Arm("SPARSE-FP4-NAIVE", True, "nvfp4_sim", "nvfp4", "sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    "SPARSE-FP4-ROUTE8":
    Arm("SPARSE-FP4-ROUTE8", True, "nvfp4_sim", "fp8_e4m3", "sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    "SPARSE-FP4-ROUTE16":
    Arm("SPARSE-FP4-ROUTE16", True, "nvfp4_sim", "bf16", "sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    # Calibration control, not a design. Identical to SPARSE-BF16 plus a
    # deterministic perturbation of the attention output at a *known* relative
    # L2 (``perturb_rel_l2``). Sweeping that knob calibrates how much final-video
    # pixel difference a given per-call attention perturbation produces, which is
    # the only way to tell "routing precision changed the video" apart from "a
    # 50-step trajectory amplifies any perturbation to saturation".
    "SPARSE-BF16-EPS":
    Arm("SPARSE-BF16-EPS", True, "bf16_perturbed", "bf16", "sparse_bf16_triton_plus_noise", "control", False),
}

# Verbatim from Phase 1/2 so all three phases label routers identically.
_ROUTER_PROVENANCE: dict[str, str] = {"bf16": "native", "nvfp4": "native", "fp8_e4m3": "simulated"}


@dataclass
class Phase5Config:
    out_dir: Path
    run_id: str
    git_commit: str
    arm_id: str
    prompt_id: str
    seed: int
    sparsity: float
    block_q: int
    block_k: int
    score_dtype: str
    shard_tag: str
    perturb_rel_l2: float = 0.0
    perturb_seed: int = 20260814
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def arm(self) -> Arm:
        return ARMS[self.arm_id]

    @classmethod
    def load(cls, path: str) -> Phase5Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arm_id = str(raw["arm"])
        if arm_id not in ARMS:
            raise ValueError(f"unknown Phase 5 arm {arm_id!r}; expected one of {sorted(ARMS)}")
        return cls(
            out_dir=Path(raw["out_dir"]),
            run_id=str(raw["run_id"]),
            git_commit=str(raw["git_commit"]),
            arm_id=arm_id,
            prompt_id=str(raw["prompt_id"]),
            seed=int(raw["seed"]),
            sparsity=float(raw.get("sparsity", 0.0)),
            block_q=int(raw.get("block_q", 128)),
            block_k=int(raw.get("block_k", 64)),
            score_dtype=str(raw.get("score_dtype", "float64")),
            shard_tag=str(raw.get("shard_tag", "shard0")),
            perturb_rel_l2=float(raw.get("perturb_rel_l2", 0.0)),
            perturb_seed=int(raw.get("perturb_seed", 20260814)),
            provenance=dict(raw.get("provenance", {})),
        )


class SparseFP4ExecAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SPARSEFP4_EXEC_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SparseFP4ExecAttentionImpl]:
        return SparseFP4ExecAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[SparseFP4ExecAttentionMetadata]:
        return SparseFP4ExecAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SparseFP4ExecAttentionMetadataBuilder]:
        return SparseFP4ExecAttentionMetadataBuilder


@dataclass
class SparseFP4ExecAttentionMetadata(AttentionMetadata):
    current_timestep: int


class SparseFP4ExecAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, current_timestep: int, **kwargs: Any) -> SparseFP4ExecAttentionMetadata:
        return SparseFP4ExecAttentionMetadata(current_timestep=current_timestep)


def compute_topk(sparsity: float, n_blocks: int) -> int:
    """Identical rule to Phase 1, Phase 2 and VSA (``video_sparse_attn.compute_topk``)."""
    return max(1, min(math.ceil((1 - sparsity) * n_blocks), n_blocks))


class _Counters:
    """Per-process aggregate receipt that the arm really ran as configured.

    A silently-ignored backend override (``STATUS.md`` trap 1) or a mask that
    quietly collapsed to dense would both be invisible in the output video, so
    the realized retained fraction is counted on every call and written next to
    the video rather than inferred from the config that requested it.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.retained_blocks = 0
        self.total_blocks = 0
        self.seq_lens: set[int] = set()
        self.layers: set[int] = set()
        self.timesteps: set[int] = set()
        self.cfg_branches: set[str] = set()
        self.k_values: set[int] = set()
        self.router_saturation_q = 0.0
        self.router_saturation_k = 0.0
        self.perturb_rel_l2_sum = 0.0
        self.perturb_calls = 0

    def as_dict(self) -> dict[str, Any]:
        realized = (self.retained_blocks / self.total_blocks) if self.total_blocks else None
        return {
            "attention_calls":
            self.calls,
            "distinct_layers":
            len(self.layers),
            "distinct_timesteps":
            len(self.timesteps),
            "cfg_branches":
            sorted(self.cfg_branches),
            "seq_lens":
            sorted(self.seq_lens),
            "k_per_query_block":
            sorted(self.k_values),
            "realized_retained_fraction":
            realized,
            "realized_sparsity":
            None if realized is None else 1.0 - realized,
            "mean_router_saturation_q": (self.router_saturation_q / self.calls) if self.calls else None,
            "mean_router_saturation_k": (self.router_saturation_k / self.calls) if self.calls else None,
            "mean_realized_perturb_rel_l2":
            ((self.perturb_rel_l2_sum / self.perturb_calls) if self.perturb_calls else None),
        }


class SparseFP4ExecAttentionImpl(AttentionImpl):
    """Runs one Phase 5 arm as the model's real attention."""

    _counters: dict[str, _Counters] = {}
    _receipt_written = False
    _config: Phase5Config | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.prefix = prefix
        self.layer_idx = layer_idx_from_prefix(prefix, default=-1)
        path = os.environ.get(PHASE5_ENV_VAR, "").strip()
        if not path:
            raise RuntimeError(f"{PHASE5_ENV_VAR} must point at a Phase 5 config JSON when "
                               "FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_EXEC_ATTN")
        self.config = Phase5Config.load(path)
        # The receipt is written from ``atexit``, by which point the config file
        # may be gone (temp dirs, cleaned scratch), so keep the parsed copy.
        type(self)._config = self.config
        self._checked_geometry = False

    @classmethod
    def counters(cls, key: str) -> _Counters:
        if key not in cls._counters:
            cls._counters[key] = _Counters()
            atexit.register(cls.write_receipt)
        return cls._counters[key]

    @classmethod
    def write_receipt(cls) -> None:
        """Dump the per-arm receipt. Idempotent; safe from ``atexit``."""
        if cls._receipt_written or not cls._counters:
            return
        cls._receipt_written = True
        config = cls._config
        if config is None:
            return
        arm = config.arm
        record = {
            "record_type": "arm_receipt",
            "run_id": config.run_id,
            "git_commit": config.git_commit,
            "prompt_id": config.prompt_id,
            "arm": config.arm_id,
            "seed": config.seed,
            "requested_sparsity": config.sparsity if arm.sparse else None,
            "sparse": arm.sparse,
            "attention_compute": arm.compute_label,
            "compute_precision_label": ("nvfp4_qk_bf16_pv" if "nvfp4" in arm.compute_label else "bf16"),
            "router_precision": arm.router,
            "router_native_or_simulated": _ROUTER_PROVENANCE.get(arm.router or "", None),
            "native_or_simulated": arm.native_or_simulated,
            "native_latency_claim_allowed": arm.latency_claim_allowed,
            "numerical_only": not arm.latency_claim_allowed,
            "block_q": config.block_q,
            "block_k": config.block_k,
            "requested_perturb_rel_l2": config.perturb_rel_l2 or None,
            "perturb_seed": config.perturb_seed if config.perturb_rel_l2 else None,
            "kernel_block": KERNEL_BLOCK,
            "score_dtype": config.score_dtype,
            "token_order": "raster_frame_y_x",
            "force_retain_diagonal": False,
            "attention_backend": "SPARSEFP4_EXEC_ATTN",
            "phase": "5",
        }
        for key, counters in sorted(cls._counters.items()):
            record[f"counters.{key}"] = counters.as_dict()
        out = config.out_dir / f"arm_receipt_{config.shard_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseFP4ExecAttentionMetadata | None,
    ) -> torch.Tensor:
        arm = self.config.arm
        if arm.compute == "bf16" and not arm.sparse:
            self._count(query, None, None)
            return dense_bf16(query, key, value, self.softmax_scale)
        if arm.compute == "nvfp4_native":
            self._count(query, None, None)
            return dense_nvfp4_native(query, key, value, self.softmax_scale)
        return self._sparse_forward(query, key, value)

    def _count(self, query: torch.Tensor, k: int | None, retained: tuple[int, int] | None) -> None:
        context = get_forward_context()
        batch = getattr(context, "forward_batch", None)
        counters = self.counters("all")
        counters.calls += 1
        counters.seq_lens.add(int(query.shape[1]))
        counters.layers.add(self.layer_idx)
        counters.timesteps.add(int(getattr(context, "current_timestep", -1) or 0))
        counters.cfg_branches.add("negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive")
        if k is not None:
            counters.k_values.add(k)
        if retained is not None:
            counters.retained_blocks += retained[0]
            counters.total_blocks += retained[1]

    @torch.no_grad()
    def _sparse_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        config = self.config
        arm = config.arm
        assert arm.router is not None
        seq_len = query.shape[1]
        if not self._checked_geometry:
            assert_kernel_scale_matches(self.head_size, self.softmax_scale)
            assert_query_grid_alignment(seq_len, config.block_q)
            self._checked_geometry = True

        # The NVFP4 dequantization is shared between the router and the compute
        # path whenever both want it, so no arm pays for it twice.
        nvfp4_qk: tuple[torch.Tensor, torch.Tensor] | None = None
        if arm.router == "nvfp4" or arm.compute == "nvfp4_sim":
            route_q, sat_q = quantize_router_input(query, "nvfp4")
            route_k, sat_k = quantize_router_input(key, "nvfp4")
            nvfp4_qk = (route_q, route_k)
        else:
            sat_q = sat_k = 0.0

        if arm.router == "nvfp4":
            assert nvfp4_qk is not None
            router_q, router_k = nvfp4_qk
        else:
            router_q, sat_router_q = quantize_router_input(query, arm.router)
            router_k, sat_router_k = quantize_router_input(key, arm.router)
            sat_q, sat_k = sat_router_q, sat_router_k

        score_dtype = getattr(torch, config.score_dtype)
        scores = block_scores(pool_blocks_1d(router_q, config.block_q), pool_blocks_1d(router_k, config.block_k),
                              self.softmax_scale, score_dtype)
        n_k_blocks = int(scores.shape[-1])
        k = compute_topk(config.sparsity, n_k_blocks)
        mask = topk_block_mask(scores, k)
        kernel_mask = expand_query_axis(mask, config.block_q // KERNEL_BLOCK).unsqueeze(0)

        if arm.compute == "nvfp4_sim":
            assert nvfp4_qk is not None
            compute_q = nvfp4_qk[0].to(query.dtype)
            compute_k = nvfp4_qk[1].to(key.dtype)
        else:
            compute_q, compute_k = query, key

        counters = self.counters("all")
        counters.router_saturation_q += sat_q
        counters.router_saturation_k += sat_k
        retained_rows = int(kernel_mask.shape[-2]) * self.num_heads
        self._count(query, k, (k * retained_rows, n_k_blocks * retained_rows))

        sizes = variable_block_sizes_for(seq_len, query.device)
        out = sparse_bf16(pad_to_kernel_block(compute_q), pad_to_kernel_block(compute_k), pad_to_kernel_block(value),
                          kernel_mask, sizes)
        out = out[:, :seq_len]
        if arm.compute == "bf16_perturbed" and config.perturb_rel_l2 > 0.0:
            out = self._perturb(out, config, timestep=int(getattr(get_forward_context(), "current_timestep", 0) or 0))
        return out

    def _perturb(self, out: torch.Tensor, config: Phase5Config, timestep: int) -> torch.Tensor:
        """Add Gaussian noise scaled to a target relative L2 of the output.

        Deterministic in ``(perturb_seed, layer, timestep)`` so the whole
        trajectory is reproducible, and the *realized* relative L2 is recorded
        rather than assumed -- the calibration curve is only meaningful if the
        x-axis is measured.
        """
        generator = torch.Generator(device=out.device)
        generator.manual_seed(config.perturb_seed + 1000 * (self.layer_idx + 1) + timestep)
        noise = torch.randn(out.shape, device=out.device, dtype=torch.float32, generator=generator)
        reference_norm = out.float().norm()
        noise_norm = noise.norm().clamp(min=1e-30)
        scaled = noise * (config.perturb_rel_l2 * reference_norm / noise_norm)
        perturbed = (out.float() + scaled).to(out.dtype)
        counters = self.counters("all")
        counters.perturb_calls += 1
        counters.perturb_rel_l2_sum += float(
            ((perturbed.float() - out.float()).norm() / reference_norm.clamp(min=1e-30)).item())
        return perturbed
