"""Research-only backend for Phase 2 of the SparseFP4 study: error decomposition.

The attention compute the model actually consumes is **dense BF16** — byte for
byte configuration A — so all arms share one denoising trajectory and every
comparison is exactly paired at the ``(prompt, layer, head, timestep, cfg_branch,
sparsity)`` level. Configurations B–F, the H3 router comparison, the
decision-margin mechanism measurement and the random-perturbation contrast
control are all computed on the side from the same captured Q/K/V.

Block scores are computed in **fp64** (``STATUS.md`` trap 8).

**Block geometry is a run parameter** (``geometry`` in the JSON config), because
Phases 1–2 measured only the raster-order ``128x64`` diagnostic geometry while
FastVideo's deployed sparse backend, VSA, uses 64-token (4,4,4) spatio-temporal
cubes in tile-contiguous token order (``STATUS.md`` trap 3). Three arms are
supported, which separates block size from token ordering:

``128x64-raster``
    Phase 2's geometry. Router scores 128-token query blocks against 64-token key
    blocks; the mask is executed on the kernel's 64-row query grid by splitting
    each query block in two.
``64x64-raster``
    Same raster token order, kernel-native ``block_q == block_k == 64``. Isolates
    the block-size change.
``64x64-cube``
    VSA's deployed geometry, built from VSA's own tiling utilities. Isolates the
    token-ordering change.

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=PRECISION_SPARSE_ATTN
    FASTVIDEO_SPARSEFP4_PHASE2=/path/to/phase2_config.json
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
from fastvideo.attention.backends.routing_probe_attn import JsonlWriter, quantize_router_input
from fastvideo.attention.backends.sparsefp4_numerics import (
    KERNEL_BLOCK, BlockGeometry, assert_kernel_scale_matches, block_attention_mass, block_scores, cube_geometry,
    dense_bf16, dense_nvfp4_native, error_metrics, expand_query_axis, from_block_layout, pool_geometry_blocks,
    random_matched_mask, raster_geometry, retained_token_fraction, sparse_bf16, to_block_layout, topk_block_mask)
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

PHASE2_ENV_VAR = "FASTVIDEO_SPARSEFP4_PHASE2"
REFERENCE_PRECISION = "bf16"
DEFAULT_GEOMETRY = "128x64-raster"
# Geometries the trap-8 tie diagnostic is evaluated at on every measured cell.
# Scoring is a pool + matmul + sort, so running all three costs almost nothing
# and makes the Phase-1-vs-Phase-2 tie-count comparison available at one place.
TIE_DIAGNOSTIC_GEOMETRIES = ("128x64-raster", "64x64-raster", "64x64-cube")

# Router provenance, verbatim from Phase 1 so the two phases label arms
# identically: the nvfp4 router decodes values a real flashinfer quantizer
# produced, while fp8 has no native attention-path quantizer to borrow.
_ROUTER_PROVENANCE: dict[str, str] = {"bf16": "native", "nvfp4": "native", "fp8_e4m3": "simulated"}

# Compute-path provenance per configuration. Sparse + NVFP4 has no native kernel
# in this repository, so those rows are numerical-only.
_COMPUTE_PROVENANCE: dict[str, tuple[str, str, bool]] = {
    # config: (compute label, native/simulated, native latency claim allowed)
    "A": ("dense_bf16_fa4", "native", True),
    "B": ("dense_nvfp4_qk_bf16_pv", "native", True),
    "B_sim": ("dense_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    "C": ("sparse_bf16_triton", "native", True),
    "C_null": ("sparse_bf16_triton", "native", True),
    "D": ("sparse_bf16_triton", "native", True),
    "D8": ("sparse_bf16_triton", "native", True),
    "C_rand": ("sparse_bf16_triton", "native", True),
    "E": ("sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    "F8": ("sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
    "F16": ("sparse_nvfp4_qk_bf16_pv_dequant_sim", "simulated", False),
}

# Mask source per configuration; None means dense (no mask).
_MASK_SOURCE: dict[str, str | None] = {
    "A": None,
    "B": None,
    "B_sim": None,
    "C": "bf16",
    "C_null": "bf16_null_control",
    "D": "nvfp4",
    "D8": "fp8_e4m3",
    "C_rand": "bf16_random_matched",
    "E": "nvfp4",
    "F8": "fp8_e4m3",
    "F16": "bf16",
}

_ISOLATES: dict[str, str] = {
    "A": "reference",
    "B": "quantization only",
    "B_sim": "quantization only (simulation control for E/F)",
    "C": "sparsification only",
    "C_null": "null control: bf16 mask vs bf16 mask, must be an exact identity",
    "D": "sparsification + NVFP4 wrong-mask",
    "D8": "sparsification + FP8 wrong-mask",
    "C_rand": "sparsification + equal-magnitude random wrong-mask",
    "E": "naive combined: NVFP4 compute + NVFP4 mask",
    "F8": "H3 arm: NVFP4 compute + FP8 router",
    "F16": "H3 arm: NVFP4 compute + BF16 router",
}

_DENSE_ARMS = ("A", "B", "B_sim")
# Ordered so C is built before the arms that are differenced against it.
_SPARSE_ARMS = ("C", "C_null", "D", "D8", "C_rand", "E", "F8", "F16")
_PHASE2_MAIN_ARMS = ("A", "B", "B_sim", "C", "D", "D8", "C_rand", "E", "F8", "F16")


@dataclass
class Phase2Config:
    out_dir: Path
    run_id: str
    git_commit: str
    prompt_id: str
    seed: int
    sparsities: tuple[float, ...]
    layers: tuple[int, ...]
    timesteps: tuple[int, ...]
    block_q: int
    block_k: int
    score_dtype: str
    shard_tag: str
    stage: str
    cfg_branches: tuple[str, ...]
    mechanism_layers: tuple[int, ...]
    mechanism_timesteps: tuple[int, ...]
    mechanism_query_blocks: int
    mechanism_sparsities: tuple[float, ...]
    random_seed: int
    geometry: str
    arms: tuple[str, ...]
    patch_size: tuple[int, int, int]
    emit_activation_stats: bool
    tie_diagnostic_geometries: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> Phase2Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def ints(key: str) -> tuple[int, ...]:
            return tuple(int(value) for value in raw.get(key, ()))

        def floats(key: str) -> tuple[float, ...]:
            return tuple(float(value) for value in raw.get(key, ()))

        geometry = str(raw.get("geometry", DEFAULT_GEOMETRY))
        arms = tuple(str(value) for value in raw.get("arms", _PHASE2_MAIN_ARMS))
        unknown = [arm for arm in arms if arm not in _COMPUTE_PROVENANCE]
        if unknown:
            raise ValueError(f"unknown configuration id(s) {unknown} in 'arms'")
        return cls(
            out_dir=Path(raw["out_dir"]),
            run_id=raw["run_id"],
            git_commit=raw["git_commit"],
            prompt_id=raw["prompt_id"],
            seed=int(raw["seed"]),
            sparsities=floats("sparsities"),
            layers=ints("layers"),
            timesteps=ints("timesteps"),
            block_q=int(raw.get("block_q", 128)),
            block_k=int(raw.get("block_k", 64)),
            score_dtype=str(raw.get("score_dtype", "float64")),
            shard_tag=str(raw.get("shard_tag", "shard0")),
            stage=str(raw.get("stage", "2")),
            cfg_branches=tuple(str(value) for value in raw.get("cfg_branches", ("positive", ))),
            mechanism_layers=ints("mechanism_layers"),
            mechanism_timesteps=ints("mechanism_timesteps"),
            mechanism_query_blocks=int(raw.get("mechanism_query_blocks", 16)),
            mechanism_sparsities=floats("mechanism_sparsities"),
            random_seed=int(raw.get("random_seed", 20260814)),
            geometry=geometry,
            arms=arms,
            patch_size=tuple(int(value) for value in raw.get("patch_size", (1, 2, 2))),  # type: ignore[arg-type]
            emit_activation_stats=bool(raw.get("emit_activation_stats", True)),
            tie_diagnostic_geometries=tuple(str(value) for value in raw.get("tie_diagnostic_geometries", ())),
            provenance=dict(raw.get("provenance", {})),
        )


class PrecisionSparseAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "PRECISION_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type[PrecisionSparseAttentionImpl]:
        return PrecisionSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[PrecisionSparseAttentionMetadata]:
        return PrecisionSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[PrecisionSparseAttentionMetadataBuilder]:
        return PrecisionSparseAttentionMetadataBuilder


@dataclass
class PrecisionSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int


class PrecisionSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, current_timestep: int, **kwargs: Any) -> PrecisionSparseAttentionMetadata:
        return PrecisionSparseAttentionMetadata(current_timestep=current_timestep)


def _compute_topk(sparsity: float, n_blocks: int) -> int:
    """Identical rule to Phase 1 and to VSA (``video_sparse_attn.compute_topk``)."""
    return max(1, min(math.ceil((1 - sparsity) * n_blocks), n_blocks))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else None


class PrecisionSparseAttentionImpl(AttentionImpl):
    """Dense BF16 attention plus side-channel A–F error decomposition."""

    _writers: dict[str, JsonlWriter] = {}

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
        path = os.environ.get(PHASE2_ENV_VAR, "").strip()
        self.config = Phase2Config.load(path) if path else None
        self._scheduler_timesteps: list[float] | None = None
        self._geometry_cache: BlockGeometry | None = None

    @classmethod
    def _writer(cls, config: Phase2Config) -> JsonlWriter:
        key = f"{config.run_id}/{config.shard_tag}"
        if key not in cls._writers:
            cls._writers[key] = JsonlWriter(config.out_dir / f"{config.shard_tag}.jsonl")
            atexit.register(cls.close_writers)
        return cls._writers[key]

    @classmethod
    def close_writers(cls) -> dict[str, int]:
        counts = {key: writer.records_written for key, writer in cls._writers.items()}
        for writer in cls._writers.values():
            writer.close()
        cls._writers.clear()
        return counts

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PrecisionSparseAttentionMetadata | None,
    ) -> torch.Tensor:
        config = self.config
        context = get_forward_context()
        timestep = int(getattr(context, "current_timestep", -1) or 0)
        batch = getattr(context, "forward_batch", None)
        cfg_branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"
        if (config is not None and self.layer_idx in config.layers and timestep in config.timesteps
                and cfg_branch in config.cfg_branches):
            self._measure(query, key, value, timestep, cfg_branch, batch)
        # Configuration A is what the model consumes, so every arm shares one
        # trajectory and the decomposition is paired by construction.
        return dense_bf16(query, key, value, self.softmax_scale)

    def _scheduler_timestep(self, batch: Any, step_index: int) -> float | None:
        if self._scheduler_timesteps is None:
            timesteps = getattr(batch, "timesteps", None)
            if timesteps is None:
                return None
            self._scheduler_timesteps = [float(value) for value in timesteps.flatten().tolist()]
        if not 0 <= step_index < len(self._scheduler_timesteps):
            return None
        return self._scheduler_timesteps[step_index]

    def _geometry(self, seq_len: int, device: torch.device, batch: Any) -> BlockGeometry:
        """Build (and cache) this run's token-to-block assignment.

        The cube arm needs the DiT latent grid, which only the forward batch
        knows, so it is resolved on the first measured cell and cached — the
        latent shape is constant within a run.
        """
        cached = self._geometry_cache
        if cached is not None:
            return cached
        config = self.config
        assert config is not None
        name = config.geometry
        if name.endswith("-raster"):
            geometry = raster_geometry(seq_len, config.block_q, device)
            if geometry.name != name:
                raise RuntimeError(f"geometry {name!r} requested but block_q={config.block_q} builds "
                                   f"{geometry.name!r}")
        elif name == "64x64-cube":
            raw_latent_shape = getattr(batch, "raw_latent_shape", None)
            if raw_latent_shape is None:
                raise RuntimeError("cube geometry needs forward_batch.raw_latent_shape and it is missing")
            patch_t, patch_h, patch_w = config.patch_size
            latent_t, latent_h, latent_w = (int(value) for value in raw_latent_shape[2:5])
            dit_seq_shape = (latent_t // patch_t, latent_h // patch_h, latent_w // patch_w)
            geometry = cube_geometry(dit_seq_shape, device)
            if geometry.seq_len != seq_len:
                raise RuntimeError(f"cube geometry for latent grid {dit_seq_shape} covers {geometry.seq_len} tokens "
                                   f"but the attention sequence is {seq_len}")
        else:
            raise ValueError(f"unknown geometry {name!r}")
        logger.info("sparsefp4 phase2: geometry %s", json.dumps(geometry.describe()))
        self._geometry_cache = geometry
        return geometry

    @torch.no_grad()
    def _measure(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, timestep: int, cfg_branch: str,
                 batch: Any) -> None:
        config = self.config
        assert config is not None
        assert_kernel_scale_matches(self.head_size, self.softmax_scale)
        scale = self.softmax_scale
        seq_len = query.shape[1]
        device = query.device
        score_dtype = getattr(torch, config.score_dtype)
        writer = self._writer(config)
        geometry = self._geometry(seq_len, device, batch)
        key_sizes = geometry.key_block_sizes
        n_k_blocks = geometry.n_k_blocks
        expand = geometry.query_expand

        laid_out_query = to_block_layout(query, geometry)
        laid_out_key = to_block_layout(key, geometry)
        laid_out_value = to_block_layout(value, geometry)

        routed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        saturation: dict[str, tuple[float, float]] = {}
        for precision in ("bf16", "nvfp4", "fp8_e4m3"):
            route_q, sat_q = quantize_router_input(query, precision)
            route_k, sat_k = quantize_router_input(key, precision)
            routed[precision] = (route_q, route_k)
            saturation[precision] = (sat_q, sat_k)

        scores = {
            precision: self._score(route_q, route_k, geometry, scale, score_dtype)
            for precision, (route_q, route_k) in routed.items()
        }
        reference_scores = scores[REFERENCE_PRECISION]
        # Null control: an independent second derivation of the BF16 mask. It must
        # be an exact identity, and unlike Phase 1's routing-only null control this
        # one also passes through the sparse kernel, so it gates the whole executed
        # path at each geometry rather than only the scorer.
        null_route_q, _ = quantize_router_input(query, REFERENCE_PRECISION)
        null_route_k, _ = quantize_router_input(key, REFERENCE_PRECISION)
        scores["bf16_null_control"] = self._score(null_route_q, null_route_k, geometry, scale, score_dtype)

        low_precision_q = routed["nvfp4"][0].to(query.dtype)
        low_precision_k = routed["nvfp4"][1].to(key.dtype)
        laid_out_lp_q = to_block_layout(low_precision_q, geometry)
        laid_out_lp_k = to_block_layout(low_precision_k, geometry)

        outputs: dict[str, torch.Tensor] = {"A": dense_bf16(query, key, value, scale)}
        if "B" in config.arms:
            outputs["B"] = dense_nvfp4_native(query, key, value, scale)
        if "B_sim" in config.arms:
            outputs["B_sim"] = dense_bf16(low_precision_q, low_precision_k, value, scale)

        common = {
            "prompt_id": config.prompt_id,
            "seed": config.seed,
            "layer": self.layer_idx,
            "timestep": timestep,
            "scheduler_timestep": self._scheduler_timestep(batch, timestep),
            "cfg_branch": cfg_branch,
            "reference_precision": REFERENCE_PRECISION,
            "reference_config": "A",
            "run_id": config.run_id,
            "git_commit": config.git_commit,
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_size,
            "softmax_scale": scale,
            "score_dtype": config.score_dtype,
            "block_q": geometry.block_q,
            "block_k": geometry.block_k,
            "n_q_blocks": geometry.n_q_blocks,
            "n_k_blocks": n_k_blocks,
            "kernel_block": KERNEL_BLOCK,
            "geometry": geometry.name,
            "token_order": geometry.token_order,
            "padded_seq_len": geometry.padded_len,
            "n_pad_slots": geometry.n_pad_slots,
            "force_retain_diagonal": False,
            "attention_backend": "PRECISION_SPARSE_ATTN",
            "stage": config.stage,
            "phase": "2",
        }

        for config_id in _DENSE_ARMS:
            if config_id in config.arms and config_id in outputs:
                self._emit_error_rows(writer, common, outputs, config_id, None, None, None)
        self._emit_score_resolution_row(writer, common, routed, config)
        for diagnostic in config.tie_diagnostic_geometries:
            self._emit_tie_diagnostic_row(writer, common, routed, config, diagnostic, seq_len, device, batch)
        if config.emit_activation_stats:
            self._emit_activation_row(writer, common, query, key, saturation)

        sparse_arms = tuple(arm for arm in _SPARSE_ARMS if arm in config.arms)
        for sparsity in config.sparsities:
            k = _compute_topk(sparsity, n_k_blocks)
            masks = {precision: topk_block_mask(score, k) for precision, score in scores.items()}
            generator = torch.Generator(device=device)
            generator.manual_seed(config.random_seed + 1000 * self.layer_idx + 10 * timestep + int(sparsity * 100))
            masks["bf16_random_matched"] = random_matched_mask(masks["bf16"], masks["nvfp4"], generator)

            low_precision_compute = {"E", "F8", "F16"}
            padded_outputs: dict[str, torch.Tensor] = {}
            for config_id in sparse_arms:
                mask_key = _MASK_SOURCE[config_id]
                assert mask_key is not None
                sparse_q = laid_out_lp_q if config_id in low_precision_compute else laid_out_query
                sparse_k = laid_out_lp_k if config_id in low_precision_compute else laid_out_key
                kernel_mask = expand_query_axis(masks[mask_key], expand).unsqueeze(0)
                padded_outputs[config_id] = sparse_bf16(sparse_q, sparse_k, laid_out_value, kernel_mask, key_sizes)
                outputs[config_id] = from_block_layout(padded_outputs[config_id], geometry)

            mask_keys = {key for key in (_MASK_SOURCE[arm] for arm in sparse_arms) if key is not None}
            retained_tokens = {key: retained_token_fraction(masks[key], key_sizes) for key in mask_keys}
            for config_id in sparse_arms:
                mask_key = _MASK_SOURCE[config_id]
                assert mask_key is not None
                self._emit_error_rows(writer, common, outputs, config_id, sparsity, k, masks[mask_key],
                                      masks[REFERENCE_PRECISION], saturation, retained_tokens.get(mask_key))

            if (self.layer_idx in config.mechanism_layers and timestep in config.mechanism_timesteps
                    and sparsity in config.mechanism_sparsities):
                self._emit_mechanism_rows(writer, common, padded_outputs, to_block_layout(outputs["A"], geometry),
                                          masks, reference_scores, laid_out_query, laid_out_key, geometry, sparsity, k)
            padded_outputs.clear()
            for config_id in sparse_arms:
                outputs.pop(config_id, None)

    @staticmethod
    def _score(route_q: torch.Tensor, route_k: torch.Tensor, geometry: BlockGeometry, scale: float,
               score_dtype: torch.dtype) -> torch.Tensor:
        """Mean-pooled block scores at ``geometry``, in ``score_dtype``.

        Pooling happens in the geometry's padded layout and divides by each
        block's true token count, so the pad slots the cube geometry introduces
        contribute nothing to a block's score.
        """
        pooled_q = pool_geometry_blocks(to_block_layout(route_q, geometry), geometry.query_block_sizes,
                                        geometry.block_q)
        pooled_k = pool_geometry_blocks(to_block_layout(route_k, geometry), geometry.key_block_sizes, geometry.block_k)
        return block_scores(pooled_q, pooled_k, scale, score_dtype)

    def _emit_activation_row(self, writer: JsonlWriter, common: dict[str, Any], query: torch.Tensor, key: torch.Tensor,
                             saturation: dict[str, tuple[float, float]]) -> None:
        """Activation range and e2m1 saturation per layer/head.

        Phase 1's "edge layers are sensitive" finding has a boring alternative
        explanation: those layers may simply have wider activations, so more
        elements clip at the e2m1 max. Recording the range and the saturation
        fraction next to the error lets the two be separated instead of assumed.
        """
        for head in range(self.num_heads):
            q_head = query[0, :, head].float()
            k_head = key[0, :, head].float()
            groups_q = q_head.unflatten(-1, (-1, 16)).abs().amax(dim=-1)
            groups_k = k_head.unflatten(-1, (-1, 16)).abs().amax(dim=-1)
            writer.write({
                **common,
                "record_type":
                "activation_stats",
                "head":
                head,
                "config":
                "activation_stats",
                "q_absmax":
                float(q_head.abs().max().item()),
                "k_absmax":
                float(k_head.abs().max().item()),
                "q_abs_median":
                float(q_head.abs().median().item()),
                "k_abs_median":
                float(k_head.abs().median().item()),
                "q_rms":
                float(q_head.pow(2).mean().sqrt().item()),
                "k_rms":
                float(k_head.pow(2).mean().sqrt().item()),
                # Dynamic range inside a microscale group is what decides how many
                # elements clip: e2m1 spends its 8 codes on [0, 6] * (amax/6).
                "q_group_amax_median":
                float(groups_q.median().item()),
                "k_group_amax_median":
                float(groups_k.median().item()),
                "q_intra_group_dynamic_range":
                float((groups_q.median() / q_head.abs().median().clamp(min=1e-12)).item()),
                "k_intra_group_dynamic_range":
                float((groups_k.median() / k_head.abs().median().clamp(min=1e-12)).item()),
                "sat_frac_q_nvfp4_layer":
                saturation["nvfp4"][0],
                "sat_frac_k_nvfp4_layer":
                saturation["nvfp4"][1],
                "sat_frac_q_fp8_layer":
                saturation["fp8_e4m3"][0],
                "sat_frac_k_fp8_layer":
                saturation["fp8_e4m3"][1],
            })

    def _emit_score_resolution_row(self, writer: JsonlWriter, common: dict[str, Any],
                                   routed: dict[str, tuple[torch.Tensor, torch.Tensor]], config: Phase2Config) -> None:
        """Document trap 8 on real Q/K rather than asserting it.

        Records the score magnitude, the discriminative spread, the number of
        exact boundary ties fp32 manufactures, and how many top-k decisions fp32
        gets wrong relative to fp64 — separately for each router precision, since
        the whole point is that fp32 does not penalize the arms equally.

        ``boundary_ties`` here is a **per-cell** count: the number of
        ``(head, query_block)`` pairs whose ``s_(k) == s_(k+1)``, out of
        ``num_heads * n_q_blocks``. Phase 1 emitted the same quantity **per head**
        instead, which is the whole of the ~13x scale difference between the two
        phases' tie counts; ``boundary_ties_fp32_per_head_*`` is recorded next to
        it so the two are directly comparable without re-deriving the denominator.
        """
        row: dict[str, Any] = {**common, "record_type": "score_resolution", "head": None, "config": "score_resolution"}
        geometry = self._geometry_cache
        assert geometry is not None
        for precision, (route_q, route_k) in routed.items():
            per_dtype = {}
            for name, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
                score = self._score(route_q, route_k, geometry, self.softmax_scale, dtype)
                ordered = torch.sort(score, dim=-1, descending=True, stable=True).values
                per_dtype[name] = (score, ordered)
            score32, ordered32 = per_dtype["fp32"]
            score64, ordered64 = per_dtype["fp64"]
            row[f"score_abs_median_{precision}"] = float(score64.abs().median().item())
            row[f"score_spread_median_{precision}"] = float((ordered64[..., 0] - ordered64[..., -1]).median().item())
            n_q_blocks = int(score64.shape[-2])
            for sparsity in config.sparsities:
                k = _compute_topk(sparsity, int(score64.shape[-1]))
                tag = f"{precision}_s{int(sparsity * 100)}"
                ties32 = (ordered32[..., k - 1] == ordered32[..., k]).sum(dim=-1)
                ties64 = (ordered64[..., k - 1] == ordered64[..., k]).sum(dim=-1)
                row[f"boundary_ties_fp32_{tag}"] = int(ties32.sum().item())
                row[f"boundary_ties_fp64_{tag}"] = int(ties64.sum().item())
                row[f"boundary_ties_fp32_per_head_{tag}"] = float(ties32.to(torch.float64).median().item())
                row[f"boundary_ties_fp64_per_head_{tag}"] = float(ties64.to(torch.float64).median().item())
                row[f"boundary_ties_denominator_{tag}"] = self.num_heads * n_q_blocks
                row[f"boundary_ties_per_head_denominator_{tag}"] = n_q_blocks
                mask32 = topk_block_mask(score32, k)
                mask64 = topk_block_mask(score64, k)
                changed = int((mask32 & ~mask64).sum().item())
                row[f"fp32_vs_fp64_blocks_changed_{tag}"] = changed
                row[f"fp32_vs_fp64_frac_changed_{tag}"] = changed / max(1, int(mask64.sum().item()))
        writer.write(row)

    def _emit_tie_diagnostic_row(self, writer: JsonlWriter, common: dict[str, Any],
                                 routed: dict[str, tuple[torch.Tensor, torch.Tensor]], config: Phase2Config,
                                 geometry_name: str, seq_len: int, device: torch.device, batch: Any) -> None:
        """Boundary-tie counts at one geometry, on both tie denominators.

        Exists to settle the Phase-1-vs-Phase-2 tie-count discrepancy from the
        same Q/K in one record: Phase 1 reported ~110 ties and Phase 2 ~1,400,
        which are candidate-explained by (a) per-head vs per-cell counting and
        (b) a different ``n_q_blocks``. Both denominators and the geometry are
        recorded explicitly here, at every geometry, so the comparison needs no
        reconstruction.
        """
        if geometry_name.endswith("-raster"):
            block_q = int(geometry_name.split("x", 1)[0])
            try:
                geometry = raster_geometry(seq_len, block_q, device)
            except RuntimeError:
                return
        elif geometry_name == "64x64-cube":
            raw_latent_shape = getattr(batch, "raw_latent_shape", None)
            if raw_latent_shape is None:
                return
            patch_t, patch_h, patch_w = config.patch_size
            latent_t, latent_h, latent_w = (int(value) for value in raw_latent_shape[2:5])
            dit_seq_shape = (latent_t // patch_t, latent_h // patch_h, latent_w // patch_w)
            geometry = cube_geometry(dit_seq_shape, device)
            if geometry.seq_len != seq_len:
                return
        else:
            return

        row: dict[str, Any] = {
            **common,
            "record_type": "tie_diagnostic",
            "head": None,
            "config": "tie_diagnostic",
            "diagnostic_geometry": geometry.name,
            "diagnostic_token_order": geometry.token_order,
            "diagnostic_block_q": geometry.block_q,
            "diagnostic_block_k": geometry.block_k,
            "diagnostic_n_q_blocks": geometry.n_q_blocks,
            "diagnostic_n_k_blocks": geometry.n_k_blocks,
        }
        for precision, (route_q, route_k) in routed.items():
            for name, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
                score = self._score(route_q, route_k, geometry, self.softmax_scale, dtype)
                ordered = torch.sort(score, dim=-1, descending=True, stable=True).values
                spread = ordered[..., 0] - ordered[..., -1]
                row[f"score_abs_median_{precision}_{name}"] = float(score.abs().median().item())
                row[f"score_spread_median_{precision}_{name}"] = float(spread.median().item())
                for sparsity in config.sparsities:
                    k = _compute_topk(sparsity, geometry.n_k_blocks)
                    tag = f"{precision}_{name}_s{int(sparsity * 100)}"
                    margin = ordered[..., k - 1] - ordered[..., k]
                    ties = (margin == 0).sum(dim=-1)
                    row[f"ties_per_cell_{tag}"] = int(ties.sum().item())
                    row[f"ties_per_head_median_{tag}"] = float(ties.to(torch.float64).median().item())
                    row[f"margin_raw_median_{tag}"] = float(margin.median().item())
                    row[f"margin_norm_median_{tag}"] = float((margin / spread.clamp(min=1e-12)).median().item())
        writer.write(row)

    def _emit_error_rows(
        self,
        writer: JsonlWriter,
        common: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        config_id: str,
        sparsity: float | None = None,
        k: int | None = None,
        mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        saturation: dict[str, tuple[float, float]] | None = None,
        retained_token_frac: float | None = None,
    ) -> None:
        compute, provenance, latency_ok = _COMPUTE_PROVENANCE[config_id]
        mask_source = _MASK_SOURCE[config_id]
        router_provenance = _ROUTER_PROVENANCE.get(mask_source or "", None)
        if mask_source == "bf16_random_matched":
            router_provenance = "synthetic_control"
        elif mask_source == "bf16_null_control":
            router_provenance = "native"
        sat_q = sat_k = None
        if saturation is not None and mask_source in saturation:
            sat_q, sat_k = saturation[mask_source]
        reference = outputs["A"]
        candidate = outputs[config_id]
        jaccard: list[float | None] = [None] * self.num_heads
        if mask is not None and reference_mask is not None:
            intersection = (mask & reference_mask).sum(dim=-1).sum(dim=-1)
            union = (mask | reference_mask).sum(dim=-1).sum(dim=-1)
            changed = (mask != reference_mask).any(dim=-1).sum(dim=-1)
            swapped = (reference_mask & ~mask).sum(dim=-1).to(torch.float64).mean(dim=-1)
            n_q_blocks = mask.shape[-2]
            jaccard = [
                float(intersection[head].item()) / max(1, int(union[head].item())) for head in range(self.num_heads)
            ]
        for head in range(self.num_heads):
            metrics = error_metrics(candidate[0, :, head], reference[0, :, head])
            writer.write({
                **common,
                "record_type":
                "error_decomposition",
                "head":
                head,
                "config":
                config_id,
                "isolates":
                _ISOLATES[config_id],
                "sparse":
                mask_source is not None,
                "sparsity":
                sparsity,
                "retained_fraction":
                None if sparsity is None else 1.0 - sparsity,
                "retained_token_fraction":
                retained_token_frac,
                "k_per_query_block":
                k,
                "attention_compute":
                compute,
                "compute_precision_label": ("nvfp4_qk_bf16_pv" if "nvfp4" in compute else "bf16"),
                "mask_source_precision":
                mask_source,
                "native_or_simulated":
                provenance,
                "router_native_or_simulated":
                router_provenance,
                "native_latency_claim_allowed":
                latency_ok,
                "numerical_only":
                not latency_ok,
                "rel_l2":
                metrics["rel_l2"],
                "cosine":
                metrics["cosine"],
                "max_abs":
                metrics["max_abs"],
                "mask_jaccard_vs_bf16":
                jaccard[head],
                "frac_query_blocks_changed": (None if mask is None else float(changed[head].item()) / n_q_blocks),
                "blocks_swapped_per_query_block": (None if mask is None else float(swapped[head].item())),
                "sat_frac_q":
                sat_q,
                "sat_frac_k":
                sat_k,
            })

    def _emit_mechanism_rows(
        self,
        writer: JsonlWriter,
        common: dict[str, Any],
        padded_outputs: dict[str, torch.Tensor],
        padded_reference: torch.Tensor,
        masks: dict[str, torch.Tensor],
        reference_scores: torch.Tensor,
        laid_out_query: torch.Tensor,
        laid_out_key: torch.Tensor,
        geometry: BlockGeometry,
        sparsity: float,
        k: int,
    ) -> None:
        """Direct test of the near-tie hypothesis, per query block.

        For every sampled query block this records (a) the exact dense attention
        mass the NVFP4 mask drops and adds, (b) the same quantity for the
        equal-magnitude random control, (c) the reference score gap between the
        dropped and added blocks, and (d) how much the attention output for those
        query rows actually moved. The prediction under Phase 1's rho = 0.9997 and
        ~0.001 median margin is that dropped mass is near zero and the random
        control removes far more.

        Everything here is computed in the geometry's padded layout, and the
        per-query-block error slice is restricted to that block's **valid** rows,
        so a short boundary tile at the cube geometry contributes only its real
        tokens.
        """
        config = self.config
        assert config is not None
        block_q = geometry.block_q
        n_q_blocks = geometry.n_q_blocks
        count = min(config.mechanism_query_blocks, n_q_blocks)
        query_blocks = [int(round(index * (n_q_blocks - 1) / max(1, count - 1))) for index in range(count)]
        # A pad-only query block has no valid rows and no defined error, so it is
        # skipped rather than recorded as zero. Cannot happen at Wan's grid (every
        # cube tile holds >= 8 real tokens) but the cube geometry permits it.
        query_blocks = [
            q_blk for q_blk in query_blocks if bool(geometry.valid[q_blk * block_q:(q_blk + 1) * block_q].any())
        ]
        reference_mask = masks["bf16"]
        candidate_mask = masks["nvfp4"]
        random_mask = masks["bf16_random_matched"]

        for head in range(self.num_heads):
            mass = block_attention_mass(laid_out_query, laid_out_key, head, query_blocks, geometry, self.softmax_scale)
            for row, q_blk in enumerate(query_blocks):
                if row >= mass.shape[0]:
                    continue
                mass_row = mass[row]
                ref_row = reference_mask[head, q_blk]
                cand_row = candidate_mask[head, q_blk]
                rand_row = random_mask[head, q_blk]
                dropped = ref_row & ~cand_row
                added = ~ref_row & cand_row
                agreed = ref_row & cand_row
                rand_dropped = ref_row & ~rand_row
                rand_added = ~ref_row & rand_row
                score_row = reference_scores[head, q_blk]
                sorted_scores = torch.sort(score_row, descending=True, stable=True).values
                spread = float((sorted_scores[0] - sorted_scores[-1]).clamp(min=1e-12).item())

                rows = slice(q_blk * block_q, (q_blk + 1) * block_q)
                row_valid = geometry.valid[rows]
                reference_out = padded_reference[0, rows, head][row_valid]
                record = {
                    **common,
                    "record_type":
                    "mechanism",
                    "head":
                    head,
                    "query_block":
                    q_blk,
                    "query_block_valid_tokens":
                    int(row_valid.sum().item()),
                    "sparsity":
                    sparsity,
                    "retained_fraction":
                    1.0 - sparsity,
                    "k_per_query_block":
                    k,
                    "n_swapped":
                    int(dropped.sum().item()),
                    "mass_dropped_total":
                    float(mass_row[dropped].sum().item()),
                    "mass_added_total":
                    float(mass_row[added].sum().item()),
                    "mass_dropped_mean":
                    _masked_mean(mass_row, dropped),
                    "mass_added_mean":
                    _masked_mean(mass_row, added),
                    "mass_agreed_mean":
                    _masked_mean(mass_row, agreed),
                    "mass_retained_total":
                    float(mass_row[ref_row].sum().item()),
                    "mass_excluded_mean":
                    _masked_mean(mass_row, ~ref_row),
                    "mass_random_dropped_total":
                    float(mass_row[rand_dropped].sum().item()),
                    "mass_random_added_total":
                    float(mass_row[rand_added].sum().item()),
                    "mass_random_dropped_mean":
                    _masked_mean(mass_row, rand_dropped),
                    "score_gap_swapped_raw": (None if not bool(dropped.any()) else float(
                        (score_row[dropped].mean() - score_row[added].mean()).item())),
                    "score_gap_swapped_norm": (None if not bool(dropped.any()) else float(
                        ((score_row[dropped].mean() - score_row[added].mean()) / spread).item())),
                    "score_gap_random_raw": (None if not bool(rand_dropped.any()) else float(
                        (score_row[rand_dropped].mean() - score_row[rand_added].mean()).item())),
                    "score_spread":
                    spread,
                }
                for config_id in ("C", "D", "C_rand", "E", "F16"):
                    if config_id not in padded_outputs:
                        continue
                    metrics = error_metrics(padded_outputs[config_id][0, rows, head][row_valid], reference_out)
                    record[f"qblock_rel_l2_{config_id}"] = metrics["rel_l2"]
                writer.write(record)
