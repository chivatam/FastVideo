"""Research-only backend for phase F1 of the SparseFP4 paper validation: scorer
arithmetic precision.

Study 1 established that NVFP4-*represented* Q/K perturb a top-k block mask
almost for free, because the swaps land on near-degenerate boundaries. It did
**not** establish that the cheap scorer can itself run at low precision: every
pooled value and every block dot product in study 1 was computed in fp64. This
backend closes that gap by crossing the two axes explicitly —

    representation in {bf16, nvfp4}   x   arithmetic in {fp64, fp32, bf16, fp8, nvfp4-like}

— and measuring, for each of the 12 resulting arms, both the mask change against
the fp64 reference and the *damage* that mask change does when executed through
the same BF16 block-sparse kernel.

Like study 1's Phase 2 backend, the attention the model actually consumes is
**dense BF16**, so every arm sees byte-identical Q/K/V from one denoising
trajectory and all comparisons are exactly paired at
``(prompt, seed, layer, head, timestep, cfg_branch, sparsity)``.

Three properties are enforced rather than hoped for:

* the retained block count ``k`` is identical across all arms at every cell
  (``mask_comparison`` raises otherwise), so precision changes *which* blocks are
  selected and never *how many*;
* an fp64 **shadow** score is always computed from R0's pooling and is the only
  source of boundary margins and tie counts, so a low-precision scorer's own ties
  cannot contaminate the scientific boundary reference (study 1's trap 8);
* the matched-random control changes exactly the same number of blocks per query
  block as the precision arm it is paired with, so the isolation ratio compares
  equal-magnitude perturbations.

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=SCORER_PRECISION_ATTN
    FASTVIDEO_SPARSEFP4_F1=/path/to/f1_config.json
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
from fastvideo.attention.backends.sparsefp4_mask_metrics import (boundary_diagnostics, deployed_tie_count, mask_hash,
                                                                 mask_comparison, spearman_rho)
from fastvideo.attention.backends.sparsefp4_numerics import (KERNEL_BLOCK, BlockGeometry, assert_kernel_scale_matches,
                                                             cube_geometry, dense_bf16, error_metrics,
                                                             expand_query_axis, from_block_layout, random_matched_mask,
                                                             raster_geometry, retained_token_fraction, sparse_bf16,
                                                             to_block_layout, topk_block_mask)
from fastvideo.attention.backends.sparsefp4_scorer_precision import (
    ARMS_BY_ID, LADDER_POSITION, REFERENCE_ARM, SCORER_ARMS, STUDY1_ARM, ScorerArm, ambient_fp32_state, autocast_state,
    declared_precision_arithmetic, pool_blocks_precision, quantize_pooled_fp8_e4m3, quantize_pooled_nvfp4,
    score_blocks_fp8_native, score_blocks_precision)
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

F1_ENV_VAR = "FASTVIDEO_SPARSEFP4_F1"
# Opt-in Q/K capture for F4.5's native-vs-simulated NVFP4 comparison. Off unless set,
# so it cannot perturb a measurement run. Points at a JSON spec:
#   {"path": "<out.pt>", "layers": [...], "timesteps": [...]}
F45_CAPTURE_ENV_VAR = "FASTVIDEO_SPARSEFP4_F45_CAPTURE"
DEFAULT_GEOMETRY = "128x64-raster"


@dataclass
class F1Config:
    out_dir: Path
    run_id: str
    git_commit: str
    prompt_id: str
    seed: int
    sparsities: tuple[float, ...]
    layers: tuple[int, ...]
    timesteps: tuple[int, ...]
    heads: tuple[int, ...]
    cfg_branches: tuple[str, ...]
    arms: tuple[str, ...]
    geometry: str
    block_q: int
    patch_size: tuple[int, int, int]
    random_seed: int
    spearman_query_block_stride: int
    shard_tag: str
    stage: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> F1Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def ints(key: str, default: tuple[int, ...] = ()) -> tuple[int, ...]:
            return tuple(int(value) for value in raw.get(key, default))

        arms = tuple(str(value) for value in raw.get("arms", tuple(arm.arm_id for arm in SCORER_ARMS)))
        unknown = [arm for arm in arms if arm not in ARMS_BY_ID]
        if unknown:
            raise ValueError(f"unknown scorer arm id(s) {unknown}")
        if REFERENCE_ARM not in arms:
            raise ValueError(f"the reference arm {REFERENCE_ARM} must always be measured")
        return cls(
            out_dir=Path(raw["out_dir"]),
            run_id=raw["run_id"],
            git_commit=raw["git_commit"],
            prompt_id=raw["prompt_id"],
            seed=int(raw["seed"]),
            sparsities=tuple(float(value) for value in raw.get("sparsities", (0.90, ))),
            layers=ints("layers"),
            timesteps=ints("timesteps"),
            heads=ints("heads"),
            cfg_branches=tuple(str(value) for value in raw.get("cfg_branches", ("positive", ))),
            arms=arms,
            geometry=str(raw.get("geometry", DEFAULT_GEOMETRY)),
            block_q=int(raw.get("block_q", 128)),
            patch_size=tuple(int(value) for value in raw.get("patch_size", (1, 2, 2))),  # type: ignore[arg-type]
            random_seed=int(raw.get("random_seed", 20260816)),
            spearman_query_block_stride=int(raw.get("spearman_query_block_stride", 32)),
            shard_tag=str(raw.get("shard_tag", "shard0")),
            stage=str(raw.get("stage", "F1")),
            provenance=dict(raw.get("provenance", {})),
        )


class ScorerPrecisionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SCORER_PRECISION_ATTN"

    @staticmethod
    def get_impl_cls() -> type[ScorerPrecisionAttentionImpl]:
        return ScorerPrecisionAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[ScorerPrecisionAttentionMetadata]:
        return ScorerPrecisionAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[ScorerPrecisionAttentionMetadataBuilder]:
        return ScorerPrecisionAttentionMetadataBuilder


@dataclass
class ScorerPrecisionAttentionMetadata(AttentionMetadata):
    current_timestep: int


class ScorerPrecisionAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, current_timestep: int, **kwargs: Any) -> ScorerPrecisionAttentionMetadata:
        return ScorerPrecisionAttentionMetadata(current_timestep=current_timestep)


def compute_topk(sparsity: float, n_blocks: int) -> int:
    """Identical rule to study 1 and to VSA's ``compute_topk``."""
    return max(1, min(math.ceil((1 - sparsity) * n_blocks), n_blocks))


class ScorerPrecisionAttentionImpl(AttentionImpl):
    """Dense BF16 attention plus a side-channel scorer-arithmetic ablation."""

    _writers: dict[str, JsonlWriter] = {}
    _fp8_native_available: bool | None = None
    _captures: list[dict[str, Any]] = []
    _capture_registered: bool = False

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
        path = os.environ.get(F1_ENV_VAR, "").strip()
        if not path:
            raise RuntimeError(f"{F1_ENV_VAR} must point at an F1 config JSON when "
                               "FASTVIDEO_ATTENTION_BACKEND=SCORER_PRECISION_ATTN")
        self.config = F1Config.load(path)
        self._geometry_cache: BlockGeometry | None = None
        self._scheduler_timesteps: list[float] | None = None
        capture_spec = os.environ.get(F45_CAPTURE_ENV_VAR, "").strip()
        self._capture: dict[str, Any] | None = json.loads(capture_spec) if capture_spec else None

    @classmethod
    def _writer(cls, config: F1Config) -> JsonlWriter:
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
        attn_metadata: ScorerPrecisionAttentionMetadata | None,
    ) -> torch.Tensor:
        config = self.config
        context = get_forward_context()
        timestep = int(getattr(context, "current_timestep", -1) or 0)
        batch = getattr(context, "forward_batch", None)
        cfg_branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"
        if (self.layer_idx in config.layers and timestep in config.timesteps and cfg_branch in config.cfg_branches):
            # Capture the ambient numerics state *before* neutralizing it, so records
            # describe the trajectory's real environment rather than the guard's.
            ambient = {**ambient_fp32_state(), **autocast_state()}
            # The denoising loop runs under autocast(bf16), which would silently cast
            # every fp32 arm's matmul down to bf16 and collapse the arithmetic ladder.
            with declared_precision_arithmetic():
                self._measure(query, key, value, timestep, cfg_branch, batch, ambient)
        if self._capture is not None and cfg_branch == "positive":
            self._maybe_capture(query, key, timestep)
        # The model consumes dense BF16, so every arm shares one trajectory.
        return dense_bf16(query, key, value, self.softmax_scale)

    def _maybe_capture(self, query: torch.Tensor, key: torch.Tensor, timestep: int) -> None:
        """Stash raw pre-backend Q/K for the F4.5 fidelity comparison.

        Kept on CPU because the tensors are large and the comparison is offline. Only
        the positive CFG branch is captured — the two branches are the same
        distribution for this purpose and capturing both would double the file for no
        added information.
        """
        spec = self._capture
        if spec is None:
            return
        if self.layer_idx not in spec.get("layers", []) or timestep not in spec.get("timesteps", []):
            return
        for which, tensor in (("query", query), ("key", key)):
            type(self)._captures.append({
                "layer": self.layer_idx,
                "timestep": timestep,
                "which": which,
                "tensor": tensor.detach().to("cpu", copy=True),
            })
        if not type(self)._capture_registered:
            type(self)._capture_registered = True
            atexit.register(type(self)._flush_captures, str(spec["path"]))

    @classmethod
    def _flush_captures(cls, path: str) -> None:
        if not cls._captures:
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"captures": cls._captures}, destination)
        logger.info("sparsefp4 F4.5: wrote %d captured tensors to %s", len(cls._captures), destination)
        cls._captures = []

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
        cached = self._geometry_cache
        if cached is not None:
            return cached
        config = self.config
        name = config.geometry
        if name.endswith("-raster"):
            geometry = raster_geometry(seq_len, config.block_q, device)
            if geometry.name != name:
                raise RuntimeError(f"geometry {name!r} requested but block_q={config.block_q} builds {geometry.name!r}")
        elif name == "64x64-cube":
            raw_latent_shape = getattr(batch, "raw_latent_shape", None)
            if raw_latent_shape is None:
                raise RuntimeError("cube geometry needs forward_batch.raw_latent_shape and it is missing")
            patch_t, patch_h, patch_w = config.patch_size
            latent_t, latent_h, latent_w = (int(value) for value in raw_latent_shape[2:5])
            geometry = cube_geometry((latent_t // patch_t, latent_h // patch_h, latent_w // patch_w), device)
            if geometry.seq_len != seq_len:
                raise RuntimeError(f"cube geometry covers {geometry.seq_len} tokens but attention has {seq_len}")
        else:
            raise ValueError(f"unknown geometry {name!r}")
        logger.info("sparsefp4 F1: geometry %s", json.dumps(geometry.describe()))
        self._geometry_cache = geometry
        return geometry

    def _arm_scores(self, arm: ScorerArm, representations: dict[str, torch.Tensor],
                    geometry: BlockGeometry) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Scores for one arm, plus that arm's own fp64 shadow and its labels.

        Two scores come back per arm and the distinction is the point of the
        phase. ``deployed`` is what a kernel implementing this arm would actually
        compute. ``shadow`` re-scores the *same pooled vectors* in fp64, so the
        difference between the two masks isolates the score-matmul precision from
        the pooling precision — an attribution that a single number could not
        provide.
        """
        route_q = representations[arm.representation + ":q"]
        route_k = representations[arm.representation + ":k"]
        pooled_q, pool_semantics = pool_blocks_precision(to_block_layout(route_q, geometry), geometry.query_block_sizes,
                                                         geometry.block_q, arm.pool_arithmetic, arm.pool_accumulate)
        pooled_k, _ = pool_blocks_precision(to_block_layout(route_k, geometry), geometry.key_block_sizes,
                                            geometry.block_k, arm.pool_arithmetic, arm.pool_accumulate)

        pooled_saturation: float | None = None
        native_or_simulated = arm.native_or_simulated
        if arm.quantize_pooled == "fp8_e4m3":
            pooled_q, sat_q = quantize_pooled_fp8_e4m3(pooled_q)
            pooled_k, sat_k = quantize_pooled_fp8_e4m3(pooled_k)
            pooled_saturation = max(sat_q, sat_k)
            pool_semantics += "+pooled_rounded_to_fp8_e4m3_per_head_amax"
        elif arm.quantize_pooled == "nvfp4":
            pooled_q, sat_q = quantize_pooled_nvfp4(pooled_q)
            pooled_k, sat_k = quantize_pooled_nvfp4(pooled_k)
            pooled_saturation = max(sat_q, sat_k)
            pool_semantics += "+pooled_rounded_to_nvfp4_e2m1_per16_e4m3_sf"

        if arm.score_arithmetic == "fp8_e4m3":
            native = score_blocks_fp8_native(pooled_q, pooled_k, self.softmax_scale)
            if native is not None:
                deployed, score_semantics = native
                type(self)._fp8_native_available = True
            else:
                # No native FP8 GEMM for these shapes: fall back to an fp32 dot on
                # FP8-rounded inputs and relabel the arm simulated. Recorded, not
                # silently substituted.
                deployed, score_semantics = score_blocks_precision(pooled_q, pooled_k, self.softmax_scale, "fp32",
                                                                   "native")
                score_semantics += "_FALLBACK_no_native_fp8_gemm"
                native_or_simulated = "simulated"
                type(self)._fp8_native_available = False
        else:
            deployed, score_semantics = score_blocks_precision(pooled_q, pooled_k, self.softmax_scale,
                                                               arm.score_arithmetic, arm.score_accumulate)

        shadow, _ = score_blocks_precision(pooled_q, pooled_k, self.softmax_scale, "fp64", "native")
        labels = {
            "representation_precision": arm.representation,
            "pool_precision": arm.pool_arithmetic,
            "pool_accumulate": arm.pool_accumulate,
            "score_precision": arm.score_arithmetic,
            "score_accumulate": arm.score_accumulate,
            "pooled_quantized_to": arm.quantize_pooled,
            "pool_semantics": pool_semantics,
            "score_semantics": score_semantics,
            "arm_label": arm.label,
            "arithmetic_ladder_position": LADDER_POSITION[arm.arm_id],
            "native_or_simulated": native_or_simulated,
            "pooled_saturation_frac": pooled_saturation,
            "purpose": arm.purpose,
        }
        return deployed, shadow, labels

    @torch.no_grad()
    def _measure(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, timestep: int, cfg_branch: str,
                 batch: Any, ambient: dict[str, Any]) -> None:
        config = self.config
        assert_kernel_scale_matches(self.head_size, self.softmax_scale)
        seq_len = query.shape[1]
        device = query.device
        writer = self._writer(config)
        geometry = self._geometry(seq_len, device, batch)
        expand = geometry.query_expand
        key_sizes = geometry.key_block_sizes
        heads = config.heads or tuple(range(self.num_heads))

        # One representation per precision, shared by every arm that uses it, so
        # the representation axis is held byte-identical across arithmetic arms.
        representations: dict[str, torch.Tensor] = {}
        repr_saturation: dict[str, float] = {}
        for precision in sorted({arm.representation for arm in (ARMS_BY_ID[a] for a in config.arms)}):
            route_q, sat_q = quantize_router_input(query, precision)
            route_k, sat_k = quantize_router_input(key, precision)
            representations[precision + ":q"] = route_q
            representations[precision + ":k"] = route_k
            repr_saturation[precision] = max(sat_q, sat_k)

        arm_scores: dict[str, torch.Tensor] = {}
        arm_shadow: dict[str, torch.Tensor] = {}
        arm_labels: dict[str, dict[str, Any]] = {}
        for arm_id in config.arms:
            arm = ARMS_BY_ID[arm_id]
            deployed, shadow, labels = self._arm_scores(arm, representations, geometry)
            arm_scores[arm_id] = deployed
            arm_shadow[arm_id] = shadow
            arm_labels[arm_id] = labels
        # The single scientific boundary reference: R0's pooling, exact arithmetic.
        reference_shadow = arm_shadow[REFERENCE_ARM]

        laid_out_query = to_block_layout(query, geometry)
        laid_out_key = to_block_layout(key, geometry)
        laid_out_value = to_block_layout(value, geometry)
        dense_reference = dense_bf16(query, key, value, self.softmax_scale)

        common = {
            "run_id": config.run_id,
            "git_sha": config.git_commit,
            "experiment_family": "scorer_precision",
            "prompt_id": config.prompt_id,
            "seed": config.seed,
            "model_id": config.provenance.get("model_id"),
            "model_revision": config.provenance.get("model_revision"),
            "resolution": config.provenance.get("resolution"),
            "frames": config.provenance.get("frames"),
            "num_steps": config.provenance.get("num_inference_steps"),
            "guidance": config.provenance.get("guidance_scale"),
            "layer": self.layer_idx,
            "timestep": timestep,
            "scheduler_timestep": self._scheduler_timestep(batch, timestep),
            "cfg_branch": cfg_branch,
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_size,
            "softmax_scale": self.softmax_scale,
            "geometry": geometry.name,
            "token_order": geometry.token_order,
            "block_q": geometry.block_q,
            "block_k": geometry.block_k,
            "n_q_blocks": geometry.n_q_blocks,
            "n_k_blocks": geometry.n_k_blocks,
            "padded_seq_len": geometry.padded_len,
            "n_pad_slots": geometry.n_pad_slots,
            "kernel_block": KERNEL_BLOCK,
            "routing_interface": "research_mean_pooled_block_scorer",
            "attention_compute_precision": "bf16",
            "resolved_attention_backend": "SCORER_PRECISION_ATTN",
            **ambient,
            "reference_arm": REFERENCE_ARM,
            "study1_arm": STUDY1_ARM,
            "force_retain_diagonal": False,
            "stage": config.stage,
            "phase": "F1",
        }

        for sparsity in config.sparsities:
            k = compute_topk(sparsity, geometry.n_k_blocks)
            masks = {arm_id: topk_block_mask(scores, k) for arm_id, scores in arm_scores.items()}
            shadow_masks = {arm_id: topk_block_mask(scores, k) for arm_id, scores in arm_shadow.items()}
            reference_mask = masks[REFERENCE_ARM]

            executed: dict[str, torch.Tensor] = {}
            random_masks: dict[str, torch.Tensor] = {}
            for arm_id in config.arms:
                kernel_mask = expand_query_axis(masks[arm_id], expand).unsqueeze(0)
                executed[arm_id] = from_block_layout(
                    sparse_bf16(laid_out_query, laid_out_key, laid_out_value, kernel_mask, key_sizes), geometry)
                if arm_id == REFERENCE_ARM:
                    continue
                # Matched-random control: same swap count per query block as this
                # arm, blocks chosen uniformly. Seeded per (layer, timestep,
                # sparsity, arm) so it is reproducible and independent of arm order.
                generator = torch.Generator(device=device)
                generator.manual_seed(config.random_seed + 1_000_000 * self.layer_idx + 1_000 * timestep +
                                      int(sparsity * 100) * 17 + sum(ord(c) for c in arm_id))
                random_masks[arm_id] = random_matched_mask(reference_mask, masks[arm_id], generator)
                random_kernel_mask = expand_query_axis(random_masks[arm_id], expand).unsqueeze(0)
                executed[f"{arm_id}:random"] = from_block_layout(
                    sparse_bf16(laid_out_query, laid_out_key, laid_out_value, random_kernel_mask, key_sizes), geometry)

            retained = {arm_id: retained_token_fraction(masks[arm_id], key_sizes) for arm_id in config.arms}
            reference_error = {
                head: error_metrics(executed[REFERENCE_ARM][0, :, head], dense_reference[0, :, head])
                for head in heads
            }

            for arm_id in config.arms:
                self._emit_arm_rows(writer, common, arm_id, arm_labels[arm_id], masks, shadow_masks, random_masks,
                                    executed, dense_reference, reference_error, reference_shadow, arm_scores[arm_id],
                                    heads, sparsity, k, retained[arm_id], repr_saturation, geometry, config)
            executed.clear()

    def _emit_arm_rows(
        self,
        writer: JsonlWriter,
        common: dict[str, Any],
        arm_id: str,
        labels: dict[str, Any],
        masks: dict[str, torch.Tensor],
        shadow_masks: dict[str, torch.Tensor],
        random_masks: dict[str, torch.Tensor],
        executed: dict[str, torch.Tensor],
        dense_reference: torch.Tensor,
        reference_error: dict[int, dict[str, float | None]],
        reference_shadow: torch.Tensor,
        deployed_scores: torch.Tensor,
        heads: tuple[int, ...],
        sparsity: float,
        k: int,
        retained_token_frac: float,
        repr_saturation: dict[str, float],
        geometry: BlockGeometry,
        config: F1Config,
    ) -> None:
        reference_mask = masks[REFERENCE_ARM]
        candidate_mask = masks[arm_id]
        study1_mask = masks.get(STUDY1_ARM)

        for head in heads:
            comparison = mask_comparison(candidate_mask[head], reference_mask[head])
            boundary = boundary_diagnostics(reference_shadow[head], candidate_mask[head], reference_mask[head], k)
            # Pooling-vs-matmul attribution: how much of this arm's mask change
            # survives re-scoring the same pooled vectors in fp64.
            pooling_only = mask_comparison(shadow_masks[arm_id][head], reference_mask[head])

            candidate_metrics = error_metrics(executed[arm_id][0, :, head], dense_reference[0, :, head])
            reference_metrics = reference_error[head]
            sparsification = reference_metrics["rel_l2"]
            wrong_mask_excess = (None if candidate_metrics["rel_l2"] is None or sparsification is None else
                                 candidate_metrics["rel_l2"] - sparsification)

            random_key = f"{arm_id}:random"
            random_excess = None
            random_metrics: dict[str, float | None] = {"rel_l2": None, "cosine": None, "max_abs": None}
            random_comparison: dict[str, float | int] | None = None
            if random_key in executed:
                random_metrics = error_metrics(executed[random_key][0, :, head], dense_reference[0, :, head])
                if random_metrics["rel_l2"] is not None and sparsification is not None:
                    random_excess = random_metrics["rel_l2"] - sparsification
                random_comparison = mask_comparison(random_masks[arm_id][head], reference_mask[head])

            vs_study1 = None
            if study1_mask is not None and arm_id != STUDY1_ARM:
                vs_study1 = mask_comparison(candidate_mask[head], study1_mask[head])["jaccard"]

            record: dict[str, Any] = {
                **common,
                **labels,
                "record_type":
                "scorer_precision",
                "arm":
                arm_id,
                "head":
                head,
                "sparsity":
                sparsity,
                "retained_fraction":
                1.0 - sparsity,
                "retained_k":
                k,
                "retained_token_fraction":
                retained_token_frac,
                "representation_saturation_frac":
                repr_saturation.get(labels["representation_precision"]),
                "mask_reference_hash":
                mask_hash(reference_mask[head]),
                "mask_candidate_hash":
                mask_hash(candidate_mask[head]),
                "null_control":
                arm_id == REFERENCE_ARM,
                # --- mask metrics vs the fp64 reference arm ---
                **{
                    f"{name}": value
                    for name, value in comparison.items()
                },
                "jaccard_vs_study1_arm":
                vs_study1,
                # --- pooling-only attribution (same pooled vectors, fp64 rescore) ---
                "jaccard_pooling_only":
                pooling_only["jaccard"],
                "num_swaps_pooling_only":
                pooling_only["num_swaps"],
                # --- boundary diagnostics, fp64 shadow ---
                **boundary,
                "deployed_score_ties":
                deployed_tie_count(deployed_scores[head], k),
                # --- damage metrics, identical BF16 sparse execution ---
                "rel_l2_vs_dense_bf16":
                candidate_metrics["rel_l2"],
                "cosine_vs_dense_bf16":
                candidate_metrics["cosine"],
                "max_abs_vs_dense_bf16":
                candidate_metrics["max_abs"],
                "sparsification_error":
                sparsification,
                "wrong_mask_excess":
                wrong_mask_excess,
                "wrong_mask_share_of_sparsification":
                (None if wrong_mask_excess is None or not sparsification else wrong_mask_excess / sparsification),
                "random_matched_rel_l2":
                random_metrics["rel_l2"],
                "random_matched_excess":
                random_excess,
                "random_matched_num_swaps": (None if random_comparison is None else random_comparison["num_swaps"]),
                # Per-cell ratio, recorded for completeness only. The reportable
                # isolation ratio is median(random_excess)/median(wrong_mask_excess)
                # per FOLLOWUP_SPEC: a per-cell quotient explodes wherever the
                # denominator is near zero, which is most cells here, so the
                # analysis must not average this column.
                "random_over_candidate_ratio_percell":
                (None if random_excess is None or not wrong_mask_excess else random_excess / wrong_mask_excess),
            }
            if head == heads[0]:
                # Spearman over *all* query blocks (no striding): the vectorized
                # tie-aware ranking costs ~0.5 ms per head, so the sampling caveat
                # the strided version would have carried is unnecessary.
                record["spearman_rho_vs_reference"] = spearman_rho(deployed_scores[head], reference_shadow[head])
                record["spearman_query_block_stride"] = 1
            writer.write(record)
