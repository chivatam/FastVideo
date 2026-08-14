"""Research-only attention backend that measures block-sparse routing stability.

Phase 1 of the SparseFP4 study (`.agents/skills/sparsefp4-video-attention/`).
The attention compute is a **pass-through to the dense BF16 FA kernel**, byte for
byte what ``FLASH_ATTN`` runs, so the denoising trajectory is unchanged and
routing metrics are computed on the side. This is the discipline
``video_sparse_attn_h3_probe`` uses: instrument without perturbing.

Q and K are observed at the only point where post-RMSNorm, post-RoPE,
post-SP-layout tensors exist as named values (``AttentionImpl.forward``), which is
what `references/EXPERIMENT_SPEC.md` 2.1 requires.

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=ROUTING_PROBE_ATTN
    FASTVIDEO_SPARSEFP4_PROBE=/path/to/probe_config.json

With the env var unset the backend is a plain dense pass-through.
"""

from __future__ import annotations

import atexit
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, compute_topk, construct_variable_block_sizes,
                                                            get_tile_partition_indices)
from fastvideo.attention.utils.flash_attn_default import flash_attn_func_compilable
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

PROBE_ENV_VAR = "FASTVIDEO_SPARSEFP4_PROBE"

E2M1_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
MXFP_BLOCK_SIZE = 16
MARGIN_EPS = 1e-12

REFERENCE_PRECISION = "bf16"
SCORER_DIAGNOSTIC = "diag_mean_pool_dot"
SCORER_VSA = "vsa_tile_mean_pool_dot"

# Per-arm provenance strings recorded verbatim in every JSONL record. `native`
# means the router consumed values a real FastVideo/kernel quantizer produced;
# `simulated` means a deterministic quant/dequant stands in for a kernel that
# does not exist on this path (EXPERIMENT_SPEC 4.4).
_ARM_PROVENANCE: dict[str, tuple[str, str]] = {
    "bf16": ("native", "identity(bf16_capture)"),
    "fp8_e4m3": ("simulated", "torch.float8_e4m3fn_cast+per_head_amax_scale"),
    "nvfp4": ("native", "flashinfer.fp4_quantize_sm100_via_nvfp4_quantize_for_fa4+host_e2m1_decode"),
    "nvfp4_sim": ("simulated", "deterministic_e2m1_roundtrip_per16_e4m3_scale"),
}


def probe_config_path() -> str | None:
    value = os.environ.get(PROBE_ENV_VAR, "").strip()
    return value or None


@dataclass
class ProbeConfig:
    out_dir: Path
    run_id: str
    git_commit: str
    prompt_id: str
    seed: int
    sparsities: tuple[float, ...]
    routing_precisions: tuple[str, ...]
    block_q: int
    block_k: int
    stage: str
    phase: str
    shard_tag: str
    patch_size: tuple[int, int, int]
    measure_timestep_stride: int
    score_dtype: str
    null_control_layer_stride: int
    null_control_timestep_stride: int
    spearman_timestep_stride: int
    run_vsa_scorer: bool
    provenance: dict[str, Any]

    @classmethod
    def load(cls, path: str) -> ProbeConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            out_dir=Path(raw["out_dir"]),
            run_id=raw["run_id"],
            git_commit=raw["git_commit"],
            prompt_id=raw["prompt_id"],
            seed=int(raw["seed"]),
            sparsities=tuple(float(value) for value in raw["sparsities"]),
            routing_precisions=tuple(str(value) for value in raw["routing_precisions"]),
            block_q=int(raw.get("block_q", 128)),
            block_k=int(raw.get("block_k", 64)),
            stage=str(raw.get("stage", "1")),
            phase=str(raw.get("phase", "1")),
            shard_tag=str(raw.get("shard_tag", "shard0")),
            patch_size=tuple(int(value) for value in raw.get("patch_size", (1, 2, 2))),  # type: ignore[arg-type]
            measure_timestep_stride=int(raw.get("measure_timestep_stride", 1)),
            score_dtype=str(raw.get("score_dtype", "float32")),
            null_control_layer_stride=int(raw.get("null_control_layer_stride", 1)),
            null_control_timestep_stride=int(raw.get("null_control_timestep_stride", 1)),
            spearman_timestep_stride=int(raw.get("spearman_timestep_stride", 10)),
            run_vsa_scorer=bool(raw.get("run_vsa_scorer", False)),
            provenance=dict(raw.get("provenance", {})),
        )


class RoutingProbeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "ROUTING_PROBE_ATTN"

    @staticmethod
    def get_impl_cls() -> type[RoutingProbeAttentionImpl]:
        return RoutingProbeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[RoutingProbeAttentionMetadata]:
        return RoutingProbeAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[RoutingProbeAttentionMetadataBuilder]:
        return RoutingProbeAttentionMetadataBuilder


@dataclass
class RoutingProbeAttentionMetadata(AttentionMetadata):
    current_timestep: int


class RoutingProbeAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, current_timestep: int, **kwargs: Any) -> RoutingProbeAttentionMetadata:
        return RoutingProbeAttentionMetadata(current_timestep=current_timestep)


class JsonlWriter:
    """One append-only JSONL shard per process, opened on first record."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any = None
        self.records_written = 0

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.records_written += 1
        # FastVideo runs the pipeline in a worker subprocess whose exit is not
        # guaranteed to run interpreter finalizers, so flush per forward-worth of
        # records rather than relying on close().
        if self.records_written % 2048 == 0:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    return torch.cat([magnitudes, -magnitudes])


def _per16_e4m3_scales(tensor: torch.Tensor) -> torch.Tensor:
    """Per-16-element E4M3 scale factors along the head-dim axis, global scale 1.0.

    Mirrors the NVFP4 recipe flashinfer's ``fp4_quantize_sm100`` implements when
    ``_nvfp4_quantize_for_fa4`` passes ``global_sf = 1.0``
    (fastvideo/attention/backends/flash_attn.py:81-82).
    """
    grouped = tensor.float().unflatten(-1, (-1, MXFP_BLOCK_SIZE))
    amax = grouped.abs().amax(dim=-1)
    return (amax / E2M1_MAX).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()


def quantize_router_input(tensor: torch.Tensor, precision: str) -> tuple[torch.Tensor, float]:
    """Return (values the router sees, fraction of elements at the format's max).

    ``tensor`` is the BF16 pre-backend Q or K in ``[B, S, H, D]``. Every arm is
    derived from this one capture so the capture point is identical across arms
    by construction (EXPERIMENT_SPEC 2.1).
    """
    if precision == "bf16":
        return tensor, 0.0
    if precision == "fp8_e4m3":
        amax = tensor.float().abs().amax(dim=(0, 1, 3), keepdim=True)
        scale = (amax / FP8_E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
        scaled = (tensor.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        quantized = scaled.to(torch.float8_e4m3fn).float()
        return quantized * scale, (quantized.abs() >= FP8_E4M3_MAX).float().mean().item()
    if precision == "nvfp4":
        seqlen = tensor.shape[1]
        packed, _ = _nvfp4_quantize_for_fa4(tensor)
        as_bytes = packed[:, :seqlen].view(torch.uint8)
        nibbles = torch.stack([(as_bytes & 0x0F).long(), ((as_bytes >> 4) & 0x0F).long()], dim=-1).flatten(-2)
        codes = _e2m1_lut(tensor.device)[nibbles]
        scales = _per16_e4m3_scales(tensor)
        dequantized = (codes.unflatten(-1, (-1, MXFP_BLOCK_SIZE)) * scales.unsqueeze(-1)).flatten(-2)
        return dequantized, (codes.abs() >= E2M1_MAX).float().mean().item()
    if precision == "nvfp4_sim":
        scales = _per16_e4m3_scales(tensor)
        safe = scales.clamp(min=2.0**-9)
        expanded = safe.unsqueeze(-1).expand(*safe.shape, MXFP_BLOCK_SIZE).flatten(-2)
        scaled = (tensor.float() / expanded).clamp(-E2M1_MAX, E2M1_MAX).abs()
        # Ties-to-even over the 8 E2M1 magnitudes: alternate strict/non-strict
        # comparisons at each midpoint so exact midpoints land on the code with
        # a zero mantissa bit.
        index = ((scaled > 0.25).long() + (scaled >= 0.75).long() + (scaled > 1.25).long() + (scaled >= 1.75).long() +
                 (scaled > 2.5).long() + (scaled >= 3.5).long() + (scaled > 5.0).long())
        magnitudes = _e2m1_lut(tensor.device)[:8][index]
        codes = torch.sign(tensor.float()) * magnitudes
        dequantized = torch.where(expanded > 0, codes * expanded, torch.zeros_like(codes))
        return dequantized, (magnitudes >= E2M1_MAX).float().mean().item()
    raise ValueError(f"unknown routing precision {precision!r}")


def pool_blocks_1d(x: torch.Tensor, block: int) -> torch.Tensor:
    """Masked-mean pool ``[B, S, H, D]`` into ``[H, ceil(S/block), D]`` fp32.

    The ragged final block is kept and averaged over its valid tokens only
    (EXPERIMENT_SPEC 3.3): zero-padding would shrink its pooled vector toward
    the origin and bias selection against it by an amount that differs per arm.
    """
    batch, seq_len, heads, dim = x.shape
    n_blocks = math.ceil(seq_len / block)
    padded = n_blocks * block
    if padded != seq_len:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padded - seq_len))
    pooled = x.view(batch, n_blocks, block, heads, dim).sum(dim=2, dtype=torch.float32)
    counts = torch.full((n_blocks, ), float(block), device=x.device, dtype=torch.float32)
    counts[-1] = float(seq_len - (n_blocks - 1) * block)
    return (pooled / counts.view(1, -1, 1, 1)).permute(0, 2, 1, 3)[0]


def pool_blocks_vsa_tiles(x: torch.Tensor, order: torch.Tensor, non_pad_index: torch.Tensor,
                          block_sizes: torch.Tensor) -> torch.Tensor:
    """Masked-mean pool ``[B, S, H, D]`` into VSA's (4,4,4) spatio-temporal tiles.

    Reproduces ``VideoSparseAttentionImpl.tile`` exactly: raster tokens are
    gathered in tile-contiguous order and scattered into a zero-filled padded
    buffer, so summing and dividing by the true tile size is the masked mean.
    """
    batch, _, heads, dim = x.shape
    n_tiles = block_sizes.numel()
    tile_elems = int(math.prod(VSA_TILE_SIZE))
    buffer = x.new_zeros((batch, n_tiles * tile_elems, heads, dim))
    buffer[:, non_pad_index] = x[:, order]
    pooled = buffer.view(batch, n_tiles, tile_elems, heads, dim).sum(dim=2, dtype=torch.float32)
    return (pooled / block_sizes.to(torch.float32).view(1, -1, 1, 1)).permute(0, 2, 1, 3)[0]


def sort_scores_descending(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable descending sort, so exact ties resolve to the lowest key-block index.

    EXPERIMENT_SPEC 1.3: low-precision quantization *creates* exact ties that do
    not exist in BF16, and an unstable tie-break would turn a deterministic
    quantization artifact into run-to-run noise that inflates measured
    instability.
    """
    return torch.sort(scores, dim=-1, descending=True, stable=True)


def average_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Tie-corrected average ranks along the last axis, for Spearman rho."""
    n = scores.shape[-1]
    order = torch.argsort(scores, dim=-1, stable=True)
    positions = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32).expand_as(scores)
    ranks = torch.empty_like(positions).scatter_(-1, order, positions)
    ordered = torch.gather(scores, -1, order)
    # Average the ranks of each run of equal values: cumulative sums of the
    # run-boundary indicator give, for every element, the first and last rank in
    # its tie group.
    new_group = torch.ones_like(ordered, dtype=torch.bool)
    new_group[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    group_id = new_group.cumsum(dim=-1) - 1
    n_groups = int(group_id.max().item()) + 1
    sums = torch.zeros(*ordered.shape[:-1], n_groups, device=scores.device, dtype=torch.float32)
    counts = torch.zeros_like(sums)
    ordered_ranks = torch.gather(ranks, -1, order)
    sums.scatter_add_(-1, group_id, ordered_ranks)
    counts.scatter_add_(-1, group_id, torch.ones_like(ordered_ranks))
    averaged = torch.gather(sums / counts, -1, group_id)
    return torch.empty_like(averaged).scatter_(-1, order, averaged)


def spearman_median(reference: torch.Tensor, candidate: torch.Tensor) -> list[float]:
    """Median over query blocks, per head, of tie-corrected rank correlation."""
    ref_ranks = average_ranks(reference)
    cand_ranks = average_ranks(candidate)
    ref_centered = ref_ranks - ref_ranks.mean(dim=-1, keepdim=True)
    cand_centered = cand_ranks - cand_ranks.mean(dim=-1, keepdim=True)
    numerator = (ref_centered * cand_centered).sum(dim=-1)
    denominator = ref_centered.norm(dim=-1) * cand_centered.norm(dim=-1)
    rho = numerator / denominator.clamp(min=MARGIN_EPS)
    return rho.median(dim=-1).values.tolist()


def margins(sorted_values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw, normalized) top-k decision margin ``s_(k) - s_(k+1)`` per query block."""
    n_key_blocks = sorted_values.shape[-1]
    if k >= n_key_blocks:
        nan = torch.full(sorted_values.shape[:-1], float("nan"), device=sorted_values.device)
        return nan, nan
    raw = sorted_values[..., k - 1] - sorted_values[..., k]
    spread = (sorted_values[..., 0] - sorted_values[..., -1]).clamp(min=MARGIN_EPS)
    return raw, raw / spread


def _median_or_none(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.median().item())


def compare_masks(reference_sorted: tuple[torch.Tensor, torch.Tensor],
                  candidate_sorted: tuple[torch.Tensor, torch.Tensor], k: int, n_key_blocks: int) -> dict[str, Any]:
    """Per-head set-overlap, margin and tie statistics for one (sparsity) budget."""
    ref_values, ref_index = reference_sorted
    cand_values, cand_index = candidate_sorted
    n_heads, n_query_blocks, _ = ref_values.shape

    ref_mask = torch.zeros((n_heads, n_query_blocks, n_key_blocks), dtype=torch.bool, device=ref_values.device)
    cand_mask = torch.zeros_like(ref_mask)
    ref_mask.scatter_(-1, ref_index[..., :k], True)
    cand_mask.scatter_(-1, cand_index[..., :k], True)
    intersection = (ref_mask & cand_mask).sum(dim=-1)

    ref_margin_raw, ref_margin_norm = margins(ref_values, k)
    cand_margin_raw, cand_margin_norm = margins(cand_values, k)
    boundary_ties = (cand_margin_raw == 0).sum(dim=-1) if torch.isfinite(cand_margin_raw).all() else None

    return {
        "intersection": intersection.sum(dim=-1).tolist(),
        "changed_query_blocks": (intersection < k).sum(dim=-1).tolist(),
        "margin_reference": [_median_or_none(row) for row in ref_margin_norm],
        "margin_candidate": [_median_or_none(row) for row in cand_margin_norm],
        "margin_raw_reference": [_median_or_none(row) for row in ref_margin_raw],
        "margin_raw_candidate": [_median_or_none(row) for row in cand_margin_raw],
        "boundary_ties": boundary_ties.tolist() if boundary_ties is not None else [None] * n_heads,
    }


class RoutingProbeAttentionImpl(AttentionImpl):
    """Dense BF16 attention plus side-channel routing-stability measurement."""

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
        config_path = probe_config_path()
        self.config = ProbeConfig.load(config_path) if config_path else None
        self._vsa_geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._scheduler_timesteps: list[float] | None = None

    @classmethod
    def _writer(cls, config: ProbeConfig) -> JsonlWriter:
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
        attn_metadata: RoutingProbeAttentionMetadata | None,
    ) -> torch.Tensor:
        if self.config is not None:
            self._measure(query, key)
        # Byte-identical to FlashAttentionImpl's dense branch
        # (fastvideo/attention/backends/flash_attn.py:325-331), so the denoising
        # trajectory matches the FLASH_ATTN baseline exactly.
        return flash_attn_func_compilable(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )

    def _vsa_tile_geometry(self, seq_len: int,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """VSA (4,4,4) tile order / pad map / true tile sizes for the latent grid."""
        if self._vsa_geometry is not None:
            return self._vsa_geometry
        context = get_forward_context()
        batch = getattr(context, "forward_batch", None)
        raw_latent_shape = getattr(batch, "raw_latent_shape", None) if batch is not None else None
        if raw_latent_shape is None:
            return None
        config = self.config
        assert config is not None
        patch_t, patch_h, patch_w = config.patch_size
        latent_t, latent_h, latent_w = (int(value) for value in raw_latent_shape[2:5])
        dit_seq_shape = (latent_t // patch_t, latent_h // patch_h, latent_w // patch_w)
        if int(math.prod(dit_seq_shape)) != seq_len:
            logger.warning_once("routing probe: latent grid %s does not match seq_len %d; skipping VSA scorer",
                                dit_seq_shape, seq_len)
            return None
        num_tiles = tuple(math.ceil(dim / tile) for dim, tile in zip(dit_seq_shape, VSA_TILE_SIZE, strict=False))
        order = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, VSA_TILE_SIZE)
        tile_elems = int(math.prod(VSA_TILE_SIZE))
        starts = torch.arange(block_sizes.numel(), device=device) * tile_elems
        slots = starts[:, None] + torch.arange(tile_elems, device=device)[None, :]
        non_pad_index = slots[torch.arange(tile_elems, device=device)[None, :] < block_sizes[:, None]]
        self._vsa_geometry = (order, non_pad_index, block_sizes)
        return self._vsa_geometry

    @torch.no_grad()
    def _measure(self, query: torch.Tensor, key: torch.Tensor) -> None:
        config = self.config
        assert config is not None
        context = get_forward_context()
        timestep = int(getattr(context, "current_timestep", -1) or 0)
        if timestep % max(1, config.measure_timestep_stride) != 0:
            return
        batch = getattr(context, "forward_batch", None)
        cfg_branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"

        writer = self._writer(config)
        seq_len = query.shape[1]
        common = {
            "prompt_id": config.prompt_id,
            "seed": config.seed,
            "layer": self.layer_idx,
            "timestep": timestep,
            "scheduler_timestep": self._scheduler_timestep(batch, timestep),
            "cfg_branch": cfg_branch,
            "reference_precision": REFERENCE_PRECISION,
            "run_id": config.run_id,
            "git_commit": config.git_commit,
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_size,
            "softmax_scale": self.softmax_scale,
            "score_dtype": config.score_dtype,
            "force_retain_diagonal": False,
            "attention_backend": "ROUTING_PROBE_ATTN",
            "attention_compute": "dense_bf16_fa4_passthrough",
            "stage": config.stage,
            "phase": config.phase,
        }

        self._measure_geometry(writer, common, query, key, SCORER_DIAGNOSTIC, timestep)
        if config.run_vsa_scorer:
            self._measure_geometry(writer, common, query, key, SCORER_VSA, timestep)

    def _pooled(self, query: torch.Tensor, key: torch.Tensor, precision: str,
                scorer: str) -> tuple[torch.Tensor, torch.Tensor, float, float] | None:
        config = self.config
        assert config is not None
        route_q, sat_q = quantize_router_input(query, precision)
        route_k, sat_k = quantize_router_input(key, precision)
        if scorer == SCORER_VSA:
            geometry = self._vsa_tile_geometry(query.shape[1], query.device)
            if geometry is None:
                return None
            order, non_pad_index, block_sizes = geometry
            pooled_q = pool_blocks_vsa_tiles(route_q, order, non_pad_index, block_sizes)
            pooled_k = pool_blocks_vsa_tiles(route_k, order, non_pad_index, block_sizes)
        else:
            pooled_q = pool_blocks_1d(route_q, config.block_q)
            pooled_k = pool_blocks_1d(route_k, config.block_k)
        return pooled_q, pooled_k, sat_q, sat_k

    def _measure_geometry(self, writer: JsonlWriter, common: dict[str, Any], query: torch.Tensor, key: torch.Tensor,
                          scorer: str, timestep: int) -> None:
        config = self.config
        assert config is not None
        seq_len = query.shape[1]
        if scorer == SCORER_VSA:
            tile_elems = int(math.prod(VSA_TILE_SIZE))
            block_q = block_k = tile_elems
        else:
            block_q, block_k = config.block_q, config.block_k

        pooled_reference = self._pooled(query, key, REFERENCE_PRECISION, scorer)
        if pooled_reference is None:
            return
        ref_q, ref_k, _, _ = pooled_reference
        # Pool in the arm's precision, matmul in fp32 (EXPERIMENT_SPEC 3.1 rule 2),
        # and apply softmax_scale so margins are in pre-softmax logit units.
        # `score_dtype` exists only so an fp64 control run can show that fp32
        # score resolution is not itself manufacturing boundary ties; the
        # pre-registered value is fp32 and every reported arm uses it.
        score_dtype = getattr(torch, config.score_dtype)
        reference_scores = (ref_q.to(score_dtype) @ ref_k.to(score_dtype).transpose(-1, -2)) * self.softmax_scale
        reference_sorted = sort_scores_descending(reference_scores)
        n_query_blocks, n_key_blocks = reference_scores.shape[-2:]
        ragged_tail = seq_len - (math.ceil(seq_len / block_k) - 1) * block_k if scorer != SCORER_VSA else None

        emit_spearman = timestep % max(1, config.spearman_timestep_stride) == 0
        for precision in config.routing_precisions:
            is_null_control = precision == REFERENCE_PRECISION
            if is_null_control and not self._null_control_due(timestep):
                continue
            pooled_candidate = self._pooled(query, key, precision, scorer)
            if pooled_candidate is None:
                continue
            cand_q, cand_k, sat_q, sat_k = pooled_candidate
            candidate_scores = ((cand_q.to(score_dtype) @ cand_k.to(score_dtype).transpose(-1, -2)) *
                                self.softmax_scale)
            candidate_sorted = sort_scores_descending(candidate_scores)
            rho = spearman_median(reference_scores, candidate_scores) if emit_spearman else None
            provenance, quantizer_impl = _ARM_PROVENANCE[precision]

            for sparsity in config.sparsities:
                k = compute_topk(sparsity, n_key_blocks)
                stats = compare_masks(reference_sorted, candidate_sorted, k, n_key_blocks)
                budget = k * n_query_blocks
                for head in range(self.num_heads):
                    intersection = int(stats["intersection"][head])
                    union = 2 * budget - intersection
                    writer.write({
                        **common,
                        "head": head,
                        "block_q": block_q,
                        "block_k": block_k,
                        "sparsity": sparsity,
                        "routing_precision": precision,
                        "scorer": scorer,
                        "intersection": intersection,
                        "union": union,
                        "selected_reference": budget,
                        "selected_candidate": budget,
                        "recall": intersection / budget,
                        "jaccard": intersection / union,
                        "decision_margin_reference": stats["margin_reference"][head],
                        "decision_margin_candidate": stats["margin_candidate"][head],
                        "decision_margin_raw_reference": stats["margin_raw_reference"][head],
                        "decision_margin_raw_candidate": stats["margin_raw_candidate"][head],
                        "frac_query_blocks_changed": stats["changed_query_blocks"][head] / n_query_blocks,
                        "boundary_ties": stats["boundary_ties"][head],
                        "spearman_rho": rho[head] if rho is not None else None,
                        "native_or_simulated": provenance,
                        "quantizer_impl": quantizer_impl,
                        "k_per_query_block": k,
                        "n_q_blocks": n_query_blocks,
                        "n_k_blocks": n_key_blocks,
                        "ragged_tail": ragged_tail,
                        "sat_frac_q": sat_q,
                        "sat_frac_k": sat_k,
                    })

    def _scheduler_timestep(self, batch: Any, step_index: int) -> float | None:
        """The scheduler's own timestep value for this step index, not the index.

        ``ForwardContext.current_timestep`` is the loop counter ``i``
        (fastvideo/pipelines/stages/denoising.py:507), so the index alone cannot be
        compared across step counts or schedulers. EXPERIMENT_SPEC 9.3(4) requires
        the scheduler value; it is cached because reading it syncs the device.
        """
        if self._scheduler_timesteps is None:
            timesteps = getattr(batch, "timesteps", None)
            if timesteps is None:
                return None
            self._scheduler_timesteps = [float(value) for value in timesteps.flatten().tolist()]
        if not 0 <= step_index < len(self._scheduler_timesteps):
            return None
        return self._scheduler_timesteps[step_index]

    def _null_control_due(self, timestep: int) -> bool:
        """Subsample the bf16-vs-bf16 null control; it is invariant by construction.

        A full enumeration would triple stage-1 output to assert one identity, so
        it runs on a deterministic lattice of layers and timesteps and every
        emitted record is still checked by the analysis script's invariant 7.
        """
        config = self.config
        assert config is not None
        return (self.layer_idx % max(1, config.null_control_layer_stride) == 0
                and timestep % max(1, config.null_control_timestep_stride) == 0)
