"""Phase F2: precision interventions at the **real** VSA routing interface.

Study 1's closest approach to VSA was a cube-geometry control that reproduced
VSA's (4,4,4) tiling but selected blocks with a research mean-pooled scorer. This
backend removes that gap: the mask is produced by the *same kernel functions* the
installed ``video_sparse_attn`` calls —

    fused_block_mean  ->  torch.matmul / sqrt(d)  ->  fused_topk_mask

— so a mask measured here is the mask VSA would route with, including VSA's fp32
bisection threshold and its index-order tie-break.

See ``artifacts/sparsefp4_followup/VSA_GATE_MAP.md`` for the code reading this is
built on. Two consequences of that map shape the whole design:

1.  ``gate_compress`` is **not** the selector. It reaches the kernel as
    ``compress_attn_weight`` and only scales the compression branch *after* the
    mask exists, so quantizing it cannot change routing. It is measured here as an
    explicitly-labelled non-routing control, never as a routing arm.
2.  VSA's deployed selector arithmetic is bf16 with fp32 accumulation — F1's R4
    condition, not study 1's fp64. So every quantity is reported against **two**
    references: ``V0`` (VSA as deployed, the honest operating point) and ``V0_FP64``
    (the exact ideal), with each row stating which one it is paired against.

Damage is measured on the **sparse branch** ``out_s`` alone. VSA's output is
``out_s + out_c * gate``, and the compression branch does not depend on the mask;
including it would add a large mask-independent constant to both sides of every
comparison and shrink every effect toward zero for purely arithmetic reasons.

Enable with::

    FASTVIDEO_ATTENTION_BACKEND=VSA_PRECISION_PROBE_ATTN
    FASTVIDEO_SPARSEFP4_F2=/path/to/f2_config.json
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

from fastvideo.attention.backends.abstract import layer_idx_from_prefix
from fastvideo.attention.backends.routing_probe_attn import JsonlWriter, quantize_router_input
from fastvideo.attention.backends.sparsefp4_mask_metrics import (boundary_diagnostics, deployed_tie_count, mask_hash,
                                                                 mask_comparison, spearman_rho)
from fastvideo.attention.backends.sparsefp4_numerics import error_metrics, random_matched_mask
from fastvideo.attention.backends.sparsefp4_scorer_precision import (ambient_fp32_state, autocast_state,
                                                                     declared_precision_arithmetic, exact_fp32_matmul)
from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, VideoSparseAttentionBackend,
                                                            VideoSparseAttentionImpl, VideoSparseAttentionMetadata,
                                                            compute_topk)
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)

F2_ENV_VAR = "FASTVIDEO_SPARSEFP4_F2"
TILE_ELEMENTS = math.prod(VSA_TILE_SIZE)

# Arms. "repr" is the representation of Q/K entering VSA's pooling; "pool"/"score"
# are the arithmetic used for VSA's own two selector stages.
#
#   V0      — VSA exactly as deployed. The reference for every deployed-relative
#             number, and the arm whose mask is checked against the real kernel.
#   V0_FP64 — the exact ideal, for measuring how far the deployed selector already
#             sits from optimal *before* any intervention is applied.
#   VA*     — Intervention A: representation of the routing input.
#   VB*     — Intervention B: selector arithmetic.
#   VC      — Intervention C: gate_compress quantization. NOT a routing arm.
#   VD      — Intervention D: torch.topk tie-break instead of VSA's.
VSA_ARMS: tuple[dict[str, Any], ...] = (
    {
        "arm": "V0",
        "repr": "bf16",
        "pool": "kernel",
        "score": "kernel",
        "kind": "reference_deployed",
        "purpose": "VSA as installed: fused_block_mean + bf16 matmul + fused_topk_mask"
    },
    {
        "arm": "V0_FP64",
        "repr": "bf16",
        "pool": "fp64",
        "score": "fp64",
        "kind": "reference_exact",
        "purpose": "exact selector on the same bf16 Q/K: how far deployed VSA already is from optimal"
    },
    {
        "arm": "VA_FP8",
        "repr": "fp8_e4m3",
        "pool": "kernel",
        "score": "kernel",
        "kind": "intervention_a",
        "purpose": "routing input quantized to FP8-E4M3, kernel arithmetic unchanged"
    },
    {
        "arm": "VA_NVFP4",
        "repr": "nvfp4",
        "pool": "kernel",
        "score": "kernel",
        "kind": "intervention_a",
        "purpose": "routing input quantized to NVFP4, kernel arithmetic unchanged"
    },
    {
        "arm": "VB_FP32",
        "repr": "bf16",
        "pool": "fp32",
        "score": "fp32",
        "kind": "intervention_b",
        "purpose": "selector arithmetic raised to fp32"
    },
    {
        "arm": "VB_BF16_LOW",
        "repr": "bf16",
        "pool": "bf16_low",
        "score": "bf16_low",
        "kind": "intervention_b",
        "purpose": "selector arithmetic lowered to true bf16 accumulation"
    },
    {
        "arm": "VA_NVFP4_VB_FP64",
        "repr": "nvfp4",
        "pool": "fp64",
        "score": "fp64",
        "kind": "intervention_ab",
        "purpose": "the H3 question at VSA's interface: can exact arithmetic rescue a quantized routing input?"
    },
    {
        "arm": "VC_GATE_NVFP4",
        "repr": "bf16",
        "pool": "kernel",
        "score": "kernel",
        "kind": "non_routing_control",
        "purpose": "gate_compress quantized to NVFP4; must NOT change the mask (falsifies the gate map if it does)"
    },
    {
        "arm": "VD_TORCH_TOPK",
        "repr": "bf16",
        "pool": "kernel",
        "score": "kernel",
        "kind": "tiebreak_control",
        "purpose": "torch.topk tie-break instead of VSA's index-order threshold rule"
    },
)
ARMS_BY_ID = {arm["arm"]: arm for arm in VSA_ARMS}
DEPLOYED_ARM = "V0"
EXACT_ARM = "V0_FP64"


@dataclass
class F2Config:
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
    random_seed: int
    shard_tag: str
    stage: str
    vsa_sparsity_for_execution: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> F2Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arms = tuple(str(value) for value in raw.get("arms", tuple(arm["arm"] for arm in VSA_ARMS)))
        unknown = [arm for arm in arms if arm not in ARMS_BY_ID]
        if unknown:
            raise ValueError(f"unknown VSA arm id(s) {unknown}")
        for required in (DEPLOYED_ARM, EXACT_ARM):
            if required not in arms:
                raise ValueError(f"arm {required} is a required reference and must be measured")
        return cls(
            out_dir=Path(raw["out_dir"]),
            run_id=raw["run_id"],
            git_commit=raw["git_commit"],
            prompt_id=raw["prompt_id"],
            seed=int(raw["seed"]),
            sparsities=tuple(float(value) for value in raw.get("sparsities", (0.90, ))),
            layers=tuple(int(value) for value in raw.get("layers", ())),
            timesteps=tuple(int(value) for value in raw.get("timesteps", ())),
            heads=tuple(int(value) for value in raw.get("heads", ())),
            cfg_branches=tuple(str(value) for value in raw.get("cfg_branches", ("positive", ))),
            arms=arms,
            random_seed=int(raw.get("random_seed", 20260816)),
            shard_tag=str(raw.get("shard_tag", "shard0")),
            stage=str(raw.get("stage", "F2")),
            vsa_sparsity_for_execution=float(raw.get("vsa_sparsity_for_execution", 0.90)),
            provenance=dict(raw.get("provenance", {})),
        )


class VSAPrecisionProbeAttentionBackend(VideoSparseAttentionBackend):
    """Subclasses VSA's backend so the denoising stage builds VSA metadata for it.

    ``DenoisingStage`` decides whether to construct ``VideoSparseAttentionMetadata``
    by checking the resolved backend against VSA; inheriting makes that check pass
    and guarantees the probe receives byte-identical metadata (tiling indices,
    ragged block sizes, sparsity) to a real VSA run.
    """
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VSA_PRECISION_PROBE_ATTN"

    @staticmethod
    def get_impl_cls() -> type[VSAPrecisionProbeAttentionImpl]:
        return VSAPrecisionProbeAttentionImpl


def vsa_pool(x: torch.Tensor, block_sizes: torch.Tensor, arithmetic: str) -> tuple[torch.Tensor, str]:
    """Block-mean pooling for one selector arm. ``x`` is ``[B, H, S, D]``.

    ``arithmetic="kernel"`` calls the installed ``fused_block_mean``, so the
    reference arm's pooling is not a reimplementation — it is the kernel's own
    code. The other modes are deliberate deviations used to move the arithmetic
    axis, and each returns a semantics string that states its accumulator.

    The non-kernel modes apply an explicit validity mask before summing. The kernel
    does **not**: ``_fused_block_mean_kernel`` loads all 64 slots unmasked and
    divides by the valid token count, which is correct only because VSA's tile
    buffer zero-fills padding (verified in ``f2_selftest``). Masking explicitly here
    is therefore a no-op on real routing input and keeps the higher-precision arms
    correct even if that invariant ever changes — the two agree bit-for-bit today.
    """
    from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean

    if arithmetic == "kernel":
        return (fused_block_mean(x, block_sizes,
                                 TILE_ELEMENTS), "pool=kernel_fused_block_mean_bf16_read_fp32_acc_bf16_write")

    batch, heads, seq_len, dim = x.shape
    n_blocks = seq_len // TILE_ELEMENTS
    grouped = x.view(batch, heads, n_blocks, TILE_ELEMENTS, dim)
    token_index = torch.arange(TILE_ELEMENTS, device=x.device)
    valid = (token_index.view(1, -1) < block_sizes.view(-1, 1)).view(1, 1, n_blocks, TILE_ELEMENTS, 1)
    denominator = block_sizes.to(torch.float64).view(1, 1, -1, 1)

    if arithmetic == "fp64":
        pooled = (grouped.to(torch.float64) * valid).sum(dim=3) / denominator
        return pooled, "pool=fp64_values_acc_fp64_valid_token_denominator"
    if arithmetic == "fp32":
        pooled = (grouped.float() * valid).sum(dim=3) / denominator.float()
        return pooled, "pool=fp32_values_acc_fp32_valid_token_denominator"
    if arithmetic == "bf16_low":
        # Genuine bf16 accumulation: round back to bf16 after every token add, which
        # is what a kernel without an fp32 accumulator would do.
        masked = (grouped * valid).to(torch.bfloat16)
        acc = torch.zeros((batch, heads, n_blocks, dim), dtype=torch.bfloat16, device=x.device)
        for token in range(TILE_ELEMENTS):
            acc = (acc + masked[:, :, :, token]).to(torch.bfloat16)
        pooled = (acc / block_sizes.to(torch.bfloat16).view(1, 1, -1, 1)).to(torch.bfloat16)
        return pooled, "pool=bf16_values_sequential_acc_bf16_valid_token_denominator"
    raise ValueError(f"unknown pool arithmetic {arithmetic!r}")


def vsa_score(pooled_q: torch.Tensor, pooled_k: torch.Tensor, dim: int, arithmetic: str) -> tuple[torch.Tensor, str]:
    """VSA's block score ``q_c @ k_c^T / sqrt(d)`` at a stated precision."""
    if arithmetic == "kernel":
        # Exactly ops.py:127 — bf16 inputs, tensor-core fp32 accumulation, bf16 out.
        scores = torch.matmul(pooled_q, pooled_k.transpose(-2, -1)) / (dim**0.5)
        return scores, "score=kernel_bf16_torch_matmul_acc_fp32_bf16_out"
    if arithmetic == "fp64":
        scores = torch.matmul(pooled_q.to(torch.float64), pooled_k.to(torch.float64).transpose(-2, -1)) / (dim**0.5)
        return scores, "score=fp64_values_acc_fp64"
    if arithmetic == "fp32":
        with exact_fp32_matmul():
            scores = torch.matmul(pooled_q.float(), pooled_k.float().transpose(-2, -1)) / (dim**0.5)
        return scores, "score=fp32_values_acc_fp32_tf32_disabled"
    if arithmetic == "bf16_low":
        left = pooled_q.to(torch.bfloat16)
        right = pooled_k.to(torch.bfloat16).transpose(-2, -1)
        acc = torch.zeros(left.shape[:-1] + (right.shape[-1], ), dtype=torch.bfloat16, device=left.device)
        for index in range(left.shape[-1]):
            acc = (acc + left[..., index:index + 1] * right[..., index:index + 1, :]).to(torch.bfloat16)
        return (acc / (dim**0.5)).to(torch.bfloat16), "score=bf16_values_sequential_rank1_acc_bf16"
    raise ValueError(f"unknown score arithmetic {arithmetic!r}")


def vsa_select(scores: torch.Tensor, topk: int, rule: str) -> tuple[torch.Tensor, str]:
    """Top-k selection over the last axis.

    Three rules, and the distinction is load-bearing:

    ``kernel``
        VSA's own Triton ``fused_topk_mask``. Only valid for bf16 scores, because
        that is the only dtype VSA ever feeds it.
    ``exact_index_order``
        VSA's *rule* — take the k largest, breaking ties toward the lower key-block
        index — evaluated at whatever precision the scores carry. A stable
        descending sort produces exactly that ordering. This is what
        higher-precision arms must use: pushing fp64 scores through the bf16 kernel
        would throw away the precision the arm exists to test, turning intervention
        B into a no-op. ``f2_selftest`` asserts this rule reproduces ``kernel``
        bit-for-bit on bf16 scores, so the substitution is verified rather than
        assumed.
    ``torch_topk``
        ``torch.topk``'s own tie behaviour, kept as intervention D's contrast.
    """
    from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask

    if rule == "kernel":
        if scores.dtype != torch.bfloat16:
            raise RuntimeError(f"the kernel selection rule is only valid for bf16 scores, got {scores.dtype}; "
                               "use rule='exact_index_order' for higher-precision arms")
        # A single +inf pins the bisection's upper bound at +inf, so the threshold never
        # converges and the kernel returns far more than topk blocks (reproduced in
        # f2_budget_debug.py, case 6). Catch it here rather than downstream, where it
        # only surfaces as an unexplained budget mismatch.
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("non-finite block scores reached fused_topk_mask; its fp32 bisection cannot "
                               "converge with +inf and would return more than topk blocks per row")
        return fused_topk_mask(scores, topk), "select=kernel_fused_topk_mask_fp32_bisect_index_order_ties"
    if rule == "exact_index_order":
        order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :topk]
        mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(-1, order, True)
        return mask, f"select=exact_stable_sort_index_order_ties_at_{scores.dtype}".replace("torch.", "")
    if rule == "torch_topk":
        indices = torch.topk(scores, topk, dim=-1).indices
        return torch.zeros(scores.shape, dtype=torch.bool,
                           device=scores.device).scatter_(-1, indices, True), "select=torch_topk_native_tiebreak"
    raise ValueError(f"unknown selection rule {rule!r}")


class VSAPrecisionProbeAttentionImpl(VideoSparseAttentionImpl):
    """Real VSA for the model, plus a side-channel precision ablation of its selector.

    Subclasses ``VideoSparseAttentionImpl`` rather than reimplementing it so the
    trajectory the model follows is genuine VSA — tiling, both branches, the
    ``gate_compress`` weighting and all. The probe runs inside ``forward`` on the
    already-tiled Q/K/V the kernel is about to consume, which is the only place
    where the routing input is available in the exact form VSA routes on.
    """

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
        super().__init__(num_heads, head_size, causal, softmax_scale, num_kv_heads, prefix, **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self.layer_idx = layer_idx_from_prefix(prefix, default=-1)
        path = os.environ.get(F2_ENV_VAR, "").strip()
        if not path:
            raise RuntimeError(f"{F2_ENV_VAR} must point at an F2 config JSON when "
                               "FASTVIDEO_ATTENTION_BACKEND=VSA_PRECISION_PROBE_ATTN")
        self.config = F2Config.load(path)
        self._logged_geometry = False
        self._budget_violations_dumped: set[tuple[str, str]] = set()

    @classmethod
    def _writer(cls, config: F2Config) -> JsonlWriter:
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

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        config = self.config
        context = get_forward_context()
        timestep = int(getattr(context, "current_timestep", -1) or 0)
        batch = getattr(context, "forward_batch", None)
        cfg_branch = "negative" if bool(getattr(batch, "is_cfg_negative", False)) else "positive"
        if (self.layer_idx in config.layers and timestep in config.timesteps and cfg_branch in config.cfg_branches):
            # Ambient numerics captured before the guard neutralizes them.
            ambient = {**ambient_fp32_state(), **autocast_state()}
            # Without this, the denoising loop's autocast(bf16) casts the fp32 arm's
            # matmul down to bf16 and VB_FP32 silently becomes a duplicate of V0.
            with declared_precision_arithmetic():
                self._measure(query, key, value, gate_compress, attn_metadata, timestep, cfg_branch, batch, ambient)
        # The model consumes real VSA, unmodified.
        return super().forward(query, key, value, gate_compress, attn_metadata)

    def _routing_inputs(self, query: torch.Tensor, key: torch.Tensor,
                        precision: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Quantize the routing representation of BHSD ``query``/``key``.

        ``quantize_router_input`` is study 1's function and is defined on ``[B, S, H, D]``
        — its fp8 scale is per-head (amax reduced over B, S, D). Feeding it BHSD would
        silently reduce over B, H, D instead, producing a per-sequence-position scale:
        a different, unrealistic granularity that would make F2's representation axis
        non-comparable to F1's. So transpose into the layout it is defined on and back,
        rather than reimplementing the quantizer.
        """
        if precision == "bf16":
            return query, key, 0.0
        route_q, sat_q = quantize_router_input(query.transpose(1, 2).contiguous(), precision)
        route_k, sat_k = quantize_router_input(key.transpose(1, 2).contiguous(), precision)
        route_q = route_q.to(query.dtype).transpose(1, 2).contiguous()
        route_k = route_k.to(key.dtype).transpose(1, 2).contiguous()
        if route_q.shape != query.shape or route_k.shape != key.shape:
            raise RuntimeError(f"routing quantization changed shape: q {tuple(query.shape)} -> "
                               f"{tuple(route_q.shape)}, k {tuple(key.shape)} -> {tuple(route_k.shape)}")
        for name, tensor in (("query", route_q), ("key", route_k)):
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{precision} routing quantization produced non-finite {name} values; "
                                   "VSA's fp32 bisection in fused_topk_mask cannot converge with +inf scores")
        return route_q, route_k, max(sat_q, sat_k)

    def _check_budget(self, mask: torch.Tensor, scores: torch.Tensor, topk: int, arm_id: str, rule: str,
                      sparsity: float, timestep: int, cfg_branch: str) -> dict[str, Any]:
        """Record — not reject — rows where VSA's selector misses its own budget.

        ``fused_topk_mask`` returns ``topk + 1`` blocks on rows whose k-th and
        (k+1)-th scores tie exactly: its fp32 bisection approaches the k-th value from
        below without reaching it, so both tied scores compare strictly above the
        threshold and the tie-fill branch never runs (``f2_kernel_topk_bug.py``
        reproduces this from the saved row, and shows it is scale-invariant). Since F2
        measures VSA's real selector, that behaviour is part of the object under study;
        aborting would discard a genuine finding. So the deviation is quantified here,
        carried on every record, and the first instance per arm is dumped for evidence.
        """
        counts = mask.sum(dim=-1)
        violating = int((counts != topk).sum())
        info: dict[str, Any] = {
            "selector_budget_violating_rows": violating,
            "selector_budget_violating_frac": violating / max(1, counts.numel()),
            "selector_budget_excess_blocks": int((counts - topk).clamp(min=0).sum()),
            "selector_budget_max_selected": int(counts.max()),
            "selector_budget_min_selected": int(counts.min()),
        }
        if not violating:
            return info
        key = (arm_id, rule)
        if key not in self._budget_violations_dumped:
            self._budget_violations_dumped.add(key)
            index = tuple(int(v) for v in (counts != topk).nonzero()[0].tolist())
            row = scores[index]
            dump = self.config.out_dir / (f"budget_violation_{arm_id}_l{self.layer_idx}_t{timestep}_"
                                          f"{cfg_branch}_sp{sparsity}.pt")
            torch.save(
                {
                    "scores_row": row.detach().cpu(),
                    "mask_row": mask[index].detach().cpu(),
                    "topk": topk,
                    "arm_id": arm_id,
                    "rule": rule,
                    "index": index,
                    "layer_idx": self.layer_idx,
                    "timestep": timestep,
                    "cfg_branch": cfg_branch,
                    "sparsity": sparsity,
                }, dump)
            logger.warning(
                "sparsefp4 F2: VSA selector missed its budget %s",
                json.dumps({
                    "arm": arm_id,
                    "rule": rule,
                    "budget": topk,
                    "selected_at_first_violation": int(counts[index]),
                    "violating_rows": violating,
                    "total_rows": int(counts.numel()),
                    "layer": self.layer_idx,
                    "timestep": timestep,
                    "cause": "fused_topk_mask fp32 bisection cannot resolve a k-th-value tie",
                    "dump": str(dump),
                }))
        return info

    @torch.no_grad()
    def _measure(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate_compress: torch.Tensor,
                 attn_metadata: VideoSparseAttentionMetadata, timestep: int, cfg_branch: str, batch: Any,
                 ambient: dict[str, Any]) -> None:
        from fastvideo_kernel import block_sparse_attn

        config = self.config
        writer = self._writer(config)
        block_sizes = attn_metadata.variable_block_sizes
        n_blocks = int(block_sizes.numel())

        # VSA consumes BHSD on Wan's 64-element tile path (VSA_GATE_MAP.md), so the
        # probe operates on exactly the layout the kernel receives.
        bhsd_q = query.transpose(1, 2).contiguous()
        bhsd_k = key.transpose(1, 2).contiguous()
        bhsd_v = value.transpose(1, 2).contiguous()
        seq_len = bhsd_q.shape[2]
        heads = config.heads or tuple(range(self.num_heads))

        if not self._logged_geometry:
            logger.info(
                "sparsefp4 F2: VSA geometry %s",
                json.dumps({
                    "tile_size": list(VSA_TILE_SIZE),
                    "tile_elements": TILE_ELEMENTS,
                    "n_blocks": n_blocks,
                    "padded_seq_len": seq_len,
                    "dit_seq_shape": list(attn_metadata.dit_seq_shape),
                    "ragged_tail_tokens": int(block_sizes.min().item()),
                }))
            self._logged_geometry = True

        arm_masks: dict[str, torch.Tensor] = {}
        arm_scores: dict[str, torch.Tensor] = {}
        arm_labels: dict[str, dict[str, Any]] = {}

        for sparsity in config.sparsities:
            topk = compute_topk(sparsity, n_blocks)
            arm_masks.clear()
            arm_scores.clear()

            for arm_id in config.arms:
                spec = ARMS_BY_ID[arm_id]
                route_q, route_k, saturation = self._routing_inputs(bhsd_q, bhsd_k, spec["repr"])
                pooled_q, pool_semantics = vsa_pool(route_q, block_sizes, spec["pool"])
                pooled_k, _ = vsa_pool(route_k, block_sizes, spec["pool"])
                scores, score_semantics = vsa_score(pooled_q, pooled_k, self.head_size, spec["score"])
                if arm_id == "VD_TORCH_TOPK":
                    rule = "torch_topk"
                elif scores.dtype == torch.bfloat16:
                    rule = "kernel"
                else:
                    rule = "exact_index_order"
                mask, select_semantics = vsa_select(scores, topk, rule)
                budget_info = self._check_budget(mask, scores, topk, arm_id, rule, sparsity, timestep, cfg_branch)
                gate_saturation: float | None = None
                if arm_id == "VC_GATE_NVFP4":
                    # Intervention C. gate_compress is quantized for real, and the
                    # resulting mask is still derived from unmodified Q/K — because per
                    # VSA_GATE_MAP.md the gate never reaches the selector. The recorded
                    # `mask_identical_to_deployed` is therefore a falsification test of
                    # the map: if quantizing the gate could move the mask, this arm
                    # would have to differ from V0, and it must not.
                    _, gate_saturation = quantize_router_input(gate_compress, "nvfp4")
                arm_masks[arm_id] = mask
                arm_scores[arm_id] = scores
                arm_labels[arm_id] = {
                    "representation_precision": spec["repr"],
                    "pool_precision": spec["pool"],
                    "score_precision": spec["score"],
                    "selection_rule": rule,
                    "intervention_kind": spec["kind"],
                    "purpose": spec["purpose"],
                    "pool_semantics": pool_semantics,
                    "score_semantics": score_semantics,
                    "select_semantics": select_semantics,
                    "representation_saturation_frac": saturation,
                    **budget_info,
                    "gate_compress_saturation_frac": gate_saturation,
                    "gate_compress_quantized": arm_id == "VC_GATE_NVFP4",
                    "gate_compress_in_selection_path": False,
                    "routing_interface": "actual_vsa_fused_block_mean_plus_fused_topk_mask",
                }

            # fused_topk_mask returns [B, H, n_q, n_k] because VSA's scores carry the
            # batch axis, so the masks are already kernel-shaped — unlike F1's
            # [H, n_q, n_k] scorer masks, which need a batch axis added. Metrics index
            # per head, so a [H, n_q, n_k] view is taken for those.
            deployed_mask = arm_masks[DEPLOYED_ARM]
            exact_scores = arm_scores[EXACT_ARM]

            executed: dict[str, torch.Tensor] = {}
            for arm_id, mask in arm_masks.items():
                executed[arm_id] = block_sparse_attn(bhsd_q, bhsd_k, bhsd_v, mask, block_sizes)[0]
            dense_reference = block_sparse_attn(
                bhsd_q, bhsd_k, bhsd_v,
                torch.ones((1, self.num_heads, n_blocks, n_blocks), dtype=torch.bool, device=bhsd_q.device),
                block_sizes)[0]

            random_masks: dict[str, torch.Tensor] = {}
            for arm_id, mask in arm_masks.items():
                if arm_id == DEPLOYED_ARM:
                    continue
                generator = torch.Generator(device=bhsd_q.device)
                generator.manual_seed(config.random_seed + 1_000_000 * self.layer_idx + 1_000 * timestep +
                                      int(sparsity * 100) * 17 + sum(ord(c) for c in arm_id))
                random_masks[arm_id] = random_matched_mask(deployed_mask, mask, generator)
                executed[f"{arm_id}:random"] = block_sparse_attn(bhsd_q, bhsd_k, bhsd_v, random_masks[arm_id],
                                                                 block_sizes)[0]

            self._emit(writer, config, ambient, arm_masks, arm_labels, arm_scores, random_masks, executed,
                       dense_reference, exact_scores, heads, sparsity, topk, n_blocks, seq_len, block_sizes, timestep,
                       cfg_branch, attn_metadata, batch)
            executed.clear()

    def _emit(self, writer: JsonlWriter, config: F2Config, ambient: dict[str, Any], arm_masks: dict[str, torch.Tensor],
              arm_labels: dict[str, dict[str, Any]], arm_scores: dict[str,
                                                                      torch.Tensor], random_masks: dict[str,
                                                                                                        torch.Tensor],
              executed: dict[str, torch.Tensor], dense_reference: torch.Tensor, exact_scores: torch.Tensor,
              heads: tuple[int,
                           ...], sparsity: float, topk: int, n_blocks: int, seq_len: int, block_sizes: torch.Tensor,
              timestep: int, cfg_branch: str, attn_metadata: VideoSparseAttentionMetadata, batch: Any) -> None:
        deployed_mask = arm_masks[DEPLOYED_ARM]
        exact_mask = arm_masks[EXACT_ARM]
        common = {
            "run_id": config.run_id,
            "git_sha": config.git_commit,
            "experiment_family": "vsa_precision",
            "prompt_id": config.prompt_id,
            "seed": config.seed,
            "model_id": config.provenance.get("model_id"),
            "model_revision": config.provenance.get("model_revision"),
            "resolution": config.provenance.get("resolution"),
            "frames": config.provenance.get("frames"),
            "num_steps": config.provenance.get("num_inference_steps"),
            "layer": self.layer_idx,
            "timestep": timestep,
            "cfg_branch": cfg_branch,
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_size,
            "softmax_scale": self.softmax_scale,
            "geometry": "vsa-4x4x4-cube-64",
            "tile_size": list(VSA_TILE_SIZE),
            "tile_elements": TILE_ELEMENTS,
            "n_blocks": n_blocks,
            "dit_seq_shape": list(attn_metadata.dit_seq_shape),
            "vsa_sparsity_for_execution": config.vsa_sparsity_for_execution,
            "resolved_attention_backend": "VSA_PRECISION_PROBE_ATTN",
            "model_trajectory_backend": "VIDEO_SPARSE_ATTN (real, unmodified)",
            "damage_measured_on": "sparse_branch_out_s_only",
            "deployed_reference_arm": DEPLOYED_ARM,
            "exact_reference_arm": EXACT_ARM,
            **ambient,
            "stage": config.stage,
            "phase": "F2",
        }

        for arm_id in config.arms:
            labels = arm_labels[arm_id]
            candidate_mask = arm_masks[arm_id]
            for head in heads:
                # allow_budget_mismatch: VSA's own kernel can return topk+1 on
                # k-th-value ties, so a mismatch here is the measurand, not a bug. The
                # per-row rate rides along on the record for F4 to gate.
                versus_deployed = mask_comparison(candidate_mask[0, head],
                                                  deployed_mask[0, head],
                                                  allow_budget_mismatch=True)
                versus_exact = mask_comparison(candidate_mask[0, head], exact_mask[0, head], allow_budget_mismatch=True)
                # Boundary margins come from the exact selector's scores: the deployed
                # bf16 scores tie often enough that using them would measure the
                # deployed arm's own quantization, not the decision's true margin.
                boundary = boundary_diagnostics(exact_scores[0, head], candidate_mask[0, head], deployed_mask[0, head],
                                                topk)

                candidate_metrics = error_metrics(executed[arm_id][:, head], dense_reference[:, head])
                deployed_metrics = error_metrics(executed[DEPLOYED_ARM][:, head], dense_reference[:, head])
                sparsification = deployed_metrics["rel_l2"]
                wrong_mask_excess = (None if candidate_metrics["rel_l2"] is None or sparsification is None else
                                     candidate_metrics["rel_l2"] - sparsification)

                random_key = f"{arm_id}:random"
                random_excess = None
                random_rel_l2 = None
                random_swaps = None
                if random_key in executed:
                    random_metrics = error_metrics(executed[random_key][:, head], dense_reference[:, head])
                    random_rel_l2 = random_metrics["rel_l2"]
                    if random_rel_l2 is not None and sparsification is not None:
                        random_excess = random_rel_l2 - sparsification
                    random_swaps = mask_comparison(random_masks[arm_id][0, head],
                                                   deployed_mask[0, head],
                                                   allow_budget_mismatch=True)["num_swaps"]

                record: dict[str, Any] = {
                    **common,
                    **labels,
                    "record_type":
                    "vsa_precision",
                    "arm":
                    arm_id,
                    "head":
                    head,
                    "sparsity":
                    sparsity,
                    "retained_k":
                    topk,
                    "retained_fraction":
                    topk / n_blocks,
                    "null_control":
                    arm_id == DEPLOYED_ARM,
                    "mask_deployed_hash":
                    mask_hash(deployed_mask[0, head]),
                    "mask_candidate_hash":
                    mask_hash(candidate_mask[0, head]),
                    "mask_identical_to_deployed":
                    bool(torch.equal(candidate_mask[0, head], deployed_mask[0, head])),
                    # --- vs VSA as deployed (the honest operating point) ---
                    "jaccard":
                    versus_deployed["jaccard"],
                    "recall":
                    versus_deployed["recall"],
                    "num_swaps":
                    versus_deployed["num_swaps"],
                    "swaps_per_query_block":
                    versus_deployed["swaps_per_query_block"],
                    "frac_decisions_changed":
                    versus_deployed["frac_decisions_changed"],
                    "frac_query_blocks_changed":
                    versus_deployed["frac_query_blocks_changed"],
                    "max_swaps_in_a_query_block":
                    versus_deployed["max_swaps_in_a_query_block"],
                    # --- vs the exact selector (how far from optimal) ---
                    "jaccard_vs_exact":
                    versus_exact["jaccard"],
                    "num_swaps_vs_exact":
                    versus_exact["num_swaps"],
                    # --- boundary diagnostics, exact-score reference ---
                    **boundary,
                    "deployed_score_ties":
                    deployed_tie_count(arm_scores[arm_id][0, head].to(torch.float64), topk),
                    # --- damage on the sparse branch ---
                    "rel_l2_vs_dense":
                    candidate_metrics["rel_l2"],
                    "cosine_vs_dense":
                    candidate_metrics["cosine"],
                    "max_abs_vs_dense":
                    candidate_metrics["max_abs"],
                    "sparsification_error":
                    sparsification,
                    "wrong_mask_excess":
                    wrong_mask_excess,
                    "abs_wrong_mask_over_sparsification":
                    (None if wrong_mask_excess is None or not sparsification else abs(wrong_mask_excess) /
                     sparsification),
                    "random_matched_rel_l2":
                    random_rel_l2,
                    "random_matched_excess":
                    random_excess,
                    "random_matched_num_swaps":
                    random_swaps,
                }
                if head == heads[0]:
                    record["spearman_rho_vs_exact"] = spearman_rho(arm_scores[arm_id][0, head].to(torch.float64),
                                                                   exact_scores[0, head])
                writer.write(record)
