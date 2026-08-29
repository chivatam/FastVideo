from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_LAYER_RE = re.compile(r"(?:blocks|transformer_blocks)\.(\d+)")


@dataclass
class RuntimeCapture:
    job_id: str | None = None
    attention_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    effective_sparsities: set[float] = field(default_factory=set)
    adaptive_decisions: list[tuple[str, int, Any, Any]] = field(default_factory=list)
    residual_decisions: list[tuple[str, int, Any, Any]] = field(default_factory=list)
    compressed_support_decisions: list[tuple[str, int, Any]] = field(default_factory=list)
    br_decisions: list[tuple[str, int, Any, Any]] = field(default_factory=list)


_CAPTURE = RuntimeCapture()
_PATCHED = False
_ADAPTIVE_POLICY = None
_RESIDUAL_POLICY = None
_BR_POLICY = None
_FINE_NATIVE_VALIDATED = False
_COMPRESSED_SUPPORT_DETAILED_TRACE = True
_BR_CANDIDATE_K = (32, 64, 96, 125, 192, 250, 375, 624)


def configure_adaptive_policy(
    *,
    retained_mass_threshold: float,
    maximum_sparsity: float,
    candidate_sparsities: tuple[float, ...] = (0.8, 0.7, 0.6, 0.4, 0.0),
    native_sparsity: float = 0.8,
) -> dict[str, Any]:
    global _ADAPTIVE_POLICY
    from research.adaptive_vsa_deadline.adaptive_attention import (
        AdaptiveVSAPolicy,
    )

    _ADAPTIVE_POLICY = AdaptiveVSAPolicy(
        retained_mass_threshold=retained_mass_threshold,
        maximum_sparsity=maximum_sparsity,
        candidate_sparsities=candidate_sparsities,
        native_sparsity=native_sparsity,
    )
    return _ADAPTIVE_POLICY.as_dict()


def configure_residual_policy(
    *,
    native_fraction: float,
    native_sparsity: float = 0.8,
    risk_formula: str = "coarse_mass_x_key_heterogeneity",
    instrument_splits: tuple[float, ...] = (),
    detailed_trace: bool = True,
    force_outside_native: bool = False,
) -> dict[str, Any]:
    global _RESIDUAL_POLICY
    from research.ra_vsa_deadline.residual_attention import (
        ResidualAwareVSAPolicy,
    )

    _RESIDUAL_POLICY = ResidualAwareVSAPolicy(
        native_fraction=native_fraction,
        native_sparsity=native_sparsity,
        risk_formula=risk_formula,
        instrument_splits=instrument_splits,
        detailed_trace=detailed_trace,
        force_outside_native=force_outside_native,
    )
    return _RESIDUAL_POLICY.as_dict()


def configure_compressed_support(
    *,
    detailed_trace: bool = True,
) -> dict[str, Any]:
    global _COMPRESSED_SUPPORT_DETAILED_TRACE
    _COMPRESSED_SUPPORT_DETAILED_TRACE = bool(detailed_trace)
    return {"detailed_trace": _COMPRESSED_SUPPORT_DETAILED_TRACE}


def configure_br_census(
    *,
    candidate_k: tuple[int, ...] = _BR_CANDIDATE_K,
) -> dict[str, Any]:
    global _BR_CANDIDATE_K
    normalized = tuple(sorted({int(value) for value in candidate_k}))
    if not normalized:
        raise ValueError("BR-VSA census requires candidate K values")
    _BR_CANDIDATE_K = normalized
    return {"candidate_k": list(_BR_CANDIDATE_K)}


def configure_br_policy(
    *,
    k_table_path: str,
) -> dict[str, Any]:
    global _BR_POLICY
    from research.br_vsa.attention import BudgetRedistributedPolicy

    _BR_POLICY = BudgetRedistributedPolicy.from_path(k_table_path)
    return _BR_POLICY.as_dict()


def begin_job(job_id: str) -> None:
    _CAPTURE.job_id = job_id
    _CAPTURE.attention_events.clear()
    _CAPTURE.rows.clear()
    _CAPTURE.effective_sparsities.clear()
    _CAPTURE.adaptive_decisions.clear()
    _CAPTURE.residual_decisions.clear()
    _CAPTURE.compressed_support_decisions.clear()
    _CAPTURE.br_decisions.clear()


def record_effective_sparsity(value: float) -> None:
    _CAPTURE.effective_sparsities.add(float(value))


def finish_job() -> tuple[float, list[dict[str, Any]], list[float]]:
    if _CAPTURE.attention_events:
        torch.cuda.synchronize()
    attention_ms = sum(start.elapsed_time(end) for start, end in _CAPTURE.attention_events)
    if _CAPTURE.adaptive_decisions:
        from research.adaptive_vsa_deadline.adaptive_attention import (
            summarize_decision,
        )

        for prefix, timestep, policy, decision in _CAPTURE.adaptive_decisions:
            decision_summary = summarize_decision(decision)
            record_effective_sparsity(decision_summary["effective_sparsity"])
            _CAPTURE.rows.append(
                {
                    "event_type": "adaptive_policy",
                    "job_id": _CAPTURE.job_id,
                    "prefix": prefix,
                    "layer": _layer_index(prefix),
                    "timestep": timestep,
                    **policy.as_dict(),
                    **decision_summary,
                }
            )
    if _CAPTURE.residual_decisions:
        from research.ra_vsa_deadline.residual_attention import (
            summarize_residual_decision,
        )

        for prefix, timestep, policy, decision in _CAPTURE.residual_decisions:
            decision_summary = summarize_residual_decision(decision)
            record_effective_sparsity(policy.native_sparsity)
            _CAPTURE.rows.append(
                {
                    "event_type": "ra_vsa_policy",
                    "job_id": _CAPTURE.job_id,
                    "prefix": prefix,
                    "layer": _layer_index(prefix),
                    "timestep": timestep,
                    **policy.as_dict(),
                    **decision_summary,
                }
            )
    if _CAPTURE.compressed_support_decisions:
        from research.compressed_halo_vsa.compressed_support import (
            summarize_compressed_support_decision,
        )

        for prefix, timestep, decision in _CAPTURE.compressed_support_decisions:
            decision_summary = summarize_compressed_support_decision(decision)
            record_effective_sparsity(0.8)
            _CAPTURE.rows.append(
                {
                    "event_type": "compressed_support_policy",
                    "job_id": _CAPTURE.job_id,
                    "prefix": prefix,
                    "layer": _layer_index(prefix),
                    "timestep": timestep,
                    **decision_summary,
                }
            )
    if _CAPTURE.br_decisions:
        from research.br_vsa.attention import (
            summarize_budget_redistributed_decision,
        )

        for prefix, timestep, policy, decision in _CAPTURE.br_decisions:
            decision_summary = summarize_budget_redistributed_decision(
                decision
            )
            _CAPTURE.rows.append(
                {
                    "event_type": "br_vsa_policy",
                    "job_id": _CAPTURE.job_id,
                    "prefix": prefix,
                    "layer": _layer_index(prefix),
                    "timestep": timestep,
                    **policy.as_dict(),
                    **decision_summary,
                }
            )
    rows = list(_CAPTURE.rows)
    effective_sparsities = sorted(_CAPTURE.effective_sparsities)
    _CAPTURE.job_id = None
    _CAPTURE.attention_events.clear()
    _CAPTURE.rows.clear()
    _CAPTURE.effective_sparsities.clear()
    _CAPTURE.adaptive_decisions.clear()
    _CAPTURE.residual_decisions.clear()
    _CAPTURE.compressed_support_decisions.clear()
    _CAPTURE.br_decisions.clear()
    return attention_ms, rows, effective_sparsities


def _timed_call(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    _CAPTURE.attention_events.append((start, end))
    return result


def fake_nvfp4_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    """Training-free NVFP4 simulation: per-16 E2M1 with E4M3 scale factors."""
    if tensor.shape[-1] % 16:
        raise ValueError(f"NVFP4 simulation requires last dimension divisible by 16, got {tensor.shape[-1]}")
    orig_dtype = tensor.dtype
    x = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 16, 16)
    scale = x.abs().amax(dim=-1, keepdim=True).div(6.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = x.div(scale)
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=x.device, dtype=x.dtype)
    levels = torch.tensor(_E2M1_LEVELS, device=x.device, dtype=x.dtype)
    bucket = torch.bucketize(normalized.abs(), boundaries)
    quantized = levels[bucket].copysign(normalized)
    return quantized.mul(scale).reshape_as(tensor).to(orig_dtype)


def _layer_index(prefix: str) -> int | None:
    match = _LAYER_RE.search(prefix)
    return int(match.group(1)) if match else None


def _selected_for_stats(prefix: str, timestep: int) -> bool:
    layers = {int(value) for value in os.getenv("FASTVIDEO_VSA_STATS_LAYERS", "0,14,29").split(",") if value}
    timesteps = {int(value) for value in os.getenv("FASTVIDEO_VSA_STATS_TIMESTEPS", "0,1,2").split(",") if value}
    layer = _layer_index(prefix)
    return layer in layers and timestep in timesteps


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "mean": values.mean().item(),
        "min": values.min().item(),
        "p10": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
    }


def _dense_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    from flash_attn.cute.interface import flash_attn_func

    output = flash_attn_func(query, key, value, softmax_scale=query.shape[-1] ** -0.5, causal=False)
    return output[0] if isinstance(output, tuple) else output


def _record_vsa_stats(
    *,
    prefix: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    metadata: Any,
) -> None:
    if _CAPTURE.job_id is None or not os.getenv("FASTVIDEO_VSA_RECORD_STATS"):
        return
    timestep = int(metadata.current_timestep)
    if not _selected_for_stats(prefix, timestep):
        return

    from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean

    block_elements = 64
    q_bhsd = query.transpose(1, 2).contiguous()
    k_bhsd = key.transpose(1, 2).contiguous()
    q_c = fused_block_mean(q_bhsd, metadata.variable_block_sizes, block_elements)
    k_c = fused_block_mean(k_bhsd, metadata.variable_block_sizes, block_elements)
    scores = torch.matmul(q_c, k_c.transpose(-2, -1)).float() / math.sqrt(query.shape[-1])

    q_fp4 = fake_nvfp4_e2m1(query)
    k_fp4 = fake_nvfp4_e2m1(key)
    q_fp4_c = fused_block_mean(q_fp4.transpose(1, 2).contiguous(), metadata.variable_block_sizes, block_elements)
    k_fp4_c = fused_block_mean(k_fp4.transpose(1, 2).contiguous(), metadata.variable_block_sizes, block_elements)
    scores_fp4 = torch.matmul(q_fp4_c, k_fp4_c.transpose(-2, -1)).float() / math.sqrt(query.shape[-1])

    probs = torch.softmax(scores, dim=-1)
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    sorted_fp4, sorted_fp4_indices = torch.sort(scores_fp4, dim=-1, descending=True)
    sorted_probs = torch.gather(probs, -1, sorted_indices)
    delta = (scores - scores_fp4).abs()
    delta_flat = delta.flatten()
    delta_q = torch.quantile(delta_flat, torch.tensor([0.95, 0.99], device=delta.device))

    dense = _dense_reference(query, key, value)
    diff = output.float() - dense.float()
    rel_l2 = diff.norm().div(dense.float().norm().clamp_min(1e-12)).item()
    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), dense.float().flatten(), dim=0).item()

    num_blocks = scores.shape[-1]
    candidates = [
        float(value)
        for value in os.getenv("FASTVIDEO_VSA_STATS_SPARSITIES", "0,0.2,0.4,0.6,0.7,0.8").split(",")
        if value
    ]
    for sparsity in candidates:
        topk = max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))
        retained = sorted_probs[..., :topk].sum(dim=-1)
        if topk < num_blocks:
            margin = sorted_scores[..., topk - 1] - sorted_scores[..., topk]
        else:
            margin = torch.full_like(retained, float("inf"))
        mask_bf16 = sorted_indices[..., :topk]
        mask_fp4 = sorted_fp4_indices[..., :topk]
        overlap = (mask_bf16.unsqueeze(-1) == mask_fp4.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
        jaccard = overlap.div(2 * topk - overlap).clamp(0, 1)
        row = {
            "job_id": _CAPTURE.job_id,
            "prefix": prefix,
            "layer": _layer_index(prefix),
            "timestep": timestep,
            "head_dim": query.shape[-1],
            "num_heads": query.shape[-2],
            "num_blocks": num_blocks,
            "sparsity": sparsity,
            "topk": topk,
            "delta_inf": delta.max().item(),
            "delta_p95": delta_q[0].item(),
            "delta_p99": delta_q[1].item(),
            "attention_output_rel_l2": rel_l2,
            "attention_output_cosine": cosine,
            **{f"retained_mass_{key}": value for key, value in _quantiles(retained).items()},
            **{f"margin_{key}": value for key, value in _quantiles(margin).items()},
            **{f"mask_jaccard_{key}": value for key, value in _quantiles(jaccard).items()},
        }
        row["risk_ratio_p50"] = 2.0 * row["delta_p99"] / max(row["margin_p50"], 1e-12)
        _CAPTURE.rows.append(row)


def install_runtime_patches(mode: str) -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    if mode in {
        "vsa_bf16",
        "sim_vsa_nvfp4",
        "adaptive_vsa",
        "ra_vsa",
        "rectified_vsa",
        "compressed_halo_vsa",
        "br_vsa_census",
        "br_vsa",
        "fine_vsa_census",
        "fine_vsa",
        "anchored_fine_vsa_census",
        "anchored_fine_vsa25",
        "anchored_fine_vsa50",
        "hierarchical_vsa_census",
        "cluster_vsa_census",
        "vector_vsa_census",
    }:
        from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionImpl

        original_vsa = VideoSparseAttentionImpl.forward

        def vsa_forward(self, query, key, value, gate_compress, attn_metadata):
            if mode == "vector_vsa_census":
                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.vector_vsa.replay import (
                        replay_vector_vsa,
                    )

                    replay = replay_vector_vsa(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        attn_metadata.non_pad_index,
                    )
                    common = {
                        "job_id": _CAPTURE.job_id,
                        "prefix": self.prefix,
                        "layer": _layer_index(self.prefix),
                        "timestep": int(
                            attn_metadata.current_timestep
                        ),
                        **replay.geometry,
                    }
                    for row in (
                        replay.error_rows
                        + replay.alignment_rows
                        + replay.structure_rows
                        + replay.benchmark_rows
                    ):
                        _CAPTURE.rows.append({**common, **row})
                return output

            if mode == "cluster_vsa_census":
                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.cluster_vsa.replay import (
                        replay_cluster_vsa,
                    )

                    capture_assignments = not any(
                        row.get("event_type") == "cluster_assignment"
                        for row in _CAPTURE.rows
                    )
                    replay = replay_cluster_vsa(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        attn_metadata.non_pad_index,
                        capture_assignments=capture_assignments,
                    )
                    common = {
                        "job_id": _CAPTURE.job_id,
                        "prefix": self.prefix,
                        "layer": _layer_index(self.prefix),
                        "timestep": int(
                            attn_metadata.current_timestep
                        ),
                        **replay.geometry,
                    }
                    for row in (
                        replay.error_rows
                        + replay.analysis_rows
                        + replay.benchmark_rows
                        + replay.assignment_rows
                    ):
                        _CAPTURE.rows.append({**common, **row})
                return output

            if mode == "hierarchical_vsa_census":
                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.hierarchical_vsa.replay import (
                        replay_hierarchical_vsa,
                    )

                    replay = replay_hierarchical_vsa(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        attn_metadata.non_pad_index,
                    )
                    common = {
                        "job_id": _CAPTURE.job_id,
                        "prefix": self.prefix,
                        "layer": _layer_index(self.prefix),
                        "timestep": int(
                            attn_metadata.current_timestep
                        ),
                        **replay.geometry,
                    }
                    for row in (
                        replay.error_rows + replay.benchmark_rows
                    ):
                        _CAPTURE.rows.append({**common, **row})
                return output

            if mode in {
                "anchored_fine_vsa25",
                "anchored_fine_vsa50",
            }:
                from research.anchored_fine_vsa.attention import (
                    anchored_fine_video_sparse_attn,
                    summarize_anchored_decision,
                )

                anchor_parent_blocks = (
                    31 if mode == "anchored_fine_vsa25" else 62
                )
                output, decision = _timed_call(
                    lambda: anchored_fine_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        anchor_parent_blocks=anchor_parent_blocks,
                    )
                )
                decision_summary = summarize_anchored_decision(
                    decision
                )
                record_effective_sparsity(
                    decision_summary["nominal_effective_sparsity"]
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.rows.append(
                        {
                            "event_type": "anchored_fine_vsa_policy",
                            "job_id": _CAPTURE.job_id,
                            "prefix": self.prefix,
                            "layer": _layer_index(self.prefix),
                            "timestep": int(
                                attn_metadata.current_timestep
                            ),
                            "mode": mode,
                            **decision_summary,
                        }
                    )
                return output

            if mode == "fine_vsa":
                from research.fine_vsa.attention import (
                    fine_video_sparse_attn,
                    summarize_fine_vsa_decision,
                )

                output, decision = _timed_call(
                    lambda: fine_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                    )
                )
                decision_summary = summarize_fine_vsa_decision(decision)
                record_effective_sparsity(
                    decision_summary["nominal_effective_sparsity"]
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.rows.append(
                        {
                            "event_type": "fine_vsa_policy",
                            "job_id": _CAPTURE.job_id,
                            "prefix": self.prefix,
                            "layer": _layer_index(self.prefix),
                            "timestep": int(
                                attn_metadata.current_timestep
                            ),
                            **decision_summary,
                        }
                    )
                return output

            if mode == "anchored_fine_vsa_census":
                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.anchored_fine_vsa.support import (
                        analyze_support_overlap,
                    )

                    support = analyze_support_overlap(
                        query,
                        key,
                        attn_metadata.variable_block_sizes,
                    )
                    common = {
                        "job_id": _CAPTURE.job_id,
                        "prefix": self.prefix,
                        "layer": _layer_index(self.prefix),
                        "timestep": int(
                            attn_metadata.current_timestep
                        ),
                        **support.geometry,
                    }
                    for row in support.rows:
                        _CAPTURE.rows.append({**common, **row})
                return output

            if mode == "fine_vsa_census":
                global _FINE_NATIVE_VALIDATED

                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.fine_vsa.replay import replay_fine_vsa

                    replay = replay_fine_vsa(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        attn_metadata.non_pad_index,
                        validate_native_kernel=not _FINE_NATIVE_VALIDATED,
                    )
                    _FINE_NATIVE_VALIDATED = True
                    timestep = int(attn_metadata.current_timestep)
                    layer = _layer_index(self.prefix)
                    common = {
                        "job_id": _CAPTURE.job_id,
                        "prefix": self.prefix,
                        "layer": layer,
                        "timestep": timestep,
                        **replay.geometry,
                    }
                    for row in (
                        replay.error_rows
                        + replay.mass_rows
                        + replay.kernel_rows
                    ):
                        _CAPTURE.rows.append({**common, **row})
                return output

            if mode == "br_vsa":
                from research.br_vsa.attention import (
                    budget_redistributed_video_sparse_attn,
                )

                if _BR_POLICY is None:
                    raise RuntimeError("BR-VSA policy was not configured before generation")
                timestep = int(attn_metadata.current_timestep)
                layer = _layer_index(self.prefix)
                if layer is None:
                    raise RuntimeError(
                        f"Could not resolve transformer layer from {self.prefix!r}"
                    )
                requested_k = _BR_POLICY.k_for(timestep, layer)
                output, decision = _timed_call(
                    lambda: budget_redistributed_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        requested_k,
                    )
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.br_decisions.append(
                        (
                            self.prefix,
                            timestep,
                            _BR_POLICY,
                            decision,
                        )
                    )
                return output

            if mode == "br_vsa_census":
                record_effective_sparsity(attn_metadata.VSA_sparsity)
                output = _timed_call(
                    lambda: original_vsa(
                        self,
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata,
                    )
                )
                if _CAPTURE.job_id is not None:
                    from research.br_vsa.sensitivity import (
                        replay_vsa_sensitivity,
                    )

                    replay = replay_vsa_sensitivity(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        attn_metadata.non_pad_index,
                        candidate_k=_BR_CANDIDATE_K,
                    )
                    timestep = int(attn_metadata.current_timestep)
                    layer = _layer_index(self.prefix)
                    for row in replay.rows:
                        _CAPTURE.rows.append(
                            {
                                "event_type": "br_vsa_sensitivity",
                                "job_id": _CAPTURE.job_id,
                                "prefix": self.prefix,
                                "layer": layer,
                                "timestep": timestep,
                                **row,
                            }
                        )
                return output

            if mode in {"rectified_vsa", "compressed_halo_vsa"}:
                from research.compressed_halo_vsa.compressed_support import (
                    compressed_support_video_sparse_attn,
                )

                support_mode = "rectified" if mode == "rectified_vsa" else "compressed_halo"
                output, decision = _timed_call(
                    lambda: compressed_support_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        mode=support_mode,
                        sparsity=float(attn_metadata.VSA_sparsity),
                        detailed_trace=(_COMPRESSED_SUPPORT_DETAILED_TRACE),
                    )
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.compressed_support_decisions.append(
                        (
                            self.prefix,
                            int(attn_metadata.current_timestep),
                            decision,
                        )
                    )
                return output

            if mode == "ra_vsa":
                from research.ra_vsa_deadline.residual_attention import (
                    residual_aware_video_sparse_attn,
                )

                if _RESIDUAL_POLICY is None:
                    raise RuntimeError("RA-VSA policy was not configured before generation")
                output, decision = _timed_call(
                    lambda: residual_aware_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        _RESIDUAL_POLICY,
                    )
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.residual_decisions.append(
                        (
                            self.prefix,
                            int(attn_metadata.current_timestep),
                            _RESIDUAL_POLICY,
                            decision,
                        )
                    )
                return output

            if mode == "adaptive_vsa":
                from research.adaptive_vsa_deadline.adaptive_attention import (
                    adaptive_video_sparse_attn,
                )

                if _ADAPTIVE_POLICY is None:
                    raise RuntimeError("Adaptive VSA policy was not configured before generation")

                output, decision = _timed_call(
                    lambda: adaptive_video_sparse_attn(
                        query,
                        key,
                        value,
                        gate_compress,
                        attn_metadata.variable_block_sizes,
                        _ADAPTIVE_POLICY,
                    )
                )
                if _CAPTURE.job_id is not None:
                    _CAPTURE.adaptive_decisions.append(
                        (
                            self.prefix,
                            int(attn_metadata.current_timestep),
                            _ADAPTIVE_POLICY,
                            decision,
                        )
                    )
                return output

            record_effective_sparsity(attn_metadata.VSA_sparsity)
            runtime_query = query
            runtime_key = key
            if mode == "sim_vsa_nvfp4":

                def operation():
                    nonlocal runtime_query, runtime_key
                    runtime_query = fake_nvfp4_e2m1(query)
                    runtime_key = fake_nvfp4_e2m1(key)
                    return original_vsa(self, runtime_query, runtime_key, value, gate_compress, attn_metadata)
            else:

                def operation():
                    return original_vsa(self, runtime_query, runtime_key, value, gate_compress, attn_metadata)

            output = _timed_call(operation)
            _record_vsa_stats(
                prefix=self.prefix,
                query=query,
                key=key,
                value=value,
                output=output,
                metadata=attn_metadata,
            )
            return output

        VideoSparseAttentionImpl.forward = vsa_forward
    else:
        from fastvideo.attention.backends.flash_attn import FlashAttentionImpl

        original_flash = FlashAttentionImpl.forward

        def flash_forward(self, query, key, value, attn_metadata):
            return _timed_call(lambda: original_flash(self, query, key, value, attn_metadata))

        FlashAttentionImpl.forward = flash_forward


def write_stats(rows: list[dict[str, Any]], path: Path) -> str | None:
    if not rows:
        return None
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return str(path)
