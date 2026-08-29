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


_CAPTURE = RuntimeCapture()
_PATCHED = False


def begin_job(job_id: str) -> None:
    _CAPTURE.job_id = job_id
    _CAPTURE.attention_events.clear()
    _CAPTURE.rows.clear()


def finish_job() -> tuple[float, list[dict[str, Any]]]:
    if _CAPTURE.attention_events:
        torch.cuda.synchronize()
    attention_ms = sum(start.elapsed_time(end) for start, end in _CAPTURE.attention_events)
    rows = list(_CAPTURE.rows)
    _CAPTURE.job_id = None
    _CAPTURE.attention_events.clear()
    _CAPTURE.rows.clear()
    return attention_ms, rows


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

    output = flash_attn_func(query, key, value, softmax_scale=query.shape[-1]**-0.5, causal=False)
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

    if mode in {"vsa_bf16", "sim_vsa_nvfp4"}:
        from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionImpl

        original_vsa = VideoSparseAttentionImpl.forward

        def vsa_forward(self, query, key, value, gate_compress, attn_metadata):
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
