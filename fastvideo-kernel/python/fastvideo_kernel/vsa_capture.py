"""Opt-in VSA top-k mask capture for the Phase-1 KV-reuse overlap study.

Captures the compact top-k KV-block indices selected by ``fused_topk_mask``
during REAL inference, immediately after selection, without altering the
production path (the boolean mask keeps flowing to the sparse kernels
unchanged; the compact indices exist only in the capture buffer).

Enabled ONLY when ``FASTVIDEO_VSA_CAPTURE_OVERLAP`` is set to an output
directory. When the variable is unset, the per-call cost is a single dict
lookup and an early return — no allocation, no RNG use, no side effects.

Optional filters (comma-separated ints, default: capture everything):
    FASTVIDEO_VSA_CAPTURE_LAYERS  e.g. "0,15,29"
    FASTVIDEO_VSA_CAPTURE_STEPS   e.g. "0,1,2"

Call-site context (layer prefix, denoising timestep, CFG branch, latent
geometry) is pushed by the framework via :func:`set_context` right before
the attention op runs; :func:`maybe_capture_topk_mask` is invoked from
``ops.video_sparse_attn`` where the mask is materialized.

One ``.pt`` shard is written per captured attention call:
    cap_step{S:03d}_layer{L:02d}_{pos|neg}_call{N:06d}.pt
containing int16/int32 indices with logical shape [B, H, Nq, K] plus
per-call metadata and two built-in sanity checks (row counts == K, and an
exact boolean-mask reconstruction comparison).
"""

from __future__ import annotations

import itertools
import os
import re
from typing import Any

import torch

_ENV_DIR = "FASTVIDEO_VSA_CAPTURE_OVERLAP"
_ENV_LAYERS = "FASTVIDEO_VSA_CAPTURE_LAYERS"
_ENV_STEPS = "FASTVIDEO_VSA_CAPTURE_STEPS"

_LAYER_RE = re.compile(r"\.(\d+)\.")

# Context pushed by the attention backend before each VSA call.
_ctx: dict[str, Any] = {}
_call_counter = itertools.count()


def enabled() -> bool:
    """True iff capture mode is requested via the environment."""
    return bool(os.environ.get(_ENV_DIR))


def capture_dir() -> str | None:
    return os.environ.get(_ENV_DIR) or None


def set_context(**kwargs: Any) -> None:
    """Record call-site metadata (layer prefix, timestep, CFG branch, ...).

    Called by the framework attention backend only when :func:`enabled`;
    the kernel package never depends on the framework.
    """
    _ctx.update(kwargs)


def _parse_filter(env_name: str) -> set[int] | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw or raw.lower() == "all":
        return None
    return {int(tok) for tok in raw.split(",") if tok.strip()}


def _layer_index(prefix: str | None) -> int:
    """Extract the block index from a module prefix like ``blocks.13.attn1.impl``."""
    if not prefix:
        return -1
    m = _LAYER_RE.search(prefix)
    return int(m.group(1)) if m else -1


def maybe_capture_topk_mask(
    mask: torch.Tensor,
    topk: int,
    variable_block_sizes: torch.Tensor | None = None,
) -> None:
    """Capture compact top-k indices from the VSA boolean mask (opt-in).

    No-op unless ``FASTVIDEO_VSA_CAPTURE_OVERLAP`` is set. Never mutates
    ``mask`` and never touches the RNG, so generation output is unaffected.
    """
    out_dir = capture_dir()
    if out_dir is None:
        return

    layer_idx = _layer_index(_ctx.get("layer_prefix"))
    timestep = int(_ctx.get("timestep", -1))
    layer_filter = _parse_filter(_ENV_LAYERS)
    step_filter = _parse_filter(_ENV_STEPS)
    if layer_filter is not None and layer_idx not in layer_filter:
        return
    if step_filter is not None and timestep not in step_filter:
        return

    B, H, Nq, Nk = mask.shape
    k = min(topk, Nk)

    # Sanity check 1: every Q row selects exactly K KV blocks. fused_topk_mask
    # can very rarely deviate on score ties; record the deviation and fall
    # back to an exact ragged encoding so reconstruction stays lossless.
    row_counts = mask.sum(dim=-1)
    counts_ok = bool((row_counts == k).all().item())

    idx_dtype = torch.int16 if Nk <= torch.iinfo(torch.int16).max else torch.int32
    # nonzero() returns indices in row-major order, so the last column
    # reshapes directly into [B, H, Nq, K] with each row sorted ascending.
    nz_cols = mask.nonzero(as_tuple=False)[:, 3]
    if counts_ok:
        idx = nz_cols.view(B, H, Nq, k)
        indices = idx.to(idx_dtype).cpu()
        indices_flat = None
        row_counts_out = None
        # Sanity check 2: compact indices exactly reconstruct the boolean mask.
        recon = torch.zeros_like(mask)
        recon.scatter_(-1, idx, True)
        reconstruct_ok = bool(torch.equal(recon, mask))
    else:
        indices = None
        indices_flat = nz_cols.to(idx_dtype).cpu()
        row_counts_out = row_counts.to(torch.int32).cpu()
        reconstruct_ok = True  # flat nonzero encoding is lossless by construction

    payload = {
        "indices": indices,
        "indices_flat": indices_flat,
        "row_counts": row_counts_out,
        "topk": k,
        "num_q_blocks": Nq,
        "num_kv_blocks": Nk,
        "batch": B,
        "heads": H,
        "counts_ok": counts_ok,
        "reconstruct_ok": reconstruct_ok,
        "variable_block_sizes": (variable_block_sizes.cpu() if variable_block_sizes is not None else None),
        "layer_index": layer_idx,
        "context": dict(_ctx),
    }

    call_id = next(_call_counter)
    branch = "neg" if _ctx.get("is_cfg_negative", False) else "pos"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"cap_step{timestep:03d}_layer{layer_idx:02d}_{branch}_call{call_id:06d}.pt"
    torch.save(payload, os.path.join(out_dir, fname))
