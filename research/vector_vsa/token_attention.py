from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _token_sparse_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    selected_index_ptr,
    selected_count_ptr,
    output_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ib,
    stride_ih,
    stride_iq,
    stride_ik,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    num_heads: tl.constexpr,
    query_sequence: tl.constexpr,
    key_sequence: tl.constexpr,
    head_dim: tl.constexpr,
    query_block: tl.constexpr,
    max_selected: tl.constexpr,
    token_tile: tl.constexpr,
):
    query_block_index = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    offsets_m = query_block_index * query_block + tl.arange(0, query_block)
    offsets_d = tl.arange(0, head_dim)
    query = tl.load(
        query_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
    )
    row_max = tl.full([query_block], -float("inf"), tl.float32)
    row_sum = tl.zeros([query_block], tl.float32)
    accumulator = tl.zeros([query_block, head_dim], tl.float32)
    scale_log2 = (1.0 / tl.sqrt(float(head_dim))) * 1.4426950408889634

    selected_count = tl.load(
        selected_count_ptr + batch * stride_cb + head * stride_ch + query_block_index * stride_cq,
    )
    selected_base = selected_index_ptr + batch * stride_ib + head * stride_ih + query_block_index * stride_iq
    tile_offsets = tl.arange(0, token_tile)
    for selected_offset in range(0, max_selected, token_tile):
        selection_offsets = selected_offset + tile_offsets
        valid_selection = selection_offsets < selected_count
        token_indices = tl.load(
            selected_base + selection_offsets * stride_ik,
            mask=valid_selection,
            other=0,
        ).to(tl.int32)
        valid_selection &= token_indices < key_sequence
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + token_indices[:, None] * stride_ks
            + offsets_d[None, :] * stride_kd,
            mask=valid_selection[:, None],
            other=0.0,
        )
        value = tl.load(
            value_ptr
            + batch * stride_vb
            + head * stride_vh
            + token_indices[:, None] * stride_vs
            + offsets_d[None, :] * stride_vd,
            mask=valid_selection[:, None],
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(key)) * scale_log2
        logits = tl.where(
            valid_selection[None, :],
            logits,
            -float("inf"),
        )

        next_max = tl.maximum(row_max, tl.max(logits, axis=1))
        alpha = tl.exp2(row_max - next_max)
        probabilities = tl.exp2(logits - next_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator *= alpha[:, None]
        accumulator = tl.dot(
            probabilities.to(tl.bfloat16),
            value,
            accumulator,
        )
        row_max = next_max

    output = accumulator / row_sum[:, None]
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od,
        output,
    )
    tl.store(
        lse_ptr + batch * stride_lb + head * stride_lh + offsets_m * stride_ls,
        row_max + tl.log2(row_sum),
    )


def token_sparse_attention(
    query_bhsd: torch.Tensor,
    key_bhsd: torch.Tensor,
    value_bhsd: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    *,
    query_block: int = 64,
    token_tile: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-only attention over per-query-block arbitrary K positions."""
    if not query_bhsd.is_cuda:
        raise ValueError("token_sparse_attention requires CUDA tensors")
    if query_bhsd.ndim != 4:
        raise ValueError("query_bhsd must use [B,H,S,D]")
    if key_bhsd.shape != value_bhsd.shape:
        raise ValueError("Key and value shapes must match")
    batch, heads, query_sequence, head_dim = query_bhsd.shape
    key_sequence = key_bhsd.shape[2]
    if query_sequence % query_block:
        raise ValueError("Query sequence must be divisible by query_block")
    expected = (batch, heads, query_sequence // query_block)
    if selected_indices.shape[:3] != expected:
        raise ValueError(f"Unexpected selected-index shape {selected_indices.shape}; expected prefix {expected}")
    if selected_counts.shape != expected:
        raise ValueError("Selected counts must match query-block geometry")
    if int(selected_counts.max().item()) > selected_indices.shape[-1]:
        raise ValueError("Selected count exceeds fixed index capacity")
    if token_tile <= 0 or token_tile & (token_tile - 1):
        raise ValueError("token_tile must be a positive power of two")

    selected_indices = selected_indices.to(torch.int32).contiguous()
    selected_counts = selected_counts.to(torch.int32).contiguous()
    output = torch.empty_like(query_bhsd)
    lse = torch.empty(
        (batch, heads, query_sequence),
        device=query_bhsd.device,
        dtype=torch.float32,
    )
    grid = (query_sequence // query_block, batch * heads)
    _token_sparse_attention_kernel[grid](
        query_bhsd,
        key_bhsd,
        value_bhsd,
        selected_indices,
        selected_counts,
        output,
        lse,
        *query_bhsd.stride(),
        *key_bhsd.stride(),
        *value_bhsd.stride(),
        *selected_indices.stride(),
        *selected_counts.stride(),
        *output.stride(),
        *lse.stride(),
        num_heads=heads,
        query_sequence=query_sequence,
        key_sequence=key_sequence,
        head_dim=head_dim,
        query_block=query_block,
        max_selected=selected_indices.shape[-1],
        token_tile=token_tile,
        num_warps=8,
        num_stages=3,
    )
    return output, lse
