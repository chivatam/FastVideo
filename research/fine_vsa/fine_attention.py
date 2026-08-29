from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fine_sparse_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    selected_index_ptr,
    selected_count_ptr,
    child_size_ptr,
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
    child_width: tl.constexpr,
    load_width: tl.constexpr,
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
        selected_count_ptr
        + batch * stride_cb
        + head * stride_ch
        + query_block_index * stride_cq,
    )
    selected_base = (
        selected_index_ptr
        + batch * stride_ib
        + head * stride_ih
        + query_block_index * stride_iq
    )
    offsets_n = tl.arange(0, load_width)
    for selected_offset in range(0, selected_count):
        child_index = tl.load(
            selected_base + selected_offset * stride_ik,
        ).to(tl.int32)
        child_size = tl.load(child_size_ptr + child_index).to(tl.int32)
        token_offsets = child_index * child_width + offsets_n
        valid_n = (
            (offsets_n < child_size)
            & (offsets_n < child_width)
            & (token_offsets < key_sequence)
        )
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + token_offsets[:, None] * stride_ks
            + offsets_d[None, :] * stride_kd,
            mask=valid_n[:, None],
            other=0.0,
        )
        value = tl.load(
            value_ptr
            + batch * stride_vb
            + head * stride_vh
            + token_offsets[:, None] * stride_vs
            + offsets_d[None, :] * stride_vd,
            mask=valid_n[:, None],
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(key)) * scale_log2
        logits = tl.where(
            valid_n[None, :],
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
        lse_ptr
        + batch * stride_lb
        + head * stride_lh
        + offsets_m * stride_ls,
        row_max + tl.log2(row_sum),
    )


def fine_sparse_attention(
    query_bhsd: torch.Tensor,
    key_bhsd: torch.Tensor,
    value_bhsd: torch.Tensor,
    selected_indices: torch.Tensor,
    child_sizes: torch.Tensor,
    *,
    child_width: int,
    query_block: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-only fixed-density sparse attention with fine KV blocks."""
    if not query_bhsd.is_cuda:
        raise ValueError("fine_sparse_attention requires CUDA tensors")
    if query_bhsd.ndim != 4:
        raise ValueError("query_bhsd must use [B,H,S,D]")
    if key_bhsd.shape != value_bhsd.shape:
        raise ValueError("Key and value shapes must match")
    batch, heads, query_sequence, head_dim = query_bhsd.shape
    key_sequence = key_bhsd.shape[2]
    if query_sequence % query_block:
        raise ValueError("Query sequence must be divisible by query_block")
    if key_sequence % child_width:
        raise ValueError("Key sequence must be divisible by child_width")
    expected = (
        batch,
        heads,
        query_sequence // query_block,
    )
    if selected_indices.shape[:3] != expected:
        raise ValueError(
            f"Unexpected selected-index shape {selected_indices.shape}; "
            f"expected prefix {expected}"
        )
    if child_sizes.numel() != key_sequence // child_width:
        raise ValueError("Child-size metadata does not cover the key sequence")
    selected_indices = selected_indices.to(torch.int32).contiguous()
    selected_counts = torch.full(
        selected_indices.shape[:-1],
        selected_indices.shape[-1],
        dtype=torch.int32,
        device=selected_indices.device,
    )
    child_sizes = child_sizes.to(torch.int32).contiguous()
    output = torch.empty_like(query_bhsd)
    lse = torch.empty(
        (batch, heads, query_sequence),
        device=query_bhsd.device,
        dtype=torch.float32,
    )
    load_width = max(16, child_width)
    grid = (query_sequence // query_block, batch * heads)
    _fine_sparse_attention_kernel[grid](
        query_bhsd,
        key_bhsd,
        value_bhsd,
        selected_indices,
        selected_counts,
        child_sizes,
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
        child_width=child_width,
        load_width=load_width,
        num_warps=8,
        num_stages=3,
    )
    return output, lse


def child_block_sizes(
    parent_sizes: torch.Tensor,
    child_width: int,
    *,
    parent_width: int = 64,
) -> torch.Tensor:
    if parent_width % child_width:
        raise ValueError("child_width must divide parent_width")
    offsets = torch.arange(
        0,
        parent_width,
        child_width,
        device=parent_sizes.device,
        dtype=parent_sizes.dtype,
    )
    return (
        parent_sizes[:, None]
        .sub(offsets[None, :])
        .clamp(min=0, max=child_width)
        .reshape(-1)
    )


def child_block_mean(
    tensor_bhsd: torch.Tensor,
    parent_sizes: torch.Tensor,
    child_width: int,
    *,
    parent_width: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = child_block_sizes(
        parent_sizes,
        child_width,
        parent_width=parent_width,
    )
    batch, heads, sequence, head_dim = tensor_bhsd.shape
    if sequence != parent_sizes.numel() * parent_width:
        raise ValueError("Tensor sequence and parent metadata disagree")
    factor = parent_width // child_width
    blocks = tensor_bhsd.view(
        batch,
        heads,
        parent_sizes.numel(),
        factor,
        child_width,
        head_dim,
    )
    valid = (
        torch.arange(
            child_width,
            device=tensor_bhsd.device,
        )[None, None, None, None, :, None]
        < sizes.view(
            1,
            1,
            parent_sizes.numel(),
            factor,
            1,
            1,
        )
    )
    means = (blocks * valid).sum(dim=-2) / sizes.clamp_min(1).view(
        1,
        1,
        parent_sizes.numel(),
        factor,
        1,
    )
    return (
        means.reshape(
            batch,
            heads,
            parent_sizes.numel() * factor,
            head_dim,
        ),
        sizes,
    )
