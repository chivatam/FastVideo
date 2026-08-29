from __future__ import annotations

import math

import pytest
import torch

from research.compressed_halo_vsa.compressed_support import (
    compressed_halo_attention,
    merge_core_halo_with_coarse,
    merge_online_outputs,
    rank_normalized_topk_mask,
    rectified_output,
)


def test_rectified_output_uses_only_omitted_coarse_mass() -> None:
    exact = torch.tensor([[[[2.0], [4.0]]]])
    attention = torch.tensor([[[[0.6, 0.3, 0.1]]]])
    values = torch.tensor([[[[10.0], [20.0], [30.0]]]])
    mask = torch.tensor([[[[True, False, False]]]])
    gate = torch.tensor([[[[0.5], [0.25]]]])

    output, retained, omitted = rectified_output(
        exact,
        attention,
        values,
        mask,
        gate,
        block_elements=2,
    )

    assert torch.allclose(retained, torch.tensor([[[0.6]]]))
    assert torch.allclose(omitted, torch.tensor([[[[9.0], [9.0]]]]))
    expected = torch.tensor([[[[5.7], [4.65]]]])
    assert torch.allclose(output, expected)


def test_online_merge_matches_concatenated_softmax() -> None:
    exact_logits = torch.tensor([[[0.2, 1.1]]])
    halo_logits = torch.tensor([[[-0.4, 0.7, 0.1]]])
    exact_values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    halo_values = torch.tensor([[[[2.0, 1.0], [1.0, 3.0], [-1.0, 2.0]]]])
    exact_prob = exact_logits.softmax(dim=-1)
    halo_prob = halo_logits.softmax(dim=-1)
    exact_output = torch.matmul(
        exact_prob.unsqueeze(-2),
        exact_values,
    ).squeeze(-2)
    halo_output = torch.matmul(
        halo_prob.unsqueeze(-2),
        halo_values,
    ).squeeze(-2)
    merged, halo_fraction = merge_online_outputs(
        exact_output,
        torch.logsumexp(exact_logits, dim=-1) / math.log(2.0),
        halo_output,
        torch.logsumexp(halo_logits, dim=-1) / math.log(2.0),
    )

    all_logits = torch.cat([exact_logits, halo_logits], dim=-1)
    all_values = torch.cat([exact_values, halo_values], dim=-2)
    expected = torch.matmul(
        all_logits.softmax(dim=-1).unsqueeze(-2),
        all_values,
    ).squeeze(-2)
    expected_halo_fraction = all_logits.softmax(dim=-1)[..., 2:].sum(dim=-1)
    assert torch.allclose(merged, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        halo_fraction,
        expected_halo_fraction,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.cuda
def test_rank_normalized_topk_is_exact_and_order_preserving() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(11)
    scores = torch.randn(
        1,
        2,
        32,
        624,
        device="cuda",
        dtype=torch.bfloat16,
    )
    scores[..., 0] = 21.0
    scores[..., 1] = -12.0
    topk = 125

    mask = rank_normalized_topk_mask(scores, topk)
    reference_indices = torch.topk(scores, topk, dim=-1).indices
    reference_mask = torch.zeros_like(
        scores,
        dtype=torch.bool,
    ).scatter_(-1, reference_indices, True)

    assert torch.equal(mask.sum(dim=-1), torch.full_like(mask.sum(dim=-1), topk))
    assert torch.equal(mask, reference_mask)


@pytest.mark.cuda
def test_fused_core_halo_merge_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(19)
    shape = (1, 2, 128, 128)
    exact = torch.randn(
        shape,
        device="cuda",
        dtype=torch.bfloat16,
    )
    halo = torch.randn_like(exact)
    coarse = torch.randn_like(exact)
    gate = torch.randn_like(exact) * 0.01
    exact_lse = torch.randn(shape[:-1], device="cuda")
    halo_lse = torch.randn(shape[:-1], device="cuda")

    output, halo_fraction = merge_core_halo_with_coarse(
        exact,
        exact_lse,
        halo,
        halo_lse,
        coarse,
        gate,
        return_halo_fraction=True,
    )
    expected_merge, expected_fraction = merge_online_outputs(
        exact,
        exact_lse,
        halo,
        halo_lse,
    )
    expected_output = expected_merge + coarse * gate

    assert halo_fraction is not None
    assert torch.allclose(
        output,
        expected_output,
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.allclose(
        halo_fraction,
        expected_fraction,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.cuda
def test_compressed_halo_kernel_matches_torch_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, heads, query_blocks, blocks, block_elements, dim = (
        1,
        2,
        2,
        8,
        64,
        128,
    )
    sequence = query_blocks * block_elements
    query = torch.randn(
        batch,
        heads,
        sequence,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        heads,
        blocks,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    mask = torch.zeros(
        batch,
        heads,
        query_blocks,
        blocks,
        device=device,
        dtype=torch.bool,
    )
    mask[..., 0] = True
    mask[..., 3] = True
    sizes = torch.tensor(
        [64, 64, 48, 64, 32, 64, 16, 64],
        device=device,
        dtype=torch.int32,
    )

    actual_output, actual_lse = compressed_halo_attention(
        query,
        key,
        value,
        mask,
        sizes,
    )

    logits = torch.matmul(
        query.float(),
        key.float().transpose(-2, -1),
    ) / math.sqrt(dim)
    logits += sizes.float().log().view(1, 1, 1, -1)
    token_mask = mask.unsqueeze(-2).expand(-1, -1, -1, block_elements, -1).reshape(batch, heads, sequence, blocks)
    logits = logits.masked_fill(token_mask, float("-inf"))
    expected_prob = logits.softmax(dim=-1)
    expected_output = torch.matmul(
        expected_prob,
        value.float(),
    ).to(torch.bfloat16)
    expected_lse = torch.logsumexp(logits, dim=-1) / math.log(2.0)

    assert torch.allclose(
        actual_output,
        expected_output,
        atol=4e-2,
        rtol=4e-2,
    )
    assert torch.allclose(
        actual_lse,
        expected_lse,
        atol=4e-2,
        rtol=4e-2,
    )
