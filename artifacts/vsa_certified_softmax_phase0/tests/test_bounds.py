"""Phase-0 certified-softmax tests (CPU/GPU, no model weights).

Run: pytest artifacts/vsa_certified_softmax_phase0/tests/ -v
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certified_bound import (LOG2E, attn_exact_max, attn_fixed_u, attn_online, block_summaries, certified_u,
                             gather_valid_scores, simulate_online_rescales, true_row_max)

BLOCK = 64
D = 128


def _case(nk=16, ksel=6, rows=8, seed=0, ragged=False, zero_radius=False):
    g = torch.Generator().manual_seed(seed)
    vbs = torch.full((nk,), BLOCK, dtype=torch.int32)
    if ragged:
        vbs = torch.randint(1, BLOCK + 1, (nk,), generator=g, dtype=torch.int32)
    k = torch.randn(1, nk * BLOCK, D, generator=g)
    if zero_radius:
        k = k.view(1, nk, BLOCK, D)[:, :, :1, :].expand(1, nk, BLOCK, D).reshape(1, nk * BLOCK, D).clone()
    # zero out padded rows, exactly like the tiled buffer
    valid = (torch.arange(BLOCK).view(1, BLOCK) < vbs.view(nk, 1)).reshape(-1)
    k[0][~valid] = 0
    q = torch.randn(rows, D, generator=g)
    sel = torch.randperm(nk, generator=g)[:ksel].sort().values
    scale = 1.0 / math.sqrt(D)
    return q, k[0], sel, vbs, scale


@pytest.mark.parametrize("ragged", [False, True], ids=["full", "ragged"])
@pytest.mark.parametrize("zero_radius", [False, True], ids=["random", "zero_radius"])
def test_bound_holds_synthetic(ragged, zero_radius):
    """Tests 1, 3, 9: Cauchy-Schwarz bound holds; boundary blocks; rho=0 blocks."""
    q, k, sel, vbs, scale = _case(ragged=ragged, zero_radius=zero_radius)
    k_bar, rho = block_summaries(k.unsqueeze(0), vbs)
    if zero_radius and not ragged:
        assert rho.abs().max() < 1e-4
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    m = true_row_max(scores)
    u = certified_u(q, k_bar[0], rho[0], sel, scale)
    assert (u - m >= -1e-4).all(), (u - m).min()


def test_block_summaries_match_explicit_valid_slicing():
    """Part 2: k_bar/rho over VALID tokens only; padding must not inflate rho."""
    q, k, sel, vbs, scale = _case(ragged=True, seed=3)
    k_bar, rho = block_summaries(k.unsqueeze(0), vbs)
    kb = k.view(-1, BLOCK, D)
    for b in range(vbs.numel()):
        n = int(vbs[b])
        ref_bar = kb[b, :n].float().mean(0)
        assert torch.allclose(k_bar[0, b], ref_bar, atol=1e-5)
        ref_rho = (kb[b, :n].float() - ref_bar).norm(dim=-1).max() if n > 0 else 0.0
        assert abs(float(rho[0, b]) - float(ref_rho)) < 1e-4


def test_adversarial_q_direction():
    """Test 10: q aligned with the worst in-block deviation stresses the bound."""
    g = torch.Generator().manual_seed(7)
    nk = 8
    k = torch.randn(nk * BLOCK, D, generator=g)
    vbs = torch.full((nk,), BLOCK, dtype=torch.int32)
    k_bar, rho = block_summaries(k.unsqueeze(0), vbs)
    b = 2
    kb = k.view(nk, BLOCK, D)
    dev = kb[b].float() - k_bar[0, b]
    worst = dev[dev.norm(dim=-1).argmax()]
    q = (worst / worst.norm() * 10).unsqueeze(0)
    sel = torch.arange(nk)
    scale = 1.0 / math.sqrt(D)
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    u = certified_u(q, k_bar[0], rho[0], sel, scale)
    m = true_row_max(scores)
    assert (u >= m - 1e-4).all()
    # adversarial alignment makes the bound TIGHT on this block
    ub = (q @ k_bar[0, b] + q.norm() * rho[0, b]) * scale
    tb = (q @ kb[b].float().T * scale).max()
    assert float(ub - tb) < 1e-3


def test_fixed_u_softmax_invariance_high_precision():
    """Test 6: softmax translation invariance -- fixed-U == exact-max in fp64."""
    q, k, sel, vbs, scale = _case(seed=5)
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    scores = scores.double()
    v = torch.randn(sel.numel() * BLOCK, D, dtype=torch.double)
    m = scores.amax(-1, keepdim=True)
    for u_off in (0.0, 3.0, 17.0):
        u = (m.squeeze(-1) + u_off)
        p_ref = torch.exp(scores - m)
        p_u = torch.exp(scores - u.unsqueeze(-1))
        o_ref = (torch.nan_to_num(p_ref) @ v) / torch.nan_to_num(p_ref).sum(-1, keepdim=True)
        o_u = (torch.nan_to_num(p_u) @ v) / torch.nan_to_num(p_u).sum(-1, keepdim=True)
        assert (o_ref - o_u).abs().max() < 1e-12


def test_scaling_matches_production():
    """Test 4: bound and scores carry the SAME 1/sqrt(D) scale."""
    q, k, sel, vbs, scale = _case(seed=9)
    k_bar, rho = block_summaries(k.unsqueeze(0), vbs)
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    u = certified_u(q, k_bar[0], rho[0], sel, scale)
    # unscaled bound vs scaled scores would violate ordering scale-dependently;
    # verify by recomputing both at scale=1 and checking delta scales linearly.
    scores1, _ = gather_valid_scores(q, k, sel, vbs, 1.0)
    u1 = certified_u(q, k_bar[0], rho[0], sel, 1.0)
    d = u - true_row_max(scores)
    d1 = u1 - true_row_max(scores1)
    assert torch.allclose(d * (1.0 / scale), d1, rtol=1e-4, atol=1e-4)


def test_loose_u_underflow_failure_mode():
    """Test 8: a deliberately huge U zeroes even the top token in bf16/exp2."""
    q, k, sel, vbs, scale = _case(seed=11)
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    v = torch.randn(sel.numel() * BLOCK, D)
    m = scores.amax(-1)
    u_loose = m + 200.0  # exp2 arg ~ -288 -> hard zero
    o = attn_fixed_u(scores, v, u_loose, bf16_p=True)
    assert not torch.isfinite(o).all() or o.abs().max() == 0 or (o.abs().max() < 1e-3)


def test_online_reference_matches_exact():
    """Test 2-adjacent: online reference == exact-max softmax in fp32."""
    q, k, sel, vbs, scale = _case(seed=13, ragged=True)
    scores, _ = gather_valid_scores(q, k, sel, vbs, scale)
    v = torch.randn(sel.numel() * BLOCK, D)
    o1 = attn_exact_max(scores, v)
    o2 = attn_online(scores, v)
    assert (o1 - o2).abs().max() < 1e-4


def test_rescale_simulation_sanity():
    """Monotone scores -> every tile raises the raw max; descending -> one."""
    R, T = 4, 512
    up = torch.arange(T).float().view(1, T).expand(R, T) * 0.1
    sim = simulate_online_rescales(up, tile=128)
    assert (sim["n_max_updates"] == T // 128).all()
    down = -up
    sim2 = simulate_online_rescales(down, tile=128)
    assert (sim2["n_max_updates"] == 1).all()
    assert (sim2["n_nontrivial_rescales"] == 0).all()
