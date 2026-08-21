"""Certified center-radius softmax reference: core math (Phase 0).

Bound (Cauchy-Schwarz), matching FastVideo VSA block semantics exactly:

    U_b(q) = [ q . k_bar_b + ||q||_2 * rho_b ] * scale
    U(q)   = max over q's selected blocks of U_b(q)  >=  max_j score(q, k_j)

where k_bar_b is the mean of the VALID tokens of block b (identical to
fused_block_mean: padding rows are zero and the sum is divided by
variable_block_sizes[b]) and rho_b = max_j ||k_j - k_bar_b|| over valid
tokens only. scale = 1/sqrt(D) exactly as the production kernel applies
via scale_log2 = sm_scale * log2(e).
"""

from __future__ import annotations

import math

import torch

BLOCK = 64
RESCALE_THRESHOLD = 8.0  # kernel: alpha == 1 iff (m_run - new_m) * scale_log2 >= -8
LOG2E = math.log2(math.e)


def block_summaries(k: torch.Tensor, vbs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """k: [H, S_pad, D] (padding rows are zero); vbs: [Nk] valid counts.

    Returns (k_bar [H, Nk, D] fp32, rho [H, Nk] fp32), valid-token semantics.
    """
    H, S, D = k.shape
    nk = vbs.numel()
    kb = k.float().view(H, nk, BLOCK, D)
    k_bar = kb.sum(dim=2) / vbs.view(1, nk, 1).float()
    diff = kb - k_bar.unsqueeze(2)
    dist = diff.norm(dim=-1)  # [H, Nk, BLOCK]
    valid = torch.arange(BLOCK, device=k.device).view(1, 1, BLOCK) < vbs.view(1, nk, 1)
    rho = torch.where(valid, dist, torch.zeros_like(dist)).amax(dim=2)
    return k_bar, rho


def certified_u(q_rows: torch.Tensor, k_bar_h: torch.Tensor, rho_h: torch.Tensor, sel: torch.Tensor,
                scale: float) -> torch.Tensor:
    """q_rows: [R, D] fp32; k_bar_h/rho_h: one head's [Nk, D]/[Nk]; sel: [K] block ids."""
    centers = q_rows @ k_bar_h[sel].T  # [R, K]
    u_b = centers + q_rows.norm(dim=-1, keepdim=True) * rho_h[sel].view(1, -1)
    return u_b.amax(dim=-1) * scale


def gather_valid_scores(q_rows: torch.Tensor, k_h: torch.Tensor, sel: torch.Tensor, vbs: torch.Tensor,
                        scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores over all selected fine tokens; invalid (padded) tokens -> -inf.

    Returns (scores [R, K*BLOCK] fp32, valid [K*BLOCK] bool). Token order is
    the kernel's: selected blocks ascending, tokens in-block."""
    Kb = sel.numel()
    ktok = k_h.view(-1, BLOCK, k_h.shape[-1])[sel].reshape(Kb * BLOCK, -1)  # [K*64, D]
    valid = (torch.arange(BLOCK, device=q_rows.device).view(1, BLOCK) < vbs[sel].view(Kb, 1)).reshape(-1)
    s = (q_rows @ ktok.float().T) * scale
    return s.masked_fill(~valid.view(1, -1), float("-inf")), valid


def true_row_max(scores: torch.Tensor) -> torch.Tensor:
    return scores.amax(dim=-1)


# ---------------------------------------------------------------------------
# Part 7: attention references (bf16-P kernel-like numerics optional)
# ---------------------------------------------------------------------------


def attn_exact_max(scores: torch.Tensor, v_tok: torch.Tensor, bf16_p: bool = False) -> torch.Tensor:
    m = scores.amax(dim=-1, keepdim=True)
    p = torch.exp2((scores - m) * LOG2E)
    p = torch.nan_to_num(p, nan=0.0)  # -inf - -inf rows
    if bf16_p:
        p = p.bfloat16().float()
    return (p @ v_tok.float()) / p.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def attn_fixed_u(scores: torch.Tensor, v_tok: torch.Tensor, u: torch.Tensor, bf16_p: bool = False) -> torch.Tensor:
    p = torch.exp2((scores - u.view(-1, 1)) * LOG2E)
    p = torch.nan_to_num(p, nan=0.0)
    if bf16_p:
        p = p.bfloat16().float()
    return (p @ v_tok.float()) / p.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def attn_online(scores: torch.Tensor, v_tok: torch.Tensor, tile: int = 128, bf16_p: bool = False) -> torch.Tensor:
    """Online softmax over `tile`-token chunks in kernel order (one stream)."""
    R, T = scores.shape
    D = v_tok.shape[-1]
    m = torch.full((R, 1), float("-inf"), device=scores.device)
    lsum = torch.zeros(R, 1, device=scores.device)
    o = torch.zeros(R, D, device=scores.device)
    for t0 in range(0, T, tile):
        s = scores[:, t0:t0 + tile]
        new_m = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        new_m = new_m.clamp(min=torch.finfo(torch.float32).min)
        alpha = torch.exp2((m - new_m) * LOG2E)
        alpha = torch.nan_to_num(alpha, nan=1.0)
        p = torch.exp2((s - new_m) * LOG2E)
        p = torch.nan_to_num(p, nan=0.0)
        if bf16_p:
            p = p.bfloat16().float()
        o = o * alpha + p @ v_tok[t0:t0 + tile].float()
        lsum = lsum * alpha + p.sum(dim=-1, keepdim=True)
        m = new_m
    return o / lsum.clamp(min=1e-30)


# ---------------------------------------------------------------------------
# Part 8: kernel-semantics online-softmax counterfactual
# ---------------------------------------------------------------------------


def simulate_online_rescales(scores: torch.Tensor, tile: int = 128) -> dict[str, torch.Tensor]:
    """Count max-updates and NONTRIVIAL rescales with the kernel's threshold
    skip: alpha != 1 iff (m_run - tile_max) * scale_log2 < -RESCALE_THRESHOLD
    (scores here already carry 1/sqrt(D); kernel folds it into scale_log2, so
    the exponent is (m_run - new_m) * log2(e) in our units)."""
    R, T = scores.shape
    m_raw = torch.full((R, ), float("-inf"), device=scores.device)  # true online max
    m_thr = torch.full((R, ), float("-inf"), device=scores.device)  # kernel's thresholded max
    n_updates = torch.zeros(R, dtype=torch.int32, device=scores.device)
    n_rescale = torch.zeros(R, dtype=torch.int32, device=scores.device)
    n_tiles = 0
    for t0 in range(0, T, tile):
        tm = scores[:, t0:t0 + tile].amax(dim=-1)
        n_updates += (tm > m_raw).int()
        m_raw = torch.maximum(m_raw, tm)
        acc_scale = (m_thr - torch.maximum(m_thr, tm)) * LOG2E
        nontrivial = (acc_scale < -RESCALE_THRESHOLD) & torch.isfinite(m_thr)
        first = ~torch.isfinite(m_thr) & torch.isfinite(tm)
        m_thr = torch.where(first | nontrivial, torch.maximum(m_thr, tm), m_thr)
        n_rescale += nontrivial.int()
        n_tiles += 1
    return {"n_tiles": torch.tensor(n_tiles), "n_max_updates": n_updates, "n_nontrivial_rescales": n_rescale}
