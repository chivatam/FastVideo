"""Phase-2 Part G: single-kernel Triton metadata construction (shared/private).

One program per (Q0, Q1) pair. Because q2k_idx rows are sorted ascending
(Part H), membership is a sorted-set test; compaction uses tl.cumsum and a
masked scatter store. Emits the kernel-style padded layout:

    shared[N, K], n_shared[N], p0[N, K], n_p0[N], p1[N, K], n_p1[N]

This is a feasibility prototype for the overhead question, not a tuned
production kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_private_kernel(
    A_ptr,
    B_ptr,
    Sh_ptr,
    NSh_ptr,
    P0_ptr,
    NP0_ptr,
    P1_ptr,
    NP1_ptr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, K_POW2)
    valid = offs < K
    a = tl.load(A_ptr + pid * K + offs, mask=valid, other=2147483647).to(tl.int32)
    b = tl.load(B_ptr + pid * K + offs, mask=valid, other=-2147483647).to(tl.int32)

    # membership via full compare (K x K); rows are small (K=144)
    eq = a[:, None] == b[None, :]
    in_b = tl.sum(eq.to(tl.int32), axis=1) > 0
    in_a = tl.sum(eq.to(tl.int32), axis=0) > 0

    sh_mask = in_b & valid
    p0_mask = (~in_b) & valid
    p1_mask = (~in_a) & (offs < K)

    sh_pos = tl.cumsum(sh_mask.to(tl.int32)) - 1
    p0_pos = tl.cumsum(p0_mask.to(tl.int32)) - 1
    p1_pos = tl.cumsum(p1_mask.to(tl.int32)) - 1

    tl.store(Sh_ptr + pid * K + sh_pos, a, mask=sh_mask)
    tl.store(P0_ptr + pid * K + p0_pos, a, mask=p0_mask)
    tl.store(P1_ptr + pid * K + p1_pos, b, mask=p1_mask)
    tl.store(NSh_ptr + pid, tl.sum(sh_mask.to(tl.int32)))
    tl.store(NP0_ptr + pid, tl.sum(p0_mask.to(tl.int32)))
    tl.store(NP1_ptr + pid, tl.sum(p1_mask.to(tl.int32)))


def build_shared_private_triton(q0_idx: torch.Tensor, q1_idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """[N,K] int (sorted rows) -> padded shared/private lists + counts (int32)."""
    N, K = q0_idx.shape
    a = q0_idx.to(torch.int32).contiguous()
    b = q1_idx.to(torch.int32).contiguous()
    sh = torch.zeros(N, K, dtype=torch.int32, device=a.device)
    p0 = torch.zeros(N, K, dtype=torch.int32, device=a.device)
    p1 = torch.zeros(N, K, dtype=torch.int32, device=a.device)
    n_sh = torch.empty(N, dtype=torch.int32, device=a.device)
    n_p0 = torch.empty(N, dtype=torch.int32, device=a.device)
    n_p1 = torch.empty(N, dtype=torch.int32, device=a.device)
    _shared_private_kernel[(N, )](a, b, sh, n_sh, p0, n_p0, p1, n_p1, K=K, K_POW2=triton.next_power_of_2(K))
    return sh, n_sh, p0, n_p0, p1, n_p1
