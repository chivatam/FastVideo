"""Phase-2 Part G: batched GPU metadata construction for both representations.

Exploits Part H's finding: q2k_idx rows are sorted ascending by KV block id
(map_to_index scans the bool map in increasing order), so intersection /
union reduce to batched searchsorted membership tests — the vectorized
equivalent of a two-pointer merge. No per-pair sort is needed in production;
`assume_sorted=False` adds one for robustness testing.

Output format mirrors the kernel's padded q2k_idx convention: fixed-width
int32 rows, valid prefix length in a count tensor, sentinel-padded tail.
"""

from __future__ import annotations

import torch

SENTINEL = torch.iinfo(torch.int32).max


def _compact(values: torch.Tensor, keep: torch.Tensor, out_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack kept entries to the row front (sorted), sentinel-pad the tail."""
    padded = torch.where(keep, values, torch.full_like(values, SENTINEL))
    packed = padded.sort(dim=-1).values[:, :out_width]
    return packed.to(torch.int32), keep.sum(dim=-1).to(torch.int32)


def build_shared_private_batched(
    q0_idx: torch.Tensor,
    q1_idx: torch.Tensor,
    assume_sorted: bool = True,
) -> tuple[torch.Tensor, ...]:
    """[N,K] x2 -> (shared[N,K], n_shared[N], p0[N,K], n_p0[N], p1[N,K], n_p1[N])."""
    a = q0_idx.long()
    b = q1_idx.long()
    if not assume_sorted:
        a = a.sort(dim=-1).values
        b = b.sort(dim=-1).values
    K = a.shape[1]
    pos = torch.searchsorted(b, a).clamp(max=K - 1)
    in_b = b.gather(1, pos) == a
    pos2 = torch.searchsorted(a, b).clamp(max=K - 1)
    in_a = a.gather(1, pos2) == b
    shared, n_shared = _compact(a, in_b, K)
    p0, n_p0 = _compact(a, ~in_b, K)
    p1, n_p1 = _compact(b, ~in_a, K)
    return shared, n_shared, p0, n_p0, p1, n_p1


def build_union_membership_batched(
    q0_idx: torch.Tensor,
    q1_idx: torch.Tensor,
    assume_sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[N,K] x2 -> (union[N,2K] int32, n_union[N], membership[N,2K] uint8 1|2|3)."""
    a = q0_idx.long()
    b = q1_idx.long()
    if not assume_sorted:
        a = a.sort(dim=-1).values
        b = b.sort(dim=-1).values
    K = a.shape[1]
    merged = torch.cat([a, b], dim=-1).sort(dim=-1).values  # [N, 2K]
    first = torch.ones_like(merged, dtype=torch.bool)
    first[:, 1:] = merged[:, 1:] != merged[:, :-1]
    union, n_union = _compact(merged, first, 2 * K)
    u = union.long().clamp(max=torch.iinfo(torch.long).max - 1)
    posb = torch.searchsorted(b, u).clamp(max=K - 1)
    posa = torch.searchsorted(a, u).clamp(max=K - 1)
    m0 = a.gather(1, posa) == u
    m1 = b.gather(1, posb) == u
    membership = (m0.to(torch.uint8) + 2 * m1.to(torch.uint8))
    valid = torch.arange(union.shape[1], device=union.device)[None, :] < n_union[:, None]
    membership = torch.where(valid, membership, torch.zeros_like(membership))
    return union, n_union, membership


def build_shared_private_bitmap(
    q0_idx: torch.Tensor,
    q1_idx: torch.Tensor,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, ...]:
    """Bitmap alternative: scatter to [N, Nk] bool, AND/XOR, re-extract.

    Ordering-agnostic; costs O(N * Nk) memory. Included as the comparison
    point for the microbench.
    """
    N, K = q0_idx.shape
    dev = q0_idx.device
    m0 = torch.zeros(N, num_kv_blocks, dtype=torch.bool, device=dev)
    m1 = torch.zeros(N, num_kv_blocks, dtype=torch.bool, device=dev)
    m0.scatter_(1, q0_idx.long(), True)
    m1.scatter_(1, q1_idx.long(), True)
    both = m0 & m1
    ids = torch.arange(num_kv_blocks, device=dev).expand(N, -1)
    shared, n_shared = _compact(ids, both, K)
    p0, n_p0 = _compact(ids, m0 & ~m1, K)
    p1, n_p1 = _compact(ids, m1 & ~m0, K)
    return shared, n_shared, p0, n_p0, p1, n_p1
