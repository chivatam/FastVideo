"""Pair-aware VSA metadata for the sm_100a local KV-reuse kernel (Phase 3).

Builds, from the SAME bool mask ``fused_topk_mask`` produces, the metadata the
``block_sparse_sm100a_pair_fwd`` kernel consumes. Two modes:

  "union"          (Strategy B2): both rows of a CTA pair walk S0 ∪ S1 in
                   ascending KV order; every K-tile is dual-consumer.
  "shared-private" (Strategy A):  rows are laid out
                   [shared asc | pad-to-4 | own-private asc]; the leading
                   ceil(|shared|/4) K-tiles are dual-consumer, the rest run
                   the baseline per-m-tile schedule with pair-max padding.

Membership/padding masking is encoded in ``block_thresholds``: per (row,
position) the number of valid tokens of the referenced KV block, or 0 to mask
the whole block (non-member in a union walk, or metadata padding). Indices
stay ascending per phase; no runtime sorting is added (top-k rows arrive
sorted from map_to_index's ascending scan — Phase 2, Part H).

Correctness-first torch implementation; production would fuse this into the
map_to_index scan.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class PairMetadata(NamedTuple):
    q2k_idx: torch.Tensor            # [B*H*Nq, width] int32, ascending per phase
    q2k_num: torch.Tensor            # [B*H*Nq] int32
    pair_shared_tiles: torch.Tensor  # [B*H*Nq//2] int32, leading dual-consumer K-tiles
    block_thresholds: torch.Tensor   # [B*H*Nq, width] int32 (0 masks the block)


BLOCKS_PER_KTILE = 4  # blk64: K_TILE=256 / BLOCK=64


def _sorted_ids(member: torch.Tensor, nk: int) -> torch.Tensor:
    """[..., Nk] bool -> ascending member ids packed to the front (sentinel nk)."""
    ids = torch.arange(nk, device=member.device, dtype=torch.int32)
    key = torch.where(member, ids, torch.full_like(ids, nk))
    return key.expand(member.shape).contiguous().sort(dim=-1).values


def build_pair_metadata(
    mask: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    mode: str = "shared-private",
) -> PairMetadata:
    """mask: [B, H, Nq, Nk] bool (rows 2p / 2p+1 are one CTA pair)."""
    assert mask.dtype == torch.bool and mask.dim() == 4
    B, H, Nq, Nk = mask.shape
    assert Nq % 2 == 0, "a CTA owns an adjacent pair of query blocks"
    dev = mask.device
    vbs = variable_block_sizes.to(device=dev, dtype=torch.int32)

    m0 = mask[:, :, 0::2, :]
    m1 = mask[:, :, 1::2, :]

    if mode == "union":
        union = m0 | m1
        n_u = union.sum(-1).to(torch.int32)                      # [B,H,P]
        width = max(BLOCKS_PER_KTILE, ((int(n_u.max()) + 3) // 4) * 4)
        row = _sorted_ids(union, Nk)
        if width > Nk:
            row = torch.nn.functional.pad(row, (0, width - Nk), value=0)
        row = row[..., :width]                                   # [B,H,P,width]
        pos = torch.arange(width, device=dev)
        valid = pos < n_u.unsqueeze(-1)
        row = torch.where(valid, row, torch.zeros_like(row))
        mem0 = m0.gather(-1, row.long()) & valid
        mem1 = m1.gather(-1, row.long()) & valid
        thr_blk = vbs[row.long()]
        thr0 = torch.where(mem0, thr_blk, torch.zeros_like(thr_blk))
        thr1 = torch.where(mem1, thr_blk, torch.zeros_like(thr_blk))
        cnt0 = cnt1 = n_u
        shared_tiles = (n_u + BLOCKS_PER_KTILE - 1) // BLOCKS_PER_KTILE
        row0 = row1 = row
    elif mode == "shared-private":
        shared = m0 & m1
        p0 = m0 & ~m1
        p1 = m1 & ~m0
        n_sh = shared.sum(-1).to(torch.int32)
        n_p0 = p0.sum(-1).to(torch.int32)
        n_p1 = p1.sum(-1).to(torch.int32)
        shared_tiles = (n_sh + BLOCKS_PER_KTILE - 1) // BLOCKS_PER_KTILE
        priv_start = shared_tiles * BLOCKS_PER_KTILE            # [B,H,P]
        cnt0 = priv_start + n_p0
        cnt1 = priv_start + n_p1
        width = max(BLOCKS_PER_KTILE, ((int(torch.max(torch.maximum(cnt0, cnt1))) + 3) // 4) * 4)

        sh_row = _sorted_ids(shared, Nk)
        # pad value: repeat the LAST shared block (threshold 0 keeps it inert)
        last_sh = sh_row.gather(-1, (n_sh.long() - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
        last_sh = torch.where(n_sh > 0, last_sh, torch.zeros_like(last_sh))
        pos = torch.arange(width, device=dev)

        def make_row(pm: torch.Tensor, cnt_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            pr_row = _sorted_ids(pm, Nk)
            in_sh = pos < n_sh.unsqueeze(-1)
            in_pad = (pos >= n_sh.unsqueeze(-1)) & (pos < priv_start.unsqueeze(-1))
            in_pr = (pos >= priv_start.unsqueeze(-1)) & (pos < cnt_m.unsqueeze(-1))
            sh_v = sh_row.gather(-1, pos.expand(*n_sh.shape, width).clamp(max=Nk - 1).long())
            pr_pos = (pos - priv_start.unsqueeze(-1)).clamp(min=0, max=Nk - 1)
            pr_v = pr_row.gather(-1, pr_pos.long())
            r = torch.where(in_sh, sh_v,
                            torch.where(in_pad, last_sh.unsqueeze(-1).expand_as(sh_v),
                                        torch.where(in_pr, pr_v, torch.zeros_like(pr_v))))
            r = r.clamp(max=Nk - 1)
            thr = torch.where(in_sh | in_pr, vbs[r.long()], torch.zeros_like(r))
            return r.to(torch.int32), thr.to(torch.int32)

        row0, thr0 = make_row(p0, cnt0)
        row1, thr1 = make_row(p1, cnt1)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    q2k_idx = torch.stack([row0, row1], dim=3).reshape(B * H * Nq, -1).to(torch.int32).contiguous()
    thresholds = torch.stack([thr0, thr1], dim=3).reshape(B * H * Nq, -1).to(torch.int32).contiguous()
    q2k_num = torch.stack([cnt0, cnt1], dim=3).reshape(B * H * Nq).to(torch.int32).contiguous()
    pair_shared_tiles = shared_tiles.reshape(B * H * (Nq // 2)).to(torch.int32).contiguous()
    return PairMetadata(q2k_idx, q2k_num, pair_shared_tiles, thresholds)
