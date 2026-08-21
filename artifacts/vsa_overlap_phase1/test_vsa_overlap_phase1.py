"""Unit tests for the Phase-1 VSA overlap capture + analysis tooling.

Run:  PYTHONPATH=fastvideo-kernel/python pytest artifacts/vsa_overlap_phase1/ -v

CPU-only; no GPU or model weights required.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_vsa_overlap import (build_pairs, interior_qblocks, masks_from_indices, pair_metrics, qblock_coords)

from fastvideo_kernel.vsa_capture import maybe_capture_topk_mask, set_context
from fastvideo_kernel.vsa_utils import (construct_variable_block_sizes, get_tile_partition_indices)

GEOMETRIES = [
    ((21, 45, 80), (6, 12, 20)),  # 720p latent
    ((21, 30, 52), (6, 8, 13)),  # 480p latent
]


@pytest.mark.parametrize("dit_seq_shape,num_tiles", GEOMETRIES)
def test_qblock_coords_match_tile_partition_order(dit_seq_shape, num_tiles):
    """Part B: q_block_id -> (t,h,w) must match get_tile_partition_indices."""
    tile = (4, 4, 4)
    T, H, W = dit_seq_shape
    part = get_tile_partition_indices(dit_seq_shape, tile, torch.device("cpu"))
    vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, torch.device("cpu"), tile)
    coords = qblock_coords(num_tiles)

    assert coords.shape[0] == vbs.numel() == math.prod(num_tiles)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), vbs.cumsum(0)[:-1].long()])
    for b in range(coords.shape[0]):
        t, h, w = coords[b].tolist()
        # First token of block b in tile order is the raster id of the
        # tile's origin voxel (t*4, h*4, w*4).
        expected_first = (t * 4) * H * W + (h * 4) * W + (w * 4)
        assert part[starts[b]].item() == expected_first, f"block {b} coord {(t, h, w)}"
        # Boundary tiles carry the right valid-token count.
        et = min(4, T - t * 4)
        eh = min(4, H - h * 4)
        ew = min(4, W - w * 4)
        assert vbs[b].item() == et * eh * ew


@pytest.mark.parametrize("dit_seq_shape,num_tiles", GEOMETRIES)
def test_neighbor_blocks_map_correctly(dit_seq_shape, num_tiles):
    """Neighboring logical blocks differ by exactly one coordinate step."""
    n_t, n_h, n_w = num_tiles
    coords = qblock_coords(num_tiles)
    for (t, h, w) in [(0, 0, 0), (1, 3, 5), (n_t - 1, n_h - 1, n_w - 2)]:
        b = t * (n_h * n_w) + h * n_w + w
        assert torch.equal(coords[b + 1], torch.tensor([t, h, w + 1]))
        if h + 1 < n_h:
            assert torch.equal(coords[b + n_w], torch.tensor([t, h + 1, w]))
        if t + 1 < n_t:
            assert torch.equal(coords[b + n_h * n_w], torch.tensor([t + 1, h, w]))


def _random_topk_mask(B: int, H: int, Nq: int, Nk: int, K: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(B, H, Nq, Nk, generator=g)
    idx = scores.topk(K, dim=-1).indices
    mask = torch.zeros(B, H, Nq, Nk, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def test_capture_roundtrip_and_disabled_noop(tmp_path, monkeypatch):
    """Sanity #2/#8: compact indices reconstruct the mask exactly; capture
    is a no-op when the env var is unset."""
    mask = _random_topk_mask(1, 2, 24, 96, K=9)
    vbs = torch.full((96, ), 64, dtype=torch.int32)

    monkeypatch.delenv("FASTVIDEO_VSA_CAPTURE_OVERLAP", raising=False)
    maybe_capture_topk_mask(mask, 9, vbs)  # disabled: must not write anything
    assert not list(tmp_path.iterdir())

    monkeypatch.setenv("FASTVIDEO_VSA_CAPTURE_OVERLAP", str(tmp_path))
    set_context(layer_prefix="blocks.7.attn1.impl",
                timestep=2,
                is_cfg_negative=False,
                num_tiles=(2, 3, 4),
                tile_size=(4, 4, 4),
                sparsity=0.9,
                topk=9,
                dit_seq_shape=(8, 12, 16))
    original = mask.clone()
    maybe_capture_topk_mask(mask, 9, vbs)
    assert torch.equal(mask, original), "capture must not mutate the mask"

    shards = glob.glob(str(tmp_path / "cap_step002_layer07_pos_*.pt"))
    assert len(shards) == 1
    payload = torch.load(shards[0], weights_only=False)
    assert payload["counts_ok"] and payload["reconstruct_ok"]
    assert payload["layer_index"] == 7
    idx = payload["indices"].long()
    assert idx.shape == (1, 2, 24, 9)
    recon = torch.zeros_like(mask)
    recon.scatter_(-1, idx, True)
    assert torch.equal(recon, mask)


def test_capture_ragged_rows_exact(tmp_path, monkeypatch):
    """A rare tie in fused_topk_mask can leave a row with != K entries; the
    capture must fall back to a lossless ragged encoding."""
    from analyze_vsa_overlap import masks_from_shard

    mask = _random_topk_mask(1, 2, 24, 96, K=9)
    mask[0, 1, 5, mask[0, 1, 5].nonzero()[0]] = False  # one row has 8 entries
    monkeypatch.setenv("FASTVIDEO_VSA_CAPTURE_OVERLAP", str(tmp_path))
    set_context(layer_prefix="blocks.3.attn1.impl",
                timestep=0,
                is_cfg_negative=False,
                num_tiles=(2, 3, 4),
                tile_size=(4, 4, 4),
                sparsity=0.9,
                topk=9,
                dit_seq_shape=(8, 12, 16))
    maybe_capture_topk_mask(mask, 9, torch.full((96, ), 64, dtype=torch.int32))

    shards = glob.glob(str(tmp_path / "cap_step000_layer03_pos_*.pt"))
    assert len(shards) == 1
    payload = torch.load(shards[0], weights_only=False)
    assert not payload["counts_ok"]
    assert payload["reconstruct_ok"]
    assert payload["indices"] is None and payload["indices_flat"] is not None
    payload["heads"], payload["num_q_blocks"] = 2, 24
    recon = masks_from_shard(payload, "cpu")
    assert torch.equal(recon, mask[0])


def test_pair_metrics_order_invariant_and_exact():
    """Sanity #3: metrics are invariant to per-row index ordering, and match
    a hand-computed example."""
    Nk = 16
    K = 4
    idx = torch.tensor([[[[1, 4, 7, 9], [1, 3, 7, 9]]]])  # S_A, S_B from the brief
    perm = torch.tensor([[[[9, 1, 7, 4], [3, 9, 1, 7]]]])  # same sets, shuffled
    for variant in (idx, perm):
        masks = masks_from_indices(variant[0], Nk)
        m = pair_metrics(masks, torch.tensor([[0, 1]]), K)
        assert m["intersection"].item() == 3  # {1,7,9}
        assert m["union"].item() == 5  # {1,3,4,7,9}
        assert m["reuse_factor"].item() == pytest.approx(8 / 5)
        assert m["jaccard"].item() == pytest.approx(3 / 5)
        assert m["overlap_k"].item() == pytest.approx(3 / 4)


def test_pr1719_pairs_are_consecutive_ids():
    """Sanity: PR#1719 pairing is (2p, 2p+1) in tile order; horizontal pairs
    never wrap a row."""
    num_tiles = (6, 12, 20)
    pairs = build_pairs(num_tiles)
    pr = pairs["pr1719_current"]
    assert torch.equal(pr[:, 0], torch.arange(0, 1440, 2))
    assert torch.equal(pr[:, 1], torch.arange(1, 1440, 2))
    # horizontal pairs never wrap a row: w+1 stays in the same (t, h) row
    hor = pairs["horizontal"]
    coords = qblock_coords(num_tiles)
    assert torch.equal(coords[hor[:, 0]][:, :2], coords[hor[:, 1]][:, :2])
    assert ((coords[hor[:, 1]][:, 2] - coords[hor[:, 0]][:, 2]) == 1).all()


def test_interior_qblocks_flags_boundary_tiles():
    """Sanity #7: partial boundary tiles are identified via vbs < 64."""
    dit_seq_shape, num_tiles = (21, 45, 80), (6, 12, 20)
    vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, torch.device("cpu"), (4, 4, 4))
    interior = interior_qblocks(vbs, 64)
    coords = qblock_coords(num_tiles)
    # T=21 -> last t-tile has 1 valid slice; H=45 -> last h-tile has 1 row.
    expected = (coords[:, 0] < 5) & (coords[:, 1] < 11)
    assert torch.equal(interior, expected)
