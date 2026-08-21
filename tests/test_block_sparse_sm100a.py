# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the sm_100a CUDA block-sparse VSA forward.

Compared against an explicit PyTorch reference rather than the Triton kernel: Triton's
block-sparse forward is hardcoded to 64-token blocks (BLOCK_M = BLOCK_N = 64) while this
extension also carries a 128-token build, so a direct comparison would be comparing two
different sparsity granularities. The reference below applies exactly the semantics the
kernel is supposed to implement -- selected blocks only, keys past variable_block_sizes
masked. Every case runs at both block sizes.

Run with: python -m pytest tests/test_block_sparse_sm100a.py -v
"""

import os
import subprocess
import sys

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_sm100a as vsa

HEAD_DIM = 128
BLOCK_SIZES = [64, 128]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0)
    or not vsa._HAS_VSA_SM100A,
    reason="requires Blackwell (sm_100a) and a built fastvideo_kernel extension",
)


def make_case(block, num_blocks=8, topk=4, heads=4, batch=1, ragged=False, seed=0):
    torch.manual_seed(seed)
    S = num_blocks * block
    shape = (batch, heads, S, HEAD_DIM) if vsa.BHSD else (batch, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    idx = torch.empty((batch * heads * num_blocks, topk), dtype=torch.int32, device="cuda")
    for r in range(idx.shape[0]):
        idx[r] = torch.randperm(num_blocks, device="cuda")[:topk].to(torch.int32).sort().values
    num = torch.full((batch * heads * num_blocks, ), topk, dtype=torch.int32, device="cuda")

    if ragged:
        vbs = torch.randint(block // 2, block + 1, (num_blocks, ), dtype=torch.int32,
                            device="cuda")
    else:
        vbs = torch.full((num_blocks, ), block, dtype=torch.int32, device="cuda")
    return q, k, v, idx, num, vbs


def reference(q, k, v, idx, num, vbs, block):
    """Dense attention restricted to the selected blocks, with padded keys masked."""
    if not vsa.BHSD:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # -> [B, H, S, D]
    B, H, S, D = q.shape
    num_blocks = vbs.numel()
    scale = 1.0 / (D**0.5)

    keep = torch.zeros((B, H, S, S), dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qb in range(num_blocks):
                row = (b * H + h) * num_blocks + qb
                for j in range(int(num[row])):
                    kb = int(idx[row, j])
                    valid = int(vbs[kb])
                    keep[b, h, qb * block:(qb + 1) * block,
                         kb * block:kb * block + valid] = True

    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = p @ v.float()
    lse = torch.logsumexp(scores, dim=-1) * 1.4426950408889634
    return out, lse


def run_and_compare(block, ragged, num_blocks=8, topk=4, heads=4, atol=0.02):
    q, k, v, idx, num, vbs = make_case(block, num_blocks=num_blocks, topk=topk, heads=heads,
                                       ragged=ragged)
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    ref, ref_lse = reference(q, k, v, idx, num, vbs, block)

    got_o = got if vsa.BHSD else got.transpose(1, 2)
    diff = (got_o.float() - ref).abs().max().item()
    assert diff < atol, f"out: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - ref_lse).abs().max().item()
    assert lse_diff < 0.05, f"lse: max |diff| = {lse_diff:.5f}"


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_forward_matches_reference(block):
    run_and_compare(block, ragged=False)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_forward_matches_reference_ragged(block):
    """variable_block_sizes is what FastVideo always passes; padded keys must be masked."""
    run_and_compare(block, ragged=True)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("topk", [1, 2, 3, 5, 7])
def test_topk_not_a_multiple_of_the_group(block, topk):
    """The kernel groups selected blocks; a ragged final group must still be correct."""
    run_and_compare(block, ragged=True, num_blocks=8, topk=topk)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", [4, 8, 16])
def test_sequence_lengths(block, num_blocks):
    run_and_compare(block, ragged=True, num_blocks=num_blocks, topk=3)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_lse_is_not_vacuous(block):
    """Guards the lse assertion: a wrong lse must actually fail the comparison."""
    q, k, v, idx, num, vbs = make_case(block, ragged=True)
    _, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    _, ref_lse = reference(q, k, v, idx, num, vbs, block)
    assert (got_lse.float() - (ref_lse + 1.0)).abs().max().item() > 0.5


def test_unsupported_is_rejected():
    q, _, _, _, _, vbs = make_case(64)
    assert not vsa.is_supported(q.float(), vbs)                 # wrong dtype
    assert not vsa.is_supported(q[..., :64].contiguous(), vbs)  # wrong head_dim
    odd = torch.full((7, ), 64, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, odd)                         # seqlen/blocks mismatch
    thirty_two = torch.full((16, ), 32, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, thirty_two)                  # block size with no build


# ---------------------------------------------------------------------------
# Per-q-tile q2k_num regressions. A CTA owns an ADJACENT PAIR of query blocks;
# the kernel must honor each row's own count -- not the even row's -- in both
# the q2k_idx window clamp and the vbs-threshold masking. These cases pin the
# two failure modes of a pair-shared count: silent corruption of the odd tile
# when the pair's rows differ, and a hang when an even row's count is zero.
# ---------------------------------------------------------------------------


def make_two_class_case(block, num_blocks=16, prefix=3, topk=4, heads=4, seed=0, zero_rows=()):
    """Two row classes, like a packed multimodal layout: `prefix` DENSE rows
    (count = num_blocks) followed by rows at a uniform smaller count
    (prefix + topk). With an ODD `prefix`, pair (prefix-1, prefix) straddles
    the classes, so the two rows of one CTA carry different counts."""
    torch.manual_seed(seed)
    S = num_blocks * block
    shape = (1, heads, S, HEAD_DIM) if vsa.BHSD else (1, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.randint(block // 2, block + 1, (num_blocks, ), dtype=torch.int32, device="cuda")

    rows = heads * num_blocks
    # Pad past each row's count with 0 -- a VALID block id, like real metadata
    # padding -- so a regression to the pair-shared count reads plausible
    # in-bounds garbage and fails by WRONG VALUES rather than by luck.
    idx = torch.zeros((rows, num_blocks), dtype=torch.int32, device="cuda")
    num = torch.zeros((rows, ), dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(seed + 1)
    for h in range(heads):
        for t in range(num_blocks):
            r = h * num_blocks + t
            if t < prefix:
                sel = torch.arange(num_blocks, dtype=torch.int32)
            else:
                vid = torch.randperm(num_blocks - prefix, generator=g)[:topk] + prefix
                sel = torch.cat([torch.arange(prefix), vid.sort().values]).to(torch.int32)
            if t in zero_rows:
                sel = sel[:0]
            idx[r, :sel.numel()] = sel.cuda()
            num[r] = sel.numel()
    return q, k, v, idx, num, vbs


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_nonuniform_counts_straddling_a_pair(block):
    """A dense row paired with a top-k row must both be exact. Regression: the
    pair-shared count computed the odd tile with the even row's count (max
    |out diff| ~0.6 on this exact case), leaving every other tile correct."""
    q, k, v, idx, num, vbs = make_two_class_case(block)
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    ref, ref_lse = reference(q, k, v, idx, num, vbs, block)
    got_o = got if vsa.BHSD else got.transpose(1, 2)
    diff = (got_o.float() - ref).abs().max().item()
    assert diff < 0.02, f"out: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - ref_lse).abs().max().item()
    assert lse_diff < 0.05, f"lse: max |diff| = {lse_diff:.5f}"


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_same_seed_determinism(block):
    """Two same-seed runs must be bitwise identical. Also documents that the
    uniform-count path is unchanged by the per-tile fix: with cnt0 == cnt1 >= 1
    the pair trip count max(cnt0, cnt1, 1) and the per-tile clamps reduce to
    the original expressions."""
    runs = []
    for _ in range(2):
        q, k, v, idx, num, vbs = make_case(block, ragged=True, seed=3)
        o, m = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
        torch.cuda.synchronize()
        runs.append((o, m))
    assert torch.equal(runs[0][0], runs[1][0]), "out differs across same-seed runs"
    assert torch.equal(runs[0][1], runs[1][1]), "lse differs across same-seed runs"


def _zero_count_case_main(block):
    """Body of test_zero_count_rows, run in a subprocess (see the test)."""
    zero_rows = (5, 6, 8, 9)  # odd member; even member with nonzero sibling 7; a whole pair
    q, k, v, idx, num, vbs = make_two_class_case(block, zero_rows=zero_rows)
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    torch.cuda.synchronize()
    ref, ref_lse = reference(q, k, v, idx, num, vbs, block)
    got_o = (got if vsa.BHSD else got.transpose(1, 2)).float()

    empty = torch.zeros(ref.shape[:-1], dtype=torch.bool, device="cuda")  # [B, H, S]
    for t in zero_rows:
        rows = slice(t * block, (t + 1) * block)
        empty[:, :, rows] = True
        zmax = got_o[:, :, rows].abs().max().item()
        assert zmax == 0.0, f"zero-count tile {t}: expected exact zeros, got max |out| = {zmax}"
    assert torch.isfinite(got_lse).all(), "lse must stay finite on zero-count rows"
    # every non-empty row -- including tile 7, whose pair sibling is empty -- is exact
    diff = (got_o - ref).abs().amax(dim=-1)
    assert diff[~empty].max().item() < 0.02
    lse_diff = (got_lse.float() - ref_lse).abs()
    assert lse_diff[~empty].max().item() < 0.05
    print("ZERO_COUNT_CASE_OK")


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_zero_count_rows(block):
    """q2k_num == 0 rows must produce exactly-zero output rows (finite lse) and
    leave every other row intact. Runs in a subprocess with a timeout because
    the failure mode this pins is a HANG (a zero count on an even row starved
    the softmax/correction mbarrier handshake); a regression must fail the
    suite, not wedge it."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--zero-count-case", str(block)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0 and "ZERO_COUNT_CASE_OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Phase-3 pair-shared (local KV reuse) forward: Strategy B2 (union walk, every
# K-tile dual-consumer) and Strategy A (shared prefix dual-consumer + baseline
# private phase). Both must reproduce the baseline kernel's sparse semantics
# on identical inputs; tolerances are bf16 reassociation only (the K-tile
# segmentation of the online softmax changes when blocks regroup).
# ---------------------------------------------------------------------------


def make_pair_mask(num_blocks, topk, heads, overlap, seed=0, batch=1):
    """Bool mask [B, H, Nq, Nk] whose CTA pairs share ~overlap*topk blocks."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch, heads, num_blocks, num_blocks, dtype=torch.bool)
    n_shared = int(round(overlap * topk))
    for b in range(batch):
        for h in range(heads):
            for p in range(num_blocks // 2):
                perm = torch.randperm(num_blocks, generator=g)
                shared = perm[:n_shared]
                pool = perm[n_shared:]
                pr0 = pool[:topk - n_shared]
                pr1 = pool[topk - n_shared:2 * (topk - n_shared)]
                mask[b, h, 2 * p, torch.cat([shared, pr0])] = True
                mask[b, h, 2 * p + 1, torch.cat([shared, pr1])] = True
    return mask.cuda()


def baseline_meta_from_mask(mask):
    """Ascending idx/num rows, exactly what map_to_index produces."""
    B, H, Nq, Nk = mask.shape
    counts = mask.sum(-1).to(torch.int32).reshape(-1)
    width = max(1, int(counts.max()))
    ids = torch.arange(Nk, device=mask.device, dtype=torch.int32)
    key = torch.where(mask, ids, torch.full_like(ids, Nk)).reshape(-1, Nk)
    idx = key.sort(dim=-1).values[:, :width].contiguous()
    idx = torch.where(torch.arange(width, device=mask.device) < counts[:, None], idx,
                      torch.zeros_like(idx))
    return idx, counts


def run_pair_case(mode, num_blocks, topk, heads, overlap, ragged, seed=0,
                  atol=0.025, lse_atol=0.05):
    from fastvideo_kernel.pair_metadata import build_pair_metadata
    torch.manual_seed(seed)
    S = num_blocks * 64
    shape = (1, heads, S, HEAD_DIM) if vsa.BHSD else (1, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    if ragged:
        vbs = torch.randint(32, 65, (num_blocks, ), dtype=torch.int32, device="cuda")
    else:
        vbs = torch.full((num_blocks, ), 64, dtype=torch.int32, device="cuda")

    mask = make_pair_mask(num_blocks, topk, heads, overlap, seed=seed)
    idx, num = baseline_meta_from_mask(mask)
    base_o, base_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)

    meta = build_pair_metadata(mask, vbs, mode=mode)
    got_o, got_lse = vsa.block_sparse_attn_sm100a_pair(
        q, k, v, meta.q2k_idx, meta.q2k_num, vbs, meta.pair_shared_tiles,
        meta.block_thresholds)

    diff = (got_o.float() - base_o.float()).abs().max().item()
    lse_diff = (got_lse.float() - base_lse.float()).abs().max().item()
    assert diff < atol, f"{mode} out vs baseline kernel: max |diff| = {diff:.5f}"
    assert lse_diff < lse_atol, f"{mode} lse vs baseline kernel: max |diff| = {lse_diff:.5f}"
    return diff, lse_diff


PAIR_MODES = ["union", "shared-private"]


@pytest.mark.parametrize("mode", PAIR_MODES)
@pytest.mark.parametrize("overlap", [1.0, 0.0, 0.6])
def test_pair_overlap_levels(mode, overlap):
    run_pair_case(mode, num_blocks=16, topk=6, heads=4, overlap=overlap, ragged=False)


@pytest.mark.parametrize("mode", PAIR_MODES)
def test_pair_one_shared_block(mode):
    run_pair_case(mode, num_blocks=16, topk=5, heads=4, overlap=1.0 / 5.0, ragged=False)


@pytest.mark.parametrize("mode", PAIR_MODES)
@pytest.mark.parametrize("overlap", [0.0, 0.6, 1.0])
def test_pair_ragged_vbs(mode, overlap):
    run_pair_case(mode, num_blocks=16, topk=6, heads=4, overlap=overlap, ragged=True)


@pytest.mark.parametrize("mode", PAIR_MODES)
@pytest.mark.parametrize("num_blocks", [4, 8, 32])
def test_pair_sequence_lengths(mode, num_blocks):
    run_pair_case(mode, num_blocks=num_blocks, topk=max(2, num_blocks // 4), heads=2,
                  overlap=0.7, ragged=True)


@pytest.mark.parametrize("mode", PAIR_MODES)
def test_pair_realistic_k144(mode):
    """720p-scale geometry (1440 blocks, K=144) vs the baseline kernel."""
    run_pair_case(mode, num_blocks=1440, topk=144, heads=2, overlap=0.715, ragged=False)


@pytest.mark.parametrize("mode", PAIR_MODES)
def test_pair_real_captured_masks(mode):
    """Real Phase-1 captured VSA rows (skips if the capture is not on disk)."""
    sample = "/mnt/nvme/outputs/phase2_pairs/sample_pairs.pt"
    if not os.path.exists(sample):
        pytest.skip("Phase-1 sample pairs not available")
    samples = torch.load(sample, weights_only=False)[:8]
    nk = samples[0]["num_kv_blocks"]
    heads = 2
    nq = 16  # embed 8 real pairs as 16 query rows per head
    mask = torch.zeros(1, heads, nq, nk, dtype=torch.bool)
    for h in range(heads):
        for i, s in enumerate(samples[h * (nq // 4):(h + 1) * (nq // 4)]):
            mask[0, h, 2 * i, s["q0_idx"].long()] = True
            mask[0, h, 2 * i + 1, s["q1_idx"].long()] = True
    mask = mask.cuda()
    torch.manual_seed(3)
    S = nk * 64
    shape = (1, heads, S, HEAD_DIM) if vsa.BHSD else (1, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    # mask has only nq query rows; place them in the first nq blocks, rest empty
    full = torch.zeros(1, heads, nk, nk, dtype=torch.bool, device="cuda")
    full[:, :, :nq] = mask
    vbs = torch.full((nk, ), 64, dtype=torch.int32, device="cuda")
    idx, num = baseline_meta_from_mask(full)
    base_o, base_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    from fastvideo_kernel.pair_metadata import build_pair_metadata
    meta = build_pair_metadata(full, vbs, mode=mode)
    got_o, got_lse = vsa.block_sparse_attn_sm100a_pair(
        q, k, v, meta.q2k_idx, meta.q2k_num, vbs, meta.pair_shared_tiles,
        meta.block_thresholds)
    diff = (got_o.float() - base_o.float()).abs().max().item()
    assert diff < 0.025, f"{mode}: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - base_lse.float()).abs()
    assert lse_diff[torch.isfinite(base_lse)].max().item() < 0.05


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--zero-count-case":
        _zero_count_case_main(int(sys.argv[2]))
