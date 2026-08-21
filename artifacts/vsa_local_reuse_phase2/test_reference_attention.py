"""Phase-2 tests: metadata correctness + exact-execution equivalence.

Run:  pytest artifacts/vsa_local_reuse_phase2/ -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_attention import (baseline_pair, decompose_shared_private, error_metrics, make_pair_tensors,
                                 strategy_a_pair, strategy_b_pair, union_membership)

SAMPLE_PAIRS = "/mnt/nvme/outputs/phase2_pairs/sample_pairs.pt"


def _sets(t: torch.Tensor) -> set[int]:
    return set(t.tolist())


def _random_sorted_rows(k: int = 144, nk: int = 1440, overlap: int = 103, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(nk, generator=g)
    shared = perm[:overlap]
    p0 = perm[overlap:k]
    p1 = perm[k:2 * k - overlap]
    q0 = torch.cat([shared, p0]).sort().values
    q1 = torch.cat([shared, p1]).sort().values
    return q0, q1


EDGE_CASES = [
    ("identical", torch.arange(0, 144), torch.arange(0, 144)),
    ("disjoint", torch.arange(0, 144), torch.arange(144, 288)),
    ("one_shared", torch.arange(0, 144), torch.cat([torch.tensor([0]), torch.arange(200, 343)])),
    ("full_overlap_shifted", torch.arange(10, 154), torch.arange(10, 154)),
    ("boundary_ids", torch.cat([torch.tensor([0]),
                                torch.arange(100, 243)]), torch.cat([torch.arange(100, 243),
                                                                     torch.tensor([1439])])),
]


@pytest.mark.parametrize("name,q0,q1", EDGE_CASES, ids=[e[0] for e in EDGE_CASES])
def test_shared_private_reconstructs_exactly(name, q0, q1):
    shared, p0, p1 = decompose_shared_private(q0, q1)
    assert _sets(shared) | _sets(p0) == _sets(q0)
    assert _sets(shared) | _sets(p1) == _sets(q1)
    assert _sets(shared) & _sets(p0) == set()
    assert _sets(shared) & _sets(p1) == set()
    # no duplicates in any output
    for t in (shared, p0, p1):
        assert t.numel() == len(_sets(t))


@pytest.mark.parametrize("name,q0,q1", EDGE_CASES, ids=[e[0] for e in EDGE_CASES])
def test_union_membership_reconstructs_exactly(name, q0, q1):
    union, member = union_membership(q0, q1)
    assert union.numel() == len(_sets(union))
    rec0 = union[(member & 1).bool()]
    rec1 = union[(member & 2).bool()]
    assert _sets(rec0) == _sets(q0)
    assert _sets(rec1) == _sets(q1)
    assert (member > 0).all() and (member <= 3).all()


def test_ordering_invariance():
    q0, q1 = _random_sorted_rows()
    g = torch.Generator().manual_seed(7)
    q0s = q0[torch.randperm(q0.numel(), generator=g)]
    q1s = q1[torch.randperm(q1.numel(), generator=g)]
    a1 = decompose_shared_private(q0, q1)
    a2 = decompose_shared_private(q0s, q1s)
    for t1, t2 in zip(a1, a2, strict=False):
        assert torch.equal(t1, t2)
    u1, m1 = union_membership(q0, q1)
    u2, m2 = union_membership(q0s, q1s)
    assert torch.equal(u1, u2) and torch.equal(m1, m2)


def test_outputs_sorted():
    q0, q1 = _random_sorted_rows(seed=3)
    shared, p0, p1 = decompose_shared_private(q0, q1)
    union, _ = union_membership(q0, q1)
    for t in (shared, p0, p1, union):
        if t.numel() > 1:
            assert (t[1:] > t[:-1]).all()


@pytest.mark.parametrize("name,q0,q1", EDGE_CASES, ids=[e[0] for e in EDGE_CASES])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_strategies_match_baseline_edge_cases(name, q0, q1, bf16):
    q0t, q1t, kv_k, kv_v, id_to_slot, scale = make_pair_tensors(q0, q1, seed=11)
    ob = baseline_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16)
    oa = strategy_a_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16)
    ob1 = strategy_b_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16, dense=False)
    ob2 = strategy_b_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16, dense=True)
    # fp32: block processing order differs (A) so allow tiny fp reassociation;
    # bf16: kernel-realistic rounding, tolerance from Part C measurement scale.
    tol = 1e-5 if not bf16 else 2e-2
    for (t0, t1) in (oa, ob1, ob2):
        m0 = error_metrics(ob[0], t0)
        m1 = error_metrics(ob[1], t1)
        assert m0["max_abs_err"] < tol, (name, m0)
        assert m1["max_abs_err"] < tol, (name, m1)
        assert m0["cosine_sim"] > 1 - 1e-6
        assert m1["cosine_sim"] > 1 - 1e-6


@pytest.mark.skipif(not os.path.exists(SAMPLE_PAIRS), reason="real sample pairs not built")
def test_real_k144_pairs_exact():
    samples = torch.load(SAMPLE_PAIRS, weights_only=False)[:8]
    for i, s in enumerate(samples):
        q0, q1 = s["q0_idx"].long(), s["q1_idx"].long()
        assert q0.numel() == 144 and q1.numel() == 144
        shared, p0, p1 = decompose_shared_private(q0, q1)
        assert shared.numel() + p0.numel() == 144
        assert shared.numel() + p1.numel() == 144
        union, member = union_membership(q0, q1)
        assert union.numel() == 288 - shared.numel()
        q0t, q1t, kv_k, kv_v, id_to_slot, scale = make_pair_tensors(q0, q1, seed=100 + i)
        ob = baseline_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16=False)
        oa = strategy_a_pair(q0t, q1t, kv_k, kv_v, q0, q1, id_to_slot, scale, bf16=False)
        assert error_metrics(ob[0], oa[0])["max_abs_err"] < 1e-5
        assert error_metrics(ob[1], oa[1])["max_abs_err"] < 1e-5


def test_gpu_batched_metadata_matches_cpu_reference():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from metadata_gpu import build_shared_private_batched, build_union_membership_batched
    torch.manual_seed(0)
    rows = [_random_sorted_rows(seed=s) for s in range(32)]
    q0 = torch.stack([r[0] for r in rows]).cuda()
    q1 = torch.stack([r[1] for r in rows]).cuda()
    sh, c_sh, p0, c_p0, p1, c_p1 = build_shared_private_batched(q0, q1)
    un, c_un, mem = build_union_membership_batched(q0, q1)
    for n in range(q0.shape[0]):
        ref_sh, ref_p0, ref_p1 = decompose_shared_private(q0[n].cpu(), q1[n].cpu())
        assert torch.equal(sh[n, :c_sh[n]].cpu(), ref_sh)
        assert torch.equal(p0[n, :c_p0[n]].cpu(), ref_p0)
        assert torch.equal(p1[n, :c_p1[n]].cpu(), ref_p1)
        ref_un, ref_mem = union_membership(q0[n].cpu(), q1[n].cpu())
        assert torch.equal(un[n, :c_un[n]].cpu(), ref_un)
        assert torch.equal(mem[n, :c_un[n]].cpu(), ref_mem)
