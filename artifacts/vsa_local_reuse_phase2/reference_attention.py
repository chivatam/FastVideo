"""Phase-2 Parts C & I: exact reference execution for the three strategies.

Implements, per real (Q0, Q1) pair over the SELECTED blocks only:

  baseline : current PR#1719 semantics — each Q walks its own sorted KV
             list with online softmax (streamed per 64-token KV block).
  A        : shared/private phased — shared blocks first (both Q), then the
             Q-private blocks; independent online-softmax state per Q.
  B1       : union walk, skipping blocks the Q did not select.
  B2       : union walk, dense: every block computed for both Q; non-member
             logits masked to -inf before softmax.

All variants keep independent per-Q running max / sum / output accumulator,
mirror the kernel's numeric layout in "bf16" mode (bf16 QK inputs, fp32
scores, bf16 P for the PV MMA, fp32 O accumulation, final 1/l scaling), and
never change sparse mask semantics. Nothing here is optimized for speed.
"""

from __future__ import annotations

import torch

HEAD_DIM = 128
BLOCK = 64

# ---------------------------------------------------------------------------
# Metadata construction (CPU/GPU reference; ordering-invariant, sorted output)
# ---------------------------------------------------------------------------


def decompose_shared_private(q0_idx: torch.Tensor,
                             q1_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(shared, q0_private, q1_private), each sorted ascending, no duplicates."""
    a = q0_idx.long().sort().values
    b = q1_idx.long().sort().values
    # membership via searchsorted on the sorted partner (two-pointer equivalent)
    pos = torch.searchsorted(b, a)
    pos = pos.clamp(max=b.numel() - 1)
    in_b = b[pos] == a
    shared = a[in_b]
    p0 = a[~in_b]
    pos2 = torch.searchsorted(a, b).clamp(max=a.numel() - 1)
    in_a = a[pos2] == b
    p1 = b[~in_a]
    return shared, p0, p1


def union_membership(q0_idx: torch.Tensor, q1_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(union sorted ascending, membership bits: 1=Q0, 2=Q1, 3=both)."""
    a = q0_idx.long().sort().values
    b = q1_idx.long().sort().values
    union = torch.unique(torch.cat([a, b]))
    m0 = torch.isin(union, a)
    m1 = torch.isin(union, b)
    return union, (m0.to(torch.uint8) + 2 * m1.to(torch.uint8))


# ---------------------------------------------------------------------------
# Streaming (online-softmax) attention over an ordered list of KV blocks
# ---------------------------------------------------------------------------


class _OnlineState:

    def __init__(self, device: torch.device) -> None:
        self.m = torch.full((BLOCK, 1), float("-inf"), dtype=torch.float32, device=device)
        self.l = torch.zeros(BLOCK, 1, dtype=torch.float32, device=device)
        self.o = torch.zeros(BLOCK, HEAD_DIM, dtype=torch.float32, device=device)

    def step(self,
             q: torch.Tensor,
             k_blk: torch.Tensor,
             v_blk: torch.Tensor,
             scale: float,
             bf16: bool,
             masked_out: bool = False) -> None:
        """One 64x64 QK + PV step. masked_out=True emulates B2's -inf mask."""
        qf = q.bfloat16().float() if bf16 else q.float()
        kf = k_blk.bfloat16().float() if bf16 else k_blk.float()
        s = (qf @ kf.T) * scale
        if masked_out:
            s = torch.full_like(s, float("-inf"))
        new_m = torch.maximum(self.m, s.max(dim=-1, keepdim=True).values)
        new_m = torch.clamp(new_m, min=torch.finfo(torch.float32).min)  # kernel's -FLT_MAX floor
        alpha = torch.exp(self.m - new_m)
        p = torch.exp(s - new_m)
        if bf16:
            p_mma = p.bfloat16()
            pv = p_mma.float() @ v_blk.bfloat16().float()
        else:
            pv = p @ v_blk.float()
        self.o = self.o * alpha + pv
        self.l = self.l * alpha + p.sum(dim=-1, keepdim=True)
        self.m = new_m

    def finalize(self) -> torch.Tensor:
        l_safe = torch.where(self.l > 0, self.l, torch.ones_like(self.l))
        return self.o / l_safe


def _run(q: torch.Tensor, kv_k: torch.Tensor, kv_v: torch.Tensor, block_ids: torch.Tensor, id_to_slot: dict[int, int],
         scale: float, bf16: bool) -> torch.Tensor:
    st = _OnlineState(q.device)
    for bid in block_ids.tolist():
        s = id_to_slot[bid]
        st.step(q, kv_k[s], kv_v[s], scale, bf16)
    return st.finalize()


def baseline_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16=False):
    """Current kernel semantics: each Q walks its own sorted list."""
    o0 = _run(q0, kv_k, kv_v, q0_idx.long().sort().values, id_to_slot, scale, bf16)
    o1 = _run(q1, kv_k, kv_v, q1_idx.long().sort().values, id_to_slot, scale, bf16)
    return o0, o1


def strategy_a_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16=False):
    """Shared phase (both Q), then q0-private, then q1-private."""
    shared, p0, p1 = decompose_shared_private(q0_idx, q1_idx)
    st0, st1 = _OnlineState(q0.device), _OnlineState(q1.device)
    for bid in shared.tolist():
        s = id_to_slot[bid]
        st0.step(q0, kv_k[s], kv_v[s], scale, bf16)
        st1.step(q1, kv_k[s], kv_v[s], scale, bf16)
    for bid in p0.tolist():
        st0.step(q0, kv_k[id_to_slot[bid]], kv_v[id_to_slot[bid]], scale, bf16)
    for bid in p1.tolist():
        st1.step(q1, kv_k[id_to_slot[bid]], kv_v[id_to_slot[bid]], scale, bf16)
    return st0.finalize(), st1.finalize()


def strategy_b_pair(q0, q1, kv_k, kv_v, q0_idx, q1_idx, id_to_slot, scale, bf16=False, dense=False):
    """Union walk. dense=False -> B1 (skip non-members); True -> B2 (mask to -inf)."""
    union, member = union_membership(q0_idx, q1_idx)
    st0, st1 = _OnlineState(q0.device), _OnlineState(q1.device)
    for bid, mem in zip(union.tolist(), member.tolist(), strict=False):
        s = id_to_slot[bid]
        is0, is1 = bool(mem & 1), bool(mem & 2)
        if dense:
            st0.step(q0, kv_k[s], kv_v[s], scale, bf16, masked_out=not is0)
            st1.step(q1, kv_k[s], kv_v[s], scale, bf16, masked_out=not is1)
        else:
            if is0:
                st0.step(q0, kv_k[s], kv_v[s], scale, bf16)
            if is1:
                st1.step(q1, kv_k[s], kv_v[s], scale, bf16)
    return st0.finalize(), st1.finalize()


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def error_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    ref = ref.double()
    test = test.double()
    diff = (ref - test).abs()
    denom = ref.norm().clamp(min=1e-30)
    return {
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_l2": float((ref - test).norm() / denom),
        "cosine_sim": float(torch.nn.functional.cosine_similarity(ref.flatten(), test.flatten(), dim=0)),
    }


def make_pair_tensors(q0_idx: torch.Tensor, q1_idx: torch.Tensor, seed: int, device: str = "cpu") -> tuple:
    """Instantiate ONLY the union blocks needed by the pair (bf16 magnitudes)."""
    union, _ = union_membership(q0_idx, q1_idx)
    g = torch.Generator(device="cpu").manual_seed(seed)
    q0 = torch.randn(BLOCK, HEAD_DIM, generator=g).to(device) * 0.5
    q1 = torch.randn(BLOCK, HEAD_DIM, generator=g).to(device) * 0.5
    kv_k = torch.randn(union.numel(), BLOCK, HEAD_DIM, generator=g).to(device) * 0.5
    kv_v = torch.randn(union.numel(), BLOCK, HEAD_DIM, generator=g).to(device) * 0.5
    id_to_slot = {int(bid): i for i, bid in enumerate(union.tolist())}
    scale = HEAD_DIM**-0.5
    return q0, q1, kv_k, kv_v, id_to_slot, scale
