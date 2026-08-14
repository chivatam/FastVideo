"""Shared numerics for the SparseFP4 study's Phase 2 error decomposition.

Kept separate from ``precision_sparse_attn.py`` so the same code paths can be
exercised by a GPU self-test (``artifacts/sparsefp4/configs/phase2_selftest.py``)
without constructing an ``AttentionImpl`` or loading a model.

Configuration vocabulary (``references/EXPERIMENT_SPEC.md`` 7.1):

======  =========  ==========================  ===============  ===================
config  sparse?    attention compute           mask source      native/simulated
======  =========  ==========================  ===============  ===================
A       no         BF16 (FA4 dense)            n/a              native (reference)
B       no         NVFP4 Q/K + BF16 PV         n/a              native
B_sim   no         NVFP4 Q/K + BF16 PV         n/a              simulated (control)
C       yes        BF16                        bf16             native
D       yes        BF16                        nvfp4            native compute
E       yes        NVFP4 Q/K + BF16 PV         nvfp4            simulated compute
F8      yes        NVFP4 Q/K + BF16 PV         fp8_e4m3         simulated compute
F16     yes        NVFP4 Q/K + BF16 PV         bf16             simulated compute
C_rand  yes        BF16                        bf16 + random    native compute
======  =========  ==========================  ===============  ===================

"NVFP4" always means **NVFP4 Q/K with BF16 PV** — that is what the FA4 kernel
implements (``artifacts/sparsefp4/PHASE0.md`` 6.1). There is no native
sparse-NVFP4 kernel in this repository, so every row whose compute is NVFP4 *and*
sparse is **numerical-only; no native latency claim**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fastvideo.attention.backends.flash_attn import _nvfp4_quantize_for_fa4
from fastvideo.attention.utils.flash_attn_cute import flash_attn_fp4_func
from fastvideo.attention.utils.flash_attn_default import flash_attn_func_compilable

# The block-sparse kernel's tile is 64 tokens on both axes and its grid is
# derived from ``Tq // 64`` (fastvideo-kernel/python/fastvideo_kernel/
# triton_kernels/block_sparse_attn_triton.py:688-703), so the executed mask is
# always 64x64 regardless of the geometry the router scored at.
KERNEL_BLOCK = 64


def pad_to_kernel_block(tensor: torch.Tensor) -> torch.Tensor:
    """Zero-pad a ``[B, S, H, D]`` tensor so ``S`` is a multiple of 64."""
    seq_len = tensor.shape[1]
    padded = math.ceil(seq_len / KERNEL_BLOCK) * KERNEL_BLOCK
    if padded == seq_len:
        return tensor
    return F.pad(tensor, (0, 0, 0, 0, 0, padded - seq_len))


def variable_block_sizes_for(seq_len: int, device: torch.device) -> torch.Tensor:
    """True token count of every 64-token key block, ragged tail included.

    The kernel uses this to mask the zero-padded columns of the final block
    (``block_sparse_attn_triton.py:133-141``), which is why zero padding the
    inputs does not bias the softmax denominator.
    """
    n_blocks = math.ceil(seq_len / KERNEL_BLOCK)
    sizes = torch.full((n_blocks, ), KERNEL_BLOCK, dtype=torch.int32, device=device)
    sizes[-1] = seq_len - (n_blocks - 1) * KERNEL_BLOCK
    return sizes


def assert_query_grid_alignment(seq_len: int, block_q: int) -> None:
    """The router's query blocks must tile the kernel's 64-row grid exactly.

    ``expand_query_axis`` splits each ``block_q``-token query block into
    ``block_q // 64`` kernel rows, so the padded 64-block count has to be a
    multiple of that factor — otherwise the expanded mask has more rows than the
    kernel's grid and the two disagree about which tokens a row covers. True at
    Wan's shape (32760 -> 512 blocks, factor 2) and asserted rather than assumed.
    """
    factor = block_q // KERNEL_BLOCK
    n_blocks = math.ceil(seq_len / KERNEL_BLOCK)
    if n_blocks % factor != 0:
        raise RuntimeError(f"seq_len {seq_len} gives {n_blocks} kernel blocks, which is not a multiple of "
                           f"block_q/{KERNEL_BLOCK} = {factor}")


# --------------------------------------------------------------------------- #
# Block geometry: how tokens are assigned to blocks.
#
# Phases 1 and 2 measured one geometry only — raster-order 128x64 blocks. That is
# *not* what FastVideo's deployed sparse backend uses: VSA partitions tokens into
# (4,4,4) spatio-temporal cubes of 64 tokens with ``block_q == block_k``, and it
# re-orders tokens into tile-contiguous order first, which is a different
# token-to-block assignment rather than merely a different block size
# (``STATUS.md`` trap 3). Two factors are therefore confounded between the two,
# and ``BlockGeometry`` exists so all three arms — ``128x64-raster``,
# ``64x64-raster``, ``64x64-cube`` — run through one code path and separate them.
# --------------------------------------------------------------------------- #

RASTER_TOKEN_ORDER = "raster_frame_y_x"
CUBE_TOKEN_ORDER = "vsa_tile_4x4x4_contiguous"


@dataclass(frozen=True)
class BlockGeometry:
    """A token-to-block assignment plus everything the kernel needs to run it.

    ``order``/``non_pad_index`` are ``None`` for the raster geometries, where the
    padded layout is just a zero-extension of the raster sequence. For the cube
    geometry they are VSA's own tile-partition indices, so the layout handed to
    the kernel is byte-for-byte the layout VSA would hand it.
    """

    name: str
    token_order: str
    block_q: int
    block_k: int
    seq_len: int
    padded_len: int
    query_block_sizes: torch.Tensor
    key_block_sizes: torch.Tensor
    valid: torch.Tensor
    order: torch.Tensor | None = None
    non_pad_index: torch.Tensor | None = None
    untile_index: torch.Tensor | None = None

    @property
    def n_q_blocks(self) -> int:
        return int(self.query_block_sizes.numel())

    @property
    def n_k_blocks(self) -> int:
        return int(self.key_block_sizes.numel())

    @property
    def query_expand(self) -> int:
        return self.block_q // KERNEL_BLOCK

    @property
    def n_pad_slots(self) -> int:
        return self.padded_len - self.seq_len

    def describe(self) -> dict[str, object]:
        sizes = self.key_block_sizes.tolist()
        return {
            "geometry": self.name,
            "token_order": self.token_order,
            "block_q": self.block_q,
            "block_k": self.block_k,
            "n_q_blocks": self.n_q_blocks,
            "n_k_blocks": self.n_k_blocks,
            "padded_seq_len": self.padded_len,
            "n_pad_slots": self.n_pad_slots,
            "key_block_size_min": min(sizes),
            "key_block_size_max": max(sizes),
            "key_block_sizes_distinct": sorted(set(sizes)),
            "all_pad_blocks": sum(1 for size in sizes if size == 0),
        }


def _ragged_sizes(seq_len: int, block: int, device: torch.device) -> torch.Tensor:
    n_blocks = math.ceil(seq_len / block)
    sizes = torch.full((n_blocks, ), block, dtype=torch.int32, device=device)
    sizes[-1] = seq_len - (n_blocks - 1) * block
    return sizes


def raster_geometry(seq_len: int, block_q: int, device: torch.device) -> BlockGeometry:
    """Phase 1/2's diagnostic geometry: contiguous raster-order blocks.

    ``block_k`` is always the kernel's 64 because the executed mask lives on the
    kernel's 64-token key grid; ``block_q`` selects the ``128x64`` (Phase 2) or
    ``64x64`` arm.
    """
    assert_query_grid_alignment(seq_len, block_q)
    padded_len = math.ceil(seq_len / block_q) * block_q
    valid = torch.arange(padded_len, device=device) < seq_len
    return BlockGeometry(
        name=f"{block_q}x{KERNEL_BLOCK}-raster",
        token_order=RASTER_TOKEN_ORDER,
        block_q=block_q,
        block_k=KERNEL_BLOCK,
        seq_len=seq_len,
        padded_len=padded_len,
        query_block_sizes=_ragged_sizes(seq_len, block_q, device),
        key_block_sizes=_ragged_sizes(seq_len, KERNEL_BLOCK, device),
        valid=valid,
    )


def cube_geometry(dit_seq_shape: tuple[int, int, int], device: torch.device) -> BlockGeometry:
    """VSA's deployed geometry: (4,4,4) spatio-temporal cubes, ``block_q == block_k``.

    Built from VSA's own utilities rather than re-derived here, so the tile
    ordering, the ragged boundary tiles and the padded slot map are the ones the
    deployed backend uses (``fastvideo_kernel/vsa_utils.py``, re-exported through
    ``video_sparse_attn``). At Wan's 480x832x81 latent grid this is
    ``(21, 30, 52) -> 6x8x13 = 624`` tiles of at most 64 tokens, i.e. a padded
    length of ``39936`` against ``32760`` real tokens.

    **Padding.** Boundary tiles are short (8/16/32 tokens here), so every one of
    the 624 blocks still holds at least 8 real tokens — there is no all-pad block
    that top-k could select as "important", and ``describe()`` records
    ``all_pad_blocks`` so that stays a checked fact rather than an assumption.
    Within a block, pad slots are excluded from the mean-pooled score by dividing
    by the true tile size (``pool_geometry_blocks``) and from the softmax by the
    kernel's per-block column mask driven by ``key_block_sizes``
    (``block_sparse_attn_triton.py:310-321``). Pad *query* rows are computed and
    then dropped by ``from_block_layout``, so they never enter an error metric.
    """
    from fastvideo.attention.backends.video_sparse_attn import (VSA_TILE_SIZE, construct_variable_block_sizes,
                                                                get_non_pad_index, get_reverse_tile_partition_indices,
                                                                get_tile_partition_indices)
    tile_elems = int(math.prod(VSA_TILE_SIZE))
    if tile_elems != KERNEL_BLOCK:
        raise RuntimeError(f"cube geometry assumes a {KERNEL_BLOCK}-token tile; VSA_TILE_SIZE={VSA_TILE_SIZE}")
    seq_len = int(math.prod(dit_seq_shape))
    num_tiles = tuple(math.ceil(dim / tile) for dim, tile in zip(dit_seq_shape, VSA_TILE_SIZE, strict=True))
    order = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    reverse = get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, VSA_TILE_SIZE)
    non_pad_index = get_non_pad_index(block_sizes, tile_elems)
    padded_len = int(block_sizes.numel()) * tile_elems
    valid = torch.zeros(padded_len, dtype=torch.bool, device=device)
    valid[non_pad_index] = True
    sizes = block_sizes.to(torch.int32)
    return BlockGeometry(
        name=f"{tile_elems}x{tile_elems}-cube",
        token_order=CUBE_TOKEN_ORDER,
        block_q=tile_elems,
        block_k=tile_elems,
        seq_len=seq_len,
        padded_len=padded_len,
        query_block_sizes=sizes,
        key_block_sizes=sizes,
        valid=valid,
        order=order,
        non_pad_index=non_pad_index,
        untile_index=non_pad_index[reverse],
    )


def to_block_layout(x: torch.Tensor, geometry: BlockGeometry) -> torch.Tensor:
    """Move a raster ``[B, S, H, D]`` tensor into the geometry's padded layout.

    Raster geometries zero-extend; the cube geometry gathers tokens in
    tile-contiguous order and scatters them into a zero-filled padded buffer,
    which is exactly ``VideoSparseAttentionImpl.tile``. Pad slots hold **zeros**
    in every case, which is what makes the masked-mean pooling below exact.
    """
    if geometry.order is None:
        pad = geometry.padded_len - x.shape[1]
        return x if pad == 0 else F.pad(x, (0, 0, 0, 0, 0, pad))
    buffer = x.new_zeros((x.shape[0], geometry.padded_len, x.shape[2], x.shape[3]))
    buffer[:, geometry.non_pad_index] = x[:, geometry.order]
    return buffer


def from_block_layout(x: torch.Tensor, geometry: BlockGeometry) -> torch.Tensor:
    """Inverse of :func:`to_block_layout`, dropping every pad slot.

    For the cube geometry this is ``VideoSparseAttentionImpl.untile``'s single
    combined gather, so pad query rows are discarded before any error metric sees
    them.
    """
    if geometry.untile_index is None:
        return x[:, :geometry.seq_len]
    return x[:, geometry.untile_index]


def pool_geometry_blocks(padded: torch.Tensor, block_sizes: torch.Tensor, slot: int) -> torch.Tensor:
    """Masked-mean pool a padded ``[B, S, H, D]`` layout into ``[H, n_blocks, D]``.

    Divides by each block's **true** token count rather than by ``slot``, so pad
    slots (which hold zeros) contribute nothing to the pooled router score. This
    is the same masked mean ``pool_blocks_1d`` applies to the raster ragged tail,
    generalized to the cube geometry's many short boundary tiles.
    """
    batch, _, heads, dim = padded.shape
    n_blocks = int(block_sizes.numel())
    pooled = padded.view(batch, n_blocks, slot, heads, dim).sum(dim=2, dtype=torch.float32)
    return (pooled / block_sizes.to(torch.float32).view(1, -1, 1, 1)).permute(0, 2, 1, 3)[0]


def retained_token_fraction(mask: torch.Tensor, key_block_sizes: torch.Tensor) -> float:
    """Fraction of *tokens* the mask retains, averaged over heads/query blocks.

    Sparsity is defined on the block axis, so at a geometry with variable-size
    blocks (the cube one) an equal block budget is **not** an equal token budget.
    Recording this makes the residual budget mismatch between geometries visible
    instead of implicit.
    """
    sizes = key_block_sizes.to(torch.float64)
    retained = (mask.to(torch.float64) * sizes).sum(dim=-1)
    return float((retained / sizes.sum()).mean().item())


def dense_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Configuration A. Byte-identical to ``FlashAttentionImpl``'s dense branch."""
    return flash_attn_func_compilable(query, key, value, softmax_scale=softmax_scale, causal=False)


def dense_nvfp4_native(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       softmax_scale: float) -> torch.Tensor:
    """Configuration B. Native NVFP4 Q/K + BF16 PV, i.e. ``_forward_nvfp4``."""
    seqlen_q, seqlen_k = query.shape[1], key.shape[1]
    q_fp4, q_sf = _nvfp4_quantize_for_fa4(query)
    k_fp4, k_sf = _nvfp4_quantize_for_fa4(key)
    output = flash_attn_fp4_func(
        q_fp4[:, :seqlen_q],
        k_fp4[:, :seqlen_k],
        value,
        q_sf,
        k_sf,
        softmax_scale=softmax_scale,
        causal=False,
    )
    return output[0] if isinstance(output, tuple) else output


def sparse_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, block_map: torch.Tensor,
                variable_block_sizes: torch.Tensor) -> torch.Tensor:
    """Block-sparse attention over an externally supplied 64x64 mask.

    ``query``/``key``/``value`` are ``[B, S, H, D]`` with ``S`` already padded to
    a multiple of 64; ``block_map`` is bool ``[B, H, S/64, S/64]``. The kernel
    hard-codes ``sm_scale = 1/sqrt(head_dim)``
    (``block_sparse_attn_triton.py:691``), which is exactly Wan's
    ``softmax_scale``; ``assert_kernel_scale_matches`` checks that at run time.

    Returns ``[B, S, H, D]``. The kernel computes in the **input dtype**, so
    passing dequantized-NVFP4 Q/K models NVFP4 Q/K + BF16 PV numerically without
    claiming a native sparse-NVFP4 kernel.
    """
    from fastvideo_kernel import block_sparse_attn
    out, _ = block_sparse_attn(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        block_map,
        variable_block_sizes,
    )
    return out.transpose(1, 2)


def assert_kernel_scale_matches(head_dim: int, softmax_scale: float) -> None:
    kernel_scale = 1.0 / math.sqrt(head_dim)
    if abs(kernel_scale - softmax_scale) > 1e-6:
        raise RuntimeError(f"block_sparse_attn hard-codes sm_scale={kernel_scale} but the layer uses "
                           f"{softmax_scale}; the sparse arms would not be comparable to dense.")


def masked_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, block_map: torch.Tensor,
                     valid: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Trusted fp32 masked-attention reference for the same 64x64 mask.

    Deliberately naive and memory-hungry (one head, one query block at a time)
    so that it shares no code with the kernel under test. Used only by the
    Phase 2 correctness gate. ``valid`` is a bool tensor over the **padded**
    sequence marking real tokens, which covers both the raster ragged tail and
    the cube geometry's short boundary tiles.
    """
    batch, seq_len, heads, dim = query.shape
    assert batch == 1
    n_blocks = seq_len // KERNEL_BLOCK
    out = torch.zeros_like(query, dtype=torch.float32)
    for head in range(heads):
        q_h = query[0, :, head].float()
        k_h = key[0, :, head].float()
        v_h = value[0, :, head].float()
        for q_blk in range(n_blocks):
            rows = slice(q_blk * KERNEL_BLOCK, (q_blk + 1) * KERNEL_BLOCK)
            scores = (q_h[rows] @ k_h.transpose(-1, -2)) * softmax_scale
            keep = block_map[0, head, q_blk].repeat_interleave(KERNEL_BLOCK)
            scores = scores.masked_fill(~(keep & valid), float("-inf"))
            # A query row whose whole key set is masked out has no defined
            # softmax; the kernel leaves such rows at zero, so match that
            # instead of producing NaN.
            finite = torch.isfinite(scores).any(dim=-1, keepdim=True)
            weights = torch.softmax(scores.masked_fill(~finite, 0.0), dim=-1)
            out[0, rows, head] = torch.where(finite, weights, torch.zeros_like(weights)) @ v_h
    return out.to(query.dtype)


def block_scores(pooled_query: torch.Tensor,
                 pooled_key: torch.Tensor,
                 softmax_scale: float,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Block scores in fp64 by default — fp32 is unsafe here.

    Phase 1 found the scores carry magnitude ~5e5 while the discriminative spread
    between competing key blocks is ~14, so an fp32 matmul quantizes the top-k
    margin onto a power-of-two grid and manufactures ~110 exact boundary ties per
    cell. That inflates apparent instability and, decisively, penalizes the FP8
    router 1.6x harder than the NVFP4 router — i.e. it biases the H3 comparison
    against H3 (``artifacts/sparsefp4/STATUS.md`` trap 8). The bf16 null control
    cannot detect it because both sides of that identity land on the same grid.
    """
    return (pooled_query.to(dtype) @ pooled_key.to(dtype).transpose(-1, -2)) * softmax_scale


def topk_block_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k key blocks per query block, ties broken by ascending key index.

    ``scores`` is ``[H, n_q_blocks, n_k_blocks]``. A stable descending sort makes
    the exact ties that low precision *creates* resolve deterministically
    (SPEC 1.3), so a quantization artifact never shows up as run-to-run noise.
    """
    index = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, index[..., :k], True)


def expand_query_axis(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """Restate a ``block_q``-row mask on the kernel's 64-row query grid.

    A 128-token query block is exactly two adjacent 64-token kernel blocks that
    share one selected key set, so Phase 1's 128x64 routing geometry is executed
    verbatim by the 64x64 kernel rather than being re-derived at 64x64.
    """
    return mask if factor == 1 else mask.repeat_interleave(factor, dim=-2)


def random_matched_mask(reference_mask: torch.Tensor, candidate_mask: torch.Tensor,
                        generator: torch.Generator) -> torch.Tensor:
    """Perturb ``reference_mask`` by the same per-query-block swap count as
    ``candidate_mask``, but with blocks chosen uniformly at random.

    This is the contrast control for the decision-margin mechanism: it holds the
    *magnitude* of the mask perturbation fixed and varies only *where* the
    perturbation lands. Equal budget is preserved exactly — ``d`` retained blocks
    are dropped and ``d`` excluded blocks are added per query block.
    """
    swaps = (reference_mask & ~candidate_mask).sum(dim=-1, keepdim=True)
    noise = torch.rand(reference_mask.shape, device=reference_mask.device, dtype=torch.float32, generator=generator)
    infinity = torch.tensor(float("inf"), device=noise.device)

    def ranks(eligible: torch.Tensor) -> torch.Tensor:
        keys = torch.where(eligible, noise, infinity)
        order = keys.argsort(dim=-1)
        positions = torch.arange(keys.shape[-1], device=keys.device).expand_as(keys)
        return torch.empty_like(positions).scatter_(-1, order, positions)

    dropped = reference_mask & (ranks(reference_mask) < swaps)
    added = ~reference_mask & (ranks(~reference_mask) < swaps)
    perturbed = (reference_mask & ~dropped) | added
    if not bool((perturbed.sum(dim=-1) == reference_mask.sum(dim=-1)).all()):
        raise RuntimeError("random_matched_mask changed the retained-block budget")
    return perturbed


def block_attention_mass(query: torch.Tensor, key: torch.Tensor, head: int, query_blocks: list[int],
                         geometry: BlockGeometry, softmax_scale: float) -> torch.Tensor:
    """Mean softmax probability mass each key block receives.

    ``query``/``key`` are in ``geometry``'s padded layout. Returns
    ``[len(query_blocks), n_k_blocks]``: for every sampled query block, the
    row-normalized attention mass of every key block, averaged over that block's
    **valid** query rows. This is the direct measurement of "what does a swapped
    block actually contribute", computed from the exact dense softmax rather than
    inferred from the block score. Pad slots are excluded on both axes, so the
    rows still sum to 1 at every geometry.
    """
    n_k_blocks = geometry.n_k_blocks
    valid = geometry.valid
    block_q = geometry.block_q
    k_h = key[0, :, head].float()
    rows_out = []
    for q_blk in query_blocks:
        start = q_blk * block_q
        stop = start + block_q
        row_valid = valid[start:stop]
        if not bool(row_valid.any()):
            continue
        scores = (query[0, start:stop, head].float()[row_valid] @ k_h.transpose(-1, -2)) * softmax_scale
        weights = torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=-1)
        rows_out.append(weights.view(-1, n_k_blocks, KERNEL_BLOCK).sum(dim=-1).mean(dim=0))
    return torch.stack(rows_out) if rows_out else query.new_zeros((0, n_k_blocks), dtype=torch.float32)


def error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | None]:
    """Per-head rel-L2 / cosine / max-abs against configuration A (SPEC 7.2)."""
    cand = candidate.float()
    ref = reference.float()
    ref_norm = ref.norm()
    if not torch.isfinite(ref_norm) or ref_norm == 0:
        return {"rel_l2": None, "cosine": None, "max_abs": None}
    diff = cand - ref
    cand_norm = cand.norm().clamp(min=1e-30)
    return {
        "rel_l2": float((diff.norm() / ref_norm).item()),
        "cosine": float(((cand * ref).sum() / (cand_norm * ref_norm)).item()),
        "max_abs": float(diff.abs().max().item()),
    }
