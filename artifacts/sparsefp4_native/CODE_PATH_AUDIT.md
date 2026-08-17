# SparseFP4 Native Composition — C0 Code-Path Audit

Repo `/mnt/nvme/FastVideo` (symlinked at `/home/ec2-user/FastVideo`), branch
`exp/sparsefp4-paper-validation`, HEAD `0a9429863ddcd9e25ed376489c908b234043a6ad`,
tree clean except untracked skill files. Host: 8x NVIDIA B200 (sm_100), CUDA 13.0,
driver 595.91.07. Environment: `/mnt/nvme/scratch/fv-venv` (Python 3.12,
torch 2.12.0+cu130, fastvideo-kernel 0.3.2,
`flash-attn-4 @ hao-ai-lab/flash-attention-fp4 @ 940bf7e5` branch
`fix/cutlass-dsl-4.5`, nvidia-cutlass-dsl 4.5.3, flashinfer 0.6.17) — identical to
studies 1 and 2 (`artifacts/sparsefp4_followup/configs/env.sh`).

This audit builds on `artifacts/sparsefp4/CODEBASE_MAP.md` (still accurate for the
FastVideo side) and adds the kernel-internals answers the native-composition study
needs. New empirical evidence: `logs/c0_probe_fp4_blocksparse_trace.log`
(probe source: `configs/probe_fp4_blocksparse_trace.py`).

---

## 1. Where does dense native NVFP4 QK execute?

- Python entry: `fastvideo/attention/backends/flash_attn.py:334-360`
  (`FlashAttentionImpl._forward_nvfp4`, gated by `FASTVIDEO_NVFP4_FA4=1` /
  `nvfp4_fa4=True`, hard-asserted to sm_100/sm_103).
- Wrapper: `fastvideo/attention/utils/flash_attn_cute.py:368-378`
  `flash_attn_fp4_func(q_fp4, k_fp4, v_bf16, sfq, sfk, ...)` → custom op
  `fastvideo::_flash_attn_cute_fp4_forward` → upstream
  `flash_attn.cute.interface._flash_attn_fwd(..., mSFQ=sfq, mSFK=sfk)`.
- Kernel: `flash_attn/cute/flash_fwd_sm100_fp4.py` `FlashAttentionForwardSm100FP4`,
  selected in `interface.py:817-846` whenever `use_blockscaled_impl`
  (`use_fp4 or mSFQ is not None or force_fp4_impl`, `interface.py:423`).
  QK MMA is a Blackwell block-scaled `tcgen05` MMA consuming packed E2M1 Q/K plus
  per-16-element E4M3 scale factors; P·V runs in BF16 by default.
- Verified live in this environment: probe step 1 (dense FP4, S=512) traces,
  compiles and returns finite BF16 output.

## 2. Which function owns packed FP4 Q/K and scales?

- Quantizer: `fastvideo/attention/backends/flash_attn.py:58-141`
  `_nvfp4_quantize_for_fa4(tensor_4d) -> (fp4, sf)`; flashinfer
  `nvfp4_quantize(..., SfLayout.layout_128x4, do_shuffle=False)`, global scale
  fixed 1.0, `sf_vec_size=16`, output dtype `torch.float4_e2m1fn_x2`, seqlen
  padded to a multiple of 128 (caller slices back).
- Inside the fork, packed Q/K enter as raw pointers (`make_ptr`,
  `interface.py:744-747, 959-966`) with explicit `q_ptr_shape`/`k_ptr_shape`;
  SFQ/SFK are `uint8` tensors in the FA4 MMA layout `(32,4,rest_m,4,rest_k,h,b)`.
- The kernel's `load_K` closure (`flash_fwd_sm100_fp4.py`, "Ki + SFKi") loads the
  K tile *and its SF tile* for a given block index through TMA — the SF fetch is
  keyed by the same `block=` index as the data fetch, which is exactly what a
  sparse gather needs.

## 3. Where does VSA produce selected block indices/masks?

- `fastvideo-kernel/python/fastvideo_kernel/ops.py:108-129` (`video_sparse_attn`):
  pooled per-(4,4,4)-tile means → `scores = q̄k̄ᵀ/√d` → `fused_topk_mask(scores,
  topk)` → bool block map `[B,H,Qtiles,KVtiles]` (64-token tiles).
- Top-k per (batch, head, q-tile) from `VSA_sparsity` via `compute_topk`
  (`fastvideo/attention/backends/video_sparse_attn.py:161-163`).
- Known upstream quirk (study 2): `fused_topk_mask` can return topk+1 blocks on
  exact score ties (~1 row in 7.5k). For frozen-mask arms we snapshot the mask
  tensor byte-for-byte, so C0/D0 see identical masks regardless.

## 4. Which kernel computes VSA fine sparse attention?

On B200 (sm_100), for the deployed `VIDEO_SPARSE_ATTN` backend with 64-token
tiles: `fastvideo_kernel.block_sparse_attn` dispatches to the **Triton** kernel
(`triton_kernels/`); the sm_90 ThunderKittens CUDA extension is absent on
sm_100. The 256-tile entry (`block_sparse_attn_256[_bshd]`) defaults to Triton
("route A": expand 256-mask to 64-blocks) and only uses the FA4 CuTe block-sparse
path when `FASTVIDEO_VSA_CUTEDSL=1`
(`fastvideo_kernel/block_sparse_attn_256.py:46-50`). All compute in input dtype
(BF16). **There is no NVFP4 anywhere in the deployed VSA fine branch.**

## 5. Is there already a block-sparse FA4/CuTe Blackwell path?

Yes — two relevant layers:

1. **Fork infrastructure**: `flash_attn/cute/block_sparsity.py` defines
   `BlockSparseTensorsTorch` (`mask_block_cnt/idx`, `full_block_cnt/idx` in
   count+index form); `interface._flash_attn_fwd(...,
   block_sparse_tensors=...)` threads it to *both* SM100 kernel classes — the
   dense/BF16 `FlashAttentionForwardSm100` **and** the block-scaled
   `FlashAttentionForwardSm100FP4` (`interface.py:919`, `interface.py:986`).
   `flash_fwd_sm100_fp4.py` imports and calls
   `produce_block_sparse_loads_sm100`, `softmax_block_sparse_sm100`,
   `get_total_block_count`, `handle_block_sparse_empty_tile_correction_sm100`
   at 4 call sites (`:2048`, `:2239`, `:2652/2697`, `:3174/3303`), guarded by
   `self.use_block_sparsity` (`:1164`).
2. **FastVideo adapter**: `fastvideo_kernel/block_sparse_attn_cute_fwd.py`
   converts VSA `(block_map, variable_block_sizes)` → `BlockSparseTensorsTorch`
   + a `mask_mod` that trims padded KV tokens, then calls `_flash_attn_fwd`.
   (Note: it passes `tile_mn=`, a kwarg the pinned interface no longer has —
   the in-repo CuTe fastpath is also stale against this pin.)

**However, the composed FP4 x block-sparse path in the pinned fork is broken by
version skew** — proven empirically by the probe:

- `probe step 2` (FP4 + block_sparse_tensors):
  `TypeError: produce_block_sparse_loads_sm100() missing 3 required positional
  arguments: 'q_producer_phase', 'qhead_per_kvhead', and 'q_subtile_factor'`.
  The FP4 kernel's four block-sparse call sites match the *old* helper
  signatures (fork commit `8fecc234`-era, raw mbar-offset protocol), while the
  pinned `block_sparse_utils.py` was upgraded (upstream merge `44d4620b`) to a
  new signature set (`seqlen_info`, `qhead_per_kvhead`, `q_subtile_factor`,
  pipeline objects instead of mbar offsets). The FP4 kernel was never migrated.
- `probe step 3` (BF16 + block_sparse_tensors on the dense SM100 kernel):
  `DSLRuntimeError: expects argument #17 (descale_tensors) ... got
  BlockSparseTensors`. `interface.py` builds one positional `compile_args` list
  for both kernel classes but the dense kernel signature gained a
  `descale_tensors` slot before `blocksparse_tensors`
  (`flash_fwd_sm100.py:380-381`) that the interface never fills.
- Additional interface bug: `get_block_sparse_expected_shapes(...,
  compute_capability)` passes the *compute capability* (10) in the `q_stage`
  parameter (`interface.py:773-775`, `:946-948`), inflating the expected
  m-granularity to `10*m_block_size`.

None of these is a missing kernel; all are repairable Python/DSL-level skew
inside the fork. The block-scaled MMA, SF-aware TMA loads, sparse load-list
iteration, and empty-tile correction all exist in `flash_fwd_sm100_fp4.py`.

## 6. Which dtypes does it support?

- QK ab dtype: `Float4E2M1FN` (NVFP4, sf_vec 16, E4M3 SF) and
  `Float8E4M3FN/E5M2` (MXFP8, sf_vec 32, E8M0 SF) via
  `is_valid_dtypes_and_scale_factor_vec_size` (`interface.py:824-830`).
- PV: BF16 default; optional FP8/NVFP4/MXFP8 PV (mSFV) exists upstream but is
  out of scope — the skill mandates BF16 PV first.
- Dense/BF16 SM100 kernel: fp16/bf16 (+ FP8 with descales).
- Block-sparse tensors: int32 count/index; per-K-tile granularity
  `n_block_size=128`, per-Q-row granularity `q_stage*m_block_size=256` tokens.

## 7. Smallest modification for retained sparse QK tiles to execute natively in NVFP4?

Repair the version skew in the fork (no new kernel needed):

1. Vendor the era-matched block-sparse helpers the FP4 kernel was written
   against (fork commit `8fecc234`: `load_block_list_sm100`,
   `produce_block_sparse_loads_sm100`, `get_total_block_count`,
   `softmax_block_sparse_sm100`, `handle_block_sparse_empty_tile_correction_sm100`)
   as a private module `block_sparse_utils_fp4.py`, and point
   `flash_fwd_sm100_fp4.py`'s imports at it. This preserves the FP4 kernel's raw
   mbar-offset synchronization protocol, which the *new* helpers no longer
   implement.
2. Fix `interface.py`: correct `q_stage` argument to
   `get_block_sparse_expected_shapes` (2 for the SM100 kernels), and insert the
   missing `descale_tensors=None` positional slot for the dense SM100 kernel's
   compile/call args (the FP4 kernel signature has no descale slot —
   `flash_fwd_sm100_fp4.py:370-398` — so its arg list stays as-is).
3. FastVideo side: a thin wrapper that quantizes Q/K with
   `_nvfp4_quantize_for_fa4` and calls `_flash_attn_fwd(q_fp4, k_fp4, v_bf16,
   mSFQ, mSFK, block_sparse_tensors=...)`. Mask conversion from a VSA bool map
   reuses `map_to_index` (Triton) exactly as
   `fastvideo_kernel/block_sparse_attn_cute_fwd.py` does.

Because `load_K`/`load_V` closures in the FP4 kernel fetch "Ki + SFKi" / "Vi +
SFVi" keyed by block index, the sparse load list automatically gathers the
correct per-tile scale factors; unselected K/V tiles (and their SFs) are never
loaded, never MMA'd, never softmaxed — the sparse loop iterates only
`mask_block_idx`/`full_block_idx` entries.

## 8. Does the path preserve online softmax across non-contiguous retained tiles?

Yes, by construction: `softmax_block_sparse_sm100` (old and new era) runs the
kernel's ordinary `softmax_step` (running row-max/row-sum in the softmax warp
group, correction rescale of the O accumulator) once per *retained* block index,
iterating the packed index list — the same online-softmax state machine as the
dense loop, just over a gathered block list. Empty-row correction
(`handle_block_sparse_empty_tile_correction_sm100`) writes zero output and
seeded LSE stats for rows with no retained blocks. Independent-per-tile softmax
does not occur anywhere in this design.

---

## Consequences for the study arms

| Arm | Path | Status at audit time |
|---|---|---|
| A0/P0 dense BF16 | FA4 `flash_attn_func` (or FA2/3) | working (studies 1-2) |
| B0/P1 dense native NVFP4 | `_forward_nvfp4` → `FlashAttentionForwardSm100FP4` | working, re-verified by probe step 1 |
| C0 sparse BF16 (frozen mask) | choose: Triton VSA-64 kernel (deployed) or FA4 dense-SM100 block-sparse (after descale fix) | Triton path working; FA4 BF16-sparse broken (descale slot bug) |
| P2 deployed VSA | `VIDEO_SPARSE_ATTN` backend, Triton fine kernel | working (study 2 ran it) |
| D0/P3 native sparse NVFP4 | `FlashAttentionForwardSm100FP4` + `block_sparse_tensors` | **broken by three localized version-skew bugs; repairable** (this study's C2) |

Mask geometry note: the FA4 sparse granularity is 256 Q-tokens x 128 K-tokens.
A VSA 64-token-tile mask maps onto it exactly as
`block_sparse_attn_cute_fwd.py` already does for BF16 (Q: `any` over 4
consecutive q-tiles; KV: VSA 64-tile mask must first be pooled to 128-token
columns — 2 tiles per column, `any`). For C0/D0 parity, C0's BF16 sparse arm
must consume the *same* 256x128 physical mask as D0, so mask coarsening happens
once, upstream of both arms.
