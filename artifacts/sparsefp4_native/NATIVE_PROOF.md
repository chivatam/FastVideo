# NATIVE_PROOF — D0/P3 native sparse NVFP4 on Blackwell

Claim under proof: the D0/P3 arm executes **native NVFP4 QK arithmetic over
only the retained sparse tiles**, with correct online softmax and BF16 PV —
not a dequantize-then-BF16 path, not dense-then-masked.

Evidence sources: `raw/performance/c3_native_proof.json` (runtime + profiler +
work-scaling receipts, generated on GPU3, B200/sm_100),
`raw/operator/c4_correctness.jsonl`, fork checkout
`/mnt/nvme/scratch/fa4-fork` branch `sparsefp4-native-composition`
(commit `e650c04e`, editable-installed into the study venv).

## 1. Source proof

All paths in the FA4 fork checkout (`flash_attn/cute/`):

| Requirement | Function / line anchor |
|---|---|
| Packed NVFP4 Q/K enter as raw FP4 pointers | `interface.py` `fp4_qk` branch: `make_ptr(Float4E2M1FN, q.data_ptr(), ...)`; the torch tensors are `torch.float4_e2m1fn_x2` produced by `fastvideo/attention/backends/flash_attn.py::_nvfp4_quantize_for_fa4` (flashinfer `nvfp4_quantize`, per-16 E4M3 SF, global scale 1.0) |
| Scale-factor use | `flash_fwd_sm100_fp4.py::load_KV` loads "Ki + SFKi" / "Vi + SFVi" — the SFK TMA fetch is keyed by the same `block=` index as the K-tile fetch; QK MMA is `tcgen05.mma...kind::mxf4nvf4.block_scale.scale_vec::4X` (see GEMM_PTX_FP4 trace lines in `logs/c4_correctness.log`) |
| Sparse-index consumption | `block_sparse_utils_fp4.py::produce_block_sparse_loads_sm100` → `load_block_list_sm100`: iterates **only** `mask_block_idx[b,h,m,:cnt]` and `full_block_idx[b,h,m,:cnt]` entries |
| Unselected K/V tiles skipped | same functions — there is no loop over all `n_block`; the load list, the MMA loop (`mma()` uses `get_total_block_count`) and the softmax loop (`softmax_block_sparse_sm100`) all iterate the retained-index list only |
| Low-precision QK MMA | `flash_fwd_sm100_fp4.py` block-scaled GEMM (`gemm_ptx_partial_fp4`, `sf_vec_size=16`, E4M3 SF); V stays BF16 (no mSFV passed) |
| Online softmax across retained tiles | `block_sparse_utils_fp4.py::softmax_block_sparse_sm100` calls the kernel's ordinary `softmax_step` (running row-max/row-sum + correction rescale) once per retained block — single softmax state machine per Q row, not per-tile softmax |
| No BF16/FP16 Q/K materialization | the call path holds only `qf4/kf4` (`torch.float4_e2m1fn_x2`) and `sfq/sfk` (uint8 E4M3); no dequantized Q/K tensor exists anywhere between quantization and the kernel (runtime receipt confirms dtypes below) |

The composed path was repaired in fork commit `e650c04e` (4 version-skew
bugs; see `CODE_PATH_AUDIT.md` §5/§7). No kernel math was changed — the
vendored helpers are byte-identical to fork commit `8fecc234`'s SM100
helpers except `cute.core.ThrMma -> cute.ThrMma` (dsl 4.5 rename) and
NamedTuple 4-field unpacking.

## 2. Runtime receipt

From `c3_native_proof.json::runtime_receipt` (B=1, S=39936 = Wan 480x832x81
VSA-tiled, H=12, D=128):

- GPU: NVIDIA B200, capability (10, 0) — sm_100.
- `q_fp4`/`k_fp4`: dtype `torch.float4_e2m1fn_x2`, shape `[1, 39936, 12, 64]`
  (packed 2/byte), contiguous.
- `sfq`/`sfk`: dtype uint8 (E4M3 bits), FA4 MMA layout
  `(32, 4, rest_m=312, 4, rest_k=2, 12, 1)` with `stride[3] == 1` —
  per-16-element scale factors.
- `v`: `torch.bfloat16` `[1, 39936, 12, 128]` (PV in BF16; no mSFV).
- Sparse lists: int32 `full_block_cnt/idx` per (batch, head, 256-token Q row)
  over 312 K-blocks of 128 tokens.

## 3. Profiler proof (torch.profiler, CUDA activity)

One dominant CUDA kernel per call; symbols distinguish the FP4 kernel class
from the BF16 one:

| Arm | Kernel symbol (prefix) | Self CUDA time |
|---|---|---|
| D0 sparse FP4 @10% retained | `..._flash_attncuteflash_fwd_sm100_fp4FlashAttentionForwardSm100_...` | 599 us |
| B0 dense FP4 | same FP4 kernel class | 5333 us |
| C0 sparse BF16 @10% | `..._flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_...` (no `_fp4`) | 695 us |
| A0 dense BF16 | same BF16 kernel class | 6468 us |

D0 runs the **FP4** kernel symbol, and its time at 10% retained is ~11% of
the dense FP4 kernel's — the work is actually skipped, not masked.

## 4. Work-scaling sanity (CUDA events, 10 warmup + 50 reps, median)

Wan-shape kernel-only latency (quantization excluded — pre-quantized inputs):

| Retained | D0 sparse FP4 | C0 sparse BF16 |
|---|---|---|
| 100% | 6.010 ms | 7.591 ms |
| 50% | 3.074 ms | 3.815 ms |
| 25% | 1.654 ms | 1.814 ms |
| 10% | 0.800 ms | 0.830 ms |

Anchors: dense-kernel path (no sparse lists) — BF16 7.414 ms, FP4 6.017 ms.

Latency tracks retained fraction near-linearly for both kernels
(6.01→0.80 ms is 7.5x for 10x work reduction). The FP4 QK advantage narrows
as sparsity rises (compute shrinks while per-tile overheads remain), which is
the expected regime behaviour, not evidence of dense work.

## 5. Correctness cross-check (C4)

`raw/operator/c4_correctness.jsonl` (32 cells: S ∈ {1024, 4096, 2048x2,
39936}, retained ∈ {1.0, 0.5, 0.25, 0.10}, 2 seeds):

| Comparison | n | cos (median/min) | rel-L2 (median/max) |
|---|---|---|---|
| D0 vs dequantized-NVFP4 fp32 oracle (identical mask) | 24 | 0.999997 / 0.999997 | 2.32e-3 / 2.35e-3 |
| B0 dense FP4 vs its dequantized oracle | 8 | 0.999997 / 0.999997 | 2.34e-3 / 2.36e-3 |
| C0 sparse BF16 vs fp32 oracle | 24 | 0.999997 / 0.999997 | 2.32e-3 / 2.35e-3 |
| A0 dense BF16 vs fp32 oracle | 8 | 0.999997 / 0.999997 | 2.34e-3 / 2.36e-3 |

The sparse-FP4 kernel's deviation from its own dequantized reference equals
the dense kernels' deviation from theirs (BF16 output-rounding floor). The
sparse path adds **no** numerical error beyond intrinsic NVFP4 QK arithmetic.
All outputs finite.

## 6. Known envelope limitation

A Q row with **zero** retained blocks deadlocks the FP4 kernel's empty-tile
correction when the persistent grid is multi-wave (repro:
`logs/c2_emptyrow_repro.log`; single-wave grids handle it, returning exact
zeros). Deployed VSA guarantees `topk >= 1` retained block per row
(`compute_topk` clamps at 1), so D0/P3 never encounter this case. All study
masks satisfy the envelope; documented here rather than silently avoided.

## Verdict

All seven native-definition conditions of the skill are met with source,
runtime, profiler, work-scaling, and correctness receipts. D0/P3 may be
labeled **native sparse NVFP4**.
