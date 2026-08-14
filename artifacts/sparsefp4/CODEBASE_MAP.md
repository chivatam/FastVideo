# SparseFP4 — Phase 0 Codebase Map

Repo: `/home/ec2-user/FastVideo`, branch `exp/sparsefp4-mask-stability`.
Every claim below is cited as `path:LINE`. Sections A–E follow the Phase 0 brief;
the last three sections are the actionable Phase 1 / Phase 3 recommendations and
the SKILL discrepancy audit.

---

## A. Attention backend architecture

### A.1 The four files that define the contract

| File | Role |
|---|---|
| `fastvideo/attention/backends/abstract.py` | `AttentionBackend` / `AttentionImpl` / `AttentionMetadata` / `AttentionMetadataBuilder` ABCs |
| `fastvideo/attention/selector.py` | `get_attn_backend()` resolution + caching + construction scope |
| `fastvideo/attention/layer.py` | `DistributedAttention`, `DistributedAttention_VSA`, `LocalAttention` — the only sanctioned model-facing entry points |
| `fastvideo/platforms/cuda.py` | enum → class-qualname mapping (`CudaPlatformBase.get_attn_backend_cls`) |

Public exports: `fastvideo/attention/__init__.py:3-16`.

### A.2 Registration: a backend needs entries in three places

1. **Enum value** — `fastvideo/platforms/interface.py:13-27` (`AttentionBackendEnum`).
   Current members, in declaration order: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`,
   `SAGE_ATTN_THREE`, `ATTN_QAT_INFER`, `ATTN_QAT_TRAIN`, `VIDEO_SPARSE_ATTN`,
   `VIDEO_SPARSE_ATTN_H3`, `BSA_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `SAGE_SLA_ATTN`,
   `NABLA_ATTN`, `NO_ATTENTION`.
2. **String → class resolution** — `fastvideo/platforms/cuda.py:112-288`. This is an
   `if/elif` chain on `selected_backend`, each arm doing a guarded import and returning a
   dotted qualname string; `fastvideo/attention/selector.py:289-292` then calls
   `current_platform.get_attn_backend_cls(...)` and `resolve_obj_by_qualname`.
   Note: `fastvideo/attention/AGENTS.md:100-108` says "wire string → class resolution in
   `selector.py`" — that is **stale**; the mapping actually lives in `platforms/cuda.py`
   (and `platforms/rocm.py:66`, `platforms/npu.py:69-70`).
3. **Layer support declaration** — a backend is only reachable from a layer whose
   `supported_attention_backends` tuple contains it. The DiT-wide default tuple is
   `fastvideo/configs/models/dits/base.py:22-29`. **`VIDEO_SPARSE_ATTN_H3`, `BSA_ATTN`,
   and `NABLA_ATTN` are absent from that default tuple** — they are opted into per model
   family (e.g. `fastvideo/configs/models/dits/minimax_h3.py:25`,
   `fastvideo/configs/models/dits/kandinsky5.py:19`).
   A requested-but-unsupported backend silently falls back with a warning:
   `fastvideo/attention/selector.py:279-288`.

### A.3 The env-var override, exactly

- Variable name: **`FASTVIDEO_ATTENTION_BACKEND`**, defined once as
  `STR_BACKEND_ENV_VAR` in `fastvideo/utils.py:57`, declared in
  `fastvideo/envs.py:22` and read at `fastvideo/envs.py:210-211`.
- Accepted values are **exactly the `AttentionBackendEnum` member names, case-sensitive
  on the enum lookup but upper-cased for typed call sites**:
  - `fastvideo/attention/selector.py:23-34` (`backend_name_to_enum`) does a raw
    `AttentionBackendEnum[backend_name]` membership test and returns `None` for anything
    unrecognized — i.e. **a typo in the env var is silently ignored** and you fall
    through to automatic selection.
  - `fastvideo/attention/selector.py:37-55` (`coerce_attn_backend`) is the strict path
    used by typed config: it `.strip().upper()`s and **raises** on an unknown name.
- Parse-once adapter: the env var is folded into `FastVideoArgs.attention_backend` in
  `fastvideo/fastvideo_args.py:274-288` (field declared at
  `fastvideo/fastvideo_args.py:138`).
- Resolution precedence — `fastvideo/attention/selector.py:177-235` plus
  `fastvideo/attention/selector.py:266-288`:
  1. explicit `requested=` (the component's own recorded decision),
  2. the construction scope `_component_attention_backend_scope`
     (`fastvideo/attention/selector.py:98-133`),
  3. `FASTVIDEO_ATTENTION_BACKEND`,
  4. the layer-declared `default_backend`,
  5. per-platform automatic selection (`fastvideo/platforms/cuda.py:250-288`).
  `NO_REQUEST` vs `None` semantics: `fastvideo/attention/selector.py:72-85`,
  `fastvideo/attention/selector.py:156-174`.
- Result caching keys on *every* selection input including device index:
  `fastvideo/attention/selector.py:225-235`, `fastvideo/attention/selector.py:238-250`.
  Never mutate the env var mid-process (`fastvideo/attention/AGENTS.md:87`).
- The resolved decision is readable off the loaded component as
  `transformer.config._resolved_attention_backend`
  (`fastvideo/configs/models/base.py:41`, written by
  `fastvideo/attention/selector.py:141-153`).

**Env-var value to select each relevant backend** (all as
`FASTVIDEO_ATTENTION_BACKEND=<value>`):

| Backend | Value | Wan2.1 reachable? |
|---|---|---|
| FlashAttention 2/3/4 dense | `FLASH_ATTN` | yes (in DiT default tuple) |
| torch SDPA | `TORCH_SDPA` | yes |
| SageAttention v1 | `SAGE_ATTN` | yes |
| SageAttention v3 (FP4) | `SAGE_ATTN_THREE` | yes |
| NVFP4 QAT inference | `ATTN_QAT_INFER` | yes |
| NVFP4 QAT training sim | `ATTN_QAT_TRAIN` | yes |
| VSA (Video Sparse Attention) | `VIDEO_SPARSE_ATTN` | yes |
| VSA for MiniMax-H3 | `VIDEO_SPARSE_ATTN_H3` | **no** — not in the DiT default tuple |
| BSA (bidirectional sparse) | `BSA_ATTN` | **no** in the default tuple, but `fastvideo/tests/inference/bsa/test_bsa_inference.py:16-20` runs it on `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, so verify empirically |
| V-MoBA | `VMOBA_ATTN` | yes |
| SLA | `SLA_ATTN` | yes |
| SageSLA (INT8 QK + FP8 V, block-sparse) | `SAGE_SLA_ATTN` | yes |
| NABLA flex-attention | `NABLA_ATTN` | **no** — Kandinsky5-only |

Two additional env gates that are **not** backend selectors but change which kernel runs:
- `FASTVIDEO_FA4=1` — opt-in to `flash_attn.cute` (FA4) for dense paths
  (`fastvideo/envs.py:217-218`, `fastvideo/attention/utils/flash_attn_default.py:43-53`).
- `FASTVIDEO_NVFP4_FA4=1` — turns the **`FLASH_ATTN`** backend into the NVFP4 path
  (`fastvideo/attention/backends/flash_attn.py:226`). Also set by the
  `nvfp4_fa4=True` kwarg (`fastvideo/entrypoints/video_generator.py:207-210`).
- `FASTVIDEO_VSA_CUTEDSL=1` / `FASTVIDEO_VSA_TRITON=1` / `FASTVIDEO_VSA_TK=1` —
  pick the VSA sparse-branch kernel
  (`fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py:37-49`,
  `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py:33-49`).
- `FASTVIDEO_DISABLE_ATTENTION_COMPILE` — default `True`, keeps attention `forward`
  out of the compile graph (`fastvideo/attention/layer.py:18-35`). **Relevant to Phase 1:
  because attention forward is `torch.compiler.disable`d by default, a python-side hook
  there is safe and cheap.**

### A.4 Backend inventory, one line each

| File | Backend name | Purpose |
|---|---|---|
| `fastvideo/attention/backends/flash_attn.py:144-166` | `FLASH_ATTN` | FA2/FA3/FA4 dense; **also hosts the NVFP4 FA4 path** (`_forward_nvfp4`, line 334) |
| `fastvideo/attention/backends/sdpa.py:13-38` | `TORCH_SDPA` | `torch.nn.functional.scaled_dot_product_attention` fallback, no head-size restriction (line 18-25) |
| `fastvideo/attention/backends/video_sparse_attn.py:115-138` | `VIDEO_SPARSE_ATTN` | VSA: 3D-tile top-k block-sparse + pooled compression branch, Wan-tuned |
| `fastvideo/attention/backends/video_sparse_attn_h3.py:109-132` | `VIDEO_SPARSE_ATTN_H3` | VSA for MiniMax-H3's packed mixed-modality sequence; **selection is pure PyTorch and the kernel takes an explicit bool mask** |
| `fastvideo/attention/backends/video_sparse_attn_h3_probe.py:34-101` | (not a backend) | Instrumentation module for the H3 VSA scores; enabled by `FASTVIDEO_H3_VSA_PROBE=<dir>` |
| `fastvideo/attention/backends/bsa_attn.py:545-568` | `BSA_ATTN` | Bidirectional Sparse Attention (arXiv:2509.01085): prunes query tokens *and* selects KV blocks; pure-PyTorch reference |
| `fastvideo/attention/backends/sla.py:118-141` | `SLA_ATTN` | Sparse-Linear Attention (arXiv:2509.24006): block-sparse branch + linear-attention branch blended by a learnable `proj_l` |
| `fastvideo/attention/backends/sla.py:347-371` | `SAGE_SLA_ATTN` | Same as SLA but the sparse branch runs SpargeAttn/SageAttention INT8-QK + FP8-V kernels |
| `fastvideo/attention/backends/nabla.py:63-79` | `NABLA_ATTN` | Kandinsky5 "nabla": 64-token pooled block map thresholded by cumulative mass, OR'd with a precomputed STA window, run through `flex_attention` `BlockMask` |
| `fastvideo/attention/backends/vmoba.py:16-34` | `VMOBA_ATTN` | Video-MoBA: per-layer rotation between temporal / spatial / spatio-temporal chunkings, top-k or threshold gate, `moba_attn_varlen` kernel |
| `fastvideo/attention/backends/sage_attn.py:13-31` | `SAGE_ATTN` | SageAttention v1 wrapper; consumes BSHD directly via `tensor_layout="NHD"` (`sage_attn.py:57-63`) |
| `fastvideo/attention/backends/sage_attn3.py:13-35` | `SAGE_ATTN_THREE` | SageAttention 3 Blackwell FP4 wrapper; transposes to BHSD in `preprocess_qkv` (line 58-70) |
| `fastvideo/attention/backends/attn_qat_infer.py:192-215` | `ATTN_QAT_INFER` | Per-arch NVFP4 inference attention: sm_12x → vendored CUTLASS SageAttn3-FP4; sm_100/sm_103 → FP4 FA4 |
| `fastvideo/attention/backends/attn_qat_train.py:112-135` | `ATTN_QAT_TRAIN` | Triton **fake-quantized** FP4 attention for QAT training (differentiable) |

`utils/`:
- `fastvideo/attention/utils/flash_attn_cute.py` — FA4 (`flash_attn.cute`) custom-op
  wrappers, incl. **`flash_attn_fp4_func`** (line 368-378) and the FP4 forward op
  (line 320-348).
- `fastvideo/attention/utils/flash_attn_default.py:43-70` — the `FASTVIDEO_FA4` gate that
  chooses FA4 vs FA2/FA3 for dense calls.
- `fastvideo/attention/utils/flash_attn_no_pad.py` — masked / varlen entry points as
  compilable custom ops.

### A.5 The ABCs a new research backend must implement

`AttentionBackend` (`fastvideo/attention/backends/abstract.py:31-65`) — four
`@staticmethod @abstractmethod`s:

```python
get_name() -> str                                  # abstract.py:38-41
get_impl_cls() -> type[AttentionImpl]              # abstract.py:43-46
get_metadata_cls() -> type[AttentionMetadata]      # abstract.py:48-51
get_builder_cls() -> type[AttentionMetadataBuilder] # abstract.py:62-65
```
Plus the optional class attr `accept_output_buffer: bool = False`
(`abstract.py:36`) and a de-facto-required
`get_supported_head_sizes() -> list[int]` (not on the ABC, but read by
`fastvideo/platforms/cuda.py:269-273`; VSA returns `[64, 128]` at
`video_sparse_attn.py:119-121`).
Backends with no metadata may `raise NotImplementedError` from
`get_metadata_cls`/`get_builder_cls` — `flash_attn.py:159-165` does exactly that.

`AttentionImpl` (`fastvideo/attention/backends/abstract.py:130-194`):

```python
def __init__(self, num_heads: int, head_size: int, softmax_scale: float,
             causal: bool = False, num_kv_heads: int | None = None,
             prefix: str = "", **extra_impl_args) -> None      # abstract.py:132-143

def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: T) -> torch.Tensor
    # abstract.py:145-161 — called AFTER the SP all-to-all, on the STACKED
    # [(3 or 4)*B, S, H, D] tensor. Default: identity.

def forward(self, query, key, value, attn_metadata: T) -> torch.Tensor
    # abstract.py:186-194 — the only abstract method.

def postprocess_output(self, output: torch.Tensor, attn_metadata: T) -> torch.Tensor
    # abstract.py:163-184 — called BEFORE the return all-to-all. Default: identity.
```

**Tensor layout contract at `AttentionImpl.forward`: `[batch, seq_len, num_heads,
head_dim]` (BSHD)**, documented at `fastvideo/attention/layer.py:96-98` and
`fastvideo/attention/layer.py:299-301`, and re-asserted by every backend
(`video_sparse_attn.py:318-334`, `bsa_attn.py:695-713`, `sla.py:277-292`,
`vmoba.py:143-149`). Backends that need BHSD transpose *themselves* inside
`forward` (`video_sparse_attn.py:331-334`) or in `preprocess_qkv`
(`sage_attn3.py:58-70`). dtype at that point for Wan is **bf16** (compute dtype from
`get_compute_dtype()`, `fastvideo/attention/layer.py:61`,
`fastvideo/configs/pipelines/wan.py:49`); `FLASH_ATTN` defensively casts non-fp16/bf16
inputs (`flash_attn.py:241-257`).

Note the VSA arity exception: `DistributedAttention_VSA` calls
`self.attn_impl.forward(q, k, v, gate_compress, ctx_attn_metadata)` — **five**
positional args (`fastvideo/attention/layer.py:234`), matched by
`video_sparse_attn.py:305-312` and `video_sparse_attn_h3.py:276-283`, both marked
`# type: ignore[override]`.

`AttentionMetadata` (`abstract.py:68-84`) is a dataclass with
`current_timestep: int` and the kw-only `VSA_sparsity: float = 0.0` (line 73) — the
sparsity knob is already on the *base* metadata class, so any new sparse backend gets it
free. `AttentionMetadataBuilder` (`abstract.py:90-109`) needs `__init__`, `prepare()`,
and `build(**kwargs) -> AttentionMetadata`.

Helper worth knowing: `layer_idx_from_prefix(prefix, default=None)`
(`abstract.py:17-28`) parses the transformer-block index out of the layer prefix
(`blocks.(\d+)`, line 14) — this is how vmoba and VSA-H3 tag per-layer behavior, and it
is how Phase 1 should recover the layer index inside an impl
(`video_sparse_attn_h3.py:248`, `vmoba.py:133-134`).

---

## B. Low-precision / NVFP4 / FlashAttention-4

### B.1 Yes — there is a native NVFP4 attention kernel, and there are two of them

**Path 1 — FP4 FlashAttention-4 (`flash_attn.cute`), the one that matters for B200/B300.**

- Kernel entry point: `fastvideo/attention/utils/flash_attn_cute.py:368-378`
  `flash_attn_fp4_func(q, k, v, sfq, sfk, softmax_scale=None, causal=False)`,
  wrapping the custom op `fastvideo::_flash_attn_cute_fp4_forward`
  (`flash_attn_cute.py:320-348`), which calls upstream `_flash_attn_fwd(..., mSFQ=sfq,
  mSFK=sfk)`. `q`/`k` are FP4-packed (`torch.float4_e2m1fn_x2`), **`v` stays BF16**
  (`flash_attn_cute.py:377`).
- Consumer 1 — the `FLASH_ATTN` backend:
  `fastvideo/attention/backends/flash_attn.py:334-360` `_forward_nvfp4`, dispatched at
  `flash_attn.py:318-319`. Gate: `self.nvfp4_fa4` from the `nvfp4_fa4` impl kwarg **or**
  `FASTVIDEO_NVFP4_FA4=1` (`flash_attn.py:226`), with hard asserts on capability
  `(10,0)`/`(10,3)` and importability (`flash_attn.py:227-232`).
- Consumer 2 — `ATTN_QAT_INFER` on datacenter Blackwell:
  `fastvideo/attention/backends/attn_qat_infer.py:276-310` `_forward_fa4_fp4`, chosen by
  `_resolved_kernel() == "fa4_fp4"` (`attn_qat_infer.py:105-120`) for capabilities in
  `_FA4_FP4_CAPABILITIES = {(10,0), (10,3)}` (`attn_qat_infer.py:63`).
- Quantization scheme, stated by the repo itself: **NVFP4 E2M1 Q/K with per-16-element
  E4M3 scale factors, BF16 P/V** — `attn_qat_infer.py:57-62` and the machine-readable
  receipt `attn_qat_infer_receipt()` → `"qk_mode=nvfp4(per-16-e4m3-sf) pv_mode=bf16"`
  (`attn_qat_infer.py:123-140`).
- **Arch requirement: sm_100a / sm_103a** (B200, B300, GB200, GB300) —
  `attn_qat_infer.py:63`, `flash_attn.py:228-229`,
  `docs/inference/optimizations.md:110`.
- **Dependency: an out-of-tree fork, not upstream flash-attn.**
  `docs/inference/optimizations.md:116-138` gives two branch/pin combinations:
  - branch `fp4` + `nvidia-cutlass-dsl==4.4.2` (+ `nvidia-cutlass-dsl-libs-base==4.4.2`),
    `quack-kernels==0.4.1`, `flashinfer-python==0.6.8` — the GB200-validated set, also
    recorded in `attn_qat_infer.py:64-76`;
  - branch `fix/cutlass-dsl-4.5` + `nvidia-cutlass-dsl>=4.5.2` + `apache-tvm-ffi`.
  Both need `CUTE_DSL_ENABLE_TVM_FFI=1` and `FASTVIDEO_FA4=1`
  (`docs/inference/optimizations.md:137-138`, `attn_qat_infer.py:68-71`).
  `attn_qat_infer.py:70-71` warns that **dsl 4.6-era installs fail at CuTe JIT trace**.
  A cutlass-dsl version skew surfaces as a loud warning re-raised as `ImportError`:
  `fastvideo/attention/utils/flash_attn_cute.py:20-31`.
- Also incompatible with FSDP: `use_fsdp_inference=True` invalidates the FP4 pointer path
  (`docs/inference/optimizations.md:160,167`,
  `examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py:43`).

**Path 2 — vendored CUTLASS SageAttention3-FP4 extension, consumer Blackwell only.**

- `fastvideo-kernel/attn_qat_infer/` (CUDA/CUTLASS sources:
  `blackwell/kernel_ws.h`, `blackwell/blockscaled_layout.h`,
  `quantization/fp4_quantization_4d.cu`), exposed as
  `attn_qat_infer.sageattn_blackwell` and imported at
  `fastvideo/attention/backends/attn_qat_infer.py:44-50`.
- Capability set `{(12,0), (12,1)}` = sm_120a / sm_121a
  (`attn_qat_infer.py:53-55`); different quantization scheme from the FA4 path, and
  `ATTN_QAT_TRAIN` simulates *this* one — so sm_100/103 deployment carries a
  measured train/inference mismatch (`attn_qat_infer.py:59-62`).
- Built via `cd fastvideo-kernel && ./build.sh` (`AGENTS.md` build section;
  `attn_qat_infer.py:54` points at `fastvideo-kernel/README.md`).
- Refuses to silently fall back: selecting `ATTN_QAT_INFER` without a usable kernel
  **raises** rather than running bf16 under an FP4 label
  (`fastvideo/platforms/cuda.py:143-154`) — precisely the scientific-integrity behavior
  the SKILL demands (SKILL rule 1/4).

### B.2 Quantizer / dequantizer helpers Phase 1 should reuse

**Reuse this one for Q/K — it is real, not fake, and works on arbitrary 4D tensors:**

```
fastvideo/attention/backends/flash_attn.py:138-141
    _nvfp4_quantize_for_fa4(tensor_4d) -> (fp4_tensor, sf_tensor)
fastvideo/attention/backends/flash_attn.py:58-104
    _nvfp4_quantize_for_fa4_impl(...)   # the real body
```

- **Input**: any `(batch, seqlen, nheads, headdim)` fp16/bf16 tensor. It is *not* tied to
  a linear layer — `flash_attn.py:340-341` calls it directly on post-RoPE `query`/`key`.
- **Granularity**: NVFP4 microscaling with `sf_vec_size = 16`
  (`flash_attn.py:71`) — one E4M3 scale factor per 16 contiguous elements along the
  `nheads*headdim` axis, plus a **global scale fixed to `1.0`**
  (`flash_attn.py:81-82`, `torch.ones(1)`). So on the FA4 path there is *no* per-tensor
  amax rescale: it is pure per-16-element block scaling.
- **Output**: `fp4_tensor` `(batch, seqlen_padded, nheads, headdim//2)` dtype
  `torch.float4_e2m1fn_x2` where `seqlen_padded` is `seqlen` rounded up to a multiple of
  128 (`flash_attn.py:72-76`); `sf_tensor` `uint8` in the FA4 MMA layout
  `(32, 4, rest_m, 4, rest_k, nheads, batch)` with `stride[3] == 1`
  (`flash_attn.py:63-65`, `flash_attn.py:88-103`). Shapes/strides pinned by
  `tests/local_tests/test_nvfp4_fa4.py:61-74`.
- **Caller must slice** `[:, :orig_seqlen]` before handing to FA4
  (`flash_attn.py:64`, `flash_attn.py:346-347`).
- It is registered as a `torch.library.custom_op`
  (`flash_attn.py:114-120`) with a fake kernel that reproduces the *strides*
  (`flash_attn.py:123-135`) — so it is torch.compile-safe.
- Underlying primitive: `flashinfer.quantization.nvfp4_quantize(t2d, global_sf,
  sfLayout=SfLayout.layout_128x4, do_shuffle=False)`, resolved lazily and memoized at
  `flash_attn.py:38-55`.

**Caveat that shapes the whole Phase 1 design: there is no matching dequantizer for
that packed/swizzled format anywhere in `fastvideo/`.** `flash_attn_fp4_func` consumes
the packed tensor directly; nothing unpacks it back to bf16. The two available
dequantize routes are:

1. `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/nvfp4_utils.py:136-239`
   `_compute_dequant(mx_tensor, scale, s_dec, ...)` — a Triton `@triton.jit` device
   function (not a host-callable op), paired with
   `_compute_quant_and_scale(...)` at `nvfp4_utils.py:12-133`. That pair *is* the real
   NVFP4 round-trip: `MXFP_BLOCK_SIZE = 16` (`nvfp4_utils.py:9`), per-16 block max / 6 →
   E4M3 scale (`nvfp4_utils.py:54-58`), optional global scale factor
   `6*448/global_max` (`nvfp4_utils.py:44-49`), and a `two_level_quant_P` SageAttn3
   row-max variant (`nvfp4_utils.py:35-41`). It is a **faithful reference for what the
   FA4 kernel sees**, and the closest thing to a "framework quantize/dequantize" pair for
   routing diagnostics. Because both halves live in the same file with the same block
   geometry, a bf16→NVFP4→bf16 round-trip built on them is *deterministic simulated
   NVFP4* — label it as such per SKILL rule 4.
2. `fastvideo/layers/fp4linear.py:53-79` — flashinfer `nvfp4_quantize` + `mm_fp4`, with
   a **per-tensor** global scale `448*6/maxabs` (`fp4linear.py:18-22`) and `block_size=16`.
   This one is **linear-layer-only** (it needs a weight and calls a GEMM); do not
   contort it onto Q/K.

**FP8 helpers** (needed for the H3 "FP8 router" arm):
- `fastvideo/layers/quantization/absmax_fp8.py`, `fp8_config.py`,
  `fp8_qat_train_config.py`; `fastvideo/layers/fp8linear.py`. All linear-oriented.
- For a tensor-level FP8 cast on Q/K, the simplest defensible route is
  `x.to(torch.float8_e4m3fn).to(torch.bfloat16)` with an explicit per-head or per-tensor
  amax scale — there is precedent in the repo for both dtypes
  (`sla.py:524-526` builds `torch.float8_e4m3fn` V with a per-`(b,h,d)` fp32 scale via
  `fused.scale_fuse_quant_cuda`).
- `nvfp4_utils.py:26-28` shows the same Triton quantizer also emits
  `tl.float8e4nv` / `tl.float8e5` — so one code path covers FP8 E4M3/E5M2 **and** NVFP4,
  which is exactly what a router-precision axis wants.

### B.3 `attn_qat_infer.py` / `attn_qat_train.py` in detail

`attn_qat_infer.py` (inference, **real** quantization):
- Arch resolution is the single source of truth: `_resolved_kernel()`
  (`attn_qat_infer.py:105-120`) → `"cutlass_sm12x"` | `"fa4_fp4"` | `None`.
- `is_attn_qat_infer_available()` (`attn_qat_infer.py:177-189`) gates on the *active
  device's* capability, not just importability — the comment explains that a CUDA-13
  wheel carries the sm_12x extension on any host.
- `AttnQatInferImpl.forward` (`attn_qat_infer.py:242-274`): BSHD in, BSHD out; the
  sm_12x arm transposes to BHSD and back (`:262-264`, `:274`), the FA4 arm does **not**
  transpose (`:283-285`).
- `_resolve_fa4_route_ops()` (`attn_qat_infer.py:146-162`) memoizes
  `(_nvfp4_quantize_for_fa4, flash_attn_fp4_func)` — i.e. it explicitly reuses the
  `flash_attn.py` quantizer. **Phase 1 should do the same.**
- `attn_qat_infer_receipt()` (`:123-140`) is a one-line provenance string
  (`arch=… kernel=… qk_mode=… pv_mode=…`). **Capture this verbatim into
  `artifacts/sparsefp4/env.json` — it is the repo's own answer to "was this really
  native NVFP4?"** Asserted in `tests/local_tests/test_nvfp4_fa4.py:168-175`.

`attn_qat_train.py` (training, **fake/simulated** quantization):
- `attn_qat_train(q_BLHD, k_BLHD, v_BLHD, is_causal=False, sm_scale=None)`
  (`attn_qat_train.py:60-109`) permutes BLHD→BHLD (`:71-73`) and calls
  `fastvideo_kernel.triton_kernels.attn_qat_train.attention` with a long positional
  flag list; the salient flags are `is_qat=True` (`:82`),
  `fake_quant_p_bwd=True` (`:84`), `use_high_prec_o=True` (`:85`),
  `use_global_sf_qkv=False` / `use_global_sf_p=False` (`:89-90`),
  `smooth_k=False` / `smooth_q=False` (`:76`, `:86`).
- `warp_specialize` is disabled on Blackwell (`:80-81`) because Triton 3.7's NVWS pass
  aborts there.
- `get_supported_head_sizes() -> [128]` only (`attn_qat_train.py:117-118`).
- **This is the closest existing "fake NVFP4 attention" and it is differentiable**, but
  it fake-quantizes *inside* a fused kernel, so you cannot read the quantized Q/K out of
  it. For routing diagnostics you need the tensor-level round-trip from B.2, not this.

### B.4 What enables the low-precision path (build / deps)

- `pyproject.toml:41` — `flashinfer-python` is a **base** dependency on Linux (so
  `nvfp4_quantize` is available without an extra).
- `pyproject.toml:125-127` — `flash-attn-4` pinned via `[tool.uv.sources]` to
  `Dao-AILab/flash-attention` rev `82d6441…`, subdirectory `flash_attn/cute`. **Note this
  is upstream FA4, not the `hao-ai-lab/flash-attention-fp4` fork that carries the FP4
  kernel** — the fork must be installed manually per
  `docs/inference/optimizations.md:116-124`.
- `pyproject.toml:189-198` — the only extra that pulls `flash-attn-4` +
  `flashinfer-python` is **`dreamverse`**. There is **no `fp4` / `nvfp4` extra**;
  `pyproject.toml:129-203` lists `swanlab, lint, test, eval-*, eval, dev, prompt-safety,
  prompt-enhancer, streaming, dreamverse, rocm`.
- `fastvideo-kernel/build.sh` + `fastvideo-kernel/CMakeLists.txt` build the CUDA
  extension (`fastvideo_kernel._C`), which supplies `sta_fwd`,
  `block_sparse_fwd`/`block_sparse_bwd` (sm_90 ThunderKittens) —
  `fastvideo-kernel/python/fastvideo_kernel/ops.py:12-16`,
  `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py:15-23`.
  On sm_100 the sm_90 TK ops are absent and VSA falls back to Triton
  (`block_sparse_attn.py:362-381`) — **relevant: VSA's 64-block kernel has no native
  Blackwell path; only the 256-block CuTe path targets sm_100+**
  (`block_sparse_attn_256.py:1-17`).
- Working example to copy: `examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py`
  (`--nvfp4_fa4`, `use_fsdp_inference=not nvfp4_fa4` at line 43, warmup loop at 55-65,
  timed run at 68-82). Sibling: `nvfp4_qat_wan2_1_1_3b.py`, `fp8_wan2_1_1_3b.py`,
  `qad_fp4_ab.py`.
- Hardware-validated reference test: `tests/local_tests/test_nvfp4_fa4.py` — shape/stride
  contract (`:61-74`), accuracy `cos > 0.95` vs BF16 FA4 at the real Wan seqlen
  (`:102-119`), and a CUDA-event kernel-only speedup check (`:121-151`) that
  **excludes quantization overhead** by pre-quantizing outside the timer (`:135-136`).
  Its constants are the exact Phase-1 study shapes: `MODEL_NHEADS = 12`,
  `MODEL_HEADDIM = 128`, `MODEL_SEQLEN = 32760` (`:44-47`).

---

## C. Sparse attention / routing

### C.1 VSA (`VIDEO_SPARSE_ATTN`) — block geometry

**The block is a 3D spatio-temporal tile of latent tokens, not a 1D token window.**

- `VSA_TILE_SIZE = (4, 4, 4)` → **64 tokens per tile**
  (`fastvideo/attention/backends/video_sparse_attn.py:28`; mirrored standalone at
  `fastvideo-kernel/python/fastvideo_kernel/vsa_utils.py:16`).
- The H3 variant uses `VSA_H3_TILE_SIZE = (4, 8, 8)` → **256 tokens per tile**
  (`fastvideo/attention/backends/video_sparse_attn_h3.py:47-48`).
- Only volumes 64 and 256 are supported:
  `vsa_utils.py:17` `_SUPPORTED_VSA_BLOCK_VOLUMES = (64, 256)`, enforced at
  `vsa_utils.py:132-134`.
- **Query block size == key block size == the tile volume.** `video_sparse_attn.py:313`
  computes `block_elements = math.prod(VSA_TILE_SIZE)` once and uses it for both axes;
  the kernel asserts `q_seq_len % block_elements == 0` *and*
  `kv_seq_len % block_elements == 0`
  (`fastvideo-kernel/python/fastvideo_kernel/ops.py:96-106`).
  So the SKILL's default "query block 128 / key block 64" (SKILL 1.2) is **not**
  expressible in VSA — see the Phase 3 recommendation.
- Tokens are reordered from raster order into tile-contiguous order before attention and
  back afterwards: `get_tile_partition_indices` (`video_sparse_attn.py:31-47`),
  `get_reverse_tile_partition_indices` (`:50-56`),
  applied in `preprocess_qkv`/`tile` (`:254-296`) and undone in `postprocess_output`
  (`:298-303`) via the fused `untile_combined_index` (`:222`).
- Boundary tiles are zero-padded to the full volume; the true token count per tile lives
  in `variable_block_sizes` (`construct_variable_block_sizes`, `:59-99`) and the
  non-pad positions in `non_pad_index` (`:102-112`). Pad slots are guaranteed zero, which
  is what makes the pooled mean exact (`video_sparse_attn_h3.py:194-206`).
- Kernel block sizes underneath: the Triton path uses **64-token** KV blocks
  (`fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py:34`), and the FA4
  CuTe BSA path uses **128-token** KV blocks (`block_sparse_attn_256.py:33`); a logical
  256-tile map is expanded 2× (`:52-74`) or 4×4 (`:77-98`) to reach them.

### C.2 VSA — the scoring rule (this is the routing function under study)

All of it is in `fastvideo-kernel/python/fastvideo_kernel/ops.py:108-129`:

```python
q_c = fused_block_mean(q, q_variable_block_sizes, block_elements)   # ops.py:109
k_c = fused_block_mean(k, variable_block_sizes,   block_elements)   # ops.py:110
v_c = fused_block_mean(v, variable_block_sizes,   block_elements)   # ops.py:111
scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)    # ops.py:113
attn   = torch.softmax(scores, dim=-1)                              # ops.py:114
out_c  = torch.matmul(attn, v_c)  # dense "compression" branch      # ops.py:115-117
mask   = fused_topk_mask(scores, topk)                              # ops.py:120
out_s  = block_sparse_attn(_256)(q, k, v, mask, variable_block_sizes)[0]  # ops.py:122-125
return out_c * compress_attn_weight + out_s                         # ops.py:127-129
```

- **Pooling = masked arithmetic mean over the tile's valid tokens, fp32 accumulation**
  (`fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py:190-207`
  `fused_block_mean`; BSHD-native equivalent inline at `ops.py:174-189`).
  This is *identical in form* to the SKILL's controlled diagnostic scorer (SKILL 1.2:
  `q.float().mean(token_axis) @ k.float().mean(token_axis).T`) — the only differences are
  the `1/sqrt(d)` factor and the pad-aware divisor. **Good news for Phase 1: the SKILL's
  simple scorer and VSA's real scorer coincide up to a positive constant, so top-k
  selections are identical.**
- **Selection = per-(batch, head, query-block) top-k over key blocks**, built by a
  randomized-pivot quickselect Triton kernel
  (`fused_compress_topk.py:225-231`, `:298-313` `fused_topk_mask`). Not a threshold, not
  cumulative mass.
- **Sparsity parameterization = `VSA_sparsity` ∈ [0,1) → topk**:
  ```python
  compute_topk(sparsity, num_blocks) = max(1, min(ceil((1-sparsity)*num_blocks), num_blocks))
  ```
  `fastvideo/attention/backends/video_sparse_attn.py:161-163`, applied per-metadata at
  `:166-167`. So `VSA_sparsity=0.9` retains 10% of key blocks — matching the SKILL's
  sparsity/retained-fraction table exactly (SKILL 1.4).
  Plumbing: `FastVideoArgs.VSA_sparsity` (`fastvideo/fastvideo_args.py:171`), CLI flag
  `--VSA_sparsity` (`fastvideo/fastvideo_args.py:644`), threaded into the builder at
  `fastvideo/pipelines/stages/denoising.py:477`, landing on
  `AttentionMetadata.VSA_sparsity` (`fastvideo/attention/backends/abstract.py:73`).
  `VSA_sparsity=0.0` ⇒ `topk == num_blocks` ⇒ mask all-True ⇒ exactly dense.
- **Compression branch**: `out_c` is dense attention over the pooled tile
  representatives, broadcast back to every token in the query tile
  (`ops.py:116-117`), scaled by `compress_attn_weight`. In Wan that weight is the
  learned `to_gate_compress` projection
  (`fastvideo/models/dits/wanvideo.py:458-462`, `:542`, `:552`, `:560`). **This branch is
  a confound for the study: it is a second, always-dense information path whose input is
  the same pooled Q/K the router uses.** Phase 2 should either hold it fixed or pass
  `gate_compress=None`, which `DistributedAttention_VSA` explicitly supports
  (`fastvideo/attention/layer.py:210-213`, `:230-233`).
- **Does the VSA *backend* accept a user-provided block mask? No.**
  `VideoSparseAttentionImpl.forward` (`video_sparse_attn.py:305-342`) passes only
  `variable_block_sizes` and `cur_topk` into `video_sparse_attn(_bshd)`; the mask is
  computed *inside* the kernel wrapper (`ops.py:120`). There is no mask argument on the
  metadata (`video_sparse_attn.py:140-158`) or the builder (`:200-235`).
- **But the layer underneath does.** These are mask-taking public entry points:
  - `fastvideo_kernel.block_sparse_attn(q, k, v, block_map, variable_block_sizes)` —
    `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py:384-393`, bool
    `[B,H,Qb,KVb]` mask, autograd-enabled;
  - `fastvideo_kernel.block_sparse_attn_from_indices(q, k, v, q2k_idx, q2k_num, vbs)` —
    `block_sparse_attn.py:347-381` (the index-native, preferred form);
  - `fastvideo_kernel.block_sparse_attn_256(...)` / `..._256_bshd(...)` —
    `block_sparse_attn_256.py:115-131` / `:134-161`, logical 256-tile bool mask.
  **This is the seam that decouples router precision from compute precision.**

### C.3 VSA-H3 (`video_sparse_attn_h3.py`) — already decoupled in Python

This is the backend that **already separates "compute block scores" from "run sparse
attention"** at the Python level:

```python
q_pooled = _pool_tiles(query, vbs)                                  # h3.py:294
k_pooled = _pool_tiles(key,   vbs)                                  # h3.py:295
scores   = q_pooled @ k_pooled.transpose(-2,-1) / sqrt(head_dim)     # h3.py:296
record_probe(probe_dir, layer_idx, query, key, scores, attn_metadata)# h3.py:297-298
mask     = _build_block_mask(scores, P, V, layer_sparsity, exempt)   # h3.py:304-310
out, _   = block_sparse_attn_256_bshd(query, key, value, mask, vbs)  # h3.py:312
```

- `_pool_tiles` (`h3.py:194-206`): fp32 sum over the 256-token axis / true tile size,
  returns `[B, H, n_tiles, D]`.
- `_build_block_mask` (`h3.py:209-232`): `topk` over the video-tile columns via
  `compute_topk` (imported from VSA, `h3.py:42-44`), plus "exempt" handling that forces
  all prefix (text/cond/audio) tiles True (`h3.py:222-231`).
- Extra scheduling knobs: `dense_layers` forces specific layers dense
  (`h3.py:142`, `:289`), and `exempt` vs `compete` is the ablation axis
  (`h3.py:16-21`, `:139`, `:168`).
- Geometry: 256-token tiles, `(4,8,8)` (`h3.py:47`).
- **Its metadata is not Wan-shaped** — it is built for a packed
  `[text | cond | audio | video]` sequence with `prefix_segments`
  (`h3.py:160-191`, `_h3_tile_geometry` at `:63-106`) and it hard-errors on a
  non-packed sequence (`h3.py:257-260`). And `VIDEO_SPARSE_ATTN_H3` is not in the Wan
  supported-backends tuple (`fastvideo/configs/models/dits/base.py:22-29`).
  So it is the right **template**, not a drop-in.

### C.4 `video_sparse_attn_h3_probe.py` — a probe backend already exists

`fastvideo/attention/backends/video_sparse_attn_h3_probe.py` is exactly the shape of
instrument Phase 1 needs, one model family over.

- **Enable**: `FASTVIDEO_H3_VSA_PROBE=<out_dir>` (`probe.py:30-31` `probe_enabled()`),
  read per-forward at `h3.py:290` and passed to `record_probe` at `h3.py:297-298`.
- **API**: `record_probe(out_dir, layer, query, key, scores, attn_metadata)`
  (`probe.py:34-42`), `@torch.no_grad()`.
- **Intended protocol**: run at **sparsity 0** so the mask selects everything and the
  model follows its exact dense trajectory while you measure what any top-k *would* have
  captured (`probe.py:4-7`). This is the right design for H1: measure routing without
  perturbing the trajectory.
- **What it records**, per `(step, layer, rank)`, from the pooled tile scores:
  - `recall_mean` — pooled softmax mass captured by top-k video tiles, per head, at
    `_FRACS = (0.05, 0.10, 0.125, 0.25, 0.50, 0.75)` (`probe.py:26`, `:48-57`);
  - `prefix_mass` — mass video queries place on non-video tiles (`probe.py:50`);
  - `true_recall@frac` — token-true validation on `_TRUE_ROWS = 128` sampled query rows
    (`probe.py:27`, `:59-85`): exact row softmax, aggregated per tile via
    `index_add_` (`:70-73`), then scored against the ranking the *pooled* proxy would
    pick (`:77-84`) — i.e. it already measures pooling-proxy failure;
  - `perfect_recall@frac` — the selection ceiling from true tile masses (`probe.py:80`).
- **Output**: one `.pt` per `(step, layer, rank)` at
  `probe_step{step:03d}_layer{layer:03d}_r{rank}.pt` (`probe.py:88-100`), tagged with
  `step = int(attn_metadata.current_timestep)` (`probe.py:44`) — the timestep tagging
  problem is already solved here.
- Deterministic row sampling seeded by `step*1000 + layer` (`probe.py:60`).
- **Gap for this study**: it records recall/mass statistics, **not** the two-precision
  mask comparison (Jaccard / top-k overlap between BF16- and NVFP4-derived masks) that
  H1 needs, and it is wired only into the H3 backend. It is a template plus a proven
  output convention, not a ready instrument.

### C.5 The other sparse backends, compared on the axes that matter

| Backend | Block geometry | Scoring rule | Sparsity knob | External mask? | Score/compute split? |
|---|---|---|---|---|---|
| **VSA** `video_sparse_attn.py` | 3D tile `(4,4,4)`=64 (or `(4,8,8)`=256), Q-block == K-block | pooled-mean Q/K, `q̄·k̄ᵀ/√d`, `ops.py:113` | `VSA_sparsity` → `compute_topk`, `:161-163` | **No** at backend; **yes** at kernel (`block_sparse_attn*`) | Split lives in the kernel wrapper (`ops.py`), not the backend |
| **VSA-H3** `video_sparse_attn_h3.py` | 3D tile `(4,8,8)`=256 + segment-pure prefix chunks | pooled-mean, `h3.py:296` | `VSA_sparsity` + `dense_layers` + `exempt` | **Yes** — builds a bool mask in Python and passes it (`h3.py:312`) | **Yes, fully, in Python** |
| **BSA** `bsa_attn.py` | 3D tile `(4,4,4)`=64 (`:44`), `num_blocks` from `:619` | two-stage: query pruning by cosine-similarity-to-block-center, keep least similar (`:88-129`); KV blocks by **cumulative softmax mass ≥ threshold** (`:132-175`) | `bsa_query_keep_ratio=0.5`, `bsa_kv_cumulative_threshold=0.9`, `bsa_min_kv_blocks=4` (`:599-601`) | No | Yes in Python (`:721-732`) but the compute is a **python loop over `(b,h,qb)`** (`:264-305`, `:329-393`) — far too slow for a sweep |
| **SLA** `sla.py` | **1D: `BLKQ=128`, `BLKK=64`** (`:200-201`) — matches the SKILL default exactly | `mean_pool` + **smooth-k** (`k - k.mean(dim=-2)`, `:99`), `q̄ @ k̄ᵀ` with **no `1/√d`** (`:102`) | `topk_ratio` (default 0.1 impl / 0.5 metadata, `:198`, `:148`); `topk = int(topk_ratio*K)` (`:105`) | No | **Yes, cleanly**: `get_block_map(q,k,topk_ratio,BLKQ,BLKK) -> (sparse_map, lut, topk)` at `:78-110` is a standalone function, then `_attention.apply(q,k,v,sparse_map,lut,...)` at `:308` |
| **SageSLA** `sla.py:379-561` | 1D, arch-dependent: `BLKQ,BLKK = 64,128` on sm90 else `128,64` (`:497-500`) | same `get_block_map` (`:503`) | `topk_ratio` | No | Yes — same split, then INT8-QK/FP8-V Sparge kernels (`:516-542`) |
| **NABLA** `nabla.py` | 1D 64-token blocks (`:44-45`, `BLOCK_SIZE=64` at `:60`) | pooled-mean, `softmax(q̄k̄ᵀ/√D)`, then **cumulative-mass binarization** at `thr` (`:48-53`), then `logical_or` with an STA window mask (`:55`) | `P` threshold (default 0.9, `:87`) | **Partially yes** — `sta_mask` is a caller-supplied block mask on the metadata (`:84-85`, `:103`), but it is OR'd in, not substituted | Yes (`nablaT_v2` at `:32-60` is standalone), but compute is `flex_attention` (`:142-147`), no low-precision path |
| **V-MoBA** `vmoba.py` | per-layer rotation of 1D-temporal / 2D-spatial / 3D chunk sizes (`:41-46`, selected at `:151-166`) | inside the `moba_attn_varlen` CUDA kernel (`:183-194`) | `moba_topk` per chunk type, or `moba_threshold=0.25` with `select_mode='threshold'` (`:80-82`) | No | **No** — scoring is fused into the kernel |

**Least-intrusive integration point for a new `PRECISION_SPARSE_ATTN`:** see the
Phase 3 section below. Short answer: subclass/copy `VideoSparseAttentionImpl` and call
`fastvideo_kernel.block_sparse_attn_256_bshd` (or `block_sparse_attn`) directly with a
mask you computed yourself — the pattern `video_sparse_attn_h3.py:292-312` already
demonstrates end-to-end, and it is the only in-tree backend that does.

---

## D. The Wan2.1 DiT attention call site

File: **`fastvideo/models/dits/wanvideo.py`** (note: `fastvideo/models/` is
pre-commit-excluded — `fastvideo/AGENTS.md:61-67`).

### D.1 Two block classes, chosen by env var at construction time

```python
attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND                         # wanvideo.py:628
transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" \
                    else WanTransformerBlock                            # wanvideo.py:629
self.blocks = nn.ModuleList([... prefix=f"{config.prefix}.blocks.{i}" ...])  # :630-641
```

- `WanTransformerBlock` — `wanvideo.py:282-434`, self-attn via
  `DistributedAttention` (`:305-309`).
- `WanTransformerBlock_VSA` — `wanvideo.py:437-582`, self-attn via
  `DistributedAttention_VSA` (`:464-468`) **plus** an extra `to_gate_compress`
  projection (`:458-462`).
- **This is a raw env read inside a model file** (an acknowledged anti-pattern per
  `fastvideo/pipelines/AGENTS.md:89`), and it means **switching to VSA changes the
  module class and the state dict** (`to_gate_compress` weights). Consequence for the
  study: a `PRECISION_SPARSE_ATTN` backend that wants the VSA gate branch must either
  add its name to this comparison or accept `gate_compress=None`.
- Layer prefix format is `Wan.blocks.{i}.attn1.impl` — `wanvideo.py:640` sets
  `prefix=f"{config.prefix}.blocks.{i}"`, `prefix="Wan"` from
  `fastvideo/configs/models/dits/wanvideo.py:114`, then `.attn1` at `wanvideo.py:309`
  and `.impl` appended by `fastvideo/attention/layer.py:69`. `layer_idx_from_prefix`
  (`abstract.py:14-28`) parses the `blocks.(\d+)` group out of it — **so the impl can
  self-identify its layer index with zero plumbing.**

### D.2 Self-attention forward, line by line (dense block)

`WanTransformerBlock.forward` — `fastvideo/models/dits/wanvideo.py:361-434`:

| Step | Line | Detail |
|---|---|---|
| AdaLN modulation | `:393` | `norm_hidden_states = (self.norm1(hidden_states.float()) * (1+scale_msa) + shift_msa).to(orig_dtype)` — FP32 LayerNorm, cast back to bf16 |
| **Q/K/V projections** | `:394-396` | `self.to_q/to_k/to_v(norm_hidden_states)` — three separate `ReplicatedLinear` (declared `:300-302`), each returning `(out, bias)`; **not fused QKV** |
| **Q/K normalization** | `:398-401` | `query = self.norm_q(query)`; `key = self.norm_k(key)`. `norm_q`/`norm_k` are `RMSNorm` declared at `:313-319`. For Wan `qk_norm == "rms_norm_across_heads"` (`fastvideo/configs/models/dits/wanvideo.py:75`), so the RMSNorm is over the **full `dim`** (`:318-319`), i.e. **applied to the flat `[B, S, dim]` tensor BEFORE the head split** |
| Head split | `:403-405` | `query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))` → `[B, S, H, D]`. Same for key, value |
| **Attention call** | `:407-413` | `attn_output, _ = self.attn1(query, key, value, original_seq_len, freqs_cis=freqs_cis)` |
| Output proj | `:414-416` | `attn_output.flatten(2)` → `self.to_out(...)` → `.squeeze(1)` |
| Residual + norm | `:418-421` | `self.self_attn_residual_norm(hidden_states, attn_output, gate_msa, null_shift, null_scale)` |

VSA block: identical, `wanvideo.py:520-582`; Q/K/V at `:539-541`, `to_gate_compress` at
`:542`, `norm_q`/`norm_k` at `:544-547`, head split incl. gate at `:549-552`,
attention call at `:554-561`.

### D.3 **Where RoPE is applied — this is the crux**

RoPE is **not** applied in the model file. It is applied **inside the attention layer,
after the sequence-parallel all-to-all**:

- The block passes `freqs_cis` (a `(cos, sin)` tuple) *into* the attention layer
  (`wanvideo.py:412`, `:559`). `freqs_cis` is built once per forward in the model:
  `get_rotary_pos_embed(...)` at `wanvideo.py:679-687`, with
  `rope_dim_list = [d - 4*(d//6), 2*(d//6), 2*(d//6)]` (`:680`) and
  `rope_theta=10000` (`:686`), cast to fp32 at `:687`.
- `DistributedAttention.forward` — `fastvideo/attention/layer.py:82-164`:
  ```
  :119  qkv = torch.cat([q, k, v], dim=0)                      # [3B, S, H, D]
  :122  qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
  :126-128  trim SP padding to original_seq_len
  :130-132  if freqs_cis is not None:
              qkv[:batch_size*2] = _apply_rotary_emb(qkv[:batch_size*2], cos, sin,
                                                     is_neox_style=False)   # <-- ROPE, Q and K only
  :134  qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)
  :137-143  optionally concat replicated (text) Q/K/V
  :145  q, k, v = qkv.chunk(3, dim=0)
  :147  output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)   # <-- BACKEND ENTRY
  ```
- `DistributedAttention_VSA.forward` — `fastvideo/attention/layer.py:172-245`, same
  shape with a 4-way stack: cat at `:213`, all-to-all `:218`, trim `:221-222`,
  **RoPE at `:224-226`** (again only `[:batch_size*2]`, i.e. Q and K, not V and not the
  gate), `preprocess_qkv` at `:228`, chunk at `:230-233`,
  `self.attn_impl.forward(q, k, v, gate_compress, ctx_attn_metadata)` at `:234`.
- `LocalAttention.forward` — `fastvideo/attention/layer.py:288-318`, RoPE at `:312-315`
  applied to `q` and `k` separately, backend call at `:317`.
- RoPE implementation: `_apply_rotary_emb` — `fastvideo/layers/rotary_embedding.py:105-149`.
  For Wan, `cos.shape[-1] == head_size//2`, so it takes the **GPT-J-style interleaved**
  branch (`is_neox_style=False`): `x1 = x[..., ::2]`, `x2 = x[..., 1::2]`
  (`rotary_embedding.py:143-144`), fp32 math, result `.type_as(x)` back to bf16
  (`:145-146`).

**Therefore: the only place in the entire codebase where post-RMSNorm, post-RoPE,
post-transpose, pre-kernel Q and K both exist as named tensors is
`fastvideo/attention/layer.py:145-147` (dense) and
`fastvideo/attention/layer.py:230-234` (VSA), and equivalently the first line of any
`AttentionImpl.forward`.** Anything captured in `wanvideo.py` is pre-RoPE and pre-SP-layout
— exactly the mistake SKILL 1.1 forbids.

### D.4 Tensor shape and layout at the backend boundary

At `AttentionImpl.forward`, for Wan2.1-T2V-1.3B at the default 480×832×81:

- Layout **`[batch, seq_len, num_heads, head_dim]` (BSHD)**, contiguous, dtype **bf16**.
- `batch = 1` per CFG branch (cond and uncond are two separate forwards —
  `fastvideo/pipelines/stages/denoising.py:513-525` and `:556-...`).
- `num_heads = 12`, `head_dim = 128` (see D.6).
- `seq_len`: latent `(T,H,W) = (21, 60, 104)` → patchified by `patch_size=(1,2,2)`
  (`fastvideo/configs/models/dits/wanvideo.py:64`) → `(21, 30, 52)` →
  **`seq_len = 21*30*52 = 32760`**. Confirmed independently by
  `tests/local_tests/test_nvfp4_fa4.py:47` (`MODEL_SEQLEN = 32760  # 480x832 video, 81 frames`).
  With `sp_size=1` there is no SP padding, so `original_seq_len == seq_len`.
- Under VSA the tensor entering `forward` is the **tiled and zero-padded** variant, since
  `preprocess_qkv` runs first (`video_sparse_attn.py:290-296`): shape
  `[B, n_tiles*64, H, D]` in tile-contiguous order, with pad slots zero. `32760 / 64 =
  511.875`, so tiles pad up: `num_tiles = (ceil(21/4), ceil(30/4), ceil(52/4)) =
  (6, 8, 13) = 624` tiles → padded seq_len `624*64 = 39936`. **This matters: under VSA,
  a Q/K capture at `forward` sees the padded/tiled sequence, not raster order — you must
  either capture before `preprocess_qkv` or carry `variable_block_sizes` /
  `untile_combined_index` alongside the capture.**

### D.5 Self-attention vs cross-attention — different paths

They are **not** the same path, which is convenient: instrumenting self-attention will not
pick up cross-attention traffic.

| | Self-attention | Cross-attention |
|---|---|---|
| Module | `self.attn1`, `DistributedAttention` / `DistributedAttention_VSA` (`wanvideo.py:305-309`, `:464-468`) | `self.attn2`, `WanT2VCrossAttention` (`wanvideo.py:342-347`) or `WanI2VCrossAttention` (`:334-339`) |
| Layer primitive | `DistributedAttention*` | `LocalAttention`, constructed in the `WanSelfAttention` base at `wanvideo.py:169-175` |
| Supported backends | the DiT-wide tuple (`self._supported_attention_backends`, passed at `wanvideo.py:308`, `:467`) | **hard-coded `(FLASH_ATTN, TORCH_SDPA)`** at `wanvideo.py:174-175` |
| RoPE | yes, `layer.py:130-132` / `:224-226` | no — `freqs_cis` is never passed (`wanvideo.py:217`, `:270-272`) |
| Q/K norm | `norm_q`/`norm_k` over full `dim` (`:398-401`) | `norm_q`/`norm_k` over full `dim` (`:200`, `:213`) |
| K/V source | video tokens (`norm_hidden_states`) | `encoder_hidden_states` (T5 text), `seq_len = 512` (`text_len=512`, `fastvideo/configs/models/dits/wanvideo.py:65`) |
| Call site | `wanvideo.py:407-413` / `:554-561` | `wanvideo.py:424` / `:572` |

**Because cross-attention pins itself to `(FLASH_ATTN, TORCH_SDPA)`, a new
`PRECISION_SPARSE_ATTN` enum member will automatically be ignored by `attn2` and engage
only on `attn1`** — i.e. the routing study targets self-attention over video tokens by
construction, with no extra filtering. (The `attn2` selector call falls back with a
warning per `fastvideo/attention/selector.py:279-288`.)

Also note `WanSelfAttention.forward` is a `pass` stub (`wanvideo.py:177-185`) — the class
exists only as a base for the cross-attention variants. Do not instrument it.

### D.6 Wan2.1-T2V-1.3B geometry — layers, heads, head_dim

**Careful: the arch-config defaults in the repo are the 14B model, not the 1.3B.**
`fastvideo/configs/models/dits/wanvideo.py:66-73` declares
`num_attention_heads=40`, `attention_head_dim=128`, `ffn_dim=13824`, `num_layers=40` —
that is Wan2.1-T2V-**14B**. The 1.3B values come from the checkpoint's
`transformer/config.json` via `update_model_arch`
(`fastvideo/configs/models/base.py:60-69`).

The 1.3B numbers, pinned explicitly in-repo by the golden-gate test
`fastvideo/tests/golden_gate/test_wan_t2v.py:20,29-32` (comment: "12 heads x 128,
Wan2.1-T2V-1.3B transformer/config.json") and corroborated by
`tests/local_tests/test_nvfp4_fa4.py:44-46`:

| Quantity | Value | Citation |
|---|---|---|
| `num_layers` (blocks) | **30** | `fastvideo/tests/golden_gate/test_wan_t2v.py:32` |
| `num_attention_heads` | **12** | `fastvideo/tests/golden_gate/test_wan_t2v.py:29`; `tests/local_tests/test_nvfp4_fa4.py:45` |
| `attention_head_dim` (head_dim) | **128** | `fastvideo/tests/golden_gate/test_wan_t2v.py:30`; `tests/local_tests/test_nvfp4_fa4.py:46` |
| `hidden_size` / `inner_dim` | **1536** (= 12 × 128) | `fastvideo/tests/golden_gate/test_wan_t2v.py:20`; computed at `fastvideo/configs/models/dits/wanvideo.py:106` |
| `ffn_dim` | **8960** | `fastvideo/tests/golden_gate/test_wan_t2v.py:31` |
| `patch_size` | `(1, 2, 2)` | `fastvideo/configs/models/dits/wanvideo.py:64` |
| `text_len` | 512 | `fastvideo/configs/models/dits/wanvideo.py:65` |
| `qk_norm` | `"rms_norm_across_heads"` | `fastvideo/configs/models/dits/wanvideo.py:75` |
| `eps` | 1e-6 | `fastvideo/configs/models/dits/wanvideo.py:76` |

**Phase 1 grid size**: 30 layers × 12 heads × 50 timesteps × 2 CFG branches = 36,000
`(layer, head, step, branch)` cells per prompt. At `VSA_sparsity` sweep of 5 values that
is 180k records/prompt — well within JSONL/Parquet, provided you write **metrics only**,
never raw Q/K (SKILL 1.1 explicitly warns about this). With SP disabled you also avoid
the per-rank head-subset bookkeeping the H3 probe needs (`probe.py:98-100`).

An SP caution: `wanvideo.py:606-607` asserts `num_attention_heads % sp_world_size == 0`,
and after the all-to-all each rank holds `12 / sp_size` heads
(`fastvideo/attention/layer.py:124-128`). **Run the routing study at `num_gpus=1` /
`sp_size=1` so head indices are global.**

### D.7 Cleanest hook point for post-RoPE, pre-backend Q/K

There is **no existing extension/hook mechanism that reaches inside `AttentionImpl.forward`**.
The two candidate mechanisms both fall short:

1. `fastvideo/hooks/hooks.py` `ModuleHookManager` / `ForwardHook`
   (`hooks.py:8-39`, `:42-114`) wraps `nn.Module.forward` and offers
   `pre_forward(module, *args, **kwargs)` (`hooks.py:33-35`) — which **can** see the
   arguments. But `AttentionImpl` is **not** an `nn.Module` in general
   (`abstract.py:130` — only `SLAAttentionImpl` and `SageSLAAttentionImpl` are, via
   `nn.Module` mixin at `sla.py:172`, `:379`), so a hook cannot be attached to the impl.
   Attaching to `DistributedAttention` (which *is* an `nn.Module`,
   `layer.py:38`) gives you `pre_forward` args — but those are the **pre-RoPE,
   pre-all-to-all** tensors. Wrong side of the boundary.
2. `fastvideo/hooks/activation_trace.py` `attach_activation_trace`
   (`:188-216`) registers `ActivationStatHook` whose only callback is
   `post_forward(module, output)` (`:137-153`) — outputs only, never inputs. It also
   only walks `model.named_modules()` (`:203`), so again it cannot see inside the impl.

**So: no existing mechanism suffices, and the minimal edit is a new
`AttentionImpl` subclass, not a monkeypatch.** Details and ranking in the
"Recommended Phase 1 instrumentation plan" below.

---

## E. Existing instrumentation and tooling to reuse

### E.1 The FastVideo activation trace (the utility the `add-model-08-trace` skill means)

- Implementation: **`fastvideo/hooks/activation_trace.py`**.
- Docs: **`docs/contributing/activation_trace.md`** (lifecycle explained at
  `activation_trace.md:208-230`).
- Hook substrate: `fastvideo/hooks/hooks.py` — `ForwardHook`
  (`hooks.py:8-39`) and `ModuleHookManager` (`hooks.py:42-114`, which replaces
  `module.forward` with a wrapper at `hooks.py:61-70`).
- API:
  ```python
  attach_activation_trace(model: nn.Module | None) -> ActivationTraceManager | None
      # activation_trace.py:188-216 ; returns None when the env gate is off
  detach_activation_trace(mgr) -> None            # activation_trace.py:219-221
  trace_step(step_idx: int)                       # activation_trace.py:50-58 (contextmanager)
  current_step_idx() -> int | None                # activation_trace.py:46-47
  ```
- Env gates (declared `fastvideo/envs.py:36-40`, parsed `:263-276`):
  `FASTVIDEO_TRACE_ACTIVATIONS` (bool gate), `FASTVIDEO_TRACE_LAYERS` (regex matched
  against `named_modules()` names, `activation_trace.py:193-194`, `:203-205`),
  `FASTVIDEO_TRACE_STATS` (default `"abs_mean,sum"`; available stats
  `abs_mean, sum, min, max, mean, std, shape, dtype` at `activation_trace.py:61-70`),
  `FASTVIDEO_TRACE_OUTPUT` (default `/tmp/fv_trace_<pid>.jsonl`; `<pid>` substituted at
  `:88-89`), `FASTVIDEO_TRACE_STEPS` (comma list of step indices, `:92-95`).
- Sink: `JsonlSink` (`activation_trace.py:98-116`), thread-safe line-buffered append.
  Record schema: `{module, tensor, step, <stats...>}` (`:141-152`).
- Wiring: attached automatically for every pipeline at
  `fastvideo/pipelines/composed_pipeline_base.py:253`
  (`self._trace_mgr = attach_activation_trace(self.modules.get("transformer"))`),
  detached at `composed_pipeline_base.py:532`.
- **`trace_step(i)` is NOT wired into the generic denoising stage.** It is wrapped only
  in the model-specific loops: `fastvideo/pipelines/basic/zimage/stages.py:253`,
  `fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py:177`,
  `fastvideo/pipelines/basic/magi_human/stages/denoising.py:167`,
  `fastvideo/pipelines/basic/magi_human/stages/sr_denoising.py:113`.
  `fastvideo/pipelines/stages/denoising.py` — the stage Wan uses — does **not** call it,
  so `current_step_idx()` returns `None` on a Wan run. **Use
  `attn_metadata.current_timestep` instead (see E.5).**
- Tests to imitate: `fastvideo/tests/hooks/test_activation_trace.py`.
- **Verdict for this study: reusable as a prior-art pattern and for JSONL conventions,
  but not as the Q/K capture mechanism** (outputs-only, module-level; see D.7).

### E.2 Benchmark harnesses and peak-memory measurement

- **End-to-end + component + peak memory**:
  `fastvideo/tests/performance/test_inference_performance.py`.
  `_run_once(...)` returns `(elapsed_s, peak_memory_mb, component_times)`
  (`:226-234`); thresholds incl. `max_peak_memory_mb` (`:169`); aggregation over
  repetitions with `max(peak_memories)` (`:469`, `:581`) and derived
  `throughput_fps = num_frames / avg_time` (`:470-473`); the emitted record includes
  `attention_backend` and `flash_attention_4_enabled`
  (`:386-387`) — **copy those two provenance fields into every SparseFP4 measurement**.
  Entry test: `test_inference_performance(cfg)` at `:631-632`.
- **Run-identity / config-hash guard**: `fastvideo/tests/performance/identity.py` — the
  env keys that define a comparable run are listed at `identity.py:32` (includes
  `FASTVIDEO_ATTENTION_BACKEND`) and read at `:141`. Companion:
  `test_inference_performance_identity.py` (e.g. `:175-176` sets
  `SAGE_ATTN` + `FASTVIDEO_FA4=1`; `:223` sets `FASTVIDEO_FA4=0`).
  **This is the mechanism that keeps SKILL rule 6 (identical settings for pairwise
  comparisons) honest — reuse it rather than inventing a run key.**
- **Platform-level peak memory helper**:
  `CudaPlatformBase.get_current_memory_usage` —
  `fastvideo/platforms/cuda.py:99-102` (`reset_peak_memory_stats` then
  `max_memory_allocated`).
- **Attention-kernel microbenchmark pattern (CUDA-event, warmup, pre-quantized to exclude
  quant overhead)**: `tests/local_tests/test_nvfp4_fa4.py:25-37` `_cuda_timer` and
  `:121-151`. This is the right template for the SKILL's "attention-kernel latency"
  metric.
- **Kernel-level VSA benchmarks**: `fastvideo-kernel/benchmarks/bench_vsa.py`,
  `bench_fused_compress_topk.py`, `benchmark_attn_qat_train.py`.
- **Serving-throughput CLI** (not what this study needs, but exists):
  `fastvideo bench` → `fastvideo/entrypoints/cli/bench.py:19-88`
  (`--dataset vbench`, `--num-prompts`, `:57-69`) and
  `fastvideo/entrypoints/cli/bench_serving.py`.
- **Torch profiler env gates**: `FASTVIDEO_TORCH_PROFILER_DIR` and friends,
  `fastvideo/envs.py:30-35`, `:226-260`; docs `docs/contributing/profiling.md`.
- Repo-wide benchmark doc: `docs/contributing/performance_benchmarks.md`.
- Baseline re-seeding procedure (if a measured shift is intentional):
  `.agents/skills/reseed-performance-baseline/SKILL.md`.

### E.3 VBench / video-quality evaluation — fully integrated

- Framework: `fastvideo/eval/` — `evaluator.py`, `worker.py`, `registry.py`,
  `metrics/`, `datasets/`, `io/`. README at `fastvideo/eval/README.md`; contributor doc
  `docs/contributing/eval-metrics.md`.
- **16 VBench dimensions**, each its own module under
  `fastvideo/eval/metrics/vbench/<dimension>/metric.py`: `subject_consistency`,
  `background_consistency`, `temporal_flickering`, `motion_smoothness`,
  `dynamic_degree`, `aesthetic_quality`, `imaging_quality`, `object_class`,
  `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`,
  `appearance_style`, `temporal_style`, `overall_consistency`. Shared helpers
  `fastvideo/eval/metrics/vbench/_utils.py`, `_grit_helper.py`.
- **Prompt corpus**: `fastvideo/eval/datasets/vbench.py` — `VBenchPromptDataset`
  (`:45-121`), "946 prompts across 16 dimensions" (`:58`), filterable by dimension
  (`:46-58`), yields `{"prompt": entry["prompt_en"], ...}` (`:121`). VBench's official
  protocol of 5 generations per prompt is encoded at `:16-18`. **This is the
  paper-scale set the SKILL asks for (SKILL Phase 5) — no external download script
  needed.**
- Reference scores for regression: `fastvideo/tests/eval/reference_scores/vbench.*.json`
  (16 files), checked by `fastvideo/tests/eval/test_metric_score_regression.py`.
- CLI: `fastvideo eval` → `fastvideo/entrypoints/cli/eval.py`.
- Runnable examples: `examples/inference/eval/bench_vbench.py`,
  `eval_ltx2_vbench.py`, `score_folder.py`, `score_video.py`, `all_metrics_demo.py`.
- Install extras: `pyproject.toml:160` `eval-vbench = ["openai-clip", "pyiqa",
  "easydict"]`, rolled up by `eval` at `:164-168`. Note the comment at
  `pyproject.toml:152-153`: detectron2 is needed for the color/object_class family and is
  not auto-installed.
- Training-time hookup (useful for the "quality recovery" arm):
  `fastvideo/train/callbacks/validation.py`.

### E.4 Prompt assets

- **`assets/prompt.txt` — 8 T2V prompts, one per line** (`wc -l` = 7 with no trailing
  newline; the 8 prompts include the raccoon/sunflowers and lion/savanna prompts used
  throughout the examples). **This is the correct existing stand-in for the SKILL's
  `assets/prompts.txt` development set** (SKILL Phase 5 asks for ~10 prompts).
- `assets/prompts/mixkit_i2v.jsonl` — 110 lines, **I2V** prompts (image-conditioned), not
  usable for a T2V routing study.
- `assets/eval/worldmodel_synthetic_flow_calibration.json` — unrelated world-model
  calibration data.
- Batch-prompt plumbing already exists: `FastVideoArgs.prompt_txt`
  (`fastvideo/fastvideo_args.py:184`).
- The single prompt used by essentially every Wan example (good "prompt_id=0" default):
  `examples/inference/basic/basic.py:29-31` and
  `examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py:51-53` (raccoon/sunflowers);
  `examples/inference/optimizations/attention_example.py:26-29` uses the Will-Smith
  prompt with `seed=1024`.
- For paper scale, prefer `VBenchPromptDataset` (E.3) over the flat text file.

### E.5 How the timestep index reaches a hook

Three mechanisms, in decreasing order of usefulness here:

1. **`attn_metadata.current_timestep`** — declared on the metadata base
   (`fastvideo/attention/backends/abstract.py:72`), set by the builder from the loop
   counter `i`: `fastvideo/pipelines/stages/denoising.py:472-473`
   (`current_timestep=i`), also `:1353-1354`, `:1358`. Read inside an impl via
   `attn_metadata.current_timestep`. **This is exactly what the H3 probe uses**
   (`video_sparse_attn_h3_probe.py:44`: `step = int(attn_metadata.current_timestep)`).
   *Caveat*: it is the **loop index** (0..num_inference_steps-1), not the sigma/t value.
   **Caveat 2**: it is only non-`None` when a metadata builder ran — and
   `fastvideo/pipelines/stages/denoising.py:499-500` sets `attn_metadata = None` for any
   backend that is neither VSA nor VMoBA. A new sparse backend must be added to that
   `if/elif` chain to receive metadata at all (see Phase 3 notes).
2. **`ForwardContext.current_timestep`** — `fastvideo/forward_context.py:31-40`, set by
   `set_forward_context(current_timestep=i, attn_metadata=..., forward_batch=batch)`
   (`forward_context.py:53-67`), entered at
   `fastvideo/pipelines/stages/denoising.py:506-511` (cond) and `:551-555` (uncond).
   Retrieved anywhere with `get_forward_context()` (`forward_context.py:45-49`).
   **This works even when `attn_metadata is None`**, and it also carries
   `forward_batch`, which exposes `batch.is_cfg_negative`
   (set `False` at `denoising.py:505`, `True` at `:550`) — i.e. **the CFG branch tag comes
   free from the same object**. The attention layers already read it
   (`fastvideo/attention/layer.py:115-116`, `:206-207`, `:309-310`).
   **Recommended: use `get_forward_context()` for `(step, cfg_branch)` tagging.**
3. `trace_step(i)` / `current_step_idx()` (`fastvideo/hooks/activation_trace.py:50-58`,
   `:46-47`) — thread-local, but **not wired into the Wan denoising stage** (see E.1), so
   it returns `None` here.

Sparsity per-step is also already schedulable: `ForwardBatch.VSA_sparsity`
(`fastvideo/pipelines/pipeline_batch_info.py:258`) and the H3 pattern of forcing early
steps dense (`fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py:159`).

### E.6 Wan2.1 T2V pipeline / stage architecture and the exact default inference args

**Pipeline**: `fastvideo/pipelines/basic/wan/wan_pipeline.py` (siblings:
`wan_dmd_pipeline.py`, `wan_i2v_pipeline.py`, `wan_causal_pipeline.py`,
`wan_v2v_pipeline.py`, `lucy_edit_pipeline.py`). Base:
`fastvideo/pipelines/composed_pipeline_base.py`. Stages are composed from
`fastvideo/pipelines/stages/` — the relevant one is
**`fastvideo/pipelines/stages/denoising.py`** (`DenoisingStage`; attention backend
resolved at construction, `denoising.py:65`).

**Registry**: `fastvideo/registry.py:927-937` maps
`"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"` → `pipeline_config_cls=WanT2V480PConfig`,
`workload_types=(WorkloadType.T2V,)`, `model_family="wan"`,
`default_preset="wan_t2v_1_3b"`.

**Pipeline config** — `fastvideo/configs/pipelines/wan.py:28-70` `WanT2V480PConfig`:
`flow_shift=3.0` (`:41`), `precision="bf16"` (`:49`), `vae_precision="fp32"` (`:54`),
`vae_decode_precision="bf16"` (`:59`), `text_encoder_precisions=("fp32",)` (`:60`),
single T5 text encoder (`:44-46`), `vae_tiling=False`/`vae_sp=False` (`:37-38`).
Frozen JSON mirror: `fastvideo/configs/wan_1.3B_t2v_pipeline.json`
(`embedded_cfg_scale: 6.0`, `flow_shift: 3`, `dit_cpu_offload: true`).

**Default inference args** — preset `wan_t2v_1_3b`,
`fastvideo/pipelines/basic/wan/presets.py:48-63`:

| Arg | Value | Citation |
|---|---|---|
| `height` | **480** | `presets.py:55` |
| `width` | **832** | `presets.py:56` |
| `num_frames` | **81** | `presets.py:57` |
| `fps` | 16 | `presets.py:58` |
| `guidance_scale` | **3.0** | `presets.py:59` |
| `num_inference_steps` | **50** | `presets.py:60` |
| `negative_prompt` | `_NEGATIVE_PROMPT_EN` | `presets.py:61` |
| `flow_shift` | 3.0 | `fastvideo/configs/pipelines/wan.py:41` |
| `seed` | **1024** (SamplingParam default; presets do not override) | `fastvideo/api/sampling_param.py:82` |

These match the SKILL's stated baseline (480×832, 81 frames) exactly, so **no override is
needed and SKILL "preserve normal inference defaults" is satisfied by doing nothing**.
Note `SamplingParam` class-level defaults are the *generic* ones (720×1280, 125 frames,
50 steps, `guidance_scale=1.0` — `sampling_param.py:85-95`); the preset is what supplies
the Wan values, so always resolve through `SamplingParam.from_pretrained(model_path)`
(`sampling_param.py:212`) or the preset rather than instantiating `SamplingParam()`.

**How inference is launched** from `examples/inference/`:
- Minimal: `examples/inference/basic/basic.py:13-32` —
  `VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1,
  use_fsdp_inference=False, dit_cpu_offload=False, vae_cpu_offload=False,
  text_encoder_cpu_offload=True, pin_cpu_memory=True)` then
  `generator.generate_video(prompt, output_path=..., save_video=True)`.
- Backend-selecting: `examples/inference/optimizations/attention_example.py:9`
  (`os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"` **before**
  `from_pretrained`, with timing around load and generate).
- NVFP4: `examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py:39-49`
  (`nvfp4_fa4=args.nvfp4_fa4`, `use_fsdp_inference=not args.nvfp4_fa4`), warmup at
  `:55-65`, timed generate at `:68-82`.
- YAML/CLI: `scripts/inference/inference_wan.yaml`, `fastvideo generate` →
  `fastvideo/entrypoints/cli/generate.py`.
- Public API entry: `VideoGenerator.from_pretrained`
  (`fastvideo/entrypoints/video_generator.py:186-241`) → `from_config` (`:243-254`);
  `nvfp4_fa4=True` is intercepted at `:207-210` and sets both
  `FASTVIDEO_NVFP4_FA4=1` and `CUTE_DSL_ENABLE_TVM_FFI=1`.

**Environment capture**: `collect_env.py` at the repo root
(`collect_env.py:1-30`, imports `fastvideo.envs.environment_variables` at `:20`) — run as
`python collect_env.py`. See the Discrepancies section: this is the real equivalent of the
SKILL's `scripts/check_env.py`.

---

## Recommended Phase 1 instrumentation plan

**Requirement restated**: capture (or compute metrics on) the *exact* Q and K that reach
the attention kernel — post-`norm_q`/`norm_k` (`wanvideo.py:398-401`), post-RoPE
(`layer.py:130-132`), post-SP-trim, in BSHD layout. Per D.7, no existing hook mechanism
reaches that point, so this is a ranked list of edits.

### Rank 1 — **New research `AttentionImpl` behind a new backend enum (recommended)**

Add `PRECISION_SPARSE_ATTN` (or `ROUTING_PROBE_ATTN` for Phase 1 only) to
`AttentionBackendEnum` (`fastvideo/platforms/interface.py:13-27`), an arm in
`CudaPlatformBase.get_attn_backend_cls` (`fastvideo/platforms/cuda.py:112-248`), the enum
in the DiT supported tuple (`fastvideo/configs/models/dits/base.py:22-29`), and a new
`fastvideo/attention/backends/precision_sparse_attn.py` whose `forward(query, key, value,
attn_metadata)` receives exactly the tensors we need as its first two arguments.

- **Exact names to implement**: `PrecisionSparseAttentionBackend(AttentionBackend)` with
  `get_name/get_impl_cls/get_metadata_cls/get_builder_cls`
  (contract: `abstract.py:31-65`) and `get_supported_head_sizes() -> [64, 128]`
  (needed by `cuda.py:269-273`);
  `PrecisionSparseAttentionImpl(AttentionImpl)` with the `__init__` signature from
  `abstract.py:132-143` and `forward` from `abstract.py:186-194`;
  `PrecisionSparseAttentionMetadata(AttentionMetadata)` +
  `...MetadataBuilder(AttentionMetadataBuilder)` modelled on
  `video_sparse_attn.py:140-235`.
- **Layer index for free**: `self.layer_idx = layer_idx_from_prefix(prefix, default=-1)`
  in `__init__` — exactly `video_sparse_attn_h3.py:248`.
- **Step + CFG branch for free**: `ctx = get_forward_context()` →
  `ctx.current_timestep`, `ctx.forward_batch.is_cfg_negative`
  (`forward_context.py:45-49`; set at `denoising.py:505-511`, `:550-555`).
- **Pass-through compute**: for a pure Phase-1 probe, forward to
  `flash_attn_func_compilable` exactly as `flash_attn.py:325-331` does, so the denoising
  trajectory is bit-comparable to the `FLASH_ATTN` baseline while metrics are computed on
  the side. This is the H3 probe's "sparsity 0, exact dense trajectory" discipline
  (`video_sparse_attn_h3_probe.py:4-7`) generalized.
- **Tradeoffs**: +1 enum member, +1 file, ~3 one-line registrations. Touches
  `fastvideo/platforms/interface.py` and `fastvideo/configs/models/dits/base.py`, which are
  shared across all model families — but additively, and the fallback path
  (`selector.py:279-288`) means other families are unaffected. Cross-attention is
  automatically excluded because `attn2` pins `(FLASH_ATTN, TORCH_SDPA)`
  (`wanvideo.py:174-175`). **Does not require a Wan model-file edit.**
  Also gets you `VSA_sparsity` free on the metadata (`abstract.py:73`) — but see the
  caveat in Rank 1's sibling note below.
- **Caveat to handle**: `fastvideo/pipelines/stages/denoising.py:466-500` only builds
  metadata for VSA and VMoBA and otherwise sets `attn_metadata = None` (`:499-500`). Either
  (a) add an arm there, or (b) read everything from `get_forward_context()` and accept
  `attn_metadata is None` — **(b) is strictly less invasive for Phase 1** and is what
  `FlashAttentionImpl` already tolerates (`flash_attn.py:266`).
- **Where the FP4 comparison happens**: inside `forward`, call
  `_nvfp4_quantize_for_fa4` (`flash_attn.py:138-141`) for the native-quantized reference,
  and/or the deterministic bf16→NVFP4→bf16 round-trip from
  `nvfp4_utils.py:12-133` + `:136-239` for a dequantized router input. Label the latter
  **simulated** (SKILL rule 4).

### Rank 2 — `ForwardHook` on the impl object, no enum changes

`ModuleHookManager.get_from_or_default(module)` (`hooks.py:56-72`) replaces
`module.forward`, and `ForwardHook.pre_forward(module, *args, **kwargs)`
(`hooks.py:33-35`) sees the arguments. `DistributedAttention` registers its impl as a
submodule **only when the impl is an `nn.Module`** (`layer.py:73-74`) — for
`FlashAttentionImpl` it is not, so `named_modules()` will not find it. But the impl is
still reachable as a plain attribute: `transformer.blocks[i].attn1.attn_impl`
(assigned at `layer.py:64-70`). You can attach a manager to it manually since
`ModuleHookManager` only needs `.forward` and `setattr` (`hooks.py:56-72`) — it does not
actually require `nn.Module` at runtime, only the type hint says so.

- **Tradeoffs**: zero source edits to `fastvideo/`; script-only. But it relies on
  duck-typing a class whose annotation says `nn.Module`, it breaks under
  `torch.compile` if `FASTVIDEO_DISABLE_ATTENTION_COMPILE=0`, and `hooks.py:89-90` rejects
  duplicate hook names so re-attachment needs care. Fragile; good for a same-day
  exploratory pass, not for the paper's production runs.

### Rank 3 — Monkeypatch `AttentionImpl.forward` of the resolved backend class

In a standalone script, wrap `FlashAttentionImpl.forward` (`flash_attn.py:234-257`) with a
metrics-computing decorator before `VideoGenerator.from_pretrained`.

- **Tradeoffs**: fastest to write, zero repo edits, and the wrapped function's `self`
  carries `softmax_scale`/`causal` so metrics are exact. But it has no access to
`prefix`/`layer_idx` (`FlashAttentionImpl.__init__` at `flash_attn.py:214-232` **discards
`prefix`**), so you cannot attribute records to a layer without also capturing
  construction order. Ruled out by that alone for H2 (per-layer localization is the whole
  point). Explicitly the option the brief asks to avoid.

### Rank 4 — Edit `fastvideo/attention/layer.py` directly

Insert an optional probe call between `layer.py:145` (`chunk`) and `layer.py:147`
(`attn_impl.forward`), gated on an env var, mirroring
`video_sparse_attn_h3.py:290` + `:297-298`.

- **Tradeoffs**: this is the single most semantically correct location — it is
  *definitionally* "post-RoPE, pre-backend", it works for **every** backend uniformly
  (so BF16/FLASH_ATTN, NVFP4/FLASH_ATTN, and VSA all report from the same line), and it
  needs ~4 lines. Counter-argument: `layer.py` is on the hot path for every model in the
  repo, and `layer.py` has three near-duplicate forwards
  (`:82-164`, `:172-245`, `:288-318`) so a correct patch touches all three. Also note
  under VSA the probe must sit **before** `preprocess_qkv` (`:228`) to see raster order,
  or after it to see what the kernel sees — pick deliberately (D.4).
- **Verdict**: use this *in addition to* Rank 1 if you need one probe that spans
  BF16-dense, NVFP4-dense and VSA in a single run. Otherwise Rank 1 is cleaner.

### Ranked summary

| Rank | Mechanism | Exact anchor | Repo edits | Layer id? | Step/CFG tag? | Verdict |
|---|---|---|---|---|---|---|
| **1** | new `AttentionImpl` behind a new enum | `abstract.py:130-194` contract; register at `interface.py:13-27`, `cuda.py:112-248`, `dits/base.py:22-29` | 3 one-liners + 1 new file | yes, via `layer_idx_from_prefix` (`abstract.py:17-28`) | yes, `get_forward_context()` | **recommended** |
| 2 | `ForwardHook` on `attn1.attn_impl` | `hooks.py:33-35`, `:56-72`; target assigned at `layer.py:64-70` | none | no (impl discards `prefix` for FLASH_ATTN) | yes | exploratory only |
| 3 | monkeypatch `FlashAttentionImpl.forward` | `flash_attn.py:234-257` | none | **no** | yes | rejected for H2 |
| 4 | inline probe in the layer | between `layer.py:145` and `:147` (+ `:233`/`:234`, `+:315`/`:317`) | ~4 lines × 3 forwards | yes (`self` has no prefix, but the *call site* does not either — use module traversal or `self.attn_impl.prefix`) | yes | use alongside Rank 1 for cross-backend single-run probes |

### Metric-computation notes (so Phase 1 does not re-derive them)

- **Reuse VSA's real scorer, do not reimplement**: `fused_block_mean` +
  `q̄k̄ᵀ/√d` + `fused_topk_mask` are importable directly as
  `from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean,
  fused_topk_mask` (`ops.py:9`), and the whole three-line recipe is `ops.py:109-120`.
  The pure-torch equivalent, if you want CPU-checkable determinism, is
  `_pool_tiles` (`video_sparse_attn_h3.py:194-206`) + `matmul` +
  `topk` (`h3.py:224`).
- **The SKILL's 128×64 geometry**: use `get_block_map(q, k, topk_ratio, BLKQ=128,
  BLKK=64)` from `fastvideo/attention/backends/sla.py:78-110` — it is a standalone
  function, already the exact default the SKILL asks for, and it returns
  `(sparse_map, lut, topk)`. Two deviations to record: it applies **smooth-k**
  (`sla.py:99`) and **omits the `1/√d` scale** (`sla.py:102`). For an
  apples-to-apples controlled diagnostic, call it with a copy that drops smooth-k.
- **Top-k count**: `compute_topk(sparsity, num_blocks)` —
  `video_sparse_attn.py:161-163`. Use it verbatim so retained fractions match VSA's own
  rounding (`ceil`, clamped to ≥1).
- **Equal-|k| masks ⇒ precision == recall.** SKILL 1.4 says not to present them as
  independent evidence; VSA's top-k is fixed-size per query block
  (`fused_topk_mask`, `fused_compress_topk.py:298-313`), so this applies.
- **Decision margin** `score[k] - score[k+1]` needs the sorted scores, which
  `fused_topk_mask` does not return — compute it with `torch.topk(scores, k+1)` on the
  fp32 `scores` you already built.
- **Write metrics, never raw Q/K.** At 32760×12×128 bf16 that is ~100 MB per tensor per
  layer per step; 30 layers × 50 steps ⇒ ~300 GB per prompt. SKILL 1.1 forbids it and the
  math agrees.
- **Record `attn_qat_infer_receipt()`** (`attn_qat_infer.py:123-140`) and the
  `identity.py:32` env key set with every run.

---

## Recommended Phase 3 integration point

**Extend VSA (`VIDEO_SPARSE_ATTN`), by adding a sibling backend that reuses VSA's
metadata/tiling and calls the mask-taking kernel directly — modelled line-for-line on
`video_sparse_attn_h3.py`.**

### Why VSA and not the others

| Candidate | Why not / why yes |
|---|---|
| **VSA** `video_sparse_attn.py` | **Chosen.** It is the SKILL's named target ("FastVideo VSA / `VIDEO_SPARSE_ATTN`", SKILL 3.1); it is the only sparse backend in the Wan supported tuple (`fastvideo/configs/models/dits/base.py:25`); the Wan model has a dedicated block class for it (`wanvideo.py:437-582`, selected at `:629`); its metadata builder is already wired into the Wan denoising loop (`fastvideo/pipelines/stages/denoising.py:466-482`); `VSA_sparsity` is already plumbed end-to-end (`fastvideo_args.py:171` → `denoising.py:477` → `abstract.py:73` → `video_sparse_attn.py:166-167`); and its scoring rule is *already* the SKILL's pooled-mean scorer (`ops.py:109-113`). The only thing it lacks is a mask *input* — and the layer directly beneath it has exactly that. |
| **VSA-H3** `video_sparse_attn_h3.py` | **The template to copy, not the thing to extend.** It already does the exact score/compute split we need (`h3.py:292-312`), already has a probe (`h3.py:297-298`), already has per-layer dense opt-outs (`h3.py:289`). But its metadata assumes a packed `[text|cond|audio|video]` sequence with `prefix_segments` (`h3.py:160-191`) and hard-errors on a non-packed one (`h3.py:257-260`), and `VIDEO_SPARSE_ATTN_H3` is not in the Wan supported tuple (`dits/base.py:22-29`). |
| **SLA** `sla.py` | Best *scoring* API in the repo (`get_block_map` at `:78-110`, exactly 128×64), and worth importing as a scorer. But rejected as the integration host: its compute path adds a **learnable** `proj_l` linear branch (`sla.py:220`, `:308-320`) whose weights do not exist in a Wan checkpoint (zero-initialized at `:240-244`), so it changes the model, not just the attention. |
| **BSA** `bsa_attn.py` | Rejected: python `for b / for h / for qb` loops in the compute path (`:216-232`, `:264-305`, `:329-393`, and again in `_reconstruct_pruned` at `:519-535`). Unusable at 624 blocks × 12 heads × 30 layers × 50 steps. Also confounds the study with query-token pruning (`:88-129`). |
| **NABLA** `nabla.py` | Rejected: `flex_attention` compute (`:142-147`) with no low-precision path, threshold-based (not top-k) selection (`:48-53`), and Kandinsky5-only registration. Its `sta_mask` metadata field (`:84-85`) is the closest thing to an external-mask input but it is OR'd in (`:55`), not substituted. |
| **V-MoBA** `vmoba.py` | Rejected: selection is fused inside `moba_attn_varlen` (`:183-194`); no Python seam to hijack. |

### The exact seam

`VideoSparseAttentionImpl.forward` (`video_sparse_attn.py:305-342`) currently delegates
mask construction to the kernel wrapper. Replace that single call with the two-step form
that `video_sparse_attn_h3.py:292-312` already ships:

```python
# scores from the ROUTER-precision Q/K
route_q, route_k = quantize_for_router(query, key, route_precision)
q_pooled = _pool_tiles(route_q, attn_metadata.variable_block_sizes)   # h3.py:194-206
k_pooled = _pool_tiles(route_k, attn_metadata.variable_block_sizes)
scores   = q_pooled @ k_pooled.transpose(-2, -1) / (query.shape[-1] ** 0.5)  # h3.py:296

# mask from top-k, using VSA's own rounding
k_keep = compute_topk(attn_metadata.VSA_sparsity, num_tiles)          # video_sparse_attn.py:161-163
mask   = torch.zeros_like(scores, dtype=torch.bool)
mask.scatter_(-1, scores.topk(k_keep, dim=-1).indices, True)          # cf. h3.py:221-226

# sparse compute at the COMPUTE precision, with OUR mask
out, _ = block_sparse_attn_256_bshd(query, key, value, mask,
                                    attn_metadata.variable_block_sizes)  # h3.py:312
```

`quantize_for_router` is the `route_precision` axis: `bf16` = identity;
`nvfp4` = the round-trip built on `nvfp4_utils.py:12-133` / `:136-239` (simulated) or
`_nvfp4_quantize_for_fa4` (`flash_attn.py:138-141`, native but not readable back);
`fp8` = an E4M3 cast with an explicit scale (precedent `sla.py:520-526`).
This gives the SKILL's two independent axes (SKILL 3.2) with the router cost being one
pooled matmul over 624×624 tiles — genuinely cheap relative to the 32760² attention.

### Concrete prerequisites, with citations

1. **Use the 256-element tile, not 64.** Set `VSA_TILE_SIZE = (4, 8, 8)`
   (as `video_sparse_attn_h3.py:47`) because:
   - `block_sparse_attn_256_bshd` is the mask-taking, BSHD-native entry point
     (`block_sparse_attn_256.py:134-161`);
   - the FA4 CuTe path — the only one targeting sm_100+ — is the 256-block path
     (`block_sparse_attn_256.py:1-17`, opt-in `FASTVIDEO_VSA_CUTEDSL=1` at `:47`);
   - the 64-block path's native kernel is sm_90 ThunderKittens only
     (`block_sparse_attn.py:26-30`, `:362-381`), so on B200 it silently runs Triton.
   `video_sparse_attn.py:26-28` documents this dispatch and
   `video_sparse_attn.py:318-326` shows the existing 256 fastpath.
   Note the mask-taking kernel is `block_sparse_attn_256_bshd`, which internally expands
   the logical 256-tile map to physical 128-token (CuTe, `:52-74`) or 64-token (Triton,
   `:77-98`) blocks — **so "block size" for latency purposes is 128 or 64, while "block
   size" for routing purposes is 256. Report both.**
2. **Add an arm to the denoising stage's metadata dispatch.**
   `fastvideo/pipelines/stages/denoising.py:466-500` builds metadata only for
   `VideoSparseAttentionBackend` (`:466`) and `VMOBAAttentionBackend` (`:483`), else
   `None` (`:499-500`). Add the new backend there, passing `raw_latent_shape[2:5]`,
   `patch_size`, `VSA_sparsity`, `device` exactly as `:472-479`.
3. **Decide the Wan block class.** `wanvideo.py:628-629` only picks
   `WanTransformerBlock_VSA` when the env var is literally `"VIDEO_SPARSE_ATTN"`. Options:
   (a) extend that comparison, which brings the `to_gate_compress` projection
   (`:458-462`) and therefore requires those weights in the checkpoint; or
   (b) run on the plain `WanTransformerBlock` (`DistributedAttention`, `layer.py:82-164`,
   4-arg `forward`) and accept no compression branch — **(b) is the cleaner science**,
   since the compression branch is a second always-dense path that confounds the sparsity
   ablation (see C.2). If you take (a), pass `gate_compress=None`, which is explicitly
   supported (`layer.py:210-213`, `:230-233`).
4. **Correctness gate before any latency claim** (SKILL Phase 4): compare against a
   trusted reference with identical Q/K/V *and identical mask*. The oracle already exists —
   `token_tile_and_valid` (`video_sparse_attn_h3.py:51-60`) is documented as "the single
   encoding of the padding contract, shared by the probe and the test oracle so they cannot
   drift". Existing kernel tests to extend:
   `fastvideo-kernel/tests/test_vsa256_forward.py`,
   `test_vsa256_forward_vbs.py`, `test_vsa256_triton.py`, `test_vsa_forward.py`.
5. **Honesty boundary.** `block_sparse_attn_256_bshd` computes in the **input dtype**
   (bf16) — there is **no** native sparse-NVFP4 kernel in this repo. Any
   `SPARSE-FP4-*` row is therefore *numerical-only* until Phase 4 lands a real kernel:
   SKILL rules 1, 3 and the Phase 3.3 note apply verbatim. The repo's own precedent for
   refusing a silent low-precision fallback is `fastvideo/platforms/cuda.py:143-154`.

---

## Discrepancies vs the SKILL

The SKILL references five paths that **do not exist** in this repo. The skill's own
support directories are empty:
`.agents/skills/sparsefp4-video-attention/{assets,references,scripts}/` all contain zero
files (only `SKILL.md` is present).

| SKILL reference | Exists? | Correct existing equivalent |
|---|---|---|
| `scripts/check_env.py` (SKILL 0.3) | **No.** `scripts/` contains only `check_docs_links.py`, `demo_anyflow_14b.py`, `ltx2_sr_alignment.py`, `verify_anyflow_fastvideo_parity.py` + subdirs | **`collect_env.py` at the repo root** (`collect_env.py:1-30`; imports `fastvideo.envs.environment_variables` at `:20`). Run `python collect_env.py`. It emits a `SystemEnv` namedtuple (`:29-40`) covering torch/CUDA/gcc/OS/pip. It has **no `--output` flag**, so redirect to `artifacts/sparsefp4/env.json` yourself (or wrap it). Supplement with `fastvideo/tests/performance/identity.py:32` for the env keys that define a comparable run, and `attn_qat_infer_receipt()` (`fastvideo/attention/backends/attn_qat_infer.py:123-140`) for the kernel-provenance line. |
| `scripts/analyze_masks.py` (SKILL 1.4) | **No.** | Nothing equivalent. The closest prior art is the **offline aggregation implied by** `fastvideo/attention/backends/video_sparse_attn_h3_probe.py:88-100` (one `.pt` per `(step, layer, rank)`, "aggregate offline" per `probe.py:19`) — but no aggregator script is checked in. **Must be written.** Reuse `fused_topk_mask` / `compute_topk` (`fastvideo-kernel/.../fused_compress_topk.py:298-313`, `video_sparse_attn.py:161-163`) so the offline analysis and the online backend agree on top-k rounding. |
| `assets/experiment_config.yaml` (SKILL "Default experiment matrix") | **No.** `assets/` contains `prompt.txt`, `prompts/`, `eval/`, `images/`, `videos/`, `logos/`, `8steps/`, and loose media — no experiment config. | Nothing equivalent. Write it to **`artifacts/sparsefp4/configs/`** (that directory already exists and is the SKILL-sanctioned artifact location). For the YAML *style* to follow, the repo's inference-config precedent is `scripts/inference/inference_wan.yaml` and the typed generator config (`fastvideo/entrypoints/cli/inference_config.py`, `VideoGenerator.from_file` at `fastvideo/entrypoints/video_generator.py:256-267`). |
| `assets/prompts.txt` (SKILL Phase 5 dev set) | **No** — but a near-exact substitute exists. | **`assets/prompt.txt`** (singular, no `s`): 8 one-per-line T2V prompts, including the raccoon/sunflowers prompt used by `examples/inference/basic/basic.py:29-31` and the lion/savanna prompt at `:37-41`. Close enough to the SKILL's "10 prompts" dev set. For the paper-scale 50–100-prompt VBench-compatible set, use `VBenchPromptDataset` (`fastvideo/eval/datasets/vbench.py:45-121`, 946 prompts, dimension-filterable) rather than a text file. Note `assets/prompts/mixkit_i2v.jsonl` is **I2V** and not usable here. |
| `references/EXPERIMENT_SPEC.md` (SKILL: "Read before editing code") | **No.** | Nothing equivalent; the SKILL body itself is the only spec. This document is intended to stand in for the repo-survey half of it. |
| `references/REPORT_TEMPLATE.md` (SKILL "Final reporting") | **No.** | Nothing equivalent. The SKILL's own "Final reporting" list (15 numbered items) is a complete substitute; write `artifacts/sparsefp4/REPORT.md` against that list directly. |

Two further SKILL-vs-repo mismatches worth flagging before Phase 1 starts:

1. **SKILL 1.2's default block geometry (query 128 / key 64) is not expressible in VSA.**
   VSA forces `q_block == k_block == prod(tile_size)` ∈ {64, 256}
   (`video_sparse_attn.py:313`, `fastvideo-kernel/python/fastvideo_kernel/ops.py:96-106`,
   `vsa_utils.py:17`). The 128×64 geometry *is* natively available, but in **SLA**
   (`fastvideo/attention/backends/sla.py:200-201`, `get_block_map` at `:78-110`).
   Recommendation: run the SKILL's controlled diagnostic at 128×64 via SLA's
   `get_block_map`, and the VSA-integrated arm at 256×256, and state the geometry on every
   table. The 64×64 ablation the SKILL asks for maps onto VSA `(4,4,4)`.
2. **The SKILL's `nvfp4_fa4` search term (SKILL 0.1) finds nothing.** There is no symbol
   `nvfp4_fa4` as a module or kernel name; it exists only as (a) a `FlashAttentionImpl`
   kwarg (`fastvideo/attention/backends/flash_attn.py:226`), (b) a
   `VideoGenerator.from_pretrained` kwarg (`fastvideo/entrypoints/video_generator.py:207`),
   and (c) the `FASTVIDEO_NVFP4_FA4` env var. The actual kernel entry point is
   `flash_attn_fp4_func` (`fastvideo/attention/utils/flash_attn_cute.py:368-378`).
