# Phase 0 — Environment Preflight + Dense Smoke Tests

**Study:** SparseFP4 Video Attention
**Status:** ✅ **PASS** — both dense baselines run; **native NVFP4 attention is available and measured** on this machine.
**Date:** 2026-08-13
**Repo:** `/home/ec2-user/FastVideo` @ `8208536cd1db7a1d32b68aaa6a679953ae23ab8b`, branch `exp/sparsefp4-mask-stability`
**Machine:** 8x NVIDIA B200, sm_100 `(10, 0)`, 183 GiB each, driver 595.91.07

---

## 1. THE INTERPRETER EVERY LATER PHASE MUST USE

```
/mnt/scratch/fv-venv/bin/python
```

CPython 3.12.13 · torch 2.12.0+cu130 · sm_100 verified with a real GPU matmul.

**Activate with this one line — it is not optional.** Several env vars (`CUDA_HOME`,
`FASTVIDEO_FA4`, `HF_HOME`) are load-bearing; without them the NVFP4 path fails and
the model is not found:

```bash
source /home/ec2-user/FastVideo/artifacts/sparsefp4/configs/env.sh
# then use "$FV_PYTHON", e.g.
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" your_script.py
```

The env file is [`configs/env.sh`](configs/env.sh). What it sets and why:

| Variable | Value | Why it is required |
|---|---|---|
| `FV_PYTHON` | `/mnt/scratch/fv-venv/bin/python` | the study interpreter |
| `CUDA_HOME` | `/usr/local/cuda-13.0` | flashinfer JIT-compiles `fp4_quantize_sm100` with `nvcc`; hard-fails without it |
| `PATH` | prepends `$CUDA_HOME/bin`, `$FV_VENV/bin` | puts `nvcc` and `ninja` where flashinfer's JIT looks |
| `HF_HOME` | `/mnt/scratch/hf-cache` | model is on the scratch disk; root volume is too small |
| `TMPDIR` | `/mnt/scratch/tmp` | build/JIT scratch off the small root volume |
| `FASTVIDEO_FA4` | `1` | **mandatory** — the flash-attention-fp4 fork ships no compiled FA2, so *every* dense attention path raises `ImportError` without the FA4 opt-in |
| `CUTE_DSL_ENABLE_TVM_FFI` | `1` | CuTeDSL kernel dispatch |

Per-run switches (do **not** put these in `env.sh`):

```bash
export FASTVIDEO_NVFP4_FA4=1          # native NVFP4 Q/K on the FLASH_ATTN backend
export FASTVIDEO_ATTENTION_BACKEND=…  # FLASH_ATTN | ATTN_QAT_INFER | VIDEO_SPARSE_ATTN | TORCH_SDPA
```

`FASTVIDEO_ATTENTION_BACKEND` is the documented registry override (`fastvideo/attention/AGENTS.md`).
No model code was edited to select a backend.

---

## 2. Verdict summary

| Question | Answer |
|---|---|
| FastVideo importable and runnable? | ✅ yes, editable install, torch 2.12.0+cu130 |
| sm_100 supported by torch? | ✅ `sm_100` in `torch.cuda.get_arch_list()`, real BF16 matmul rel-err 1.7e-3, `_scaled_mm` FP8 OK |
| Model downloaded? | ✅ rev `0fad780a534b6463e45facd96134c9f345acfa5b` |
| Dense BF16 smoke | ✅ PASS |
| Dense NVFP4 smoke | ✅ PASS |
| **Native NVFP4 attention?** | ✅ **AVAILABLE — native, not simulated.** Evidence in §6 |
| H4 (systems composability) | **available** — a native FP4 attention kernel exists to extend |
| CUDA toolkit for Phase 4? | ✅ CUDA 13.0.88 (`nvcc`) installed during Phase 0 |
| Sparse backends for Phase 3? | ✅ VSA + block-sparse kernels all import |
| Blockers needing a decision | **none** |

---

## 3. Building the environment — verbatim commands

### 3.1 Storage first (this was a real constraint)

The root volume has **25 GB total / ~9 GB free**. The venv (6.6 GB) + model (27 GB) +
CUDA toolkit (4.9 GB) do not fit. The host has eight **unmounted, empty** 3.5 TB
instance-store NVMe drives (`blkid` reported no filesystem, `wipefs -n` no signatures),
so one was formatted and mounted as scratch:

```bash
sudo mkfs.xfs -f -L fvscratch /dev/nvme1n1
sudo mkdir -p /mnt/scratch
sudo mount -o noatime /dev/nvme1n1 /mnt/scratch
sudo chown ec2-user:ec2-user /mnt/scratch
mkdir -p /mnt/scratch/hf-cache /mnt/scratch/uv-cache /mnt/scratch/tmp
```

⚠️ **`/mnt/scratch` is instance store — it does NOT survive an instance stop/start.**
The venv, model cache and CUDA toolkit would all need rebuilding. Nothing under
`artifacts/` depends on it.

### 3.2 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # -> uv 0.12.4 at ~/.local/bin/uv
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/scratch/uv-cache
export TMPDIR=/mnt/scratch/tmp
```

### 3.3 FastVideo (the install that worked)

```bash
cd /home/ec2-user/FastVideo
uv venv /mnt/scratch/fv-venv --python 3.12
VIRTUAL_ENV=/mnt/scratch/fv-venv UV_TORCH_BACKEND=cu130 uv pip install -e ".[dev]"
```

Resolved cleanly in ~11 s. `pyproject.toml` was **not modified**.
Installed `torch==2.12.0+cu130`, `torchvision 0.27.0+cu130`, `torchaudio 2.11.0+cu130`,
`transformers 5.15.0`, `diffusers 0.39.0`, `fastvideo-kernel 0.3.2`, `triton 3.7.0`.

### 3.4 Verify sm_100 with real GPU work (not just `is_available()`)

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/scratch/fv-venv/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())
a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); b=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
c=a@b; torch.cuda.synchronize(); ref=a.float()@b.float()
print('rel_err', ((c.float()-ref).norm()/ref.norm()).item())"
```

```
2.12.0+cu130 13.0 (10, 0) ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
REAL BF16 MATMUL OK rel_err= 0.0016622886760160327
FP8 scaled_mm OK torch.bfloat16 torch.Size([256, 256])
float4_e2m1fn_x2 dtype present: True
```

**No fallback to `--system-site-packages` against `/opt/pytorch` was needed.** The
clean cu130 install supports sm_100. `/opt/pytorch` (Python 3.13, torch 2.13.0+cu130)
is unused by this study.

### 3.5 Native NVFP4 attention stack

```bash
# the FP4 fork; https:// instead of the docs' git+ssh:// (no SSH key on this host)
VIRTUAL_ENV=/mnt/scratch/fv-venv uv pip install --no-deps \
  "git+https://github.com/hao-ai-lab/flash-attention-fp4.git@fix/cutlass-dsl-4.5#subdirectory=flash_attn/cute"

# quack-kernels 0.5.0 is the release that pins nvidia-cutlass-dsl>=4.5.2 (see §4.3)
VIRTUAL_ENV=/mnt/scratch/fv-venv uv pip install --no-deps "quack-kernels==0.5.0"

VIRTUAL_ENV=/mnt/scratch/fv-venv uv pip install ninja
```

`nvidia-cutlass-dsl 4.5.3`, `nvidia-cutlass-dsl-libs-base 4.5.3`, `apache-tvm-ffi
0.1.13.post3` and `flashinfer-python 0.6.17` were **already present** from the base
install, and `--no-deps` kept torch untouched (verified `2.12.0+cu130` after each step).

### 3.6 CUDA toolkit 13.0

```bash
# bind-mount so ~7 GB of toolkit lands on scratch, not the 9 GB-free root volume
sudo mkdir -p /mnt/scratch/cuda/13.0 /usr/local/cuda-13.0
sudo mount --bind /mnt/scratch/cuda/13.0 /usr/local/cuda-13.0
sudo dnf install -y cuda-toolkit-13-0      # ~3 min
/usr/local/cuda-13.0/bin/nvcc --version    # -> release 13.0, V13.0.88
```

⚠️ The bind mount is **not** in `/etc/fstab`; after a reboot re-run the `mount --bind`
(or `nvcc` disappears and the NVFP4 quantizer fails).

### 3.7 Model

```bash
df -h            # checked FIRST: 3.4 T free on /mnt/scratch
HF_HOME=/mnt/scratch/hf-cache /mnt/scratch/fv-venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', max_workers=8))"
```

31 files, **27 GB**, no authentication or license gate.

```
revision: 0fad780a534b6463e45facd96134c9f345acfa5b
path:     /mnt/scratch/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
```

Framework defaults confirmed from `SamplingParam.from_pretrained`: **480x832, 81 frames,
50 steps, guidance 3.0, seed 1024** — matches the experiment spec.

---

## 4. Install problems hit, and how each was resolved

Recorded in full, including the dead ends.

### 4.1 Python 3.11 → `fastvideo-kernel` build failed (no nvcc)

First attempt used `--python 3.11` (the FP4 docs say "Python 3.10 or 3.11"). uv had to
build `fastvideo-kernel==0.3.2` from sdist and CMake died:

```
CMake Error … CMakeCUDAFindToolkit.cmake:104 (message):
  Failed to find nvcc.  Compiler requires the CUDA toolkit.
```

**Cause:** PyPI publishes only **cp312** wheels for `fastvideo-kernel 0.3.2`
(`fastvideo_kernel-0.3.2-cp312-cp312-manylinux…x86_64.whl` + sdist). On 3.11 there is no
wheel, so it builds from source and needs nvcc.

**Resolution:** use **Python 3.12** → prebuilt wheel, no compile.
The docs' "Python 3.10 or 3.11" line is stale for this repo state: every FP4 dependency
(`nvidia-cutlass-dsl`, `quack-kernels`, `flashinfer-python`) ships `py3` wheels, and the
whole stack works on 3.12 — verified end-to-end in §5/§6. **Recorded discrepancy, not a
blocker.**

### 4.2 `ModuleNotFoundError: No module named 'quack'`

`flash_attn/cute/utils.py` does `import quack.activation`. The fork is installed with
`--no-deps`, so `quack-kernels` must be installed explicitly. The docs' validated set
says `quack-kernels==0.4.1`, but that pins `nvidia-cutlass-dsl>=4.4.2` and pairs with the
`@fp4` branch on dsl **4.4.2**.

**Resolution:** the `fix/cutlass-dsl-4.5` branch requires dsl `>=4.5.2`, so use the
matching quack release. Walked every `quack-kernels` release's `requires_dist`:

| quack-kernels | pins nvidia-cutlass-dsl |
|---|---|
| 0.4.1 | `>=4.4.2` (docs' set, for branch `@fp4`) |
| **0.5.0** | **`>=4.5.2`** ← chosen, matches installed 4.5.3 |
| 0.5.1–0.5.3 | `==4.6.0.dev0` — **4.6-era is explicitly unsupported** (`cute.make_fragment` removed; fails at CuTe JIT trace) |
| 0.6.x | `==4.6.x` — same 4.6 exclusion |

`quack-kernels==0.5.0` + `nvidia-cutlass-dsl==4.5.3`. **Do not upgrade either to 4.6.x.**

### 4.3 `git+ssh://` in the docs is not usable here

`docs/inference/optimizations.md` gives
`pip install … "git+ssh://git@github.com/hao-ai-lab/flash-attention-fp4.git@…"`.
No SSH key is configured on this host. Switched to `git+https://` — same repo, same
branch, resolved to commit `940bf7e511375ec160bc2d7188bef35915ded1e3` (pinned in
`env.json` for reproducibility).

### 4.4 `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

Raised from `flashinfer/jit/cpp_ext.py::get_cuda_path` inside
`nvfp4_quantize` → `fp4_quantize_sm100`. The NVFP4 **Q/K quantizer** (not the attention
kernel itself) is a flashinfer JIT module compiled on first use, so a CUDA toolkit is a
**runtime** requirement for the NVFP4 path, not just for kernel development.

**Resolution:** installed CUDA 13.0 (§3.6) and exported `CUDA_HOME`.

### 4.5 `FileNotFoundError: … 'ninja'`

Same JIT path — after finding nvcc, flashinfer shells out to `ninja`.
**Resolution:** `uv pip install ninja`. First JIT build then took ~70 s (one-time; cached
under `/mnt/scratch` afterwards).

### 4.6 `scripts/check_env.py` does not exist — SKILL discrepancy

The SKILL instructs `python scripts/check_env.py --output artifacts/sparsefp4/env.json`.
Confirmed absent (`ls: cannot access 'scripts/check_env.py'`). The repo ships
**`collect_env.py` at its root** instead, which is human-readable and takes no `--output`.

**Resolution, using what exists:**
- `collect_env.py` output → [`env/collect_env.txt`](env/collect_env.txt)
- machine-readable half written by a purpose-built
  [`configs/write_env_json.py`](configs/write_env_json.py) → [`env.json`](env.json)

### 4.7 Peak memory initially read 0.00 GiB

`torch.cuda.max_memory_allocated()` in the driver process returned 0: FastVideo runs the
pipeline in a **worker subprocess**, so the parent's allocator never sees the activations.
**Resolution:** sample device-level `nvidia-smi memory.used` on a background thread
(`GpuMemorySampler` in `configs/phase0_smoke.py`). Both smoke runs were re-run after this
fix, so the reported numbers come from the same code path.

---

## 5. Smoke-test results

Driver: [`configs/phase0_smoke.py`](configs/phase0_smoke.py). Single GPU.

```bash
cd /home/ec2-user/FastVideo
source artifacts/sparsefp4/configs/env.sh

# 1. dense BF16
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase0_smoke.py \
  --mode bf16 --steps 4 \
  --out-dir artifacts/sparsefp4/videos \
  --metrics-json artifacts/sparsefp4/raw/phase0_smoke_bf16.json \
  > artifacts/sparsefp4/logs/phase0_smoke_bf16.log 2>&1

# 2. dense native NVFP4
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase0_smoke.py \
  --mode nvfp4 --steps 4 \
  --out-dir artifacts/sparsefp4/videos \
  --metrics-json artifacts/sparsefp4/raw/phase0_smoke_nvfp4.json \
  > artifacts/sparsefp4/logs/phase0_smoke_nvfp4.log 2>&1
```

| | dense BF16 | dense NVFP4 |
|---|---|---|
| Precision label | **native BF16** (FA4 CuTe) | **native NVFP4** (E2M1 Q/K, per-16 E4M3 SF, BF16 V) |
| Simulated? | no | **no** |
| Result | ✅ PASS | ✅ PASS |
| Generation wall-clock | 10.83 s | 11.54 s |
| Model load | 34.44 s | 34.29 s |
| Peak GPU-0 memory | 14.38 GiB | 14.38 GiB |
| Output | [`videos/phase0_smoke_bf16.mp4`](videos/phase0_smoke_bf16.mp4) | [`videos/phase0_smoke_nvfp4.mp4`](videos/phase0_smoke_nvfp4.mp4) |
| Raw metrics | [`raw/phase0_smoke_bf16.json`](raw/phase0_smoke_bf16.json) | [`raw/phase0_smoke_nvfp4.json`](raw/phase0_smoke_nvfp4.json) |
| Log | [`logs/phase0_smoke_bf16.log`](logs/phase0_smoke_bf16.log) | [`logs/phase0_smoke_nvfp4.log`](logs/phase0_smoke_nvfp4.log) |

Both produced a valid 81-frame 480x832 video (`(81, 480, 832, 3)` decoded).

### ⚠️ These are SMOKE settings, not experiment settings

| Knob | Smoke value | Experiment value |
|---|---|---|
| `num_inference_steps` | **4** | **50** (framework default) |
| height x width | 480x832 | 480x832 ✅ same |
| `num_frames` | 81 | 81 ✅ same |
| seed | 1024 | 1024 ✅ same |

Only the step count was reduced. Spatial dims and frame count were deliberately kept at
the experiment defaults so the attention kernels JIT-compile and execute at the real
seqlen (32760).

**The wall-clock numbers above are liveness checks, NOT a benchmark.** One repetition, no
warmup/measure separation, and CuTeDSL JIT compile time lands inside the measured window.
The 0.7 s BF16-vs-NVFP4 difference is **not** a speedup measurement and must not be
reported as one. For a properly warmed attention-kernel measurement see §6.3.

---

## 6. Native NVFP4 availability — VERDICT: ✅ AVAILABLE (native, not simulated)

H1–H3 can use the **real** kernel; no fake quantization is required for the attention
compute path. **H4 is available**, not blocked.

### 6.1 What is native, precisely

| Component | Precision | Native? |
|---|---|---|
| Q, K | NVFP4 **E2M1**, per-16-element **E4M3** scale factors | ✅ native, `tcgen05.mma…kind::mxf4nvf4.block_scale.scale_vec::4X` |
| QKᵀ MMA | FP4 block-scaled, 5th-gen tensor cores | ✅ native |
| V / PV | BF16 | native BF16 (kernel is `qk_mode=nvfp4, pv_mode=bf16`) |
| Q/K quantizer | flashinfer `fp4_quantize_sm100` (JIT CUDA) | ✅ native |

There is **no quantize-dequantize simulation** anywhere in this path.
⚠️ Note for Phase 2/3 bookkeeping: **PV is BF16, not FP4.** "Dense NVFP4" here means
*NVFP4 Q/K with BF16 PV*, which is what the FastVideo/FA4 kernel implements — label it
that way in every table rather than as fully-FP4 attention.

### 6.2 Evidence

**(a) The framework's own resolution receipt** (`attn_qat_infer_receipt()`):

```
arch=sm_100 kernel=flash-attention-fp4 qk_mode=nvfp4(per-16-e4m3-sf) pv_mode=bf16 train_sim_mismatch=measured
```

with `is_attn_qat_infer_available() == True`, `fa_version == "4"`, `_FA4_FP4_AVAILABLE == True`.
`sm_100` is in `_FA4_FP4_CAPABILITIES = {(10, 0), (10, 3)}` in
`fastvideo/attention/backends/attn_qat_infer.py`.

**(b) Blackwell FP4 MMA instructions in the emitted kernel.** The CuTeDSL trace prints
the actual PTX MMA kind:

```
GEMM_PTX_FP4 kind=tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X is_ts=False
```

`mxf4nvf4.block_scale` is the hardware block-scaled-FP4 tensor-core op — it cannot be
produced by a simulated path. Present in
[`logs/phase0_nvfp4_kernel_probe.log`](logs/phase0_nvfp4_kernel_probe.log) and in
[`logs/phase0_smoke_nvfp4.log`](logs/phase0_smoke_nvfp4.log) (4 occurrences); **0
occurrences** in the BF16 log.

**(c) Genuinely FP4-typed tensors** — `torch.float4_e2m1fn_x2`, half-width last dim,
`uint8` scale-factor tensor in the FA4 MMA layout:

```
QUANTIZED: torch.float4_e2m1fn_x2 (1, 1024, 12, 64)  sf: torch.uint8 (32, 4, 8, 4, 2, 12, 1)
```

**(d) The pipeline really used it.** The NVFP4 run logs `NVFP4 FA4 enabled for
FlashAttentionImpl (quant_qk only)` per attention layer, and the two videos differ in a
way consistent with FP4 quantization noise — identical seed/prompt/steps, so BF16 and a
no-op path would be bit-identical:

| | value |
|---|---|
| md5 bf16 vs nvfp4 | `69cd6078…` vs `4c5a7cb4…` (differ) |
| mean abs pixel diff (0–255) | **5.06** |
| pixel correlation | **0.99582** |
| bit-identical | **False** |

**(e) Kernel-level correctness at the real Wan2.1 attention shape**
(B=1, seqlen=32760, 12 heads, head_dim=128), vs a BF16 SDPA reference —
[`raw/phase0_nvfp4_kernel_probe.json`](raw/phase0_nvfp4_kernel_probe.json):

| Path | cosine sim | rel. L2 | max abs err |
|---|---|---|---|
| native NVFP4 | **0.99050** | 0.13783 | 0.01196 |
| native BF16 FA4 | 0.999995 | 0.00311 | 0.00024 |

cos ≈ 0.99 matches the documented "~0.99 per-call cosine similarity vs BF16" exactly —
independent confirmation the kernel behaves as its authors describe. **This ~0.99 / 0.138
rel-L2 gap is precisely the perturbation H1 asks about**, so a real effect exists to
measure rather than a numerically-null path.

### 6.3 Measured attention-kernel latency (warmed, properly benchmarked)

Probe: [`configs/phase0_nvfp4_kernel_probe.py`](configs/phase0_nvfp4_kernel_probe.py).
Attention kernel **only**, 3 warmup iters (JIT compile excluded), CUDA-synchronized,
median of 20 reps, identical shapes:

| Path | median | stdev |
|---|---|---|
| native NVFP4 | **4.013 ms** | 0.141 |
| native BF16 FA4 | **5.135 ms** | 0.334 |
| **NVFP4 speedup** | **1.28x** | |

Consistent with the documented ~1.31x. Scope, stated explicitly: this is a **measured
attention-kernel microbenchmark**, not an end-to-end generation speedup, and not a
theoretical FLOP reduction. It is the honest dense-NVFP4 baseline any Phase 4 sparse
kernel must beat.

```bash
CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase0_nvfp4_kernel_probe.py \
  --out artifacts/sparsefp4/raw/phase0_nvfp4_kernel_probe.json \
  > artifacts/sparsefp4/logs/phase0_nvfp4_kernel_probe.log 2>&1
```

---

## 7. CUDA toolkit for Phase 4 kernel work — ✅ AVAILABLE

**At Phase 0 start: ABSENT.** `nvcc` not on PATH, no `/usr/local/cuda*` at all.

**Now: PRESENT** — CUDA Toolkit **13.0.88** at `/usr/local/cuda-13.0`, installed from the
already-configured `cuda-rhel9-x86_64` dnf repo. 13.0 was chosen deliberately to match
torch's `cu130`. Details in [`env/nvcc.txt`](env/nvcc.txt).

Phase 4 readiness:

- ✅ `nvcc` 13.0.88, `ninja`, g++ 11.5.0, CMake 4.4.2 — full toolchain
- ✅ `sudo` available (13.1/13.2/13.3 also installable if a fork needs a newer nvcc)
- ✅ `fastvideo-kernel` source tree in-repo at `fastvideo-kernel/` with `build.sh`
- ✅ CuTeDSL FP4 kernel source is editable Python at
  `/mnt/scratch/fv-venv/lib/python3.12/site-packages/flash_attn/cute/` — the Phase 4
  tile-skipping edit is a Python/CuTeDSL change, **not** a C++ recompile
- ⚠️ Two caveats: the `/usr/local/cuda-13.0` bind mount must be re-created after reboot
  (§3.6), and the toolkit lives on ephemeral instance store

---

## 8. Constraints on later phases

### Blocking a decision from you: **none**

### Storage — the main operational risk

| Filesystem | Size | Used | Free | Note |
|---|---|---|---|---|
| `/` | 25 GB | 19 GB | **6.4 GB** | ⚠️ small; keep large writes off it |
| `/mnt/scratch` (`/dev/nvme1n1`) | 3.5 TB | 70 GB | **3.4 TB** | ⚠️ **ephemeral instance store** |

- Room is ample for Phase 1/2 raw records and Phase 5 videos — put them under
  `/mnt/scratch` if large, and keep only aggregates + selected videos in `artifacts/`
  (which is on the root volume).
- **7 more unmounted 3.5 TB NVMe drives** are available if ever needed
  (`/dev/nvme2n1`…`/dev/nvme8n1`).
- ⚠️ An instance stop/start wipes the venv, the 27 GB model and the CUDA toolkit.
  §3 is a complete rebuild recipe.

### Dependency pins — do not drift

| Package | Pinned | Why |
|---|---|---|
| `nvidia-cutlass-dsl` | **4.5.3** | 4.6-era removes `cute.make_fragment`; fails at CuTe JIT trace |
| `quack-kernels` | **0.5.0** | only release pinning dsl `>=4.5.2` without forcing 4.6 |
| `flash-attn-4` | `fix/cutlass-dsl-4.5` @ `940bf7e5…` | dsl-4.5-compatible fork |
| `torch` | **2.12.0+cu130** | matches CUDA 13.0 + sm_100 |
| Python | **3.12** | only version with a prebuilt `fastvideo-kernel` wheel |

⚠️ **No FlashAttention-2 is installed** (`flash_attn_2_installed: false`) — the FP4 fork
replaces it. Consequences:
- `FASTVIDEO_FA4=1` must stay set or dense attention raises `ImportError`.
- Any path needing FA2 specifically (attention backward / training / GQA below sm90)
  is unavailable. Not needed for this inference-only study.

### Other

- **Model licensing:** no gate. `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` downloaded
  unauthenticated (HF warned only about rate limits). Set `HF_TOKEN` if Phase 5 pulls a
  VBench prompt set and hits limits.
- **FSDP:** `use_fsdp_inference=False` is **required** for the FP4 path (sharding
  invalidates FP4 tensor pointers). The smoke driver disables it for *both* modes so the
  pair stays comparable — keep that symmetry in all later paired comparisons.
- **Sparse backends for Phase 3 are ready** (checked, all import): `video_sparse_attn`
  and `video_sparse_attn_bshd` from `fastvideo_kernel`, plus `block_sparse_attn`,
  `block_sparse_attn_256`, `block_sparse_attn_cute_fwd`, `block_sparse_attn_varlen`,
  `vmoba`, `vsa_utils`. Registry exposes `VIDEO_SPARSE_ATTN`, `VIDEO_SPARSE_ATTN_H3`,
  `BSA_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `NABLA_ATTN`. Note `video_sparse_attn_h3.py`
  targets sm_10x via the FA4 CuTe 256-tile path — the natural Phase 3/4 starting point.
- **GPU capacity:** all 8 B200s idle; one run uses ~14.4 GiB of 179 GiB, so Phase 1
  sweeps can run 8-way in parallel with `CUDA_VISIBLE_DEVICES`.
- **`num_gpus=1`** suffices; the 1.3B model needs no sharding.

---

## 9. Artifact index

| Path | Contents |
|---|---|
| [`env.json`](env.json) | machine-readable environment: interpreter, torch, capability, driver, CUDA, git commit/branch/dirty, model id + revision, full FA4 stack versions, native-NVFP4 probe |
| [`env/collect_env.txt`](env/collect_env.txt) | repo `collect_env.py` output |
| [`env/nvidia-smi.txt`](env/nvidia-smi.txt) | `nvidia-smi -q`, all 8 GPUs |
| [`env/pip-freeze.txt`](env/pip-freeze.txt) | frozen package list from the study interpreter |
| [`env/nvcc.txt`](env/nvcc.txt) | CUDA toolkit: absent → installed, with Phase 4 assessment |
| [`configs/env.sh`](configs/env.sh) | **the activation script every phase must source** |
| [`configs/phase0_smoke.py`](configs/phase0_smoke.py) | BF16 / NVFP4 smoke driver |
| [`configs/phase0_nvfp4_kernel_probe.py`](configs/phase0_nvfp4_kernel_probe.py) | native-NVFP4 kernel evidence + warmed microbenchmark |
| [`configs/write_env_json.py`](configs/write_env_json.py) | regenerates `env.json` |
| `raw/phase0_smoke_{bf16,nvfp4}.json` | smoke metrics |
| `raw/phase0_nvfp4_kernel_probe.json` | accuracy + latency evidence |
| `logs/phase0_smoke_{bf16,nvfp4}.log` | full smoke stdout/stderr |
| `logs/phase0_nvfp4_kernel_probe.log` | kernel probe log incl. `GEMM_PTX_FP4` lines |
| `videos/phase0_smoke_{bf16,nvfp4}.mp4` | smoke outputs (81f, 480x832) |

Nothing was committed to git. `STATUS.md`, `CODEBASE_MAP.md` and `.agents/skills/` were
not touched. No files were created at the repo root.

---

## 10. Recommended next action (Phase 1)

Instrument Q/K **at the attention-backend boundary** — after Q/K norm and RoPE. The clean
seam is `FlashAttentionImpl.forward` / `_forward_nvfp4` in
`fastvideo/attention/backends/flash_attn.py`, where `_nvfp4_quantize_for_fa4(query)` is
called: the tensors handed to that function are exactly the post-transform Q/K the kernel
consumes, satisfying the SKILL's §1.1 requirement.

Real NVFP4 quantization is reachable for routing diagnostics — reuse
`_nvfp4_quantize_for_fa4` (native, flashinfer `fp4_quantize_sm100`) rather than writing a
fake quantizer, and label any FP8 routing variant separately depending on whether it uses
`torch._scaled_mm`-style native FP8 (confirmed working on this machine) or simulation.
