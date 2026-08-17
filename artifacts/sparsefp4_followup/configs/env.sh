# SparseFP4 FOLLOW-UP study — canonical environment for every follow-up phase.
#   source /home/ec2-user/FastVideo/artifacts/sparsefp4_followup/configs/env.sh
#
# WHY THIS FILE EXISTS SEPARATELY FROM artifacts/sparsefp4/configs/env.sh:
# the original study's scratch filesystem (/mnt/scratch, instance-store NVMe on
# /dev/nvme1n1) did NOT survive the instance stop/start between the two studies,
# exactly as PHASE0.md §3.1 warned. On the rebuilt host the eight instance-store
# NVMes are assembled as a 28 TB RAID array mounted at /mnt/nvme, and the repo
# itself now lives at /mnt/nvme/FastVideo (/home/ec2-user/FastVideo is a symlink).
# The original env.sh is left untouched as the historical record of study 1.
#
# Dependency pins are IDENTICAL to study 1 (verified package-by-package):
#   Python 3.12.14 · torch 2.12.0+cu130 · fastvideo-kernel 0.3.2
#   flash-attn-4 @ 940bf7e511375ec160bc2d7188bef35915ded1e3 (fix/cutlass-dsl-4.5)
#   nvidia-cutlass-dsl 4.5.3 · quack-kernels 0.5.0 · flashinfer-python 0.6.17
#   apache-tvm-ffi 0.1.13.post3

export FV_SCRATCH=/mnt/nvme/scratch
export FV_VENV="$FV_SCRATCH/fv-venv"
export FV_PYTHON="$FV_VENV/bin/python"

# CUDA toolkit 13.0 (matches torch's cu130). Bind-mounted off the 25 GB root
# volume. Required at RUNTIME by flashinfer's JIT, which shells out to nvcc +
# ninja to build fp4_quantize_sm100 — the NVFP4 Q/K quantizer.
#   sudo mkdir -p /usr/local/cuda-13.0
#   sudo mount --bind /mnt/nvme/scratch/cuda/13.0 /usr/local/cuda-13.0
# The bind mount is NOT in fstab and must be re-created after every reboot.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$FV_VENV/bin:$HOME/.local/bin:$PATH"

# Big-disk caches (root volume has <10 GB free).
export HF_HOME="$FV_SCRATCH/hf-cache"
export TMPDIR="$FV_SCRATCH/tmp"
export UV_CACHE_DIR="$FV_SCRATCH/uv-cache"

# FA4 / NVFP4 attention. FASTVIDEO_FA4=1 is mandatory here: the
# flash-attention-fp4 fork ships no compiled FlashAttention-2, so without the
# opt-in every dense attention path raises ImportError.
export FASTVIDEO_FA4=1
export CUTE_DSL_ENABLE_TVM_FFI=1

# Where bulky raw records go (never the root volume).
export FV_RAW_ROOT="$FV_SCRATCH/sparsefp4_followup"

# Per-run switches — do NOT set these globally:
#   export FASTVIDEO_NVFP4_FA4=1          # native NVFP4 Q/K on FLASH_ATTN
#   export FASTVIDEO_ATTENTION_BACKEND=…  # FLASH_ATTN | VIDEO_SPARSE_ATTN | ...
