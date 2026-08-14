# SparseFP4 study — canonical environment for every phase.
#   source /home/ec2-user/FastVideo/artifacts/sparsefp4/configs/env.sh
#
# Chosen interpreter: /mnt/scratch/fv-venv/bin/python  (CPython 3.12.13, torch 2.12.0+cu130)
# /mnt/scratch is an xfs filesystem on local instance-store NVMe (/dev/nvme1n1, 3.5T).
# The root volume only has ~6 GB free, so the venv, HF cache, CUDA toolkit and
# CuTeDSL/flashinfer JIT caches all live on /mnt/scratch.

export FV_VENV=/mnt/scratch/fv-venv
export FV_PYTHON=/mnt/scratch/fv-venv/bin/python

# CUDA toolkit 13.0 (matches torch's cu130). Installed via dnf into a bind mount
# backed by /mnt/scratch/cuda/13.0 so it does not consume the small root volume.
# Required by flashinfer's JIT (it shells out to nvcc + ninja to build
# fp4_quantize_sm100, which the NVFP4 Q/K quantizer calls).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$FV_VENV/bin:$HOME/.local/bin:$PATH"

# Big-disk caches.
export HF_HOME=/mnt/scratch/hf-cache
export TMPDIR=/mnt/scratch/tmp
export UV_CACHE_DIR=/mnt/scratch/uv-cache

# FA4 / NVFP4 attention. FASTVIDEO_FA4=1 is mandatory here: the
# flash-attention-fp4 fork ships no compiled FlashAttention-2, so without the
# opt-in every dense attention path raises ImportError.
export FASTVIDEO_FA4=1
export CUTE_DSL_ENABLE_TVM_FFI=1

# Enable native NVFP4 Q/K attention per-run instead of globally:
#   export FASTVIDEO_NVFP4_FA4=1      # env-var switch on the FLASH_ATTN backend
# or pass nvfp4_fa4=True to VideoGenerator.from_pretrained.
# Backend override (see fastvideo/attention/AGENTS.md):
#   export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN | ATTN_QAT_INFER | VIDEO_SPARSE_ATTN | SDPA
