"""Emit artifacts/sparsefp4/env.json — the machine-readable Phase 0 environment record.

Run with the study interpreter:
    source artifacts/sparsefp4/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4/configs/write_env_json.py

Note: the SKILL references `python scripts/check_env.py --output artifacts/sparsefp4/env.json`,
but no `scripts/check_env.py` exists in this repo. The repo ships `collect_env.py` at
its root (human-readable, saved to env/collect_env.txt); this script supplies the
machine-readable half the SKILL asks for.
"""

import json
import os
import platform
import subprocess
import sys

import torch

REPO = "/home/ec2-user/FastVideo"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT = os.path.join(REPO, "artifacts/sparsefp4/env.json")


def sh(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=REPO, check=True).stdout.strip()
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    try:
        return __import__("importlib.metadata", fromlist=["version"]).version(name)
    except Exception:
        return None


def fa4_fp4_probe() -> dict[str, object]:
    """Native NVFP4 attention availability, recorded as evidence rather than assumed."""
    probe: dict[str, object] = {}
    try:
        from fastvideo.attention.backends.attn_qat_infer import (
            attn_qat_infer_receipt,
            is_attn_qat_infer_available,
        )
        probe["attn_qat_infer_receipt"] = attn_qat_infer_receipt()
        probe["attn_qat_infer_available"] = is_attn_qat_infer_available()
    except Exception as exc:
        probe["attn_qat_infer_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from fastvideo.attention.backends.flash_attn import _FA4_FP4_AVAILABLE, fa_version
        probe["fa_version"] = fa_version
        probe["fa4_fp4_available"] = _FA4_FP4_AVAILABLE
    except Exception as exc:
        probe["fa4_fp4_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def main() -> None:
    caps = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    probe = fa4_fp4_probe()

    snapshot = sh(
        "bash", "-lc", f"ls -d {os.environ.get('HF_HOME','')}/hub/models--"
        f"{MODEL_ID.replace('/', '--')}/snapshots/* 2>/dev/null | head -1")
    revision = os.path.basename(snapshot) if snapshot else None

    env = {
        "study": "sparsefp4-video-attention",
        "phase": "0",
        "recorded_at_utc": sh("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
        "interpreter": {
            "path":
            sys.executable,
            "activation_command":
            f"source {REPO}/artifacts/sparsefp4/configs/env.sh",
            "python_version":
            platform.python_version(),
            "venv":
            "/mnt/scratch/fv-venv",
            "venv_filesystem":
            "xfs on local instance-store NVMe /dev/nvme1n1 (3.5T), "
            "used because the root volume has <10 GB free",
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "arch_list": torch.cuda.get_arch_list(),
            "sm_100_in_arch_list": "sm_100" in torch.cuda.get_arch_list(),
            "cuda_available": torch.cuda.is_available(),
        },
        "gpu": {
            "count":
            torch.cuda.device_count(),
            "count_note":
            "torch.cuda.device_count() reflects CUDA_VISIBLE_DEVICES at record time; "
            "the host has 8x NVIDIA B200 (see nvidia_smi_gpu_count).",
            "nvidia_smi_gpu_count":
            len((sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") or "").splitlines()),
            "names":
            names,
            "device_capabilities":
            caps,
            "device_capability":
            caps[0] if caps else None,
            "driver_version": (sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
                               or "").splitlines()[:1],
            "total_memory_mib_per_gpu": (sh("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
                                         or "").splitlines(),
        },
        "cuda_toolkit": {
            "nvcc_on_path":
            sh("which", "nvcc"),
            "nvcc_version_raw":
            sh("nvcc", "--version"),
            "cuda_home":
            os.environ.get("CUDA_HOME"),
            "installed_during_phase0":
            True,
            "note":
            "Absent at Phase 0 start; installed as dnf 'cuda-toolkit-13-0' onto a bind "
            "mount backed by /mnt/scratch. Needed by flashinfer's JIT and available for "
            "Phase 4 kernel work.",
        },
        "fastvideo": {
            "repo": REPO,
            "git_commit": sh("git", "rev-parse", "HEAD"),
            "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_files": (sh("git", "status", "--short") or "").splitlines(),
            "version": pkg_version("fastvideo"),
            "install_mode": "editable (uv pip install -e '.[dev]', UV_TORCH_BACKEND=cu130)",
        },
        "model": {
            "id": MODEL_ID,
            "revision": revision,
            "local_snapshot": snapshot,
            "hf_home": os.environ.get("HF_HOME"),
            "default_height": 480,
            "default_width": 832,
            "default_num_frames": 81,
            "default_num_inference_steps": 50,
            "default_guidance_scale": 3.0,
            "default_seed": 1024,
        },
        "attention_stack": {
            "flash_attn_4_fork": "hao-ai-lab/flash-attention-fp4 @ fix/cutlass-dsl-4.5",
            "flash_attn_4_commit": "940bf7e511375ec160bc2d7188bef35915ded1e3",
            "flash_attn_4_version": pkg_version("flash-attn-4"),
            "nvidia_cutlass_dsl": pkg_version("nvidia-cutlass-dsl"),
            "quack_kernels": pkg_version("quack-kernels"),
            "flashinfer_python": pkg_version("flashinfer-python"),
            "apache_tvm_ffi": pkg_version("apache-tvm-ffi"),
            "fastvideo_kernel": pkg_version("fastvideo-kernel"),
            "flash_attn_2_installed": pkg_version("flash-attn") is not None,
            "required_env": {
                "FASTVIDEO_FA4": "1",
                "CUTE_DSL_ENABLE_TVM_FFI": "1",
                "FASTVIDEO_NVFP4_FA4": "1 to enable native NVFP4 Q/K attention (or nvfp4_fa4=True)",
            },
            "probe": probe,
        },
        "key_package_versions": {
            name: pkg_version(name)
            for name in ("transformers", "diffusers", "accelerate", "triton", "numpy", "huggingface-hub")
        },
        "storage": {
            "scratch_mount": "/mnt/scratch",
            "scratch_df": sh("bash", "-lc", "df -h /mnt/scratch | tail -1"),
            "root_df": sh("bash", "-lc", "df -h / | tail -1"),
        },
    }

    with open(OUT, "w") as fh:
        json.dump(env, fh, indent=2)
    print(f"wrote {OUT}")
    print(json.dumps(probe, indent=2))


if __name__ == "__main__":
    main()
