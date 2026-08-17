"""Emit artifacts/sparsefp4_followup/env.json — the machine-readable F0 environment record.

Run with the follow-up study interpreter:
    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/write_env_json.py

Unlike study 1's `write_env_json.py` this also emits an explicit *drift receipt*
against `artifacts/sparsefp4/env/pip-freeze.txt`, because the follow-up runs on a
rebuilt host: the original `/mnt/scratch` instance-store filesystem did not
survive an instance stop/start, so the venv, the 27 GB model snapshot and the
CUDA toolkit were all reinstalled. Whether the reinstall reproduced study 1's
dependency set is a load-bearing fact for every paired comparison against
study 1's numbers, so it is measured rather than asserted.
"""

import json
import os
import platform
import re
import subprocess
import sys
from typing import Any

import torch

REPO = "/home/ec2-user/FastVideo"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STUDY1_FREEZE = os.path.join(REPO, "artifacts/sparsefp4/env/pip-freeze.txt")
FOLLOWUP_FREEZE = os.path.join(REPO, "artifacts/sparsefp4_followup/env/pip-freeze.txt")
OUT = os.path.join(REPO, "artifacts/sparsefp4_followup/env.json")

# Pins that study 1 declared load-bearing (PHASE0.md §8 "Dependency pins — do not drift").
CRITICAL_PINS = {
    "torch": "2.12.0+cu130",
    "nvidia-cutlass-dsl": "4.5.3",
    "quack-kernels": "0.5.0",
    "flashinfer-python": "0.6.17",
    "apache-tvm-ffi": "0.1.13.post3",
    "fastvideo-kernel": "0.3.2",
}
FA4_COMMIT = "940bf7e511375ec160bc2d7188bef35915ded1e3"


def sh(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=REPO, check=True).stdout.strip()
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    try:
        return __import__("importlib.metadata", fromlist=["version"]).version(name)
    except Exception:
        return None


def parse_freeze(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Using Python"):
                continue
            m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:==|\s@\s)\s*(.+)$", line)
            if m:
                out[m.group(1).lower()] = m.group(2)
    return out


def freeze_drift() -> dict[str, object]:
    """Package-by-package comparison against study 1's frozen environment."""
    a, b = parse_freeze(STUDY1_FREEZE), parse_freeze(FOLLOWUP_FREEZE)
    if not a or not b:
        return {"available": False, "reason": "one or both freeze files missing"}
    changed = {k: {"study1": a[k], "followup": b[k]} for k in sorted(a.keys() & b.keys()) if a[k] != b[k]}
    critical_ok = {}
    for name, want in CRITICAL_PINS.items():
        got = pkg_version(name)
        critical_ok[name] = {"required": want, "installed": got, "match": got == want}
    return {
        "available": True,
        "study1_freeze": STUDY1_FREEZE,
        "followup_freeze": FOLLOWUP_FREEZE,
        "n_packages_study1": len(a),
        "n_packages_followup": len(b),
        "only_in_study1": sorted(a.keys() - b.keys()),
        "only_in_followup": sorted(b.keys() - a.keys()),
        "version_changed": changed,
        "n_version_changed": len(changed),
        "critical_pins": critical_ok,
        "all_critical_pins_match": all(v["match"] for v in critical_ok.values()),
    }


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


def vsa_probe() -> dict[str, object]:
    """VSA availability — the follow-up's F2 phase depends on it, so record it up front."""
    probe: dict[str, object] = {}
    try:
        from fastvideo_kernel import vsa_utils
        probe["vsa_tile_size"] = list(getattr(vsa_utils, "VSA_TILE_SIZE", ()))
        probe["vsa_utils_file"] = vsa_utils.__file__
    except Exception as exc:
        probe["vsa_utils_error"] = f"{type(exc).__name__}: {exc}"
    for fn in ("video_sparse_attn", "video_sparse_attn_bshd"):
        try:
            mod = __import__("fastvideo_kernel", fromlist=[fn])
            probe[f"{fn}_importable"] = hasattr(mod, fn)
        except Exception as exc:
            probe[f"{fn}_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def main() -> None:
    caps = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    snapshot = sh(
        "bash", "-lc", f"ls -d {os.environ.get('HF_HOME','')}/hub/models--"
        f"{MODEL_ID.replace('/', '--')}/snapshots/* 2>/dev/null | head -1")
    revision = os.path.basename(snapshot) if snapshot else None

    env: dict[str, Any] = {
        "study":
        "sparsefp4-paper-validation",
        "phase":
        "F0",
        "parent_study": {
            "artifacts": "artifacts/sparsefp4/",
            "report": "artifacts/sparsefp4/REPORT.md",
            "git_commit_at_study1_report_time": "8208536cd1db7a1d32b68aaa6a679953ae23ab8b",
            "branch_at_study1": "exp/sparsefp4-mask-stability",
        },
        "recorded_at_utc":
        sh("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
        "host_rebuild_note":
        "The host was stopped/restarted between study 1 and this follow-up. Study 1's "
        "/mnt/scratch (xfs on /dev/nvme1n1) is GONE, exactly as PHASE0.md §3.1 warned. "
        "The eight instance-store NVMes are now a 28 TB md RAID at /mnt/nvme, and the "
        "repo lives at /mnt/nvme/FastVideo (/home/ec2-user/FastVideo is a symlink). The "
        "venv, CUDA 13.0 toolkit and the 27 GB model snapshot were rebuilt from "
        "PHASE0.md §3. Code and artifacts survived because they were committed.",
        "interpreter": {
            "path": sys.executable,
            "activation_command": f"source {REPO}/artifacts/sparsefp4_followup/configs/env.sh",
            "python_version": platform.python_version(),
            "python_version_study1": "3.12.13",
            "venv": os.environ.get("FV_VENV"),
            "venv_filesystem": sh("bash", "-lc", "df -hT /mnt/nvme | tail -1"),
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
            "note":
            "Reinstalled for this follow-up as dnf 'cuda-toolkit-13-0' onto a bind mount "
            "backed by /mnt/nvme/scratch/cuda/13.0. Needed at runtime by flashinfer's JIT "
            "(fp4_quantize_sm100 = the NVFP4 Q/K quantizer). Bind mount is not in fstab.",
        },
        "fastvideo": {
            "repo": REPO,
            "repo_realpath": os.path.realpath(REPO),
            "git_commit": sh("git", "rev-parse", "HEAD"),
            "git_short": sh("git", "rev-parse", "--short=7", "HEAD"),
            "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_files": (sh("git", "status", "--short") or "").splitlines(),
            "version": pkg_version("fastvideo"),
            "install_mode": "editable (uv pip install -e '.[dev]', UV_TORCH_BACKEND=cu130)",
        },
        "model": {
            "id": MODEL_ID,
            "revision": revision,
            "revision_study1": "0fad780a534b6463e45facd96134c9f345acfa5b",
            "revision_matches_study1": revision == "0fad780a534b6463e45facd96134c9f345acfa5b",
            "local_snapshot": snapshot,
            "hf_home": os.environ.get("HF_HOME"),
            "default_height": 480,
            "default_width": 832,
            "default_num_frames": 81,
            "default_num_inference_steps": 50,
            "default_guidance_scale": 3.0,
        },
        "attention_stack": {
            "flash_attn_4_fork": "hao-ai-lab/flash-attention-fp4 @ fix/cutlass-dsl-4.5",
            "flash_attn_4_commit_required": FA4_COMMIT,
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
                "FASTVIDEO_NVFP4_FA4": "1 to enable native NVFP4 Q/K attention",
            },
            "probe": fa4_fp4_probe(),
            "vsa": vsa_probe(),
        },
        "key_package_versions": {
            name: pkg_version(name)
            for name in ("transformers", "diffusers", "accelerate", "triton", "numpy", "huggingface-hub", "scipy")
        },
        "environment_drift_vs_study1":
        freeze_drift(),
        "storage": {
            "scratch_mount": os.environ.get("FV_SCRATCH"),
            "scratch_df": sh("bash", "-lc", "df -h /mnt/nvme | tail -1"),
            "root_df": sh("bash", "-lc", "df -h / | tail -1"),
            "raw_root": os.environ.get("FV_RAW_ROOT"),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(env, fh, indent=2)
    print(f"wrote {OUT}")
    drift = env["environment_drift_vs_study1"]
    print(
        json.dumps(
            {
                "attention_probe": env["attention_stack"]["probe"],
                "vsa": env["attention_stack"]["vsa"],
                "all_critical_pins_match": drift.get("all_critical_pins_match"),
                "n_version_changed": drift.get("n_version_changed"),
                "version_changed": drift.get("version_changed"),
                "model_revision_matches_study1": env["model"]["revision_matches_study1"],
            },
            indent=2))


if __name__ == "__main__":
    main()
