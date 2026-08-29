from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.adaptive_vsa_fp4.scripts.worker import MODEL_REVISIONS


def _run(command: list[str], *, cwd: Path | None = None, required: bool = True) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        if required:
            raise
        return None


def _git_commit(path: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=path, required=False)


def _python_environment(python: Path) -> dict[str, Any]:
    script = r"""
import importlib.metadata as metadata
import json
import platform
import torch

names = [
    "fastvideo",
    "fastvideo-kernel",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "triton",
    "flashinfer-python",
    "quack-kernels",
    "torch-c-dlpack-ext",
    "decord",
    "pandas",
    "pyarrow",
]
installed = {
    dist.metadata["Name"].lower(): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "packages": {name: installed.get(name.lower()) for name in names},
}))
"""
    return json.loads(_run([str(python), "-c", script]) or "{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--main-python",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/venvs/adaptive-vsa-fp4/bin/python"),
    )
    parser.add_argument(
        "--fp4-python",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/venvs/adaptive-vsa-fp4-dense-nvfp4/bin/python"),
    )
    parser.add_argument(
        "--fp4-repo",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/src/flash-attention-fp4"),
    )
    parser.add_argument(
        "--t2v-compbench-repo",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/src/T2V-CompBench"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    vbench = repo / "fastvideo/third_party/eval/vbench"
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": _run(["uname", "-a"]),
        "architecture": platform.machine(),
        "fastvideo": {
            "repo": str(repo),
            "commit": _git_commit(repo),
            "branch": _run(["git", "branch", "--show-current"], cwd=repo),
            "remote": _run(["git", "remote", "get-url", "origin"], cwd=repo),
        },
        "fastvideo_kernel": {
            "source": "FastVideo subtree",
            "fastvideo_commit": _git_commit(repo),
            "cutlass_submodule": _git_commit(repo / "fastvideo-kernel/include/cutlass"),
            "tilelang_kernel_submodule": _git_commit(repo / "fastvideo-kernel/include/tk"),
        },
        "attention_sources": {
            "flash_attention_4_commit": "14c377950125c70b7a9dabf9c561fca53715ac7d",
            "flash_attention_fp4_repo": str(args.fp4_repo),
            "flash_attention_fp4_commit": _git_commit(args.fp4_repo),
        },
        "cuda": {
            "nvcc": _run(["/usr/local/cuda-13.0/bin/nvcc", "--version"]),
            "driver_and_gpus": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version,pstate,"
                    "clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "topology": _run(["nvidia-smi", "topo", "-m"]),
            "gpu_count": int(_run(["nvidia-smi", "--list-gpus"]).count("\n") + 1),
        },
        "environments": {
            "vsa_and_evaluation": _python_environment(args.main_python),
            "dense_nvfp4": _python_environment(args.fp4_python),
        },
        "models": MODEL_REVISIONS,
        "datasets": {
            "vbench_repo": str(vbench),
            "vbench_revision": _git_commit(vbench),
            "prompt_manifest": "artifacts/adaptive_vsa_fp4/phase0/vbench_subject_consistency_prompts.json",
            "prompt_count": 72,
            "t2v_compbench_repo": str(args.t2v_compbench_repo),
            "t2v_compbench_revision": _git_commit(args.t2v_compbench_repo),
        },
        "storage": {
            "artifact_root": str((repo / "artifacts/adaptive_vsa_fp4").resolve()),
            "hf_cache": os.environ.get("HF_HUB_CACHE", "/mnt/fastvideo-gpu0/hf-cache/hub"),
            "triton_cache": os.environ.get("TRITON_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/triton"),
            "cute_dsl_cache": os.environ.get("CUTE_DSL_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/cute"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
