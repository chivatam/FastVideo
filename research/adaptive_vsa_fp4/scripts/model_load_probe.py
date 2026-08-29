from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import time
from pathlib import Path

from research.adaptive_vsa_fp4.scripts.worker import MODEL_REVISIONS, _load_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REVISIONS)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dense_bf16_fa4", "vsa_bf16", "dense_nvfp4_fa4", "sim_vsa_nvfp4"],
    )
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)
    os.environ.setdefault("HF_HOME", "/mnt/fastvideo-gpu0/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/fastvideo-gpu0/hf-cache/hub")
    os.environ.setdefault("TRITON_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/triton")
    os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/cute")

    payload = {
        "model": args.model,
        "model_revision": MODEL_REVISIONS[args.model],
        "sparsity": args.sparsity,
    }
    started = time.time()
    status = {
        "model": args.model,
        "revision": payload["model_revision"],
        "mode": args.mode,
        "sparsity": args.sparsity,
        "status": "running",
    }
    generator = None
    try:
        generator = _load_generator(payload, args.mode)
        status.update(
            status="ok",
            load_seconds=time.time() - started,
            fastvideo_args_model_path=generator.fastvideo_args.model_path,
        )
    except BaseException as exc:
        status.update(status="failed", load_seconds=time.time() - started, error=repr(exc))
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
