"""C5 capture driver: run Wan2.1 under genuine VSA and dump QKV + frozen masks.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=1 "$FV_PYTHON" artifacts/sparsefp4_native/configs/c5_capture_run.py \
        --run-id c5-capture --vsa-sparsity 0.90
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_LAYERS = (0, 7, 15, 22, 29)
DEFAULT_TIMESTEPS = (0, 12, 25, 37, 49)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", default=(
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--cfg-branches", nargs="+", default=["positive"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--vsa-sparsity", type=float, default=0.90)
    parser.add_argument("--raw-root", type=Path,
                        default=Path(os.environ.get("FV_RAW_ROOT_NATIVE",
                                                    "/mnt/nvme/scratch/sparsefp4_native")))
    args = parser.parse_args()

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    capture_config = {
        "out_dir": str(out_dir / "cells"),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "cfg_branches": list(args.cfg_branches),
    }
    config_path = out_dir / "capture_config.json"
    config_path.write_text(json.dumps(capture_config, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                              text=True).strip(),
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": 3.0,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "vsa_sparsity": args.vsa_sparsity,
        "capture": capture_config,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                             encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_CAPTURE_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_CAPTURE"] = str(config_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    torch.manual_seed(args.seed)
    started = time.time()
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        VSA_sparsity=args.vsa_sparsity,
    )
    load_s = time.time() - started

    sp = SamplingParam.from_pretrained(MODEL_ID)
    sp.num_inference_steps = args.steps
    sp.seed = args.seed
    sp.height = args.height
    sp.width = args.width
    sp.num_frames = args.num_frames
    sp.save_video = False
    sp.return_frames = False
    sp.prompt = args.prompt

    gen_started = time.time()
    generator.generate_video(args.prompt, sampling_param=sp)
    gen_s = time.time() - gen_started

    cells = sorted((out_dir / "cells").glob("cell_*.pt"))
    summary = {
        "run_id": args.run_id,
        "cells_written": len(cells),
        "cells": [c.name for c in cells],
        "model_load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
