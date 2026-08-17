"""Priority-2 capture: QKV + exact-10% VSA256 masks from a genuine P4G
(sparse BF16 fine) trajectory, at 480p and/or 720p.

    CUDA_VISIBLE_DEVICES=1 $FV_PYTHON c5b_capture256_run.py --run-id cap256-480 \
        --height 480 --width 832
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15, 22, 29])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[0, 12, 25, 37, 49])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sparsity", type=float, default=0.90)
    ap.add_argument("--prompt", default=(
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."))
    ap.add_argument("--raw-root", type=Path,
                    default=Path("/mnt/nvme/scratch/sparsefp4_native"))
    args = ap.parse_args()

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"out_dir": str(out_dir / "cells"), "layers": args.layers,
           "timesteps": args.timesteps, "cfg_branches": ["positive"]}
    cfg_path = out_dir / "capture256_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_VSA256_FA4_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_FINE"] = "bf16"   # genuine sparse-BF16 trajectory
    os.environ["FASTVIDEO_SPARSEFP4_CAPTURE256"] = str(cfg_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    torch.manual_seed(args.seed)
    gen = VideoGenerator.from_pretrained(
        MODEL_ID, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=True,
        VSA_sparsity=args.sparsity)
    sp = SamplingParam.from_pretrained(MODEL_ID)
    sp.num_inference_steps = args.steps
    sp.seed = args.seed
    sp.height = args.height
    sp.width = args.width
    sp.save_video = False
    sp.return_frames = False
    sp.prompt = args.prompt
    t0 = time.time()
    gen.generate_video(args.prompt, sampling_param=sp)
    cells = sorted((out_dir / "cells").glob("cell_*.pt"))
    print(json.dumps({"cells_written": len(cells), "gen_s": round(time.time() - t0, 1)}))
    gen.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
