"""Run REAL Wan VSA inference with Phase-1 top-k overlap capture enabled.

One process per (prompt, resolution, seed) run so the capture directory and
the environment are fully isolated:

    PYTHONPATH=fastvideo-kernel/python python artifacts/vsa_overlap_phase1/run_capture.py \
        --resolution 720p --prompt-id p0 --seed 1024 \
        --capture-root /mnt/nvme/outputs/vsa_capture

`--no-capture` runs the identical generation without instrumentation and is
used for the capture-on/off output-identity sanity check (frames are hashed
in both modes via ``return_frames=True``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
VSA_SPARSITY = 0.9

# 81 frames -> latent T = (81-1)/4 + 1 = 21.
# 720p: H_lat = 720/8/2 = 45, W_lat = 1280/8/2 = 80  -> (21, 45, 80), 6*12*20 = 1440 blocks
# 480p: H_lat = 480/8/2 = 30, W_lat =  832/8/2 = 52  -> (21, 30, 52), 6*8*13  =  624 blocks
RESOLUTIONS = {
    "720p": {
        "height": 720,
        "width": 1280,
        "num_frames": 81
    },
    "480p": {
        "height": 480,
        "width": 832,
        "num_frames": 81
    },
}

PROMPTS = {
    "p0": ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
           "wide with interest. Soft natural light, warm cheerful tones, mid-shot, cinematic."),
    "p1": ("A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. The puddles "
           "reflect glowing signs advertising ramen, karaoke, and VR arcades. A woman in a "
           "translucent raincoat walks briskly with an LED umbrella. Steam rises from a street "
           "food cart, and a cat darts across the screen."),
    "p2": ("A majestic lion strides across the golden savanna, its powerful frame glistening "
           "under the warm afternoon sun. The tall grass ripples gently in the breeze. Low "
           "angle, steady tracking shot, cinematic."),
}


def frames_hash(frames) -> str:
    """Deterministic content hash of the generated frames (pre-encoding)."""
    import numpy as np
    h = hashlib.sha256()
    arr = np.asarray(frames)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="720p")
    ap.add_argument("--prompt-id", choices=PROMPTS, default="p0")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--capture-root", default="/mnt/nvme/outputs/vsa_capture")
    ap.add_argument("--no-capture", action="store_true", help="disable instrumentation (identity check)")
    ap.add_argument("--output-path", default="/mnt/nvme/outputs/vsa_capture_videos")
    args = ap.parse_args()

    run_id = f"{args.resolution}_{args.prompt_id}_seed{args.seed}"
    capture_dir = os.path.join(args.capture_root, run_id)

    # Env must be set BEFORE fastvideo import / worker spawn.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    if args.no_capture:
        os.environ.pop("FASTVIDEO_VSA_CAPTURE_OVERLAP", None)
    else:
        os.environ["FASTVIDEO_VSA_CAPTURE_OVERLAP"] = capture_dir
        os.makedirs(capture_dir, exist_ok=True)

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    generator = VideoGenerator.from_pretrained(
        MODEL,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=VSA_SPARSITY,
    )

    sampling_param = SamplingParam.from_pretrained(MODEL)
    res = RESOLUTIONS[args.resolution]
    sampling_param.height = res["height"]
    sampling_param.width = res["width"]
    sampling_param.num_frames = res["num_frames"]
    sampling_param.seed = args.seed

    t0 = time.perf_counter()
    result = generator.generate_video(
        PROMPTS[args.prompt_id],
        output_path=args.output_path,
        save_video=True,
        return_frames=True,
        sampling_param=sampling_param,
    )
    gen_time = time.perf_counter() - t0

    fhash = None
    if isinstance(result, dict) and result.get("frames") is not None:
        fhash = frames_hash(result["frames"])

    meta = {
        "model": MODEL,
        "prompt_id": args.prompt_id,
        "prompt": PROMPTS[args.prompt_id],
        "seed": args.seed,
        "resolution": args.resolution,
        "pixel_shape": res,
        "VSA_sparsity": VSA_SPARSITY,
        "vsa_tile_size": [4, 4, 4],
        "num_inference_steps": sampling_param.num_inference_steps,
        "guidance_scale": sampling_param.guidance_scale,
        "flattening_convention": "q_block_id = t_tile*(n_h*n_w) + h_tile*n_w + w_tile "
        "(t outer, w inner; from get_tile_partition_indices)",
        "capture_enabled": not args.no_capture,
        "frames_sha256": fhash,
        "generation_time_s": gen_time,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }
    meta_path = (os.path.join(capture_dir, "run_meta.json") if not args.no_capture else os.path.join(
        args.output_path, f"{run_id}_nocapture_meta.json"))
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[run_capture] {run_id} capture={not args.no_capture} "
          f"gen_time={gen_time:.1f}s frames_sha256={fhash}")
    generator.shutdown()


if __name__ == "__main__":
    main()
