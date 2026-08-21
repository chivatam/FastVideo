"""Phase-0 (certified softmax): capture real Q/K + selected indices from Wan.

One process per (prompt, resolution). Layer/step-filtered raw Q/K capture
(FASTVIDEO_VSA_CAPTURE_QK) via fastvideo_kernel.vsa_capture; execution is
unchanged (opt-in hook, Phase-1 identity check covers the mechanism).

    PYTHONPATH=fastvideo-kernel/python python \
        artifacts/vsa_certified_softmax_phase0/capture_or_extract.py \
        --resolution 720p --prompt-id p0 --qk-root /mnt/nvme/outputs/vsa_qk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "artifacts", "vsa_overlap_phase1"))

CAPTURE_LAYERS = "0,3,14,15,27,29"  # early x2, middle x2, late x2 (30-layer Wan1.3B)
CAPTURE_STEPS = "0,1,2"  # 3-step DMD schedule: early / middle / late


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", choices=("720p", "480p"), default="720p")
    ap.add_argument("--prompt-id", choices=("p0", "p1", "p2"), default="p0")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--qk-root", default="/mnt/nvme/outputs/vsa_qk")
    args = ap.parse_args()

    from run_capture import MODEL, PROMPTS, RESOLUTIONS, VSA_SPARSITY

    run_id = f"{args.resolution}_{args.prompt_id}_seed{args.seed}"
    qk_dir = os.path.join(args.qk_root, run_id)
    os.makedirs(qk_dir, exist_ok=True)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_VSA_CAPTURE_QK"] = qk_dir
    os.environ["FASTVIDEO_VSA_CAPTURE_LAYERS"] = CAPTURE_LAYERS
    os.environ["FASTVIDEO_VSA_CAPTURE_STEPS"] = CAPTURE_STEPS
    os.environ.pop("FASTVIDEO_VSA_CAPTURE_OVERLAP", None)

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    generator = VideoGenerator.from_pretrained(MODEL,
                                               num_gpus=1,
                                               use_fsdp_inference=False,
                                               text_encoder_cpu_offload=True,
                                               pin_cpu_memory=True,
                                               dit_cpu_offload=False,
                                               vae_cpu_offload=False,
                                               VSA_sparsity=VSA_SPARSITY)
    sp = SamplingParam.from_pretrained(MODEL)
    res = RESOLUTIONS[args.resolution]
    sp.height, sp.width, sp.num_frames = res["height"], res["width"], res["num_frames"]
    sp.seed = args.seed

    t0 = time.perf_counter()
    generator.generate_video(PROMPTS[args.prompt_id],
                             output_path="/mnt/nvme/outputs/vsa_qk_videos",
                             save_video=True,
                             sampling_param=sp)
    meta = {
        "model": MODEL,
        "prompt_id": args.prompt_id,
        "seed": args.seed,
        "resolution": args.resolution,
        "VSA_sparsity": VSA_SPARSITY,
        "capture_layers": CAPTURE_LAYERS,
        "capture_steps": CAPTURE_STEPS,
        "gen_time_s": time.perf_counter() - t0
    }
    with open(os.path.join(qk_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[qk-capture] {run_id} done in {meta['gen_time_s']:.1f}s")
    generator.shutdown()


if __name__ == "__main__":
    main()
