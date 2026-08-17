"""Paper-scale paired generation: one process = one arm + one prompt shard,
model loaded once, VBench prompts (7 scorable dims), fixed seed.

    paper_run.py --arm P4 --shard 0 --num-shards 8 --run-id paper-s090
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]
ARMS = {
    "P0": dict(backend="FLASH_ATTN", nvfp4=False, fine=None, sparse=False),
    "P1": dict(backend="FLASH_ATTN", nvfp4=True, fine=None, sparse=False),
    "P2": dict(backend="VIDEO_SPARSE_ATTN", nvfp4=False, fine=None, sparse=True),
    "P4G": dict(backend="SPARSEFP4_VSA256_FA4_ATTN", nvfp4=False, fine="bf16", sparse=True),
    "P4": dict(backend="SPARSEFP4_VSA256_FA4_ATTN", nvfp4=False, fine="nvfp4", sparse=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--run-id", default="paper-s090")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sparsity", type=float, default=0.90)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/mnt/nvme/scratch/sparsefp4_native/paper_videos"))
    args = ap.parse_args()

    spec = ARMS[args.arm]
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = spec["backend"]
    os.environ["FASTVIDEO_FA4"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if spec["nvfp4"]:
        os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
    else:
        os.environ.pop("FASTVIDEO_NVFP4_FA4", None)
    if spec["fine"]:
        os.environ["FASTVIDEO_SPARSEFP4_FINE"] = spec["fine"]

    import numpy as np
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.eval.datasets.vbench import VBenchPromptDataset

    prompts = list(VBenchPromptDataset(dimensions=DIMS))
    mine = [(i, p) for i, p in enumerate(prompts) if i % args.num_shards == args.shard]
    out_dir = args.out_root / args.run_id / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
                       vae_cpu_offload=False, text_encoder_cpu_offload=True,
                       pin_cpu_memory=True)
    if spec["sparse"]:
        init_kwargs["VSA_sparsity"] = args.sparsity
    gen = VideoGenerator.from_pretrained(MODEL_ID, **init_kwargs)

    sp = SamplingParam.from_pretrained(MODEL_ID)
    sp.num_inference_steps = args.steps
    sp.save_video = False
    sp.return_frames = True

    manifest = []
    for idx, entry in mine:
        tag = f"v{idx:04d}"
        npy = out_dir / f"{tag}.f16.npy"
        if npy.is_file():
            continue
        prompt = entry["prompt"]
        sp.seed = args.seed
        sp.prompt = prompt
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        result = gen.generate_video(prompt, sampling_param=sp)
        samples = result.get("samples")
        arr = samples.detach().to(torch.float16).cpu().numpy()
        np.save(npy, arr)
        manifest.append(dict(idx=idx, prompt=prompt, dimensions=entry["dimensions"],
                             gen_s=round(time.perf_counter() - t0, 1)))
        (out_dir / f"manifest_shard{args.shard}.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        del result, samples, arr
        gc.collect()
        print(f"[{args.arm} shard{args.shard}] {tag} done", flush=True)

    print(f"SHARD_DONE arm={args.arm} shard={args.shard} n={len(mine)}")
    gen.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
