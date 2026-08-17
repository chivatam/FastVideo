"""Smoke test for the P3 backend: 5-step 480p generation, both fine precisions.

Usage: CUDA_VISIBLE_DEVICES=3 $FV_PYTHON p3_smoke.py --fine nvfp4
"""
import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", choices=["nvfp4", "bf16"], default="nvfp4")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", default="/mnt/nvme/scratch/sparsefp4_native/p3_smoke")
    args = parser.parse_args()

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_NATIVE_VSA_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_FINE"] = args.fine

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    torch.manual_seed(1234)
    gen = VideoGenerator.from_pretrained(
        model_id, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
        vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=True,
        VSA_sparsity=0.9)
    sp = SamplingParam.from_pretrained(model_id)
    sp.num_inference_steps = args.steps
    sp.seed = 1234
    sp.save_video = True
    sp.output_path = f"{args.out}_{args.fine}"
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers."
    sp.prompt = prompt
    t0 = time.time()
    gen.generate_video(prompt, sampling_param=sp)
    print(f"SMOKE_OK fine={args.fine} gen_s={time.time()-t0:.1f}")
    gen.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
