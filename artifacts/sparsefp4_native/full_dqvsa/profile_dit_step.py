"""Profile one native-P4 DiT forward (standalone transformer) and dump a
chrome trace for the runtime breakdown. Mirrors the serving operator:
SPARSEFP4_VSA256_FA4_ATTN, fine=nvfp4, VSA sparsity 0.90.

  profile_dit_step.py --res 480 --out /tmp/trace480.json
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29610")
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_VSA256_FA4_ATTN"
os.environ["FASTVIDEO_FA4"] = "1"
os.environ["FASTVIDEO_SPARSEFP4_FINE"] = "nvfp4"

import torch

from fastvideo.attention.backends.sparsefp4_vsa256_fa4 import SparseFP4VSA256FA4MetadataBuilder
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   maybe_init_distributed_environment_and_model_parallel)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

RES = {  # (latent T,H,W)
    "480": (21, 60, 104),
    "720": (21, 90, 160),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=("480", "720"), required=True)
    ap.add_argument("--model-path", default="checkpoints/dqvsa_T3/checkpoint-500/transformer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = torch.device("cuda:0")
    fv_args = FastVideoArgs(model_path=args.model_path, dit_cpu_offload=False,
                            pipeline_config=PipelineConfig(dit_config=WanVideoConfig(),
                                                           dit_precision="bf16"))
    fv_args.device = device
    model = TransformerLoader().load(args.model_path, fv_args).to(dtype=torch.bfloat16).eval()
    if args.compile:
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = torch.compile(blk, mode="max-autotune-no-cudagraphs",
                                            dynamic=False)

    t, h, w = RES[args.res]
    hidden = torch.randn(1, 16, t, h, w, device=device, dtype=torch.bfloat16)
    text = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)
    meta = SparseFP4VSA256FA4MetadataBuilder().build(
        current_timestep=0, raw_latent_shape=(t, h, w), patch_size=(1, 2, 2),
        VSA_sparsity=0.90, device=device, cache_tile_buf=False)
    batch = ForwardBatch(data_type="dummy")

    def fwd():
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=meta,
                                                  forward_batch=batch):
            model(hidden_states=hidden, encoder_hidden_states=text, timestep=timestep)

    for _ in range(2):  # warmup / JIT
        fwd()
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.reps):
            fwd()
        torch.cuda.synchronize()
    prof.export_chrome_trace(args.out)
    print(f"trace written to {args.out}")
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
