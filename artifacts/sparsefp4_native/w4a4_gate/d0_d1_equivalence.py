"""D0-vs-D1 output equivalence check (same weights, compiled vs eager)."""
from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29640")
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

maybe_init_distributed_environment_and_model_parallel(1, 1)
device = torch.device("cuda:0")
fv_args = FastVideoArgs(model_path="checkpoints/dqvsa_T3/checkpoint-500/transformer",
                        dit_cpu_offload=False,
                        pipeline_config=PipelineConfig(dit_config=WanVideoConfig(),
                                                       dit_precision="bf16"))
fv_args.device = device
model = TransformerLoader().load(fv_args.model_path, fv_args).to(dtype=torch.bfloat16).eval()

t, h, w = 21, 60, 104
torch.manual_seed(7)
hidden = torch.randn(1, 16, t, h, w, device=device, dtype=torch.bfloat16)
text = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)
meta = SparseFP4VSA256FA4MetadataBuilder().build(
    current_timestep=0, raw_latent_shape=(t, h, w), patch_size=(1, 2, 2),
    VSA_sparsity=0.90, device=device, cache_tile_buf=False)
fb = ForwardBatch(data_type="dummy")


def fwd():
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=meta,
                                              forward_batch=fb):
        return model(hidden_states=hidden, encoder_hidden_states=text, timestep=timestep)


out_eager = fwd().float()
for i, blk in enumerate(model.blocks):
    model.blocks[i] = torch.compile(blk, mode="max-autotune-no-cudagraphs", dynamic=False)
out_compiled = fwd().float()

rel = (out_eager - out_compiled).norm() / out_eager.norm()
cos = torch.nn.functional.cosine_similarity(out_eager.flatten(), out_compiled.flatten(), dim=0)
print(f"D0 vs D1: rel-L2 {rel.item():.3e}  cosine {cos.item():.6f}  "
      f"max-abs {(out_eager - out_compiled).abs().max().item():.3e}  "
      f"finite={torch.isfinite(out_compiled).all().item()}")
cleanup_dist_env_and_memory()
