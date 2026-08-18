"""W4A4 gate Phase 4/5: swap target linears for native NVFP4 W4A4 GEMMs
(flashinfer mm_fp4) inside the real D1 transformer and measure forward wall.

Ladder: --w4a4 ffn | ffn+o | all   (W1 / W2 / W3); omit for W0 baseline.

  w4a4_forward_bench.py --res 480 --w4a4 ffn --compile --out w1_480.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29650")
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

RES = {"480": (21, 60, 104), "720": (21, 90, 160)}
FP4_MAX = 448.0 * 6.0


class W4A4Linear(torch.nn.Module):
    """Native NVFP4 W4A4 linear: production weight packing (offline) +
    production activation quantization (per call) + mm_fp4 block-scaled GEMM.
    No dequant-to-BF16 GEMM anywhere."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        from flashinfer import SfLayout, nvfp4_quantize
        w = weight.detach().to(torch.bfloat16).cuda()  # [n, k]
        self.n, self.k = w.shape
        w_gs = torch.tensor([FP4_MAX / w.float().abs().max().item()], device=w.device)
        # weight path: cudnn backend consumes the UNshuffled 128x4 SF layout
        w_fp4, w_sf = nvfp4_quantize(w, w_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        self.register_buffer("w_fp4_t", w_fp4.t())
        self.register_buffer("w_sf_t", w_sf.t())
        self.register_buffer("w_gs", w_gs)
        self.register_buffer("bias", bias.detach().to(torch.bfloat16).cuda()
                             if bias is not None else None)
        # activation global scale calibrated on first call, then frozen
        self.register_buffer("a_gs", torch.zeros(1, device=w.device))

    @torch._dynamo.disable
    def forward(self, x: torch.Tensor):
        from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
        shp = x.shape
        x2 = x.reshape(-1, self.k)
        if float(self.a_gs) == 0.0:
            self.a_gs = torch.tensor([FP4_MAX / (x2.float().abs().max().item() + 1e-8)],
                                     device=x2.device)
        a_fp4, a_sf = nvfp4_quantize(x2.to(torch.bfloat16), self.a_gs,
                                     sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        alpha = (1.0 / (self.a_gs * self.w_gs)).float()
        y = mm_fp4(a_fp4, self.w_fp4_t, a_sf, self.w_sf_t, alpha,
                   torch.bfloat16, None, 16, False, "cudnn", enable_pdl=False)
        if self.bias is not None:
            y = y + self.bias
        # FastVideo ReplicatedLinear returns (out, bias_placeholder)
        return y.reshape(*shp[:-1], self.n), None


def select_names(model: torch.nn.Module, mode: str) -> list[str]:
    names = []
    for name, mod in model.named_modules():
        if not mod.__class__.__name__.endswith("Linear"):
            continue
        is_ffn = ".ffn" in name
        is_o = name.endswith("to_out") and ".attn2" not in name
        is_qkv = name.endswith(("to_q", "to_k", "to_v")) and ".attn2" not in name
        if mode == "ffn" and is_ffn:
            names.append(name)
        elif mode == "ffn+o" and (is_ffn or is_o):
            names.append(name)
        elif mode == "all" and (is_ffn or is_o or is_qkv):
            names.append(name)
    return names


def ckpt_key(fv_name: str) -> str:
    """FastVideo module name -> diffusers-layout checkpoint key prefix."""
    import re
    m = re.match(r"blocks\.(\d+)\.(.*)", fv_name)
    assert m, fv_name
    i, rest = m.group(1), m.group(2)
    table = {
        "to_q": f"blocks.{i}.attn1.to_q", "to_k": f"blocks.{i}.attn1.to_k",
        "to_v": f"blocks.{i}.attn1.to_v", "to_out": f"blocks.{i}.attn1.to_out.0",
        "ffn.fc_in": f"blocks.{i}.ffn.net.0.proj", "ffn.fc_out": f"blocks.{i}.ffn.net.2",
        "attn2.to_q": f"blocks.{i}.attn2.to_q", "attn2.to_k": f"blocks.{i}.attn2.to_k",
        "attn2.to_v": f"blocks.{i}.attn2.to_v", "attn2.to_out": f"blocks.{i}.attn2.to_out.0",
    }
    return table[rest]


def swap(model: torch.nn.Module, names: list[str], safetensors_path: str) -> int:
    from safetensors import safe_open
    count = 0
    with safe_open(safetensors_path, "pt") as f:
        for name in names:
            key = ckpt_key(name)
            weight = f.get_tensor(f"{key}.weight")
            bias = f.get_tensor(f"{key}.bias") if f"{key}.bias" in f.keys() else None
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], W4A4Linear(weight, bias))
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=("480", "720"), required=True)
    ap.add_argument("--w4a4", choices=("none", "ffn", "ffn+o", "all"), default="none")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model-path", default="checkpoints/dqvsa_T3/checkpoint-500/transformer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    device = torch.device("cuda:0")
    fv_args = FastVideoArgs(model_path=args.model_path, dit_cpu_offload=False,
                            pipeline_config=PipelineConfig(dit_config=WanVideoConfig(),
                                                           dit_precision="bf16"))
    fv_args.device = device
    model = TransformerLoader().load(args.model_path, fv_args).to(dtype=torch.bfloat16).eval()

    n_swapped = 0
    if args.w4a4 != "none":
        st_path = os.path.join(args.model_path, "diffusion_pytorch_model.safetensors")
        n_swapped = swap(model, select_names(model, args.w4a4), st_path)
    if args.compile:
        mode = os.environ.get("W4A4_COMPILE_MODE", "default")
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = torch.compile(blk, mode=mode, dynamic=False)

    t, h, w = RES[args.res]
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

    for _ in range(3):
        out = fwd()
    torch.cuda.synchronize()
    walls = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000)
    walls.sort()
    result = dict(res=args.res, w4a4=args.w4a4, compile=args.compile,
                  n_swapped=n_swapped,
                  wall_ms_median=walls[len(walls) // 2], wall_ms_all=walls,
                  out_finite=bool(torch.isfinite(out).all().item()),
                  peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "wall_ms_all"}, indent=2))
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
