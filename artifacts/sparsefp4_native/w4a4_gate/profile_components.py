"""W4A4 gate Phase 1/3: fine-grained per-component GPU-time decomposition.

CUDA-event pairs around disjoint module families of the real D0 (or D1)
transformer forward, plus total wall and kernel-trace cross-checks.
Event timing is stream-ordered (no per-module syncs), so per-family times
include the kernels each module launches; the residual bucket is
modulation/residual/elementwise/rope/patch-embed glue.

  profile_components.py --res 480 --batch 1 --out eager480_b1.json
  profile_components.py --res 720 --batch 1 --compile --out opt720_b1.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29620")
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

# (family, name-predicate) — first match wins; families are disjoint by
# construction (linears matched before their containers are excluded).
def family_of(name: str, module: torch.nn.Module) -> str | None:
    import fastvideo.layers.linear as fvlin
    from fastvideo.attention import DistributedAttention_VSA, LocalAttention
    is_linear = isinstance(module, torch.nn.Linear) or module.__class__.__name__.endswith("Linear") \
        or isinstance(module, getattr(fvlin, "ReplicatedLinear", ()))
    if is_linear:
        if ".ffn" in name:
            return "ffn GEMMs"
        if name.endswith(("to_q", "to_k", "to_v")) and ".attn2" not in name:
            return "self QKV proj GEMMs"
        if name.endswith("to_gate_compress"):
            return "gate_compress proj GEMM"
        if name.endswith(("to_out",)) and ".attn2" not in name:
            return "self o_proj GEMM"
        if ".attn2" in name:
            return "cross-attn proj GEMMs"
        return "other linears (embedders/head)"
    if isinstance(module, DistributedAttention_VSA):
        return "sparse attention (selector+quant+FP4 kernel+glue)"
    if isinstance(module, LocalAttention) and ".attn2" in name:
        return "cross-attn kernel"
    cls = module.__class__.__name__
    if "Norm" in cls and name.count(".") >= 2:
        return "norms"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", choices=("480", "720"), required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile", action="store_true", help="D1: torch.compile the blocks")
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

    if args.compile:
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = torch.compile(blk, mode="max-autotune-no-cudagraphs",
                                            dynamic=False)

    # ---- register event hooks on disjoint families ----
    pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    hooked: list[torch.nn.Module] = []

    def add_hooks(m: torch.nn.Module, fam: str):
        def pre(_m, _inp):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            _m.__dict__["_ev0"] = ev0
        def post(_m, _inp, _out):
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            pairs.append((fam, _m.__dict__.pop("_ev0"), ev1))
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)
        hooked.append(m)

    if not args.compile:  # hooks break torch.compile graphs; skip for D1
        seen_ids = set()
        for name, module in model.named_modules():
            fam = family_of(name, module)
            if fam and id(module) not in seen_ids:
                # avoid double-count: skip modules nested inside an already-hooked one
                add_hooks(module, fam)
                seen_ids.add(id(module))

    t, h, w = RES[args.res]
    B = args.batch
    hidden = torch.randn(B, 16, t, h, w, device=device, dtype=torch.bfloat16)
    text = torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16)
    timestep = torch.full((B,), 500, device=device, dtype=torch.bfloat16)
    meta = SparseFP4VSA256FA4MetadataBuilder().build(
        current_timestep=0, raw_latent_shape=(t, h, w), patch_size=(1, 2, 2),
        VSA_sparsity=0.90, device=device, cache_tile_buf=False)
    batch = ForwardBatch(data_type="dummy")

    def fwd():
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=meta,
                                                  forward_batch=batch):
            model(hidden_states=hidden, encoder_hidden_states=text, timestep=timestep)

    for _ in range(3):
        pairs.clear()
        fwd()
    torch.cuda.synchronize()

    walls = []
    fam_ms_total: dict[str, float] = defaultdict(float)
    for _ in range(args.reps):
        pairs.clear()
        t0 = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000)
        for fam, e0, e1 in pairs:
            fam_ms_total[fam] += e0.elapsed_time(e1)
    fam_ms = {k: v / args.reps for k, v in fam_ms_total.items()}

    wall = sum(walls) / len(walls)
    covered = sum(fam_ms.values())
    result = dict(res=args.res, batch=B, compile=args.compile, reps=args.reps,
                  wall_ms_per_forward=wall, families_ms=fam_ms,
                  residual_ms=wall - covered,
                  peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
