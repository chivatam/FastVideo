"""P4 root-cause step 2: time the actual SparseFP4VSA256FA4Impl.forward at
720p geometry, fine=nvfp4 vs fine=bf16, in one process (impl-level wall clock).

If both are ~5-7 ms/call, the 46 ms/call E2E gap is external to the impl and
the next step is a torch.profiler capture of a real DiT step.
"""
import json
import os
import time

import torch

os.environ["FASTVIDEO_SPARSEFP4_FINE"] = "nvfp4"  # will flip per-arm below

from fastvideo.attention.backends.sparsefp4_vsa256_fa4 import (SparseFP4VSA256FA4Impl,
                                                               SparseFP4VSA256FA4MetadataBuilder)

B, H, D = 1, 12, 128
raw_latent = (21, 90, 160)  # 720x1280x81 latents (T, H/8, W/8)
patch = (1, 2, 2)

builder = SparseFP4VSA256FA4MetadataBuilder()
meta = builder.build(current_timestep=0, raw_latent_shape=raw_latent, patch_size=patch,
                     VSA_sparsity=0.9, device=torch.device("cuda"), cache_tile_buf=False)
S = int(meta.variable_block_sizes.numel()) * 256
print("tiles:", meta.variable_block_sizes.numel(), "padded S:", S)

torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
gate = torch.randn_like(q)

results = {}
for fine in ("bf16", "nvfp4"):
    os.environ["FASTVIDEO_SPARSEFP4_FINE"] = fine
    impl = SparseFP4VSA256FA4Impl(num_heads=H, head_size=D, causal=False,
                                  softmax_scale=D**-0.5)
    for _ in range(5):
        impl.forward(q, k, v, gate, meta)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        impl.forward(q, k, v, gate, meta)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    results[fine] = ms
    print(f"impl.forward fine={fine}: {ms:.3f} ms/call", flush=True)

print(f"delta x 3000 calls: {(results['nvfp4']-results['bf16'])*3:.1f} s/video")
json.dump(results, open("artifacts/sparsefp4_native/raw/performance/p4_impl_wall_720p.json", "w"), indent=2)
print("IMPL_WALL_DONE")
