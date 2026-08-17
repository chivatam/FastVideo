"""F4.5: how far is simulated NVFP4 from the native quantizer, on real activations?

The scorer arms R8/R9 emulate NVFP4 arithmetic because there is no native NVFP4
block-dot to call. F4.5 requires that such simulation be compared against a native
twin where one exists and that median/p90/max disagreement be reported. A native twin
*does* exist for the representation step: ``quantize_router_input(..., "nvfp4")`` goes
through the FA4 fork's real quantize/dequantize, while ``"nvfp4_sim"`` is the
arithmetic emulation.

Synthetic Gaussians would understate the gap, because NVFP4's per-16 E4M3 scaling is
sensitive to the outlier structure of real activations. So this captures Q/K from the
actual model at the same layers and timesteps the study measures, then compares.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=<free gpu> "$FV_PYTHON" \
        artifacts/sparsefp4_followup/configs/f4_representation_fidelity.py \
        --out artifacts/sparsefp4_followup/raw/f4_representation_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_ENV = "FASTVIDEO_SPARSEFP4_F45_CAPTURE"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
PROMPTS = REPO_ROOT / ".agents/skills/sparsefp4-video-attention/assets/prompts.txt"


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p90": (statistics.quantiles(ordered, n=10, method="inclusive")[8] if len(ordered) >= 10 else ordered[-1]),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def compare_captures(capture_path: Path) -> dict[str, Any]:
    import torch

    from fastvideo.attention.backends.routing_probe_attn import quantize_router_input

    payload = torch.load(capture_path, map_location="cuda")
    per_tensor: list[dict[str, Any]] = []
    rel_disagreements: list[float] = []
    score_disagreements: list[float] = []

    for item in payload["captures"]:
        tensor = item["tensor"].cuda()
        native, native_saturation = quantize_router_input(tensor, "nvfp4")
        simulated, simulated_saturation = quantize_router_input(tensor, "nvfp4_sim")
        native32, simulated32 = native.float(), simulated.float()

        # Relative disagreement between the two dequantized representations, scaled by
        # the native representation's own magnitude: this is the quantity that would
        # propagate into a block score.
        denominator = native32.norm().clamp(min=torch.finfo(torch.float32).tiny)
        rel = float((simulated32 - native32).norm() / denominator)
        max_abs = float((simulated32 - native32).abs().max())
        exact_match = float((simulated32 == native32).float().mean())

        # And the consequence that actually matters: does the *pooled block vector* the
        # scorer actually dots differ? Q/K arrive as [B, S, H, D]; pooling is over the
        # sequence axis within a block, so pool over S with the study's block size and
        # compare the resulting per-head block vectors.
        block = 128
        batch, seq_len, heads, dim = native32.shape
        usable = (seq_len // block) * block
        pooled_native = native32[:, :usable].reshape(batch, usable // block, block, heads, dim).mean(dim=2)
        pooled_sim = simulated32[:, :usable].reshape(batch, usable // block, block, heads, dim).mean(dim=2)
        score_rel = float(
            (pooled_sim - pooled_native).norm() / pooled_native.norm().clamp(min=torch.finfo(torch.float32).tiny))

        rel_disagreements.append(rel)
        score_disagreements.append(score_rel)
        per_tensor.append({
            "layer": item["layer"],
            "timestep": item["timestep"],
            "which": item["which"],
            "shape": list(tensor.shape),
            "rel_l2_disagreement": rel,
            "max_abs_disagreement": max_abs,
            "frac_elements_bit_identical": exact_match,
            "pooled_rel_disagreement": score_rel,
            "native_saturation_frac": native_saturation,
            "simulated_saturation_frac": simulated_saturation,
        })
        del tensor, native, simulated, native32, simulated32

    rel_stats = quantiles(rel_disagreements)
    pooled_stats = quantiles(score_disagreements)
    return {
        "n_tensors": len(per_tensor),
        "summary": {
            "median_rel_disagreement": rel_stats["median"],
            "p90_rel_disagreement": rel_stats["p90"],
            "max_rel_disagreement": rel_stats["max"],
            "median_pooled_rel_disagreement": pooled_stats["median"],
            "p90_pooled_rel_disagreement": pooled_stats["p90"],
            "max_pooled_rel_disagreement": pooled_stats["max"],
            "native_twin": "quantize_router_input(..., 'nvfp4') via the FA4 fork's real quantize/dequantize",
            "simulated": "quantize_router_input(..., 'nvfp4_sim'), the arithmetic emulation",
            "latency_policy": "simulated arms are never used for latency claims",
        },
        "per_tensor": per_tensor,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=Path("/mnt/nvme/scratch/sparsefp4_followup/f45_capture/qk.pt"))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 15, 28, 29])
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 25, 48])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reuse-capture", action="store_true")
    args = parser.parse_args()

    if not args.capture.is_file() or not args.reuse_capture:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        # The capture rides on the F1 backend, which is already the thing that defines
        # the study's capture point — so the tensors compared here are exactly the
        # tensors the scorer arms quantize, not a re-derivation of them.
        f1_config = {
            "out_dir": str(args.capture.parent),
            "run_id": "f45-capture",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "prompt_id": "p01",
            "seed": 1234,
            "sparsities": [0.90],
            "layers": args.layers,
            "timesteps": args.timesteps,
            "heads": [0],
            "cfg_branches": ["positive"],
            "arms": ["R0"],
            "geometry": "128x64-raster",
            "block_q": 128,
            "spearman_query_block_stride": 64,
            "shard_tag": "f45_capture",
            "stage": "F4.5-capture",
            "random_seed": 20260816,
            "provenance": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "num_inference_steps": args.steps,
                "guidance_scale": 3.0,
                "resolution": "480x832",
                "frames": 81,
                "prompt": "",
                "affected_layers": [],
                "unaffected_layers": [],
                "broad_layers": [],
            },
        }
        config_path = args.capture.parent / "f45_f1_config.json"
        config_path.write_text(json.dumps(f1_config, indent=2) + "\n", encoding="utf-8")

        os.environ[CAPTURE_ENV] = json.dumps({
            "path": str(args.capture),
            "layers": args.layers,
            "timesteps": args.timesteps,
        })
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SCORER_PRECISION_ATTN"
        os.environ["FASTVIDEO_SPARSEFP4_F1"] = str(config_path)
        print(f"capturing real Q/K at layers {args.layers}, timesteps {args.timesteps}")

        import torch

        from fastvideo import VideoGenerator
        from fastvideo.api.sampling_param import SamplingParam

        prompt = next(line.strip() for line in PROMPTS.read_text(encoding="utf-8").splitlines()
                      if line.strip() and not line.startswith("#"))
        torch.manual_seed(1234)
        generator = VideoGenerator.from_pretrained(MODEL_ID,
                                                   num_gpus=1,
                                                   use_fsdp_inference=False,
                                                   dit_cpu_offload=False,
                                                   vae_cpu_offload=False,
                                                   text_encoder_cpu_offload=True,
                                                   pin_cpu_memory=True)
        sampling_param = SamplingParam.from_pretrained(MODEL_ID)
        sampling_param.num_inference_steps = args.steps
        sampling_param.seed = 1234
        sampling_param.save_video = False
        sampling_param.return_frames = False
        sampling_param.prompt = prompt
        generator.generate_video(prompt, sampling_param=sampling_param)

    if not args.capture.is_file():
        # The DiT runs in a worker subprocess whose atexit flush can land after the
        # driver's generate call returns, so poll briefly rather than racing it.
        for _ in range(60):
            if args.capture.is_file():
                break
            time.sleep(1.0)
    if not args.capture.is_file():
        raise SystemExit(f"no capture written to {args.capture}; the capture hook did not fire")

    result = compare_captures(args.capture)
    result.update({
        "phase": "F4.5",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "capture": str(args.capture),
        "layers": args.layers,
        "timesteps": args.timesteps,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    summary = result["summary"]
    print(f"\nnative vs simulated NVFP4 on {result['n_tensors']} real tensors:")
    print(f"  representation rel-L2: median {summary['median_rel_disagreement']:.4e}  "
          f"p90 {summary['p90_rel_disagreement']:.4e}  max {summary['max_rel_disagreement']:.4e}")
    print(f"  pooled-vector rel:     median {summary['median_pooled_rel_disagreement']:.4e}  "
          f"p90 {summary['p90_pooled_rel_disagreement']:.4e}  max {summary['max_pooled_rel_disagreement']:.4e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
