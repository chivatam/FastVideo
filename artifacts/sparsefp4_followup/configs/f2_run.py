"""F2 driver: run Wan2.1 under **real** VSA with the selector-precision probe.

The model's attention is genuine ``VIDEO_SPARSE_ATTN`` — the probe subclasses it
and only adds a side-channel measurement — so the trajectory is a real
sparse-attention trajectory, not a dense one. That is the point: F2's external
validity comes from measuring VSA's own selector on VSA's own trajectory.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f2_run.py \
        --run-id <id> --prompt-index 0 --stage F2-diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = REPO_ROOT / ".agents/skills/sparsefp4-video-attention/assets/prompts.txt"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
DEFAULT_TIMESTEPS = (0, 1, 10, 25, 40, 48)
ALL_ARMS = ("V0", "V0_FP64", "VA_FP8", "VA_NVFP4", "VB_FP32", "VB_BF16_LOW", "VA_NVFP4_VB_FP64", "VC_GATE_NVFP4",
            "VD_TORCH_TOPK")


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.90])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--heads", type=int, nargs="*", default=[])
    parser.add_argument("--cfg-branches", nargs="+", default=["positive", "negative"])
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", default="F2-full")
    # The sparsity VSA itself executes with, which sets the model's trajectory. Kept
    # separate from --sparsities (the measured mask budgets) so the trajectory is one
    # fixed, declared operating point rather than varying per measurement.
    parser.add_argument("--vsa-sparsity", type=float, default=0.90)
    parser.add_argument("--raw-root",
                        type=Path,
                        default=Path(os.environ.get("FV_RAW_ROOT", "/mnt/nvme/scratch/sparsefp4_followup")))
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit(f"--prompt-index {args.prompt_index} out of range for {len(prompts)} prompts")
    prompt = prompts[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    shard_tag = f"{prompt_id}_s{args.seed}"

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "out_dir": str(out_dir),
        "run_id": args.run_id,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsities": list(args.sparsities),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "heads": list(args.heads),
        "cfg_branches": list(args.cfg_branches),
        "arms": list(args.arms),
        "shard_tag": shard_tag,
        "stage": args.stage,
        "vsa_sparsity_for_execution": args.vsa_sparsity,
        "random_seed": 20260816 + args.prompt_index + 100000 * args.seed,
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "num_inference_steps": args.steps,
            "guidance_scale": 3.0,
            "resolution": "480x832",
            "frames": 81,
            "prompt": prompt,
        },
    }
    config_path = out_dir / f"f2_config_{shard_tag}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VSA_PRECISION_PROBE_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_F2"] = str(config_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    # TF32 left at the worker's ambient value on purpose (see f1_run.py); the fp32
    # selector arm establishes exactness locally and every record carries the state.
    torch.manual_seed(args.seed)

    started = time.time()
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        VSA_sparsity=args.vsa_sparsity,
    )
    load_seconds = time.time() - started

    sampling_param = SamplingParam.from_pretrained(MODEL_ID)
    sampling_param.num_inference_steps = args.steps
    sampling_param.seed = args.seed
    sampling_param.save_video = False
    sampling_param.return_frames = False
    sampling_param.prompt = prompt

    generate_started = time.time()
    generator.generate_video(prompt, sampling_param=sampling_param)
    generate_seconds = time.time() - generate_started

    shard = out_dir / f"{shard_tag}.jsonl"
    summary = {
        "run_id": args.run_id,
        "prompt_id": prompt_id,
        "seed": args.seed,
        "shard": str(shard),
        "records_provisional": sum(1 for _ in shard.open(encoding="utf-8")) if shard.is_file() else 0,
        "records_count_note": "provisional; worker flush may trail this read — see f2_validate.py",
        "shard_bytes": shard.stat().st_size if shard.is_file() else 0,
        "model_load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 2),
        "vsa_sparsity_for_execution": args.vsa_sparsity,
        "arms": list(args.arms),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "sparsities": list(args.sparsities),
        "stage": args.stage,
    }
    (out_dir / f"run_summary_{shard_tag}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
