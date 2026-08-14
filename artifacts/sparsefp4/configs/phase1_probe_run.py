"""Phase 1 driver: run Wan2.1 denoising with the routing probe attached.

One process per GPU, each at ``sp_size=1`` so head indices are global. The
attention compute is dense BF16 (pass-through), identical to the ``FLASH_ATTN``
baseline; the probe backend writes one JSONL record per
(layer, head, timestep, cfg_branch, sparsity, routing_precision) cell.

Usage (after ``source artifacts/sparsefp4/configs/env.sh``)::

    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_probe_run.py \
        --run-id 20260813-000000-8208536-p1-stage1 --prompt-index 0 \
        --sparsities 0.80 0.90 --steps 50
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


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-index", type=int, required=True, help="0-based index into the prompt file")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.80, 0.90])
    parser.add_argument("--routing-precisions", nargs="+", default=["bf16", "fp8_e4m3", "nvfp4", "nvfp4_sim"])
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", default="1")
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "artifacts/sparsefp4/raw")
    parser.add_argument("--measure-timestep-stride",
                        type=int,
                        default=1,
                        help="record only every Nth denoising step (1 = every step)")
    parser.add_argument("--score-dtype",
                        default="float32",
                        choices=("float32", "float64"),
                        help="block-score matmul dtype; float32 is pre-registered, float64 is a control only")
    parser.add_argument("--null-control-layer-stride", type=int, default=1)
    parser.add_argument("--null-control-timestep-stride", type=int, default=1)
    parser.add_argument("--spearman-timestep-stride", type=int, default=10)
    parser.add_argument("--run-vsa-scorer", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-video", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit(f"--prompt-index {args.prompt_index} out of range for {len(prompts)} prompts")
    prompt = prompts[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_config = {
        "out_dir": str(out_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsities": list(args.sparsities),
        "routing_precisions": list(args.routing_precisions),
        "block_q": args.block_q,
        "block_k": args.block_k,
        "stage": args.stage,
        "phase": "1",
        "shard_tag": prompt_id,
        "patch_size": [1, 2, 2],
        "measure_timestep_stride": args.measure_timestep_stride,
        "score_dtype": args.score_dtype,
        "null_control_layer_stride": args.null_control_layer_stride,
        "null_control_timestep_stride": args.null_control_timestep_stride,
        "spearman_timestep_stride": args.spearman_timestep_stride,
        "run_vsa_scorer": args.run_vsa_scorer,
        "provenance": {
            "model_id": MODEL_ID,
            "num_inference_steps": args.steps,
            "prompt": prompt,
        },
    }
    config_path = out_dir / f"probe_config_{prompt_id}.json"
    config_path.write_text(json.dumps(probe_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ROUTING_PROBE_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_PROBE"] = str(config_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

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
    )
    load_seconds = time.time() - started

    sampling_param = SamplingParam.from_pretrained(MODEL_ID)
    sampling_param.num_inference_steps = args.steps
    sampling_param.seed = args.seed
    sampling_param.save_video = args.save_video
    sampling_param.return_frames = False
    sampling_param.prompt = prompt
    if args.save_video and args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        sampling_param.output_path = str(args.output_video.parent)

    generate_started = time.time()
    generator.generate_video(prompt, sampling_param=sampling_param)
    generate_seconds = time.time() - generate_started

    shard = out_dir / f"{prompt_id}.jsonl"
    summary = {
        "run_id": args.run_id,
        "prompt_id": prompt_id,
        "shard": str(shard),
        "records": sum(1 for _ in shard.open(encoding="utf-8")) if shard.is_file() else 0,
        "shard_bytes": shard.stat().st_size if shard.is_file() else 0,
        "model_load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 2),
        "steps": args.steps,
        "seed": args.seed,
        "sparsities": list(args.sparsities),
        "routing_precisions": list(args.routing_precisions),
    }
    (out_dir / f"run_summary_{prompt_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
