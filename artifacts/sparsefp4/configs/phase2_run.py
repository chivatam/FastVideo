"""Phase 2 driver: run Wan2.1 denoising with the A-F error-decomposition probe.

One process per GPU at ``sp_size=1``. The attention the model consumes is dense
BF16 (configuration A), so the trajectory is identical to the ``FLASH_ATTN``
baseline and to every Phase 1 run; configurations B-F, the H3 router comparison,
the decision-margin mechanism measurement and the random-perturbation contrast
control are computed on the side at the selected cells.

Block scores are fp64 by default (``artifacts/sparsefp4/STATUS.md`` trap 8).

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_run.py \
        --run-id 20260814-000000-8208536-p2-main --prompt-index 0
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

# Phase 1's refined targets (PHASE1.md 9): the affected regions are the network
# edges, the unaffected controls are mid-stack, and both sets are measured so the
# H2-style claim stays falsifiable rather than selected-for.
AFFECTED_LAYERS = (0, 1, 2, 27, 28, 29)
UNAFFECTED_LAYERS = (5, 6, 10, 11, 13)
BROAD_LAYERS = (8, 16, 20, 23, 24, 25)


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.80, 0.90, 0.95])
    parser.add_argument("--layers",
                        type=int,
                        nargs="+",
                        default=sorted(AFFECTED_LAYERS + UNAFFECTED_LAYERS + BROAD_LAYERS))
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 1, 10, 25, 40])
    parser.add_argument("--cfg-branches", nargs="+", default=["positive", "negative"])
    parser.add_argument("--mechanism-layers", type=int, nargs="+", default=[0, 5, 13, 24, 28, 29])
    parser.add_argument("--mechanism-timesteps", type=int, nargs="+", default=[0, 25])
    parser.add_argument("--mechanism-sparsities", type=float, nargs="+", default=[0.80, 0.90, 0.95])
    parser.add_argument("--mechanism-query-blocks", type=int, default=12)
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--geometry",
                        default="128x64-raster",
                        choices=("128x64-raster", "64x64-raster", "64x64-cube"),
                        help="token-to-block assignment; the cube arm is VSA's deployed geometry")
    parser.add_argument("--arms",
                        nargs="+",
                        default=["A", "B", "B_sim", "C", "D", "D8", "C_rand", "E", "F8", "F16"],
                        help="configuration ids to measure; Phase 2B runs A C C_null D C_rand")
    parser.add_argument("--tie-diagnostic-geometries",
                        nargs="*",
                        default=[],
                        help="extra geometries to emit boundary-tie diagnostics at on every measured cell")
    parser.add_argument("--no-activation-stats",
                        action="store_true",
                        help="skip the per-head activation/saturation rows (already measured in Phase 2)")
    parser.add_argument("--score-dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", default="2-main")
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit(f"--prompt-index {args.prompt_index} out of range for {len(prompts)} prompts")
    prompt = prompts[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    # The geometry name is the single source of truth for block sizes, so a run
    # cannot ask for "64x64-raster" and silently score at 128.
    block_q, block_k = (args.block_q, args.block_k)
    if args.geometry == "64x64-raster":
        block_q = block_k = 64
    elif args.geometry == "128x64-raster":
        block_q, block_k = 128, 64
    elif args.geometry == "64x64-cube":
        block_q = block_k = 64

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    phase2_config = {
        "out_dir": str(out_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsities": list(args.sparsities),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "cfg_branches": list(args.cfg_branches),
        "mechanism_layers": list(args.mechanism_layers),
        "mechanism_timesteps": list(args.mechanism_timesteps),
        "mechanism_sparsities": list(args.mechanism_sparsities),
        "mechanism_query_blocks": args.mechanism_query_blocks,
        "block_q": block_q,
        "block_k": block_k,
        "geometry": args.geometry,
        "arms": list(args.arms),
        "tie_diagnostic_geometries": list(args.tie_diagnostic_geometries),
        "emit_activation_stats": not args.no_activation_stats,
        "score_dtype": args.score_dtype,
        "shard_tag": prompt_id,
        "stage": args.stage,
        "random_seed": 20260814 + args.prompt_index,
        "provenance": {
            "model_id": MODEL_ID,
            "num_inference_steps": args.steps,
            "prompt": prompt,
            "affected_layers": list(AFFECTED_LAYERS),
            "unaffected_layers": list(UNAFFECTED_LAYERS),
            "broad_layers": list(BROAD_LAYERS),
        },
    }
    config_path = out_dir / f"phase2_config_{prompt_id}.json"
    config_path.write_text(json.dumps(phase2_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "PRECISION_SPARSE_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_PHASE2"] = str(config_path)

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
    sampling_param.save_video = False
    sampling_param.return_frames = False
    sampling_param.prompt = prompt

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
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "geometry": args.geometry,
        "arms": list(args.arms),
        "score_dtype": args.score_dtype,
    }
    (out_dir / f"run_summary_{prompt_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
