"""F1 driver: run Wan2.1 denoising with the scorer-arithmetic-precision probe.

One process per GPU at ``sp_size=1``. The attention the model consumes is dense
BF16, so the trajectory is identical to study 1's baseline and every scorer arm
sees byte-identical Q/K.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_run.py \
        --run-id 20260816-000000-abcdef0-f1-diag --prompt-index 0 --stage F1-diagnostic
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

# Study 1's Phase 2 layer partition, reused verbatim so "affected" / "unaffected"
# mean the same thing in both studies and the regions are not re-selected here
# after seeing follow-up data.
AFFECTED_LAYERS = (0, 1, 2, 27, 28, 29)
UNAFFECTED_LAYERS = (5, 6, 10, 11, 13)
BROAD_LAYERS = (8, 16, 20, 23, 24, 25)

# Six timesteps spanning the trajectory, as F1.3 requires: first, early, middle,
# late-middle, late, near-final. Study 1's Phase 2 used [0, 1, 10, 25, 40]; this
# adds step 48 so the near-final region is covered.
DEFAULT_TIMESTEPS = (0, 1, 10, 25, 40, 48)


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
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.90])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--heads", type=int, nargs="*", default=[], help="empty means all heads")
    parser.add_argument("--cfg-branches", nargs="+", default=["positive", "negative"])
    parser.add_argument("--arms",
                        nargs="+",
                        default=["R0", "R1", "R2", "R3", "R4", "R5", "R4L", "R5L", "R6", "R7", "R8", "R9"])
    parser.add_argument("--geometry", default="128x64-raster", choices=("128x64-raster", "64x64-raster", "64x64-cube"))
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--spearman-query-block-stride", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", default="F1-full")
    # F3B varies the token count. Defaults are the study-1 operating point, so an
    # unflagged run stays byte-comparable with everything measured so far.
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--raw-root",
                        type=Path,
                        default=Path(os.environ.get("FV_RAW_ROOT", "/mnt/nvme/scratch/sparsefp4_followup")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit(f"--prompt-index {args.prompt_index} out of range for {len(prompts)} prompts")
    prompt = prompts[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    shard_tag = f"{prompt_id}_s{args.seed}"

    block_q = args.block_q
    if args.geometry in ("64x64-raster", "64x64-cube"):
        block_q = 64
    elif args.geometry == "128x64-raster":
        block_q = 128

    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_config = {
        "out_dir": str(out_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsities": list(args.sparsities),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "heads": list(args.heads),
        "cfg_branches": list(args.cfg_branches),
        "arms": list(args.arms),
        "geometry": args.geometry,
        "block_q": block_q,
        "spearman_query_block_stride": args.spearman_query_block_stride,
        "shard_tag": shard_tag,
        "stage": args.stage,
        "random_seed": 20260816 + args.prompt_index + 100000 * args.seed,
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "num_inference_steps": args.steps,
            "guidance_scale": 3.0,
            "resolution": f"{args.height}x{args.width}",
            "frames": args.frames,
            "prompt": prompt,
            "affected_layers": list(AFFECTED_LAYERS),
            "unaffected_layers": list(UNAFFECTED_LAYERS),
            "broad_layers": list(BROAD_LAYERS),
        },
    }
    config_path = out_dir / f"f1_config_{shard_tag}.json"
    config_path.write_text(json.dumps(f1_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SCORER_PRECISION_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_F1"] = str(config_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    # NOTE: TF32 is deliberately *not* mutated here. FastVideo runs the pipeline
    # in a worker subprocess, so a driver-side setting would not reach the DiT
    # anyway, and forcing it would change the model's own matmuls away from the
    # trajectory study 1 measured. The scorer's fp32/fp8 arms establish exactness
    # locally via `exact_fp32_matmul()`, and each raw record carries the worker's
    # ambient TF32 state so the distinction is auditable.
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
    sampling_param.height = args.height
    sampling_param.width = args.width
    sampling_param.num_frames = args.frames

    generate_started = time.time()
    generator.generate_video(prompt, sampling_param=sampling_param)
    generate_seconds = time.time() - generate_started

    shard = out_dir / f"{shard_tag}.jsonl"
    summary = {
        "run_id": args.run_id,
        "prompt_id": prompt_id,
        "seed": args.seed,
        "shard": str(shard),
        # Provisional: the DiT runs in a worker subprocess whose JSONL writer flushes
        # on its own atexit, which can be after the driver reads the file. Treat
        # f1_validate.py's count as authoritative for the record total.
        "records_provisional": sum(1 for _ in shard.open(encoding="utf-8")) if shard.is_file() else 0,
        "records_count_note": "provisional; worker flush may trail this read — see f1_validate.py",
        "shard_bytes": shard.stat().st_size if shard.is_file() else 0,
        "model_load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 2),
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "arms": list(args.arms),
        "layers": list(args.layers),
        "timesteps": list(args.timesteps),
        "sparsities": list(args.sparsities),
        "geometry": args.geometry,
        "stage": args.stage,
    }
    (out_dir / f"run_summary_{shard_tag}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
