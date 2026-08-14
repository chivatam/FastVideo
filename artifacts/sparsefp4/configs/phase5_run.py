"""Phase 5 driver: generate one end-to-end video for one (prompt, arm, seed).

One process per generation. The arm is fixed at model-load time because
``SparseFP4ExecAttentionImpl`` reads its config in ``__init__``, so a separate
process per arm is what keeps the configuration honest -- and it also gives a
clean per-arm peak-memory number instead of a high-water mark shared across arms.

Everything except the attention op is held identical across arms (SKILL rule 6):
same checkpoint revision, scheduler, steps, resolution, frame count, guidance,
seed, negative prompt, and no compile/CUDA-graph settings anywhere. The only
difference between two arms is which attention the DiT consumes.

Trap 1 (a typo in ``FASTVIDEO_ATTENTION_BACKEND`` is *silently ignored*) is
guarded twice: the backend refuses to construct without the Phase 5 config, and
the run summary records both the loader's resolved-backend log line and the
existence of the backend's own ``arm_receipt``. If the override had been ignored,
no receipt would exist and the summary would say so.

Decoded frames are saved as float16 rather than only as an mp4: the routing
precision effect Phase 2 predicts is ~0.02% of attention error, and H.264
quantization noise would comfortably swamp it. The mp4 is written too, for
looking at.

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_run.py \
        --run-id 20260814-XXXXXX-8208536-p5-main --prompt-index 0 --arm DENSE-BF16
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

ARM_IDS = (
    "DENSE-BF16",
    "DENSE-FP4",
    "SPARSE-BF16",
    "SPARSE-FP4-NAIVE",
    "SPARSE-FP4-ROUTE8",
    "SPARSE-FP4-ROUTE16",
    "SPARSE-BF16-EPS",
)


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--arm", required=True, choices=ARM_IDS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--score-dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage", default="5-main")
    parser.add_argument("--perturb-rel-l2", type=float, default=0.0)
    parser.add_argument("--perturb-seed", type=int, default=20260814)
    parser.add_argument("--video-root", type=Path, default=Path("/mnt/scratch/sparsefp4-videos"))
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--keep-float-frames", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    if not 0 <= args.prompt_index < len(prompts):
        raise SystemExit(f"--prompt-index {args.prompt_index} out of range for {len(prompts)} prompts")
    prompt = prompts[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    sparse = args.arm.startswith("SPARSE")
    tag = f"{prompt_id}_{args.arm}_s{args.seed}"
    if args.perturb_rel_l2 > 0.0:
        tag = f"{tag}_eps{args.perturb_rel_l2:g}"

    raw_dir = args.raw_root / args.run_id
    video_dir = args.video_root / args.run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    phase5_config = {
        "out_dir": str(raw_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "arm": args.arm,
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsity": args.sparsity if sparse else 0.0,
        "block_q": args.block_q,
        "block_k": args.block_k,
        "score_dtype": args.score_dtype,
        "shard_tag": tag,
        "perturb_rel_l2": args.perturb_rel_l2,
        "perturb_seed": args.perturb_seed,
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "num_inference_steps": args.steps,
            "prompt": prompt,
            "stage": args.stage,
        },
    }
    config_path = raw_dir / f"phase5_config_{tag}.json"
    config_path.write_text(json.dumps(phase5_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_EXEC_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_PHASE5"] = str(config_path)
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    import imageio
    import numpy as np
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
    sampling_param.return_frames = True
    sampling_param.prompt = prompt

    generate_started = time.time()
    result = generator.generate_video(prompt, sampling_param=sampling_param)
    generate_seconds = time.time() - generate_started

    samples = result.get("samples")
    frame_path = video_dir / f"{tag}.f16.npy"
    video_path = video_dir / f"{tag}.mp4"
    frame_stats: dict[str, object] = {}
    if samples is not None:
        # [B, C, T, H, W] in [0, 1]; keep float16 for metric sensitivity.
        array = samples.detach().to(torch.float16).cpu().numpy()
        np.save(frame_path, array)
        frame_stats = {
            "frames_shape": list(array.shape),
            "frames_dtype": str(array.dtype),
            "frames_bytes": int(frame_path.stat().st_size),
            "pixel_min": float(np.min(array.astype(np.float32))),
            "pixel_max": float(np.max(array.astype(np.float32))),
            "pixel_mean": float(np.mean(array.astype(np.float32))),
        }
    frames = result.get("frames")
    if frames:
        imageio.mimsave(video_path, frames, fps=sampling_param.fps, format="mp4")

    receipt_path = raw_dir / f"arm_receipt_{tag}.json"
    # The receipt is written by the *worker* process that owns the attention
    # impl, so shut the executor down before reading it back -- otherwise the
    # parent races the worker's final flush and records a false negative.
    generator.shutdown()
    for _ in range(40):
        if receipt_path.is_file():
            break
        time.sleep(0.25)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    summary = {
        "run_id": args.run_id,
        "stage": args.stage,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "arm": args.arm,
        "sparse": sparse,
        "requested_sparsity": args.sparsity if sparse else None,
        "requested_perturb_rel_l2": args.perturb_rel_l2 or None,
        "realized_perturb_rel_l2": (receipt or {}).get("counters.all", {}).get("mean_realized_perturb_rel_l2"),
        "tag": tag,
        "seed": args.seed,
        "steps": args.steps,
        "height": sampling_param.height,
        "width": sampling_param.width,
        "num_frames": sampling_param.num_frames,
        "guidance_scale": sampling_param.guidance_scale,
        "fps": sampling_param.fps,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "git_commit": phase5_config["git_commit"],
        "attention_backend_requested": "SPARSEFP4_EXEC_ATTN",
        "flash_attention_4_enabled": os.environ.get("FASTVIDEO_FA4") == "1",
        "nvfp4_fa4_env": os.environ.get("FASTVIDEO_NVFP4_FA4"),
        "torch_compile": False,
        "cuda_graphs": False,
        "arm_receipt_written": receipt_path.is_file(),
        "arm_receipt_path": str(receipt_path) if receipt_path.is_file() else None,
        # Copied up from the worker so a single file answers "did the arm the
        # config named actually run, at the budget it asked for?"
        "realized_sparsity": (receipt or {}).get("counters.all", {}).get("realized_sparsity"),
        "attention_calls": (receipt or {}).get("counters.all", {}).get("attention_calls"),
        "distinct_layers": (receipt or {}).get("counters.all", {}).get("distinct_layers"),
        "distinct_timesteps": (receipt or {}).get("counters.all", {}).get("distinct_timesteps"),
        "native_or_simulated": (receipt or {}).get("native_or_simulated"),
        "native_latency_claim_allowed": (receipt or {}).get("native_latency_claim_allowed"),
        "attention_compute": (receipt or {}).get("attention_compute"),
        "router_precision": (receipt or {}).get("router_precision"),
        "model_load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 3),
        "e2e_latency_seconds": result.get("e2e_latency"),
        "generation_time_seconds": result.get("generation_time"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "frame_path": str(frame_path) if samples is not None else None,
        "video_path": str(video_path) if frames else None,
        "config_path": str(config_path),
        **frame_stats,
    }
    logging_info = result.get("logging_info")
    stages = getattr(logging_info, "stages", None)
    if stages:
        summary["stage_execution_times"] = {
            name: info.get("execution_time")
            for name, info in stages.items() if isinstance(info, dict)
        }
    (raw_dir / f"run_summary_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
