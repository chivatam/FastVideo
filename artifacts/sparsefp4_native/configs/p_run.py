"""P-arm driver: one process per (arm, prompt) — quality frames + E2E timing.

Arms (only the attention path differs; checkpoint/scheduler/steps/resolution/
frames/guidance/seed/negative-prompt identical):

  P0   dense BF16 FA4            FLASH_ATTN, FASTVIDEO_FA4=1
  P1   dense native NVFP4        FLASH_ATTN, FASTVIDEO_FA4=1, FASTVIDEO_NVFP4_FA4=1
  P2   deployed VSA (BF16)       VIDEO_SPARSE_ATTN, VSA_sparsity
  P2G  geometry control          SPARSEFP4_NATIVE_VSA_ATTN, FINE=bf16
  P3   native sparse NVFP4 fine  SPARSEFP4_NATIVE_VSA_ATTN, FINE=nvfp4

Writes per (arm,prompt):
  <video_root>/<run>/<tag>.f16.npy   float16 frames [B,C,T,H,W] for metrics
  <video_root>/<run>/<tag>.mp4       for eyeballing
  <raw_root>/<run>/summary_<tag>.json  timings, peak memory, provenance

Perf mode (--perf-reps N) repeats generation N times after 1 warmup and
records per-rep E2E seconds + DiT-stage seconds (FASTVIDEO_STAGE_LOGGING).
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

ARMS = {
    "P0": dict(backend="FLASH_ATTN", nvfp4=False, fine=None, sparse=False),
    "P1": dict(backend="FLASH_ATTN", nvfp4=True, fine=None, sparse=False),
    "P2": dict(backend="VIDEO_SPARSE_ATTN", nvfp4=False, fine=None, sparse=True),
    "P2G": dict(backend="SPARSEFP4_NATIVE_VSA_ATTN", nvfp4=False, fine="bf16", sparse=True),
    "P3": dict(backend="SPARSEFP4_NATIVE_VSA_ATTN", nvfp4=False, fine="nvfp4", sparse=True),
    # VSA-on-FA4 geometry-native arms: 256-token tile selector, exact FA4
    # sparse mapping (no mask coarsening), FA4 Blackwell fine kernel.
    "P4G": dict(backend="SPARSEFP4_VSA256_FA4_ATTN", nvfp4=False, fine="bf16", sparse=True),
    "P4": dict(backend="SPARSEFP4_VSA256_FA4_ATTN", nvfp4=False, fine="nvfp4", sparse=True),
}


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--prompt-index", type=int, required=True)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--sparsity", type=float, default=0.90)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--perf-reps", type=int, default=0)
    ap.add_argument("--model-path", default=MODEL_ID,
                    help="pipeline dir override (e.g. QAT-recovered checkpoint)")
    ap.add_argument("--video-root", type=Path,
                    default=Path("/mnt/nvme/scratch/sparsefp4_native/videos"))
    ap.add_argument("--raw-root", type=Path,
                    default=Path("/mnt/nvme/scratch/sparsefp4_native/p_runs"))
    args = ap.parse_args()

    spec = ARMS[args.arm]
    prompts = load_prompts(args.prompts)
    prompt = prompts[args.prompt_index]
    tag = f"p{args.prompt_index:02d}_{args.arm}_s{args.seed}"

    raw_dir = args.raw_root / args.run_id
    video_dir = args.video_root / args.run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = spec["backend"]
    os.environ["FASTVIDEO_FA4"] = "1"
    if spec["nvfp4"]:
        os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
    else:
        os.environ.pop("FASTVIDEO_NVFP4_FA4", None)
    if spec["fine"]:
        os.environ["FASTVIDEO_SPARSEFP4_FINE"] = spec["fine"]
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    import imageio
    import numpy as np
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    torch.manual_seed(args.seed)
    t0 = time.time()
    init_kwargs = dict(num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
                       vae_cpu_offload=False, text_encoder_cpu_offload=True,
                       pin_cpu_memory=True)
    if spec["sparse"]:
        init_kwargs["VSA_sparsity"] = args.sparsity
    generator = VideoGenerator.from_pretrained(args.model_path, **init_kwargs)
    load_s = time.time() - t0

    sp = SamplingParam.from_pretrained(MODEL_ID)  # sampling preset stays canonical
    sp.num_inference_steps = args.steps
    sp.seed = args.seed
    if args.height:
        sp.height = args.height
    if args.width:
        sp.width = args.width
    sp.save_video = False
    sp.return_frames = True
    sp.prompt = prompt

    def one_generation():
        start = time.perf_counter()
        result = generator.generate_video(prompt, sampling_param=sp)
        elapsed = time.perf_counter() - start
        info = result.get("logging_info")
        stages = (info.get("stages", {}) if isinstance(info, dict)
                  else getattr(info, "stages", {}) or {})
        dit_s = None
        for name, data in (stages or {}).items():
            if isinstance(data, dict) and data.get("stage_class", name).startswith("Denoising"):
                dit_s = data.get("execution_time")
        return result, elapsed, dit_s, result.get("peak_memory_mb")

    result, gen_s, dit_s, peak_mb = one_generation()

    perf = []
    if args.perf_reps > 0:
        for rep in range(args.perf_reps):
            _, e2e, dstep, pmb = one_generation()
            perf.append(dict(rep=rep, e2e_s=e2e, dit_s=dstep, peak_memory_mb=pmb))

    samples = result.get("samples")
    frames = result.get("frames")
    frame_stats = {}
    if samples is not None:
        arr = samples.detach().to(torch.float16).cpu().numpy()
        np.save(video_dir / f"{tag}.f16.npy", arr)
        frame_stats = dict(frames_shape=list(arr.shape),
                           pixel_mean=float(np.mean(arr.astype(np.float32))))
    if frames:
        imageio.mimsave(video_dir / f"{tag}.mp4", frames, fps=sp.fps, format="mp4")

    summary = {
        "run_id": args.run_id, "arm": args.arm, "tag": tag,
        "prompt_index": args.prompt_index, "prompt": prompt,
        "backend": spec["backend"], "fine": spec["fine"],
        "nvfp4_env": os.environ.get("FASTVIDEO_NVFP4_FA4"),
        "sparsity": args.sparsity if spec["sparse"] else None,
        "seed": args.seed, "steps": args.steps,
        "height": sp.height, "width": sp.width, "num_frames": sp.num_frames,
        "guidance_scale": sp.guidance_scale, "fps": sp.fps,
        "model_id": MODEL_ID,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=REPO_ROOT, text=True).strip(),
        "load_s": round(load_s, 2), "first_gen_s": round(gen_s, 2),
        "first_dit_s": dit_s, "first_peak_memory_mb": peak_mb,
        "perf_reps": perf, **frame_stats,
    }
    (raw_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "perf_reps"}, indent=2))
    generator.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
