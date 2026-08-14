"""Phase 5 performance: end-to-end latency and peak memory, native arms only.

**Only ``DENSE-BF16`` and ``DENSE-FP4`` appear here.** The three
``SPARSE-FP4-*`` arms have no native sparse-NVFP4 kernel in this environment and
are simulated by dequantizing NVFP4 Q/K back to BF16 before a BF16 block-sparse
kernel; timing them would measure the simulation, not a design. ``SPARSE-BF16``
runs on a real block-sparse kernel but on a *research* mask path with fp64
scoring per call, so its wall clock is not a deployable number either -- it is
recorded as ``advisory`` and kept out of the headline table. This mirrors
``GO_NO_GO.md`` scoping item 3 and SKILL integrity rules 2 and 3.

Protocol (SKILL "Benchmark protocol"): warm up first so first-call CuTeDSL JIT
compilation is excluded, then multiple measured repetitions, CUDA-synchronized
around each, identical shapes and prompt and seed and step count throughout,
median plus dispersion, and compile/CUDA-graph state recorded. Reuses the
provenance fields ``attention_backend`` and ``flash_attention_4_enabled`` that
``fastvideo/tests/performance/test_inference_performance.py`` records, and its
``_extract_component_times`` for the per-stage DiT breakdown.

Peak memory is ``torch.cuda.max_memory_allocated`` inside the worker (what
``multiproc_executor`` reports as ``peak_memory_mb``), taken as the max over
measured repetitions exactly as the repo harness does.

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_perf.py \
        --run-id 20260814-XXXXXX-p5-perf --arm DENSE-BF16
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
DEFAULT_PROMPTS = REPO_ROOT / ".agents/skills/sparsefp4-video-attention/assets/prompts.txt"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"

NATIVE_ARMS = ("DENSE-BF16", "DENSE-FP4")
ADVISORY_ARMS = ("SPARSE-BF16", )


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def dispersion(values: list[float]) -> dict[str, float | None]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "iqr": (statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0]) if len(values) >= 4 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=NATIVE_ARMS + ADVISORY_ARMS)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    args = parser.parse_args()

    prompt = load_prompts(args.prompts)[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    sparse = args.arm.startswith("SPARSE")
    tag = f"{prompt_id}_{args.arm}"
    raw_dir = args.raw_root / args.run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase5_config = {
        "out_dir": str(raw_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "arm": args.arm,
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsity": args.sparsity if sparse else 0.0,
        "block_q": 128,
        "block_k": 64,
        "score_dtype": "float64",
        "shard_tag": f"perf_{tag}",
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "stage": "5-perf"
        },
    }
    config_path = raw_dir / f"phase5_perf_config_{tag}.json"
    config_path.write_text(json.dumps(phase5_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SPARSEFP4_EXEC_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_PHASE5"] = str(config_path)
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.tests.performance.test_inference_performance import _extract_component_times

    torch.manual_seed(args.seed)
    load_started = time.time()
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    load_seconds = time.time() - load_started

    def make_param() -> SamplingParam:
        param = SamplingParam.from_pretrained(MODEL_ID)
        param.num_inference_steps = args.steps
        param.seed = args.seed
        param.save_video = False
        # Frame return would add a ~194 MB D->H copy per repetition, which is
        # not part of the generation being timed.
        param.return_frames = False
        param.prompt = prompt
        return param

    def one_run() -> dict[str, Any]:
        param = make_param()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = generator.generate_video(prompt, sampling_param=param)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "wall_seconds": elapsed,
            "reported_generation_time": result.get("generation_time"),
            "reported_e2e_latency": result.get("e2e_latency"),
            "peak_memory_mb": result.get("peak_memory_mb"),
            "component_times": _extract_component_times(result),
        }

    warmups = [one_run() for _ in range(args.warmup)]
    measured = [one_run() for _ in range(args.reps)]
    param = make_param()

    wall = [run["wall_seconds"] for run in measured]
    peaks = [run["peak_memory_mb"] for run in measured if run["peak_memory_mb"]]
    dit_times = [
        run["component_times"]["dit_time_s"] for run in measured if run["component_times"].get("dit_time_s") is not None
    ]

    payload: dict[str, Any] = {
        "run_id": args.run_id,
        "stage": "5-perf",
        "arm": args.arm,
        "native_latency_claim_allowed": args.arm in NATIVE_ARMS,
        "reporting_class": "headline" if args.arm in NATIVE_ARMS else "advisory",
        "why": ("native end-to-end path" if args.arm in NATIVE_ARMS else
                "real block-sparse kernel but research mask path with per-call fp64 scoring; "
                "not a deployable latency number"),
        "prompt_id": prompt_id,
        "prompt": prompt,
        "seed": args.seed,
        "steps": args.steps,
        "height": param.height,
        "width": param.width,
        "num_frames": param.num_frames,
        "guidance_scale": param.guidance_scale,
        "requested_sparsity": args.sparsity if sparse else None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "git_commit": phase5_config["git_commit"],
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "attention_backend": "SPARSEFP4_EXEC_ATTN",
        "flash_attention_4_enabled": os.environ.get("FASTVIDEO_FA4") == "1",
        "torch_compile": False,
        "cuda_graphs": False,
        "compile_mode": None,
        "num_warmup_runs": args.warmup,
        "num_measurement_runs": args.reps,
        "cuda_synchronized": True,
        "model_load_seconds": round(load_seconds, 2),
        "warmup_wall_seconds": [round(run["wall_seconds"], 3) for run in warmups],
        "e2e_wall_seconds": [round(value, 3) for value in wall],
        "e2e_wall_seconds_stats": dispersion(wall),
        "peak_memory_mb_max": max(peaks) if peaks else None,
        "peak_memory_mb_all": peaks,
        "dit_time_s_all": dit_times,
        "dit_time_s_stats": dispersion(dit_times) if dit_times else None,
        "per_step_dit_ms_median": (statistics.median(dit_times) / args.steps * 1000.0) if dit_times else None,
        "component_times_measured": [run["component_times"] for run in measured],
        "throughput_fps_median": param.num_frames / statistics.median(wall),
        "note": ("Warmed (first-call CuTeDSL JIT excluded), CUDA-synchronized, identical shapes/prompt/seed/steps. "
                 "This is measured wall clock, never a FLOP-count claim."),
    }

    out_path = raw_dir / f"phase5_perf_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "component_times_measured"}, indent=2))
    generator.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
