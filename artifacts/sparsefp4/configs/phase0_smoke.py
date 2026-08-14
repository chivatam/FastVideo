"""Phase 0 smoke test: dense BF16 vs dense native-NVFP4 attention on Wan2.1-T2V-1.3B.

SMOKE SETTINGS, NOT EXPERIMENT SETTINGS. Spatial dims and frame count are kept at
the framework defaults (480x832, 81 frames) so the attention kernels JIT-compile
and run at the real experiment shapes, but `--steps` is reduced far below the
framework default of 50. Wall-clock numbers from this script are therefore a
liveness check, not a benchmark: no warmup separation, one repetition, and the
FA4 CuTeDSL JIT compile cost lands inside the measured window.

Precision labelling (per the study's scientific-integrity rules):
  --mode bf16   -> native BF16 attention (FA4 CuTe BF16 kernel)
  --mode nvfp4  -> native NVFP4 attention (flash-attention-fp4, NVFP4 E2M1 Q/K
                   with per-16 E4M3 scale factors, BF16 V). Not simulated.
"""

import argparse
import json
import os
import subprocess
import threading
import time

from fastvideo import VideoGenerator

PROMPT = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
          "wide with interest. The playful yet serene atmosphere is complemented by soft "
          "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")


class GpuMemorySampler:
    """Polls nvidia-smi for device-level peak memory.

    In-process ``torch.cuda.max_memory_allocated`` reads 0 here because
    FastVideo runs the pipeline in a worker subprocess, so the parent's CUDA
    allocator never sees the activations.
    """

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.25) -> None:
        self._gpu_index = gpu_index
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_mib = 0

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self._gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
                self.peak_mib = max(self.peak_mib, int(out.stdout.strip().splitlines()[0]))
            except Exception:  # sampling must never break the run
                pass
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "GpuMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--steps", type=int, default=4, help="SMOKE value; framework default is 50")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    args = parser.parse_args()

    nvfp4 = args.mode == "nvfp4"
    print(f"[phase0-smoke] mode={args.mode} precision_label="
          f"{'native NVFP4 (E2M1 Q/K, per-16 E4M3 SF, BF16 V)' if nvfp4 else 'native BF16'}")
    print(f"[phase0-smoke] SMOKE steps={args.steps} (framework default 50); "
          f"dims={args.height}x{args.width} frames={args.num_frames} (both experiment defaults)")

    os.makedirs(args.out_dir, exist_ok=True)

    with GpuMemorySampler(gpu_index=0) as mem:
        load_t0 = time.perf_counter()
        generator = VideoGenerator.from_pretrained(
            args.model,
            num_gpus=1,
            nvfp4_fa4=nvfp4,
            # FSDP is disabled for BOTH modes: it is incompatible with the FP4
            # pointer path, and matching it across modes keeps the pair comparable.
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )
        load_s = time.perf_counter() - load_t0
        print(f"[phase0-smoke] load_seconds={load_s:.2f}")

        video_path = os.path.join(args.out_dir, f"phase0_smoke_{args.mode}.mp4")
        gen_t0 = time.perf_counter()
        generator.generate(
            request={
                "prompt": PROMPT,
                "sampling": {
                    "num_inference_steps": args.steps,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "seed": args.seed,
                },
                "output": {
                    "save_video": True,
                    "output_path": video_path,
                },
            })
        gen_s = time.perf_counter() - gen_t0
    peak_gib = mem.peak_mib / 1024

    metrics = {
        "mode":
        args.mode,
        "precision_label":
        ("native NVFP4 attention (E2M1 Q/K, per-16 E4M3 scale factors, BF16 V)" if nvfp4 else "native BF16 attention"),
        "is_simulated_quantization":
        False,
        "smoke_settings":
        True,
        "steps":
        args.steps,
        "framework_default_steps":
        50,
        "height":
        args.height,
        "width":
        args.width,
        "num_frames":
        args.num_frames,
        "seed":
        args.seed,
        "load_seconds":
        round(load_s, 2),
        "generate_seconds":
        round(gen_s, 2),
        "peak_gpu0_memory_gib_nvidia_smi":
        round(peak_gib, 2),
        "video_path":
        video_path,
        "video_exists":
        os.path.exists(video_path),
        "note": ("Wall-clock includes first-call FA4 CuTeDSL JIT compilation and is a liveness "
                 "check only; it is NOT a benchmark and must not be reported as a speedup."),
    }
    with open(args.metrics_json, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[phase0-smoke] METRICS " + json.dumps(metrics))

    generator.shutdown()


if __name__ == "__main__":
    main()
