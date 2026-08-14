"""Phase 5 trajectory-divergence control: why the pixel numbers are large.

The Phase 5 pixel comparison found that swapping the *router* precision at fixed
sparsity moves the final video by a pixel MAE only ~5x smaller than sparsifying
at all -- while Phase 2 measured the wrong-mask error term at 0.02%-0.03% of
attention error, i.e. 200-500x below the H3 threshold. Those two facts are not
in conflict, but only one of two explanations can be right, and they have
opposite implications:

**(a) Amplification.** A 50-step denoising loop is a chaotic map. Any
per-step perturbation, however small, is amplified until the two trajectories
decorrelate, at which point the pixel difference saturates at "two different
samples from the same model". In that regime the pixel difference measures
*whether* the trajectories diverged, not *by how much* the attention differed,
so it cannot rank perturbation magnitudes at all.

**(b) Routing precision genuinely matters end-to-end**, and Phase 2's
single-step decomposition missed a real accumulating effect.

The discriminating measurement: run the *reference* trajectory (dense BF16,
exactly configuration A) and at each of a set of steps compute, from the same
captured Q/K/V, the attention output each arm *would* produce. That is a
single-step, exactly-paired quantity -- the same thing Phase 2 measured -- but
now sampled along a real trajectory at all layers. If the arms' single-step
divergences reproduce Phase 2's ordering (routing precision ~1e-4 of the
sparsity effect) while the free-running pixel differences do not, explanation
(a) is established and the pixel numbers must be reported as saturated.

The second discriminator is a step-truncation sweep, run separately: apply the
arm's attention for only the first N steps and dense BF16 afterwards. Under (a)
the pixel difference grows with N and saturates; under (b) it scales with the
per-step error.

Emits one JSONL row per (layer, timestep, cfg_branch, head, arm-pair).

    FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_EXEC_ATTN is *not* used here; this
    probe rides on the Phase 5 backend's sibling config so the compute is
    dense BF16 and every arm is a side computation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = REPO_ROOT / ".agents/skills/sparsefp4-video-attention/assets/prompts.txt"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 5, 12, 25, 37, 49])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    args = parser.parse_args()

    prompt = load_prompts(args.prompts)[args.prompt_index]
    prompt_id = f"p{args.prompt_index + 1:02d}"
    out_dir = args.raw_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the Phase 2 probe backend verbatim: dense BF16 compute, side-channel
    # A-F decomposition, fp64 scores, exactly-paired arms. The only difference
    # from the Phase 2 main run is coverage -- all 30 layers, 6 timesteps, at the
    # single matched sparsity Phase 5 uses.
    phase2_config = {
        "out_dir": str(out_dir),
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "prompt_id": prompt_id,
        "seed": args.seed,
        "sparsities": [args.sparsity],
        "layers": args.layers,
        "timesteps": args.timesteps,
        "cfg_branches": ["positive", "negative"],
        "mechanism_layers": [],
        "mechanism_timesteps": [],
        "mechanism_sparsities": [],
        "mechanism_query_blocks": 0,
        "block_q": 128,
        "block_k": 64,
        "score_dtype": "float64",
        "shard_tag": prompt_id,
        "stage": "5-singlestep",
        "random_seed": 20260814 + args.prompt_index,
        "provenance": {
            "model_id": MODEL_ID,
            "num_inference_steps": args.steps,
            "prompt": prompt,
            "purpose": ("single-step exactly-paired attention divergence along the real dense-BF16 "
                        "trajectory, at the same sparsity as the Phase 5 free-running video runs"),
        },
    }
    config_path = out_dir / f"phase5_singlestep_config_{prompt_id}.json"
    config_path.write_text(json.dumps(phase2_config, indent=2) + "\n", encoding="utf-8")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "PRECISION_SPARSE_ATTN"
    os.environ["FASTVIDEO_SPARSEFP4_PHASE2"] = str(config_path)

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    torch.manual_seed(args.seed)
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    sampling_param = SamplingParam.from_pretrained(MODEL_ID)
    sampling_param.num_inference_steps = args.steps
    sampling_param.seed = args.seed
    sampling_param.save_video = False
    sampling_param.return_frames = False
    sampling_param.prompt = prompt
    generator.generate_video(prompt, sampling_param=sampling_param)
    generator.shutdown()

    shard = out_dir / f"{prompt_id}.jsonl"
    records = 0
    by_config: dict[str, list[float]] = {}
    if shard.is_file():
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records += 1
                if row.get("record_type") != "error_decomposition" or row.get("rel_l2") is None:
                    continue
                by_config.setdefault(row["config"], []).append(float(row["rel_l2"]))

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "stage": "5-singlestep",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "seed": args.seed,
        "sparsity": args.sparsity,
        "layers": args.layers,
        "timesteps": args.timesteps,
        "records": records,
        "shard": str(shard),
        "median_rel_l2_by_config": {name: statistics.median(values)
                                    for name, values in sorted(by_config.items())},
        "n_by_config": {name: len(values) for name, values in sorted(by_config.items())},
        "config_legend": {
            "A": "reference: dense BF16",
            "B": "dense NVFP4 Q/K + BF16 PV (native)",
            "C": "sparse BF16 compute, BF16 router",
            "D": "sparse BF16 compute, NVFP4 router",
            "D8": "sparse BF16 compute, FP8 router",
            "E": "SPARSE-FP4-NAIVE single-step equivalent (simulated compute, NVFP4 router)",
            "F8": "SPARSE-FP4-ROUTE8 single-step equivalent (simulated compute, FP8 router)",
            "F16": "SPARSE-FP4-ROUTE16 single-step equivalent (simulated compute, BF16 router)",
            "C_rand": "equal-magnitude random wrong-mask contrast control",
        },
    }
    (out_dir / f"phase5_singlestep_summary_{prompt_id}.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
