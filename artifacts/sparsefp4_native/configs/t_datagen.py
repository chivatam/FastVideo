"""DQ-VSA T-matrix training corpus: motion-diverse synthetic clips.

Teacher-generated data (P0 dense BF16 operator, Wan2.1-1.3B) over the full
946-prompt VBench corpus (all 16 dimensions -> broad camera/object/human
motion, spatial detail, scene diversity), with motion-heavy prompts
(dynamic_degree/human_action) oversampled with two extra seeds. Generation
seeds (100-102) are disjoint from the evaluation seed (1234).

One process per shard:
    t_datagen.py --shard 0 --num-shards 8
Then once, to write merge.txt/videos2caption.json:
    t_datagen.py --finalize
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CORPUS = Path("/mnt/nvme/scratch/sparsefp4_native/t_corpus")
VIDEOS = CORPUS / "videos"
BASE_SEED = 100
MOTION_DIMS = {"dynamic_degree", "human_action"}
HEIGHT, WIDTH, FRAMES, FPS, STEPS = 480, 832, 81, 16, 50


def job_list():
    from fastvideo.eval.datasets.vbench import VBenchPromptDataset
    prompts = list(VBenchPromptDataset(dimensions="all"))
    jobs = []
    for i, entry in enumerate(prompts):
        seeds = [BASE_SEED]
        if MOTION_DIMS & set(entry["dimensions"]):
            seeds += [BASE_SEED + 1, BASE_SEED + 2]
        for s in seeds:
            jobs.append((i, s, entry["prompt"]))
    return jobs


def generate(shard: int, num_shards: int) -> None:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["FASTVIDEO_FA4"] = "1"
    os.environ.pop("FASTVIDEO_NVFP4_FA4", None)

    import imageio

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    VIDEOS.mkdir(parents=True, exist_ok=True)
    jobs = [j for k, j in enumerate(job_list()) if k % num_shards == shard]
    todo = [(i, s, p) for i, s, p in jobs
            if not (VIDEOS / f"v{i:04d}_s{s}.mp4").is_file()]
    print(f"shard {shard}: {len(todo)}/{len(jobs)} clips to generate", flush=True)
    if not todo:
        print("DATAGEN_SHARD_DONE", flush=True)
        return

    gen = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=True)
    sp = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    sp.num_inference_steps = STEPS
    sp.height, sp.width = HEIGHT, WIDTH
    sp.save_video = False
    sp.return_frames = True

    meta = []
    meta_path = CORPUS / f"meta.shard{shard}.jsonl"
    for i, seed, prompt in todo:
        sp.seed = seed
        sp.prompt = prompt
        result = gen.generate_video(prompt, sampling_param=sp)
        frames = result.get("frames")
        out = VIDEOS / f"v{i:04d}_s{seed}.mp4"
        imageio.mimsave(out, frames, fps=sp.fps, format="mp4")
        meta.append(dict(path=out.name,
                         resolution=dict(width=WIDTH, height=HEIGHT),
                         size=out.stat().st_size, fps=float(sp.fps),
                         duration=len(frames) / sp.fps,
                         num_frames=len(frames), cap=[prompt]))
        with open(meta_path, "w") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        print(f"shard {shard}: wrote {out.name}", flush=True)
    gen.shutdown()
    print("DATAGEN_SHARD_DONE", flush=True)


def finalize() -> None:
    entries, seen = [], set()
    for f in sorted(CORPUS.glob("meta.shard*.jsonl")):
        for line in open(f):
            m = json.loads(line)
            if m["path"] in seen or not (VIDEOS / m["path"]).is_file():
                continue
            seen.add(m["path"])
            entries.append(m)
    (CORPUS / "videos2caption.json").write_text(json.dumps(entries, indent=1))
    (CORPUS / "merge.txt").write_text(f"{VIDEOS},{CORPUS / 'videos2caption.json'}\n")
    print(f"finalized {len(entries)} clips -> {CORPUS / 'videos2caption.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
    else:
        generate(args.shard, args.num_shards)


if __name__ == "__main__":
    main()
