"""Assemble tables/c8_performance.md from the collected performance receipts."""

import json
import statistics
from pathlib import Path

ART = Path("artifacts/sparsefp4_native")
PERF_DIR = Path("/mnt/nvme/scratch/sparsefp4_native/p_runs/perf-s090")

ARM_LABELS = {
    "P0": "P0 dense BF16 (FA4)",
    "P1": "P1 dense native NVFP4",
    "P2": "P2 deployed VSA@0.9 (Triton fine)",
    "P2G": "P2G VSA sel. + FA4 BF16 fine (24% kept)",
    "P3": "P3 VSA sel. + native NVFP4 fine (24% kept)",
    "P4G": "P4G VSA256-FA4 BF16 fine (10% kept, exact)",
    "P4": "P4 VSA256-FA4 native NVFP4 fine (10% kept, exact)",
}


def main():
    lines = ["# C8 — Performance", ""]

    # kernel-level from c3
    c3 = json.load(open(ART / "raw/performance/c3_native_proof.json"))
    lines += ["## Attention-kernel latency (CUDA events, median of 50, Wan shape "
              "B=1 S=39936 H=12 D=128, pre-quantized inputs)", "",
              "| Arm | retained | median ms | p90 ms |", "|---|---|---|---|"]
    for r in c3["work_scaling"]:
        ret = f"{r.get('retained', 1.0):.2f}" if "retained" in r else "dense"
        lines.append(f"| {r['arm']} | {ret} | {r['median_ms']:.3f} | {r['p90_ms']:.3f} |")

    # fine-branch wall decomposition
    try:
        fw = json.load(open(ART / "raw/performance/fine_branch_wall.json"))
        qo = json.load(open(ART / "raw/performance/quant_overhead.json"))
        lines += ["", "## Fine-branch wall-clock per call (24%-kept mask, incl. host overhead)", "",
                  "| Branch | wall ms | CUDA-event ms |", "|---|---|---|"]
        for k, v in fw.items():
            lines.append(f"| {k} | {v['wall_ms']:.3f} | {v['event_ms']:.3f} |")
        lines.append("")
        lines.append(f"Quantize overhead: {qo['per_call_qk_ms']:.3f} ms per call (Q+K), "
                     f"~{qo['per_video_s']:.1f} s per 50-step CFG video.")
    except FileNotFoundError:
        pass

    # E2E arms — 480p and 720p
    for res, perf_dir in (("480x832x81", PERF_DIR),
                          ("720x1280x81", PERF_DIR.parent / "perf720-s090")):
        lines += ["", f"## End-to-end at {res} (median steady-state reps; 50 steps, "
                  "seed 1234, prompt p00; first gen excluded as warmup/JIT)", "",
                  "| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB |",
                  "|---|---|---|---|---|"]
        base = None
        for arm in ("P0", "P1", "P2", "P2G", "P3", "P4G", "P4"):
            p = perf_dir / f"summary_p00_{arm}_s1234.json"
            if not p.is_file():
                if res.startswith("480") or arm in ("P0", "P1", "P2", "P4G", "P4"):
                    lines.append(f"| {ARM_LABELS[arm]} | (n/a) | | | |")
                continue
            d = json.load(open(p))
            reps = d["perf_reps"]
            e2e = statistics.median(r["e2e_s"] for r in reps)
            dit = statistics.median(r["dit_s"] for r in reps if r["dit_s"])
            mb = max((r["peak_memory_mb"] or 0) for r in reps)
            if arm == "P0":
                base = e2e
            sp = f"{base / e2e:.3f}x" if base else ""
            lines.append(f"| {ARM_LABELS[arm]} | {e2e:.2f} | {sp} | {dit:.2f} | {mb:.0f} |")

    lines += ["",
              "Notes: all arms share checkpoint/scheduler/steps/resolution/frames/",
              "guidance/seed/negative prompt; no torch.compile/CUDA graphs; one",
              "process per arm. Kernel table excludes quantization (pre-quantized);",
              "E2E includes everything. Never inferred from FLOPs."]

    out = ART / "tables/c8_performance.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
