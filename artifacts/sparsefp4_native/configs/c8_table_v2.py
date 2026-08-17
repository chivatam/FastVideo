"""Assemble tables/c8_performance_v2.md from the V2 controlled perf receipts.

All E2E arms ran under the SAME allocator configuration
(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True); no pre/post-allocator-fix
numbers are mixed. Kernel-level tables are reused from the c3 receipts
(allocator-independent: pre-quantized inputs, CUDA-event timed).
"""

import json
import statistics
from pathlib import Path

ART = Path("artifacts/sparsefp4_native")
PERF480 = Path("/mnt/nvme/scratch/sparsefp4_native/p_runs/perf480-v2")
PERF720 = Path("/mnt/nvme/scratch/sparsefp4_native/p_runs/perf720-v2")

ARM_LABELS = {
    "P0": "P0 dense BF16 (FA4)",
    "P1": "P1 dense native NVFP4",
    "P2": "P2 deployed VSA@0.9 (Triton fine)",
    "P4G": "P4G VSA256-FA4 BF16 fine (10% kept, exact)",
    "P4": "P4 VSA256-FA4 native NVFP4 fine (10% kept, exact)",
}


def main():
    lines = ["# C8 — Performance (V2, unified allocator configuration)", "",
             "Every E2E arm below ran with `PYTORCH_CUDA_ALLOC_CONF="
             "expandable_segments:True` in a fresh process (receipts: "
             "`raw/performance/perf_v2/`, logs: `logs/perf_v2/`). The old "
             "`tables/c8_performance.md` mixed pre-allocator-fix numbers "
             "(e.g. P4 720p 250.9 s) and is retained as historical/root-cause "
             "evidence only — see `P4_PERF_ROOT_CAUSE.md`.", ""]

    # kernel-level from c3 (unchanged receipts; allocator-independent)
    c3 = json.load(open(ART / "raw/performance/c3_native_proof.json"))
    lines += ["## Attention-kernel latency (CUDA events, median of 50, Wan shape "
              "B=1 S=39936 H=12 D=128, pre-quantized inputs)", "",
              "| Arm | retained | median ms | p90 ms |", "|---|---|---|---|"]
    for r in c3["work_scaling"]:
        ret = f"{r.get('retained', 1.0):.2f}" if "retained" in r else "dense"
        lines.append(f"| {r['arm']} | {ret} | {r['median_ms']:.3f} | {r['p90_ms']:.3f} |")

    for res, perf_dir in (("480x832x81", PERF480), ("720x1280x81", PERF720)):
        lines += ["", f"## End-to-end at {res} (median of 3 steady-state reps; 50 steps, "
                  "seed 1234, prompt p00; first gen excluded as warmup/JIT; "
                  "expandable_segments allocator in every arm)", "",
                  "| System | E2E s | E2E speedup vs P0 | DiT s | Peak MB | alloc_conf |",
                  "|---|---|---|---|---|---|"]
        base = None
        for arm in ("P0", "P1", "P2", "P4G", "P4"):
            p = perf_dir / f"summary_p00_{arm}_s1234.json"
            d = json.load(open(p))
            reps = d["perf_reps"]
            e2e = statistics.median(r["e2e_s"] for r in reps)
            dit = statistics.median(r["dit_s"] for r in reps if r["dit_s"])
            mb = max((r["peak_memory_mb"] or 0) for r in reps)
            alloc = d.get("alloc_conf") or "(unset)"
            if arm == "P0":
                base = e2e
            sp = f"{base / e2e:.3f}x" if base else ""
            lines.append(f"| {ARM_LABELS[arm]} | {e2e:.2f} | {sp} | {dit:.2f} | {mb:.0f} | {alloc} |")

    lines += ["",
              "Notes: all arms share checkpoint/scheduler/steps/resolution/frames/",
              "guidance/seed/negative prompt; no torch.compile/CUDA graphs; one",
              "process per arm; identical `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.",
              "Kernel table excludes quantization (pre-quantized); E2E includes",
              "everything. Never inferred from FLOPs."]

    out = ART / "tables/c8_performance_v2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
