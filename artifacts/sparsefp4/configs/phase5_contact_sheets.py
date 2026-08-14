"""Phase 5 qualitative artifacts: paired contact sheets across the six arms.

A reader of the report will look at these before reading any table, so the sheet
is built to be read: one row per arm, one column per sampled frame, arm labels
burned into the left margin along with the provenance the labelling rules require
("NVFP4 Q/K + BF16 PV", "simulated -- no latency claim"). Every row is the same
prompt, the same seed, the same frame indices, so any visible difference between
rows is the arm and nothing else.

Also emits a difference sheet (``|arm - DENSE-BF16|``, amplified by a stated
gain) because at these magnitudes the raw frames of two arms can look identical
while differing structurally, and the amplified difference is what shows *where*
they differ.

    source artifacts/sparsefp4/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_contact_sheets.py \
        --run-id 20260814-032700-8208536-p5-main --prompts p01 p03 p05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REFERENCE_ARM = "DENSE-BF16"
ARM_ORDER = (
    "DENSE-BF16",
    "DENSE-FP4",
    "SPARSE-BF16",
    "SPARSE-FP4-NAIVE",
    "SPARSE-FP4-ROUTE8",
    "SPARSE-FP4-ROUTE16",
)
ARM_CAPTION = {
    "DENSE-BF16": "dense BF16 (native, reference)",
    "DENSE-FP4": "dense NVFP4 Q/K + BF16 PV (native)",
    "SPARSE-BF16": "sparse 0.90, BF16 compute, BF16 router (native kernel)",
    "SPARSE-FP4-NAIVE": "sparse 0.90, NVFP4 Q/K + BF16 PV, NVFP4 router (simulated)",
    "SPARSE-FP4-ROUTE8": "sparse 0.90, NVFP4 Q/K + BF16 PV, FP8 router (simulated)",
    "SPARSE-FP4-ROUTE16": "sparse 0.90, NVFP4 Q/K + BF16 PV, BF16 router (simulated)",
}
LABEL_WIDTH = 430
HEADER_HEIGHT = 46
PAD = 4


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_frames(path: Path) -> np.ndarray:
    """``[B, C, T, H, W]`` float16 -> ``(T, H, W, C)`` uint8."""
    array = np.load(path).astype(np.float32)
    if array.ndim == 5:
        array = array[0]
    array = np.transpose(array, (1, 2, 3, 0))
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def build_sheet(
    frames_by_arm: dict[str, np.ndarray],
    frame_indices: list[int],
    title: str,
    scale: float,
    difference: bool,
    gain: float,
) -> Image.Image:
    arms = [arm for arm in ARM_ORDER if arm in frames_by_arm]
    sample = frames_by_arm[arms[0]]
    height = int(sample.shape[1] * scale)
    width = int(sample.shape[2] * scale)
    sheet_w = LABEL_WIDTH + len(frame_indices) * (width + PAD) + PAD
    sheet_h = HEADER_HEIGHT + len(arms) * (height + PAD) + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD + 4, 10), title, fill=(255, 255, 255), font=_font(20))

    reference = frames_by_arm.get(REFERENCE_ARM)
    for row, arm in enumerate(arms):
        top = HEADER_HEIGHT + row * (height + PAD)
        draw.text((PAD + 4, top + 6), arm, fill=(255, 235, 120), font=_font(17))
        draw.text((PAD + 4, top + 28), ARM_CAPTION[arm], fill=(190, 190, 200), font=_font(12))
        for column, index in enumerate(frame_indices):
            frame = frames_by_arm[arm][index]
            if difference and reference is not None:
                delta = np.abs(frame.astype(np.int16) - reference[index].astype(np.int16))
                frame = np.clip(delta.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            image = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
            sheet.paste(image, (LABEL_WIDTH + column * (width + PAD), top))
            if row == 0:
                draw.text((LABEL_WIDTH + column * (width + PAD) + 4, HEADER_HEIGHT - 16),
                          f"frame {index}",
                          fill=(150, 150, 160),
                          font=_font(12))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompts", nargs="+", default=["p01", "p03", "p05", "p07"])
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 20, 40, 60, 80])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=0.42)
    parser.add_argument("--diff-gain", type=float, default=6.0)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/scratch/sparsefp4"))
    parser.add_argument("--video-root", type=Path, default=Path("/mnt/scratch/sparsefp4-videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sparsefp4/videos"))
    args = parser.parse_args()

    raw_dir = args.raw_root / args.run_id
    video_dir = args.video_root / args.run_id
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_text: dict[str, str] = {}
    for path in raw_dir.glob("run_summary_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        prompt_text[record["prompt_id"]] = record.get("prompt", "")

    written: list[dict[str, object]] = []
    for prompt in args.prompts:
        frames_by_arm: dict[str, np.ndarray] = {}
        for arm in ARM_ORDER:
            path = video_dir / f"{prompt}_{arm}_s{args.seed}.f16.npy"
            if path.is_file():
                frames_by_arm[arm] = load_frames(path)
        if len(frames_by_arm) < 2:
            print(f"skip {prompt}: only {len(frames_by_arm)} arms present")
            continue
        indices = [i for i in args.frames if i < next(iter(frames_by_arm.values())).shape[0]]
        caption = (prompt_text.get(prompt, "")[:110] + "...") if prompt_text.get(prompt) else ""
        for difference, suffix in ((False, "frames"), (True, "absdiff")):
            title = (f"SparseFP4 Phase 5 | {prompt} | seed {args.seed} | 480x832x81, 50 steps, sparsity 0.90 | "
                     f"{'|arm - DENSE-BF16| x' + str(args.diff_gain) if difference else caption}")
            sheet = build_sheet(frames_by_arm, indices, title, args.scale, difference, args.diff_gain)
            out_path = args.out_dir / f"contact_sheet_{prompt}_s{args.seed}_{suffix}.png"
            sheet.save(out_path, optimize=True)
            written.append({
                "prompt_id": prompt,
                "kind": suffix,
                "path": str(out_path),
                "arms": list(frames_by_arm),
                "frames": indices,
                "bytes": out_path.stat().st_size,
            })
            print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    manifest = args.out_dir / f"contact_sheets_{args.run_id}.json"
    manifest.write_text(json.dumps(
        {
            "run_id": args.run_id,
            "seed": args.seed,
            "diff_gain": args.diff_gain,
            "scale": args.scale,
            "note": ("Rows are arms at identical prompt/seed/frame index. SPARSE-FP4-* compute is "
                     "simulated NVFP4 Q/K + BF16 PV; no latency claim attaches to those rows."),
            "sheets": written,
        },
        indent=2,
    ) + "\n",
                          encoding="utf-8")
    print(f"manifest {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
