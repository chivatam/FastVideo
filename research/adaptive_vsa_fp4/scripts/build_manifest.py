from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fastvideo.eval.datasets import get_dataset

DEFAULT_ARTIFACT_ROOT = Path("artifacts/adaptive_vsa_fp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()

    rows = list(get_dataset("vbench", dimensions=["subject_consistency"]))
    prompts = []
    for index, row in enumerate(rows):
        digest = hashlib.sha256(row["prompt"].encode()).hexdigest()
        prompts.append({
            "prompt_id": f"vbench-subject-{index:03d}-{digest[:12]}",
            "index": index,
            "prompt": row["prompt"],
            "sha256": digest,
            "dimensions": row["dimensions"],
            "n_samples": row["n_samples"],
        })
    if len(prompts) != 72:
        raise RuntimeError(f"Expected 72 subject_consistency prompts, found {len(prompts)}")

    out = args.artifact_root / "phase0" / "vbench_subject_consistency_prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompts, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
