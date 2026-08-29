from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SELECTION_SALT = "coretail-vsa-vbench2-external-v1"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_external_prompts(
    source_path: Path,
    evaluation_path: Path,
    *,
    calibration_count: int = 32,
    heldout_count: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = json.loads(source_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    excluded = {str(row["prompt"]).strip().casefold() for row in evaluation}
    rows = []
    for prompt, metadata in source.items():
        prompt = str(metadata.get("caption", prompt)).strip()
        if not prompt or prompt.casefold() in excluded:
            continue
        prompt_hash = _hash(prompt)
        selection_hash = _hash(f"{SELECTION_SALT}:{prompt}")
        rows.append({
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "selection_sha256": selection_hash,
            "source": "VBench-2.0 full text prompt benchmark",
            "source_dimensions": json.dumps(
                metadata.get("dimension", []),
                sort_keys=True,
            ),
        })
    frame = (pd.DataFrame(rows).drop_duplicates("prompt_sha256").sort_values("selection_sha256").reset_index(drop=True))
    required = calibration_count + heldout_count
    if len(frame) < required:
        raise RuntimeError(f"External prompt pool has {len(frame)} rows; need {required}")
    selected = frame.iloc[:required].copy()
    selected["selection_rank"] = range(1, required + 1)
    selected["split"] = [
        *("external_calibration" for _ in range(calibration_count)),
        *("heldout_offline" for _ in range(heldout_count)),
    ]
    selected["prompt_id"] = [
        f"vbench2-coretail-{index:03d}-{digest[:12]}" for index, digest in enumerate(selected["prompt_sha256"])
    ]
    calibration = selected.loc[selected["split"].eq("external_calibration")].copy()
    heldout = selected.loc[selected["split"].eq("heldout_offline")].copy()
    if set(calibration["prompt_sha256"]) & set(heldout["prompt_sha256"]):
        raise RuntimeError("Calibration and held-out prompt sets overlap")
    return calibration, heldout


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [{
        "prompt_id":
        row.prompt_id,
        "prompt":
        row.prompt,
        "sha256":
        row.prompt_sha256,
        "source":
        row.source,
        "source_dimensions":
        json.loads(row.source_dimensions),
        "selection_method": (f"lowest SHA256({SELECTION_SALT}:prompt), excluding all "
                             "72 subject-consistency evaluation prompt texts"),
        "selection_rank":
        int(row.selection_rank),
        "split":
        row.split,
    } for row in frame.itertuples()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("fastvideo/third_party/eval/vbench/VBench-2.0/prompts/"
                     "VBench2_full_text_info.json"),
    )
    parser.add_argument(
        "--evaluation-prompts",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/adaptive_vsa_fp4/phase0/"
                     "vbench_subject_consistency_prompts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration, heldout = select_external_prompts(
        args.source,
        args.evaluation_prompts,
    )
    calibration.to_csv(
        args.output / "external_calibration_prompts.csv",
        index=False,
    )
    heldout.to_csv(
        args.output / "heldout_offline_prompts.csv",
        index=False,
    )
    (args.output / "external_calibration_prompts.json").write_text(json.dumps(_records(calibration), indent=2) + "\n")
    (args.output / "heldout_offline_prompts.json").write_text(json.dumps(_records(heldout), indent=2) + "\n")
    hashes = pd.concat([calibration, heldout]).loc[:, ["split", "prompt_id", "prompt_sha256"]]
    hashes.to_csv(
        args.output / "prompt_hashes.txt",
        sep="\t",
        index=False,
    )
    print(f"calibration={len(calibration)} heldout={len(heldout)} "
          f"overlap=0")


if __name__ == "__main__":
    main()
