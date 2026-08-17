"""Extract the few columns the statistics need from multi-gigabyte JSONL shards.

The raw shards are ~4.7 GB (F1) and ~3.5 GB (F2) because every record carries full
provenance and semantics strings. Bootstrap resampling only needs five numeric columns
plus the grouping keys, and re-parsing the JSON on every analysis pass costs ~10
minutes per phase. This writes a compact ``.npz`` per phase-sparsity so the statistics
can be re-run in seconds.

Columns are stored as parallel arrays with string keys factorized to integer codes, so
the cache is a few megabytes rather than gigabytes.

    source artifacts/sparsefp4_followup/configs/env.sh
    "$FV_PYTHON" artifacts/sparsefp4_followup/configs/build_stats_cache.py \
        --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
        --out artifacts/sparsefp4_followup/raw/cache/f1_full.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Everything the F3/F4 statistics consume. Kept explicit so a column added here is a
# deliberate act rather than an accidental cache bloat.
NUMERIC = ("wrong_mask_excess", "random_matched_excess", "sparsification_error", "jaccard", "recall",
           "swaps_per_query_block", "sparsity", "seq_len", "retained_k")
CATEGORICAL = ("prompt_id", "arm", "cfg_branch", "geometry", "resolution", "native_or_simulated")
INTEGER = ("layer", "timestep", "head", "seed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    numeric: dict[str, list[float]] = {name: [] for name in NUMERIC}
    integer: dict[str, list[int]] = {name: [] for name in INTEGER}
    categorical: dict[str, list[str]] = {name: [] for name in CATEGORICAL}
    n_rows = 0

    for shard in args.shard:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                n_rows += 1
                for name in NUMERIC:
                    value = record.get(name)
                    numeric[name].append(float("nan") if value is None else float(value))
                for name in INTEGER:
                    value = record.get(name)
                    integer[name].append(-1 if value is None else int(value))
                for name in CATEGORICAL:
                    categorical[name].append(str(record.get(name)))
        print(f"  {shard.name}: {n_rows} rows cumulative")

    payload: dict[str, np.ndarray] = {}
    for name, values in numeric.items():
        payload[f"num_{name}"] = np.asarray(values, dtype=np.float64)
    for name, ivalues in integer.items():
        payload[f"int_{name}"] = np.asarray(ivalues, dtype=np.int32)
    for name, svalues in categorical.items():
        levels, codes = np.unique(np.asarray(svalues), return_inverse=True)
        payload[f"cat_{name}_codes"] = codes.astype(np.int32)
        payload[f"cat_{name}_levels"] = levels

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size_mb = args.out.stat().st_size / 1048576
    print(f"\nwrote {args.out} ({size_mb:.1f} MB) for {n_rows} rows from {len(args.shard)} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
