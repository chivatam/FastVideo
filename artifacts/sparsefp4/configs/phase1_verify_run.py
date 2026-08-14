"""Verify a SparseFP4 Phase-1 run before any aggregate is quoted from it.

Two failure modes this exists to catch:

1. **A silently-ignored backend override.** A typo in ``FASTVIDEO_ATTENTION_BACKEND``
   falls through to auto-selection instead of erroring
   (``fastvideo/attention/selector.py``), so a run can confidently measure the
   *default* attention path. The probe only writes a record when
   ``RoutingProbeAttentionImpl.forward`` actually runs, so a **complete** (layer,
   head, timestep, cfg_branch, sparsity, routing_precision) lattice is positive
   proof that every self-attention layer resolved onto the probe. A partial
   lattice means some layers fell back.
2. **A harness regression.** The bf16-routing-vs-bf16-reference arm is an
   identity by construction; if it ever deviates from Jaccard 1.0 the measurement
   is not trustworthy and the run must be debugged, not reported.

Usage::

    "$FV_PYTHON" artifacts/sparsefp4/configs/phase1_verify_run.py \
        --raw-dir /mnt/scratch/sparsefp4/<run_id> --expect-layers 30 --expect-heads 12
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any

INVARIANT_TOLERANCE = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--expect-layers", type=int, default=30)
    parser.add_argument("--expect-heads", type=int, default=12)
    parser.add_argument("--expect-timesteps",
                        type=int,
                        default=None,
                        help="number of distinct measured step indices (default: infer and report)")
    parser.add_argument("--expect-backend", default="ROUTING_PROBE_ATTN")
    parser.add_argument("--out", type=Path, default=None, help="write the verdict JSON here as well as stdout")
    return parser


def check_invariants(record: dict[str, Any]) -> list[str]:
    """EXPERIMENT_SPEC 6.4, re-checked on disk rather than trusted from the writer."""
    problems: list[str] = []
    intersection = float(record["intersection"])
    union = float(record["union"])
    reference = float(record["selected_reference"])
    candidate = float(record["selected_candidate"])
    if reference != candidate:
        problems.append("unequal_budget")
    if not 0.0 <= intersection <= min(reference, candidate):
        problems.append("intersection_out_of_range")
    if abs(union - (reference + candidate - intersection)) > INVARIANT_TOLERANCE:
        problems.append("union_mismatch")
    if abs(float(record["recall"]) - intersection / reference) > INVARIANT_TOLERANCE:
        problems.append("recall_mismatch")
    if abs(float(record["jaccard"]) - intersection / union) > INVARIANT_TOLERANCE:
        problems.append("jaccard_mismatch")
    if record["reference_precision"] != "bf16":
        problems.append("bad_reference_precision")
    if not 0.0 <= float(record["sparsity"]) < 1.0:
        problems.append("sparsity_out_of_range")
    if record["native_or_simulated"] not in ("native", "simulated"):
        problems.append("bad_provenance_enum")
    return problems


def main() -> int:
    args = build_parser().parse_args()
    # Accept both the live uncompressed shards and the gzipped archival copies, so
    # the same gate can re-verify what actually shipped under artifacts/.
    shards = sorted({*args.raw_dir.glob("*.jsonl"), *args.raw_dir.glob("*.jsonl.gz")})
    if not shards:
        raise SystemExit(f"no *.jsonl / *.jsonl.gz shards under {args.raw_dir}")

    total = 0
    malformed = 0
    invariant_failures: collections.Counter[str] = collections.Counter()
    null_control_records = 0
    null_control_failures = 0
    layers: set[int] = set()
    heads: set[int] = set()
    timesteps: set[int] = set()
    branches: set[str] = set()
    prompts: set[str] = set()
    sparsities: set[float] = set()
    precisions: set[str] = set()
    backends: set[str] = set()
    seq_lens: set[int] = set()
    scorers: set[str] = set()
    provenance: dict[str, set[str]] = collections.defaultdict(set)
    # k must be geometry-only, hence identical across precision arms for a cell.
    k_by_cell: dict[tuple[Any, ...], set[int]] = collections.defaultdict(set)
    per_shard: dict[str, int] = {}
    cell_counts: collections.Counter[tuple[Any, ...]] = collections.Counter()

    for shard in shards:
        shard_records = 0
        opener = gzip.open if shard.suffix == ".gz" else open
        with opener(shard, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                shard_records += 1
                for problem in check_invariants(record):
                    invariant_failures[problem] += 1
                layers.add(int(record["layer"]))
                heads.add(int(record["head"]))
                timesteps.add(int(record["timestep"]))
                branches.add(str(record.get("cfg_branch", "unknown")))
                prompts.add(str(record["prompt_id"]))
                sparsities.add(float(record["sparsity"]))
                precisions.add(str(record["routing_precision"]))
                backends.add(str(record.get("attention_backend", "unknown")))
                seq_lens.add(int(record["seq_len"]))
                scorers.add(str(record.get("scorer", "unknown")))
                provenance[str(record["routing_precision"])].add(str(record["native_or_simulated"]))
                cell = (record["prompt_id"], record["layer"], record["head"], record["timestep"],
                        record.get("cfg_branch"), record["sparsity"])
                k_by_cell[cell].add(int(record["k_per_query_block"]))
                cell_counts[(record["prompt_id"], record["routing_precision"], record["sparsity"])] += 1
                if record["routing_precision"] == record["reference_precision"]:
                    null_control_records += 1
                    if (float(record["jaccard"]) != 1.0 or float(record["recall"]) != 1.0
                            or float(record.get("frac_query_blocks_changed", 0.0)) != 0.0):
                        null_control_failures += 1
        per_shard[shard.name] = shard_records

    k_disagreements = sum(1 for values in k_by_cell.values() if len(values) > 1)
    non_null_precisions = sorted(precisions - {"bf16"})
    expected_non_null = (len(prompts) * len(layers) * len(heads) * len(timesteps) * len(branches) * len(sparsities) *
                         len(non_null_precisions))
    observed_non_null = sum(count for (_, precision, _), count in cell_counts.items() if precision != "bf16")

    verdict = {
        "raw_dir": str(args.raw_dir),
        "shards": per_shard,
        "records_total": total,
        "malformed_lines": malformed,
        "prompts": sorted(prompts),
        "n_layers": len(layers),
        "n_heads": len(heads),
        "n_timesteps": len(timesteps),
        "timesteps": sorted(timesteps),
        "cfg_branches": sorted(branches),
        "sparsities": sorted(sparsities),
        "routing_precisions": sorted(precisions),
        "provenance_by_arm": {
            key: sorted(value)
            for key, value in sorted(provenance.items())
        },
        "attention_backends": sorted(backends),
        "seq_lens": sorted(seq_lens),
        "scorers": sorted(scorers),
        "null_control_records": null_control_records,
        "null_control_failures": null_control_failures,
        "invariant_failures": dict(invariant_failures),
        "k_disagreements_across_arms": k_disagreements,
        "non_null_records_expected": expected_non_null,
        "non_null_records_observed": observed_non_null,
    }

    failures: list[str] = []
    if malformed:
        failures.append(f"{malformed} malformed line(s)")
    if invariant_failures:
        failures.append(f"schema invariant failures: {dict(invariant_failures)}")
    if null_control_failures:
        failures.append(f"NULL CONTROL FAILED on {null_control_failures} record(s)")
    if not null_control_records:
        failures.append("no null-control (bf16 vs bf16) records present")
    if k_disagreements:
        failures.append(f"k differs across precision arms in {k_disagreements} cell(s)")
    if backends != {args.expect_backend}:
        failures.append(f"attention_backend is {sorted(backends)}, expected [{args.expect_backend!r}]")
    if len(seq_lens) != 1:
        failures.append(f"seq_len is not constant within the run: {sorted(seq_lens)}")
    if len(layers) != args.expect_layers:
        failures.append(f"incomplete layer lattice: {len(layers)} of {args.expect_layers} "
                        "(a missing layer means that layer fell back off the probe backend)")
    if len(heads) != args.expect_heads:
        failures.append(f"incomplete head lattice: {len(heads)} of {args.expect_heads}")
    if args.expect_timesteps is not None and len(timesteps) != args.expect_timesteps:
        failures.append(f"incomplete timestep lattice: {len(timesteps)} of {args.expect_timesteps}")
    if observed_non_null != expected_non_null:
        failures.append(f"non-null record count {observed_non_null} != full lattice {expected_non_null}")

    verdict["failures"] = failures
    verdict["verdict"] = "PASS" if not failures else "FAIL"
    text = json.dumps(verdict, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
