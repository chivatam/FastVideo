from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from research.adaptive_vsa_fp4.scripts.worker import MODEL_REVISIONS, canonical_job_id, run_worker

DEFAULT_ARTIFACT_ROOT = Path("artifacts/adaptive_vsa_fp4")
MODE_PRECISION = {
    "dense_bf16_fa4": "bf16",
    "vsa_bf16": "bf16",
    "adaptive_vsa": "bf16",
    "ra_vsa": "bf16",
    "rectified_vsa": "bf16",
    "compressed_halo_vsa": "bf16",
    "br_vsa_census": "bf16",
    "br_vsa": "bf16",
    "fine_vsa_census": "bf16",
    "fine_vsa": "bf16",
    "anchored_fine_vsa_census": "bf16",
    "anchored_fine_vsa25": "bf16",
    "anchored_fine_vsa50": "bf16",
    "hierarchical_vsa_census": "bf16",
    "dense_nvfp4_fa4": "nvfp4_qk",
    "sim_vsa_nvfp4": "sim_nvfp4_qk",
}


def _code_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            ordinal INTEGER NOT NULL,
            phase INTEGER NOT NULL,
            mode TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            worker_id INTEGER,
            hostname TEXT,
            started_at REAL,
            finished_at REAL,
            result_path TEXT,
            error TEXT
        )
        """
    )
    conn.commit()
    return conn


def _mode_sparsities(
    mode: str,
    values: list[float],
) -> list[float]:
    return [0.0] if mode.startswith("dense_") else values


def prepare_jobs(args: argparse.Namespace) -> int:
    prompts = json.loads(args.prompts.read_text())
    if args.limit is not None:
        prompts = prompts[: args.limit]
    model_revision = MODEL_REVISIONS[args.model]
    code_commit = _code_commit()
    conn = _init_db(args.db)
    ordinal = conn.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM jobs").fetchone()[0]
    created = 0
    for mode in args.modes:
        sparsities = (
            [args.adaptive_floor_sparsity]
            if mode == "adaptive_vsa"
            else [args.ra_native_sparsity]
            if mode == "ra_vsa"
            else [0.8]
            if mode
            in {
                "rectified_vsa",
                "compressed_halo_vsa",
                "br_vsa_census",
                "br_vsa",
                "fine_vsa_census",
                "fine_vsa",
                "anchored_fine_vsa_census",
                "anchored_fine_vsa25",
                "anchored_fine_vsa50",
                "hierarchical_vsa_census",
            }
            else _mode_sparsities(
                mode,
                args.sparsities,
            )
        )
        for sparsity in sparsities:
            for prompt in prompts:
                payload: dict[str, Any] = {
                    "phase": args.phase,
                    "mode": mode,
                    "model": args.model,
                    "model_revision": model_revision,
                    "prompt_id": prompt["prompt_id"],
                    "prompt": prompt["prompt"],
                    "seed": args.seed,
                    "sparsity": sparsity,
                    "topk": None,
                    "precision": MODE_PRECISION[mode],
                    "adaptive_p": (args.adaptive_p if mode == "adaptive_vsa" else None),
                    "adaptive_floor_sparsity": (args.adaptive_floor_sparsity if mode == "adaptive_vsa" else None),
                    "adaptive_candidate_sparsities": (
                        args.adaptive_candidate_sparsities if mode == "adaptive_vsa" else None
                    ),
                    "adaptive_native_sparsity": (args.adaptive_native_sparsity if mode == "adaptive_vsa" else None),
                    "ra_native_fraction": (args.ra_native_fraction if mode == "ra_vsa" else None),
                    "ra_native_sparsity": (args.ra_native_sparsity if mode == "ra_vsa" else None),
                    "ra_risk_formula": (args.ra_risk_formula if mode == "ra_vsa" else None),
                    "ra_instrument_splits": (args.ra_instrument_splits if mode == "ra_vsa" else None),
                    "ra_detailed_trace": (not args.ra_minimal_trace if mode == "ra_vsa" else None),
                    "ra_force_outside_native": (args.ra_force_outside_native if mode == "ra_vsa" else None),
                    "cs_detailed_trace": (
                        not args.cs_minimal_trace
                        if mode
                        in {
                            "rectified_vsa",
                            "compressed_halo_vsa",
                        }
                        else None
                    ),
                    "br_candidate_k": (
                        args.br_candidate_k
                        if mode == "br_vsa_census"
                        else None
                    ),
                    "br_k_table_path": (
                        str(args.br_k_table)
                        if mode == "br_vsa" and args.br_k_table is not None
                        else None
                    ),
                    "height": args.height,
                    "width": args.width,
                    "frames": args.frames,
                    "fps": args.fps,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "scheduler": "model_default",
                    "code_commit": code_commit,
                    "save_video": not args.no_save_video,
                    "dry_run_seconds": args.dry_run_seconds,
                }
                job_id = canonical_job_id(payload)
                payload["video_path"] = str(
                    args.artifact_root
                    / f"phase{args.phase}"
                    / "videos"
                    / mode
                    / f"s{sparsity:.2f}"
                    / f"{prompt['prompt_id']}-{job_id[:10]}.mp4"
                )
                payload["stats_path"] = str(args.artifact_root / f"phase{args.phase}" / "stats" / f"{job_id}.parquet")
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs(job_id, ordinal, phase, mode, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, ordinal, args.phase, mode, json.dumps(payload, sort_keys=True)),
                )
                if cursor.rowcount:
                    ordinal += 1
                    created += 1
    conn.commit()
    conn.close()
    return created


def reset_stale(db: Path, phase: int, mode: str, stale_after_s: float) -> int:
    conn = _init_db(db)
    cutoff = time.time() - stale_after_s
    cursor = conn.execute(
        """
        UPDATE jobs SET status='pending', worker_id=NULL
        WHERE phase=? AND mode=? AND status='running' AND started_at<?
        """,
        (phase, mode, cutoff),
    )
    conn.commit()
    changed = cursor.rowcount
    conn.close()
    return changed


def _worker_entry(kwargs: dict[str, Any]) -> None:
    run_worker(**kwargs)


def run_mode(args: argparse.Namespace, mode: str) -> None:
    reset_stale(args.db, args.phase, mode, args.stale_after)
    context = mp.get_context("spawn")
    workers = []
    for gpu_id in range(args.num_workers):
        kwargs = {
            "db_path": args.db,
            "phase": args.phase,
            "mode": mode,
            "worker_id": gpu_id,
            "artifact_root": args.artifact_root,
            "dry_run": args.dry_run,
        }
        process = context.Process(target=_worker_entry, args=(kwargs,), name=f"gpu-{gpu_id}-{mode}")
        process.start()
        workers.append(process)
    for process in workers:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"{process.name} exited with {process.exitcode}")


def print_summary(db: Path, phase: int) -> None:
    conn = _init_db(db)
    rows = conn.execute(
        "SELECT mode, status, COUNT(*) FROM jobs WHERE phase=? GROUP BY mode, status ORDER BY mode, status",
        (phase,),
    ).fetchall()
    conn.close()
    for row in rows:
        print(*row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--model", default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers", choices=MODEL_REVISIONS)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dense_bf16_fa4", "vsa_bf16", "dense_nvfp4_fa4", "sim_vsa_nvfp4"],
        choices=MODE_PRECISION,
    )
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.7, 0.8])
    parser.add_argument("--adaptive-p", type=float, default=0.97)
    parser.add_argument("--adaptive-floor-sparsity", type=float, default=0.8)
    parser.add_argument("--adaptive-native-sparsity", type=float, default=0.8)
    parser.add_argument(
        "--adaptive-candidate-sparsities",
        type=float,
        nargs="+",
        default=[0.8, 0.7, 0.6, 0.4, 0.0],
    )
    parser.add_argument("--ra-native-fraction", type=float, default=0.75)
    parser.add_argument("--ra-native-sparsity", type=float, default=0.8)
    parser.add_argument(
        "--ra-risk-formula",
        default="coarse_mass_x_key_heterogeneity",
    )
    parser.add_argument(
        "--ra-instrument-splits",
        type=float,
        nargs="*",
        default=[],
    )
    parser.add_argument("--ra-minimal-trace", action="store_true")
    parser.add_argument(
        "--ra-force-outside-native",
        action="store_true",
    )
    parser.add_argument("--cs-minimal-trace", action="store_true")
    parser.add_argument(
        "--br-candidate-k",
        type=int,
        nargs="+",
        default=[32, 64, 96, 125, 192, 250, 375, 624],
    )
    parser.add_argument("--br-k-table", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--stale-after", type=float, default=3600)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-seconds", type=float, default=0.25)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    args.db = args.db or args.artifact_root / f"phase{args.phase}" / "jobs.sqlite"
    args.prompts = args.prompts or args.artifact_root / "phase0" / "vbench_subject_consistency_prompts.json"
    if "br_vsa" in args.modes and args.br_k_table is None:
        parser.error("--br-k-table is required for br_vsa")

    if not args.run_only:
        print(f"created={prepare_jobs(args)}")
    if not args.prepare_only:
        for mode in args.modes:
            run_mode(args, mode)
    print_summary(args.db, args.phase)


if __name__ == "__main__":
    main()
