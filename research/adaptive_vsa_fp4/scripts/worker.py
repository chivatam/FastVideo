from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import socket
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any

MODEL_REVISIONS = {
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": "25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef",
    "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers": "c505014d0f8fe673ddf2f8cc5307eabf791e1f9d",
    "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers": "ab7d6149c10bb41ba39226e2262cd2e9ee85b88e",
    "FastVideo/LTX-2.3-Distilled-Diffusers": "22b09fb1860a944bf10fa21f033d957d9ab9ec20",
}


def _rpc_begin_capture(
    worker: Any,
    *,
    job_id: str,
    mode: str,
    sparsity: float,
    adaptive_p: float | None = None,
    adaptive_floor_sparsity: float | None = None,
    adaptive_candidate_sparsities: list[float] | None = None,
    adaptive_native_sparsity: float | None = None,
    ra_native_fraction: float | None = None,
    ra_native_sparsity: float | None = None,
    ra_risk_formula: str | None = None,
    ra_instrument_splits: list[float] | None = None,
    ra_detailed_trace: bool = True,
    ra_force_outside_native: bool = False,
    cs_detailed_trace: bool = True,
) -> dict[str, Any]:
    import torch

    from research.adaptive_vsa_fp4.scripts.runtime import (
        begin_job,
        configure_adaptive_policy,
        configure_compressed_support,
        configure_residual_policy,
        install_runtime_patches,
    )

    install_runtime_patches(mode)
    worker.fastvideo_args.VSA_sparsity = float(sparsity)
    policy = None
    if mode == "adaptive_vsa":
        if adaptive_p is None or adaptive_floor_sparsity is None:
            raise ValueError("Adaptive VSA requires p and floor sparsity")
        policy = configure_adaptive_policy(
            retained_mass_threshold=float(adaptive_p),
            maximum_sparsity=float(adaptive_floor_sparsity),
            candidate_sparsities=tuple(adaptive_candidate_sparsities or [0.8, 0.7, 0.6, 0.4, 0.0]),
            native_sparsity=float(
                adaptive_native_sparsity if adaptive_native_sparsity is not None else adaptive_floor_sparsity
            ),
        )
    elif mode == "ra_vsa":
        if ra_native_fraction is None:
            raise ValueError("RA-VSA requires a native fraction")
        policy = configure_residual_policy(
            native_fraction=float(ra_native_fraction),
            native_sparsity=float(ra_native_sparsity if ra_native_sparsity is not None else sparsity),
            risk_formula=(ra_risk_formula or "coarse_mass_x_key_heterogeneity"),
            instrument_splits=tuple(ra_instrument_splits or ()),
            detailed_trace=bool(ra_detailed_trace),
            force_outside_native=bool(ra_force_outside_native),
        )
    elif mode in {"rectified_vsa", "compressed_halo_vsa"}:
        policy = configure_compressed_support(
            detailed_trace=bool(cs_detailed_trace),
        )
    begin_job(job_id)
    torch.cuda.reset_peak_memory_stats()
    return {
        "status": "capture_started",
        "rank": worker.rpc_rank,
        "effective_sparsity": float(worker.fastvideo_args.VSA_sparsity),
        "adaptive_policy": policy,
    }


def _rpc_prepare_runtime(
    worker: Any,
    *,
    mode: str,
    sparsity: float,
    adaptive_p: float | None = None,
    adaptive_floor_sparsity: float | None = None,
    adaptive_candidate_sparsities: list[float] | None = None,
    adaptive_native_sparsity: float | None = None,
    ra_native_fraction: float | None = None,
    ra_native_sparsity: float | None = None,
    ra_risk_formula: str | None = None,
    ra_instrument_splits: list[float] | None = None,
    ra_detailed_trace: bool = True,
    ra_force_outside_native: bool = False,
    cs_detailed_trace: bool = True,
) -> dict[str, Any]:
    from research.adaptive_vsa_fp4.scripts.runtime import (
        configure_adaptive_policy,
        configure_compressed_support,
        configure_residual_policy,
        install_runtime_patches,
    )

    install_runtime_patches(mode)
    worker.fastvideo_args.VSA_sparsity = float(sparsity)
    policy = None
    if mode == "adaptive_vsa":
        if adaptive_p is None or adaptive_floor_sparsity is None:
            raise ValueError("Adaptive VSA requires p and floor sparsity")
        policy = configure_adaptive_policy(
            retained_mass_threshold=float(adaptive_p),
            maximum_sparsity=float(adaptive_floor_sparsity),
            candidate_sparsities=tuple(adaptive_candidate_sparsities or [0.8, 0.7, 0.6, 0.4, 0.0]),
            native_sparsity=float(
                adaptive_native_sparsity if adaptive_native_sparsity is not None else adaptive_floor_sparsity
            ),
        )
    elif mode == "ra_vsa":
        if ra_native_fraction is None:
            raise ValueError("RA-VSA requires a native fraction")
        policy = configure_residual_policy(
            native_fraction=float(ra_native_fraction),
            native_sparsity=float(ra_native_sparsity if ra_native_sparsity is not None else sparsity),
            risk_formula=(ra_risk_formula or "coarse_mass_x_key_heterogeneity"),
            instrument_splits=tuple(ra_instrument_splits or ()),
            detailed_trace=bool(ra_detailed_trace),
            force_outside_native=bool(ra_force_outside_native),
        )
    elif mode in {"rectified_vsa", "compressed_halo_vsa"}:
        policy = configure_compressed_support(
            detailed_trace=bool(cs_detailed_trace),
        )
    return {
        "status": "runtime_prepared",
        "rank": worker.rpc_rank,
        "effective_sparsity": float(worker.fastvideo_args.VSA_sparsity),
        "adaptive_policy": policy,
    }


def _rpc_finish_capture(worker: Any) -> dict[str, Any]:
    import torch

    from research.adaptive_vsa_fp4.scripts.runtime import finish_job

    attention_ms, stats_rows, effective_sparsities = finish_job()
    return {
        "status": "capture_finished",
        "rank": worker.rpc_rank,
        "attention_ms": attention_ms,
        "stats_rows": stats_rows,
        "effective_sparsities": effective_sparsities,
        "peak_hbm_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _validate_effective_sparsity(response: list[dict[str, Any]], expected: float, status: str) -> None:
    if not response or response[0].get("status") != status:
        raise RuntimeError(f"Inner-worker runtime setup failed: {response!r}")
    actual = float(response[0]["effective_sparsity"])
    if abs(actual - expected) > 1e-9:
        raise RuntimeError(f"Requested VSA sparsity {expected}, inner worker reported {actual}.")


def canonical_job_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def claim_job(conn: sqlite3.Connection, phase: int, mode: str, worker_id: int) -> tuple[str, dict[str, Any]] | None:
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job_id, payload FROM jobs
                WHERE phase=? AND mode=? AND status='pending'
                ORDER BY ordinal LIMIT 1
                """,
                (phase, mode),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                """
                UPDATE jobs SET status='running', worker_id=?, started_at=?,
                    attempts=attempts+1, hostname=?
                WHERE job_id=? AND status='pending'
                """,
                (worker_id, time.time(), socket.gethostname(), row[0]),
            )
            if conn.total_changes:
                conn.commit()
                return row[0], json.loads(row[1])
            conn.rollback()
        except sqlite3.OperationalError:
            conn.rollback()
            time.sleep(0.1)


def finish_job(conn: sqlite3.Connection, job_id: str, status: str, result_path: str | None, error: str | None) -> None:
    conn.execute(
        "UPDATE jobs SET status=?, finished_at=?, result_path=?, error=? WHERE job_id=?",
        (status, time.time(), result_path, error, job_id),
    )
    conn.commit()


def _extract_component_times(result: dict[str, Any]) -> dict[str, float | None]:
    mapping = {
        "TextEncodingStage": "text_encoder_time_s",
        "DenoisingStage": "dit_time_s",
        "DecodingStage": "vae_decode_time_s",
        "WanDenoisingStage": "dit_time_s",
        "WanDecodingStage": "vae_decode_time_s",
    }
    output = {"text_encoder_time_s": None, "dit_time_s": None, "vae_decode_time_s": None}
    logging_info = result.get("logging_info")
    stages = getattr(logging_info, "stages", None)
    if stages is None and isinstance(logging_info, dict):
        stages = logging_info.get("stages")
    for stage_name, data in (stages or {}).items():
        if not isinstance(data, dict):
            continue
        key = data.get("component_metric") or mapping.get(data.get("stage_class", stage_name))
        if key not in output:
            continue
        value = data.get("execution_time")
        if value is not None:
            output[key] = float(value) + float(output[key] or 0.0)
    return output


def _configure_mode(mode: str) -> None:
    os.environ["FASTVIDEO_FA4"] = "1"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    os.environ["CUTE_DSL_ENABLE_TVM_FFI"] = "1"
    if mode in {
        "vsa_bf16",
        "sim_vsa_nvfp4",
        "adaptive_vsa",
        "ra_vsa",
        "rectified_vsa",
        "compressed_halo_vsa",
    }:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        os.environ["FASTVIDEO_VSA_SM100A"] = "1"
        os.environ["FASTVIDEO_NVFP4_FA4"] = "0"
    elif mode == "dense_nvfp4_fa4":
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
        os.environ["FASTVIDEO_NVFP4_FA4"] = "1"
        os.environ["FASTVIDEO_RESEARCH_ALLOW_UNUSED_VSA_GATES"] = "1"
    elif mode == "dense_bf16_fa4":
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
        os.environ["FASTVIDEO_NVFP4_FA4"] = "0"
        os.environ["FASTVIDEO_RESEARCH_ALLOW_UNUSED_VSA_GATES"] = "1"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _load_generator(payload: dict[str, Any], mode: str):
    _configure_mode(mode)
    from fastvideo import VideoGenerator
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=payload["model"],
        revision=payload["model_revision"],
        cache_dir=os.environ["HF_HUB_CACHE"],
        local_files_only=True,
    )

    return VideoGenerator.from_pretrained(
        snapshot,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        enable_torch_compile=False,
        nvfp4_fa4=mode == "dense_nvfp4_fa4",
        VSA_sparsity=float(payload["sparsity"]),
    )


def _warm_generator(generator: Any, payload: dict[str, Any], mode: str) -> None:
    sparsity = float(payload["sparsity"])
    generator.fastvideo_args.VSA_sparsity = sparsity
    prepared = generator.executor.collective_rpc(
        _rpc_prepare_runtime,
        kwargs={
            "mode": mode,
            "sparsity": sparsity,
            "adaptive_p": payload.get("adaptive_p"),
            "adaptive_floor_sparsity": payload.get("adaptive_floor_sparsity"),
            "adaptive_candidate_sparsities": payload.get("adaptive_candidate_sparsities"),
            "adaptive_native_sparsity": payload.get("adaptive_native_sparsity"),
            "ra_native_fraction": payload.get("ra_native_fraction"),
            "ra_native_sparsity": payload.get("ra_native_sparsity"),
            "ra_risk_formula": payload.get("ra_risk_formula"),
            "ra_instrument_splits": payload.get("ra_instrument_splits"),
            "ra_detailed_trace": payload.get(
                "ra_detailed_trace",
                True,
            ),
            "ra_force_outside_native": payload.get(
                "ra_force_outside_native",
                False,
            ),
            "cs_detailed_trace": payload.get(
                "cs_detailed_trace",
                True,
            ),
        },
    )
    _validate_effective_sparsity(prepared, sparsity, "runtime_prepared")
    generator.generate_video(
        payload["prompt"],
        save_video=False,
        seed=int(payload["seed"]),
        num_frames=int(payload["frames"]),
        height=int(payload["height"]),
        width=int(payload["width"]),
        fps=int(payload["fps"]),
        num_inference_steps=int(payload["steps"]),
        guidance_scale=float(payload["cfg"]),
    )


def _run_generation(generator, job_id: str, payload: dict[str, Any], mode: str, worker_id: int) -> dict[str, Any]:
    from research.adaptive_vsa_fp4.scripts.runtime import write_stats

    sparsity = float(payload["sparsity"])
    generator.fastvideo_args.VSA_sparsity = sparsity
    output_path = Path(payload["video_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_start = generator.executor.collective_rpc(
        _rpc_begin_capture,
        kwargs={
            "job_id": job_id,
            "mode": mode,
            "sparsity": sparsity,
            "adaptive_p": payload.get("adaptive_p"),
            "adaptive_floor_sparsity": payload.get("adaptive_floor_sparsity"),
            "adaptive_candidate_sparsities": payload.get("adaptive_candidate_sparsities"),
            "adaptive_native_sparsity": payload.get("adaptive_native_sparsity"),
            "ra_native_fraction": payload.get("ra_native_fraction"),
            "ra_native_sparsity": payload.get("ra_native_sparsity"),
            "ra_risk_formula": payload.get("ra_risk_formula"),
            "ra_instrument_splits": payload.get("ra_instrument_splits"),
            "ra_detailed_trace": payload.get(
                "ra_detailed_trace",
                True,
            ),
            "ra_force_outside_native": payload.get(
                "ra_force_outside_native",
                False,
            ),
            "cs_detailed_trace": payload.get(
                "cs_detailed_trace",
                True,
            ),
        },
    )
    _validate_effective_sparsity(capture_start, sparsity, "capture_started")
    start = time.perf_counter()
    result = generator.generate_video(
        payload["prompt"],
        output_path=str(output_path),
        save_video=payload["save_video"],
        seed=int(payload["seed"]),
        num_frames=int(payload["frames"]),
        height=int(payload["height"]),
        width=int(payload["width"]),
        fps=int(payload["fps"]),
        num_inference_steps=int(payload["steps"]),
        guidance_scale=float(payload["cfg"]),
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    capture_finish = generator.executor.collective_rpc(_rpc_finish_capture)
    if not capture_finish or capture_finish[0].get("status") != "capture_finished":
        raise RuntimeError(f"Failed to finish inner-worker capture: {capture_finish!r}")
    capture = capture_finish[0]
    attention_ms = float(capture["attention_ms"])
    stats_rows = capture["stats_rows"]
    effective_sparsities = [float(value) for value in capture["effective_sparsities"]]
    if mode in {"vsa_bf16", "sim_vsa_nvfp4"} and effective_sparsities != [sparsity]:
        raise RuntimeError(f"Requested VSA sparsity {sparsity}, attention metadata observed {effective_sparsities!r}.")
    adaptive_rows = [row for row in stats_rows if row.get("event_type") == "adaptive_policy"]
    if mode == "adaptive_vsa" and not adaptive_rows:
        raise RuntimeError("Adaptive VSA produced no policy trace rows")
    residual_rows = [row for row in stats_rows if row.get("event_type") == "ra_vsa_policy"]
    if mode == "ra_vsa" and not residual_rows:
        raise RuntimeError("RA-VSA produced no policy trace rows")
    compressed_rows = [row for row in stats_rows if row.get("event_type") == "compressed_support_policy"]
    if mode in {"rectified_vsa", "compressed_halo_vsa"} and not compressed_rows:
        raise RuntimeError("Compressed-support VSA produced no policy trace rows")
    invalid_compressed_k = [
        row
        for row in compressed_rows
        if (
            int(row["selected_count_min"]) != 125 or int(row["selected_count_max"]) != 125 or int(row["exact_k"]) != 125
        )
    ]
    if invalid_compressed_k:
        raise RuntimeError(f"Compressed-support VSA violated native K=125: {invalid_compressed_k[:3]!r}")
    if residual_rows:
        invalid_fixed_k = [
            row
            for row in residual_rows
            if (
                int(row["selected_count_min"]) != int(row["total_slots"])
                or int(row["selected_count_max"]) != int(row["total_slots"])
            )
        ]
        if invalid_fixed_k:
            raise RuntimeError(f"RA-VSA violated the fixed-K invariant in policy traces: {invalid_fixed_k[:3]!r}")
        forced_rows = [row for row in residual_rows if bool(row.get("force_outside_native"))]
        invalid_replacement = [
            row
            for row in forced_rows
            if (
                int(row["replacement_count_min"]) != int(row["rescue_slots"])
                or int(row["replacement_count_max"]) != int(row["rescue_slots"])
            )
        ]
        if invalid_replacement:
            raise RuntimeError(f"Forced RA-VSA violated the exact replacement invariant: {invalid_replacement[:3]!r}")
    if adaptive_rows:
        total_rows = sum(int(row["num_query_rows"]) for row in adaptive_rows)
        effective_sparsity = (
            sum(float(row["effective_sparsity"]) * int(row["num_query_rows"]) for row in adaptive_rows) / total_rows
        )
    else:
        effective_sparsity = (
            effective_sparsities[0] if len(effective_sparsities) == 1 else (0.0 if mode.startswith("dense_") else None)
        )
    component = _extract_component_times(result)
    stats_path = Path(payload["stats_path"])
    written_stats = write_stats(stats_rows, stats_path)
    actual_video = result.get("video_path")
    record = {
        "job_id": job_id,
        "phase": payload["phase"],
        "model": payload["model"],
        "model_revision": payload["model_revision"],
        "prompt_id": payload["prompt_id"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
        "sparsity": payload["sparsity"],
        "effective_sparsity": effective_sparsity,
        "topk": payload.get("topk"),
        "precision": payload["precision"],
        "resolution": f"{payload['height']}x{payload['width']}",
        "frames": payload["frames"],
        "steps": payload["steps"],
        "cfg": payload["cfg"],
        "attention_backend": os.environ["FASTVIDEO_ATTENTION_BACKEND"],
        "kernel_path": mode,
        "adaptive_p": payload.get("adaptive_p"),
        "adaptive_floor_sparsity": payload.get("adaptive_floor_sparsity"),
        "adaptive_candidate_sparsities": payload.get("adaptive_candidate_sparsities"),
        "adaptive_native_sparsity": payload.get("adaptive_native_sparsity"),
        "ra_native_fraction": payload.get("ra_native_fraction"),
        "ra_native_sparsity": payload.get("ra_native_sparsity"),
        "ra_risk_formula": payload.get("ra_risk_formula"),
        "ra_instrument_splits": payload.get("ra_instrument_splits"),
        "ra_detailed_trace": payload.get("ra_detailed_trace"),
        "ra_force_outside_native": payload.get("ra_force_outside_native"),
        "cs_detailed_trace": payload.get("cs_detailed_trace"),
        "gpu_id": worker_id,
        "warm": True,
        "wall_ms": wall_ms,
        "dit_ms": None if component["dit_time_s"] is None else component["dit_time_s"] * 1000.0,
        "attention_ms": attention_ms,
        "peak_hbm_bytes": int(capture["peak_hbm_bytes"]),
        "video_path": actual_video or (str(output_path) if output_path.exists() else None),
        "stats_path": written_stats,
        "status": "ok",
        "error": None,
        "hostname": socket.gethostname(),
        "started_at": start,
    }
    return record


def run_worker(
    *,
    db_path: Path,
    phase: int,
    mode: str,
    worker_id: int,
    artifact_root: Path,
    dry_run: bool = False,
) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("HF_HOME", "/mnt/fastvideo-gpu0/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/mnt/fastvideo-gpu0/hf-cache/hub")
    os.environ.setdefault("TRITON_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/triton")
    os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/mnt/fastvideo-gpu0/jit-cache/cute")
    conn = connect(db_path)
    generator = None
    loaded_model = None
    warmed = False
    while True:
        claimed = claim_job(conn, phase, mode, worker_id)
        if claimed is None:
            break
        job_id, payload = claimed
        result_path = artifact_root / f"phase{phase}" / "records" / f"{job_id}.json"
        try:
            if dry_run:
                time.sleep(float(payload.get("dry_run_seconds", 0.25)))
                record = {
                    "job_id": job_id,
                    "phase": phase,
                    "model": payload["model"],
                    "prompt_id": payload["prompt_id"],
                    "gpu_id": worker_id,
                    "status": "ok",
                    "dry_run": True,
                }
            else:
                if generator is None or loaded_model != payload["model"]:
                    if generator is not None:
                        generator.shutdown()
                    generator = _load_generator(payload, mode)
                    loaded_model = payload["model"]
                    warmed = False
                if not warmed:
                    _warm_generator(generator, payload, mode)
                    warmed = True
                record = _run_generation(generator, job_id, payload, mode, worker_id)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = result_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
            os.replace(tmp, result_path)
            finish_job(conn, job_id, "ok", str(result_path), None)
        except BaseException as exc:
            error = "".join(traceback.format_exception(exc))
            attempts = conn.execute("SELECT attempts FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            if attempts < 2:
                conn.execute(
                    "UPDATE jobs SET status='pending', worker_id=NULL, error=? WHERE job_id=?",
                    (error, job_id),
                )
                conn.commit()
            else:
                finish_job(conn, job_id, "failed", None, error)
    if generator is not None:
        generator.shutdown()
    conn.close()
