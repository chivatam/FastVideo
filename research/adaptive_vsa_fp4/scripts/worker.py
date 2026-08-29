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


def _rpc_begin_capture(worker: Any, *, job_id: str, mode: str) -> dict[str, Any]:
    import torch

    from research.adaptive_vsa_fp4.scripts.runtime import begin_job, install_runtime_patches

    install_runtime_patches(mode)
    begin_job(job_id)
    torch.cuda.reset_peak_memory_stats()
    return {"status": "capture_started", "rank": worker.rpc_rank}


def _rpc_prepare_runtime(worker: Any, *, mode: str) -> dict[str, Any]:
    from research.adaptive_vsa_fp4.scripts.runtime import install_runtime_patches

    install_runtime_patches(mode)
    return {"status": "runtime_prepared", "rank": worker.rpc_rank}


def _rpc_finish_capture(worker: Any) -> dict[str, Any]:
    import torch

    from research.adaptive_vsa_fp4.scripts.runtime import finish_job

    attention_ms, stats_rows = finish_job()
    return {
        "status": "capture_finished",
        "rank": worker.rpc_rank,
        "attention_ms": attention_ms,
        "stats_rows": stats_rows,
        "peak_hbm_bytes": int(torch.cuda.max_memory_allocated()),
    }


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
    if mode in {"vsa_bf16", "sim_vsa_nvfp4"}:
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
    generator.fastvideo_args.VSA_sparsity = float(payload["sparsity"])
    prepared = generator.executor.collective_rpc(_rpc_prepare_runtime, kwargs={"mode": mode})
    if not prepared or prepared[0].get("status") != "runtime_prepared":
        raise RuntimeError(f"Failed to prepare inner-worker runtime: {prepared!r}")
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

    generator.fastvideo_args.VSA_sparsity = float(payload["sparsity"])
    output_path = Path(payload["video_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_start = generator.executor.collective_rpc(
        _rpc_begin_capture,
        kwargs={"job_id": job_id, "mode": mode},
    )
    if not capture_start or capture_start[0].get("status") != "capture_started":
        raise RuntimeError(f"Failed to start inner-worker capture: {capture_start!r}")
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
        "topk": payload.get("topk"),
        "precision": payload["precision"],
        "resolution": f"{payload['height']}x{payload['width']}",
        "frames": payload["frames"],
        "steps": payload["steps"],
        "cfg": payload["cfg"],
        "attention_backend": os.environ["FASTVIDEO_ATTENTION_BACKEND"],
        "kernel_path": mode,
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
