from __future__ import annotations

from types import SimpleNamespace

from research.adaptive_vsa_fp4.scripts import runtime
from research.adaptive_vsa_fp4.scripts.worker import (
    _rpc_prepare_runtime,
    _validate_effective_sparsity,
)


def test_prepare_runtime_updates_inner_worker_sparsity(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "install_runtime_patches", lambda mode: None)
    worker = SimpleNamespace(
        rpc_rank=0,
        fastvideo_args=SimpleNamespace(VSA_sparsity=0.2),
    )

    response = _rpc_prepare_runtime(worker, mode="vsa_bf16", sparsity=0.8)

    assert response["effective_sparsity"] == 0.8
    assert worker.fastvideo_args.VSA_sparsity == 0.8


def test_capture_records_observed_sparsity() -> None:
    runtime.begin_job("job")
    runtime.record_effective_sparsity(0.6)
    runtime.record_effective_sparsity(0.6)

    attention_ms, rows, effective_sparsities = runtime.finish_job()

    assert attention_ms == 0.0
    assert rows == []
    assert effective_sparsities == [0.6]


def test_validate_effective_sparsity_rejects_mismatch() -> None:
    response = [
        {
            "status": "runtime_prepared",
            "effective_sparsity": 0.2,
        }
    ]

    try:
        _validate_effective_sparsity(response, 0.8, "runtime_prepared")
    except RuntimeError as error:
        assert "inner worker reported 0.2" in str(error)
    else:
        raise AssertionError("Expected a sparsity mismatch to fail.")
