"""Phase 5 correctness gate: verify all six arms before any video is generated.

A Phase 5 video costs real GPU minutes, and the whole point of the phase is that
the arm the config *names* is the arm the model *runs*. A silently-ignored
backend override (``STATUS.md`` trap 1), a mask that collapsed to dense, or an
arm whose "NVFP4" path was actually BF16 would all produce a confidently-wrong
null. So before generating anything, drive
:class:`SparseFP4ExecAttentionImpl.forward` directly at Wan's real attention
shape and check:

1. ``DENSE-BF16`` is **bit-identical** to the ``FLASH_ATTN`` dense branch, so the
   reference arm introduces no backend artefact of its own.
2. ``DENSE-FP4`` matches the native NVFP4 kernel bit for bit and *differs* from
   BF16 -- i.e. the low-precision path is really engaged.
3. Every sparse arm realizes the requested retained-block budget exactly.
4. The three ``SPARSE-FP4-*`` arms are **bit-identical at sparsity 0**, which is
   the check that actually proves they share one compute path and differ only in
   router precision.
5. The mask disagreement between routers is recorded, so the arms are known to be
   genuinely different runs rather than an accidental alias.

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase5_selftest.py \
        --out artifacts/sparsefp4/raw/phase5_selftest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

WAN_SEQ_LEN = 32760
WAN_HEADS = 12
HEAD_DIM = 128
SPARSITY = 0.90


def _write_config(tmp: Path, arm: str, sparsity: float) -> Path:
    path = tmp / f"phase5_config_{arm}.json"
    path.write_text(
        json.dumps({
            "out_dir": str(tmp),
            "run_id": "selftest",
            "git_commit": "selftest",
            "arm": arm,
            "prompt_id": "p00",
            "seed": 0,
            "sparsity": sparsity,
            "block_q": 128,
            "block_k": 64,
            "score_dtype": "float64",
            "shard_tag": f"selftest-{arm}",
        }),
        encoding="utf-8",
    )
    return path


def _run_arm(arm: str, sparsity: float, tensors: tuple[torch.Tensor, ...], tmp: Path) -> tuple[torch.Tensor, Any]:
    """Instantiate the impl fresh per arm; the config is read in ``__init__``."""
    from fastvideo.attention.backends.sparsefp4_exec_attn import SparseFP4ExecAttentionImpl
    from fastvideo.forward_context import set_forward_context

    os.environ["FASTVIDEO_SPARSEFP4_PHASE5"] = str(_write_config(tmp, arm, sparsity))
    SparseFP4ExecAttentionImpl._counters = {}
    SparseFP4ExecAttentionImpl._receipt_written = False
    impl = SparseFP4ExecAttentionImpl(
        num_heads=WAN_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        prefix="transformer_blocks.7.attn1",
    )
    query, key, value = tensors
    with set_forward_context(current_timestep=3, attn_metadata=None, forward_batch=None):
        out = impl.forward(query, key, value, None)
    counters = SparseFP4ExecAttentionImpl._counters["all"].as_dict()
    return out, counters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, default=SPARSITY)
    args = parser.parse_args()

    from fastvideo.attention.backends.sparsefp4_exec_attn import ARMS, compute_topk
    from fastvideo.attention.backends.routing_probe_attn import pool_blocks_1d, quantize_router_input
    from fastvideo.attention.backends.sparsefp4_numerics import (block_scores, dense_bf16, dense_nvfp4_native,
                                                                 error_metrics, topk_block_mask)

    device = torch.device("cuda")
    torch.manual_seed(4242)
    scale = HEAD_DIM**-0.5
    # A shared common-mode component reproduces the large-magnitude / small-spread
    # block-score regime that makes fp64 scoring mandatory (trap 8).
    common = 6.0 * torch.randn(1, 1, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) + common
    key = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) + common
    value = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    tensors = (query, key, value)

    results: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    receipts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(dir="/mnt/scratch/tmp") as raw_tmp:
        tmp = Path(raw_tmp)
        for arm in ARMS:
            out, counters = _run_arm(arm, args.sparsity, tensors, tmp)
            outputs[arm] = out
            receipts[arm] = counters

    reference_bf16 = dense_bf16(query, key, value, scale)
    reference_fp4 = dense_nvfp4_native(query, key, value, scale)

    results.append({
        "check": "DENSE-BF16 is bit-identical to the FLASH_ATTN dense branch",
        "bit_identical": bool(torch.equal(outputs["DENSE-BF16"], reference_bf16)),
        "pass": bool(torch.equal(outputs["DENSE-BF16"], reference_bf16)),
    })
    results.append({
        "check": "DENSE-FP4 is bit-identical to the native NVFP4 kernel",
        "bit_identical": bool(torch.equal(outputs["DENSE-FP4"], reference_fp4)),
        "native_or_simulated": "native",
        "pass": bool(torch.equal(outputs["DENSE-FP4"], reference_fp4)),
    })
    fp4_vs_bf16 = error_metrics(outputs["DENSE-FP4"], reference_bf16)
    results.append({
        "check": "DENSE-FP4 really engages the low-precision path (differs from BF16)",
        **fp4_vs_bf16,
        "pass": fp4_vs_bf16["rel_l2"] is not None and fp4_vs_bf16["rel_l2"] > 1e-3,
    })

    n_k_blocks = -(-WAN_SEQ_LEN // 64)
    expected_k = compute_topk(args.sparsity, n_k_blocks)
    expected_retained = expected_k / n_k_blocks
    for arm, spec in ARMS.items():
        counters = receipts[arm]
        if not spec.sparse:
            results.append({
                "check": f"{arm}: dense arm reports no mask budget",
                "realized_retained_fraction": counters["realized_retained_fraction"],
                "pass": counters["realized_retained_fraction"] is None,
            })
            continue
        realized = counters["realized_retained_fraction"]
        ok = realized is not None and abs(realized - expected_retained) < 1e-9 and counters["k_per_query_block"] == [
            expected_k
        ]
        results.append({
            "check": f"{arm}: realized retained-block budget equals the request",
            "requested_sparsity": args.sparsity,
            "expected_k": expected_k,
            "n_k_blocks": n_k_blocks,
            "realized_retained_fraction": realized,
            "expected_retained_fraction": expected_retained,
            "k_per_query_block": counters["k_per_query_block"],
            "pass": bool(ok),
        })

    naive = outputs["SPARSE-FP4-NAIVE"]
    for arm in ("SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16"):
        metrics = error_metrics(outputs[arm], naive)
        results.append({
            "check": f"{arm} differs from SPARSE-FP4-NAIVE (the router really changed the mask)",
            "native_or_simulated": "simulated",
            **metrics,
            "pass": metrics["rel_l2"] is not None and metrics["rel_l2"] > 0.0,
        })

    # The three SPARSE-FP4-* arms share one compute path and differ only in
    # router precision. Removing the mask (sparsity 0 => every block retained)
    # therefore has to make them **bit-identical**. This is the check that
    # actually isolates "router-only", and it does not depend on the score
    # distribution of the test tensors.
    with tempfile.TemporaryDirectory(dir="/mnt/scratch/tmp") as raw_tmp:
        dense_equiv = {
            arm: _run_arm(arm, 0.0, tensors, Path(raw_tmp))[0]
            for arm in ("SPARSE-FP4-NAIVE", "SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16")
        }
    for arm in ("SPARSE-FP4-ROUTE8", "SPARSE-FP4-ROUTE16"):
        identical = bool(torch.equal(dense_equiv[arm], dense_equiv["SPARSE-FP4-NAIVE"]))
        results.append({
            "check": f"{arm} shares SPARSE-FP4-NAIVE's compute path exactly (bit-identical at sparsity 0)",
            "native_or_simulated": "simulated",
            "bit_identical": identical,
            "pass": identical,
        })

    sparsity_effect = error_metrics(outputs["SPARSE-BF16"], outputs["DENSE-BF16"])["rel_l2"]
    routing_effect = error_metrics(outputs["SPARSE-FP4-NAIVE"], outputs["SPARSE-FP4-ROUTE16"])["rel_l2"]
    results.append({
        "check": "sparsity effect dominates routing-precision effect (plumbing sanity, random tensors)",
        "sparsity_rel_l2_vs_dense": sparsity_effect,
        "routing_precision_rel_l2": routing_effect,
        "ratio": (sparsity_effect / routing_effect) if routing_effect else None,
        # Random tensors have no block structure for the router to find, so this
        # is recorded rather than gated: an i.i.d. score field makes every block
        # equally (un)important, which is the one regime where routing noise is
        # NOT cheap. Phase 2 measured the real-activation answer.
        "pass": True,
        "gated": False,
    })

    failures = [row for row in results if not row.get("pass")]

    # How much do the routers actually disagree? Recorded so the three FP4 arms
    # are provably distinct configurations and not an accidental alias.
    masks = {}
    for precision in ("bf16", "nvfp4", "fp8_e4m3"):
        route_q, _ = quantize_router_input(query, precision)
        route_k, _ = quantize_router_input(key, precision)
        scores = block_scores(pool_blocks_1d(route_q, 128), pool_blocks_1d(route_k, 64), scale, torch.float64)
        masks[precision] = topk_block_mask(scores, expected_k)
    router_agreement = {}
    for precision in ("nvfp4", "fp8_e4m3"):
        intersection = int((masks[precision] & masks["bf16"]).sum().item())
        union = int((masks[precision] | masks["bf16"]).sum().item())
        router_agreement[f"jaccard_{precision}_vs_bf16"] = intersection / max(1, union)

    payload = {
        "verdict": "PASS" if not failures else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "seq_len": WAN_SEQ_LEN,
        "num_heads": WAN_HEADS,
        "head_dim": HEAD_DIM,
        "sparsity": args.sparsity,
        "note": ("Random tensors, not model activations: this gates the harness plumbing, "
                 "not the research result. SPARSE-FP4-* rows are simulated compute; no latency claim."),
        "checks": results,
        "router_mask_agreement_random_tensors": router_agreement,
        "arm_receipts": receipts,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
