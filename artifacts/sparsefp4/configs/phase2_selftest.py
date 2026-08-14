"""Phase 2 correctness gate: verify every kernel before any error is attributed.

Nothing in Phase 2 may be trusted until the block-sparse kernel provably ignores
masked-out blocks and reproduces an independent masked-attention reference. Also
checks the padding/ragged-tail handling, the equal-budget invariant of the random
perturbation control, and how closely the simulated NVFP4 round-trip tracks the
native FA4 NVFP4 kernel (the premise that lets E/F be simulated at all).

    source artifacts/sparsefp4/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4/configs/phase2_selftest.py \
        --out artifacts/sparsefp4/raw/phase2_selftest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from fastvideo.attention.backends.sparsefp4_numerics import (
    KERNEL_BLOCK, BlockGeometry, assert_kernel_scale_matches, assert_query_grid_alignment, block_attention_mass,
    block_scores, cube_geometry, dense_bf16, dense_nvfp4_native, error_metrics, expand_query_axis, from_block_layout,
    masked_reference, pad_to_kernel_block, pool_geometry_blocks, random_matched_mask, raster_geometry,
    retained_token_fraction, sparse_bf16, to_block_layout, topk_block_mask, variable_block_sizes_for)
from fastvideo.attention.backends.routing_probe_attn import pool_blocks_1d, quantize_router_input

WAN_SEQ_LEN = 32760
WAN_HEADS = 12
HEAD_DIM = 128
# Wan2.1-T2V-1.3B at 480x832x81 with patch (1,2,2): the DiT token grid VSA tiles.
WAN_DIT_SEQ_SHAPE = (21, 30, 52)


def _mask_from_scores(query: torch.Tensor, key: torch.Tensor, block_q: int, block_k: int, softmax_scale: float,
                      precision: str, k: int) -> torch.Tensor:
    route_q, _ = quantize_router_input(query, precision)
    route_k, _ = quantize_router_input(key, precision)
    scores = block_scores(pool_blocks_1d(route_q, block_q), pool_blocks_1d(route_k, block_k), softmax_scale)
    return topk_block_mask(scores, k)


def _geometry_mask(query: torch.Tensor, key: torch.Tensor, geometry: BlockGeometry, softmax_scale: float,
                   precision: str, k: int) -> torch.Tensor:
    """Router mask scored in ``geometry``'s layout, exactly as the backend does."""
    route_q, _ = quantize_router_input(query, precision)
    route_k, _ = quantize_router_input(key, precision)
    pooled_q = pool_geometry_blocks(to_block_layout(route_q, geometry), geometry.query_block_sizes, geometry.block_q)
    pooled_k = pool_geometry_blocks(to_block_layout(route_k, geometry), geometry.key_block_sizes, geometry.block_k)
    return topk_block_mask(block_scores(pooled_q, pooled_k, softmax_scale), k)


def check_small_sparse_vs_reference(device, results: list[dict]) -> None:
    """Kernel vs an independent fp32 masked softmax, including a ragged tail."""
    torch.manual_seed(7)
    seq_len = 31 * KERNEL_BLOCK + 56
    heads = 4
    scale = HEAD_DIM**-0.5
    query = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    padded = [pad_to_kernel_block(t) for t in (query, key, value)]
    sizes = variable_block_sizes_for(seq_len, device)
    n_blocks = sizes.numel()
    valid = torch.arange(padded[0].shape[1], device=device) < seq_len

    for sparsity in (0.0, 0.50, 0.80, 0.95):
        k = max(1, round((1 - sparsity) * n_blocks))
        mask = _mask_from_scores(padded[0], padded[1], KERNEL_BLOCK, KERNEL_BLOCK, scale, "bf16", k).unsqueeze(0)
        kernel_out = sparse_bf16(padded[0], padded[1], padded[2], mask, sizes)[:, :seq_len]
        reference = masked_reference(padded[0], padded[1], padded[2], mask, valid, scale)[:, :seq_len]
        metrics = error_metrics(kernel_out, reference)
        results.append({
            "check": "sparse_kernel_vs_masked_reference",
            "seq_len": seq_len,
            "heads": heads,
            "sparsity": sparsity,
            "k": k,
            "n_blocks": n_blocks,
            **metrics,
            "pass": metrics["rel_l2"] is not None and metrics["rel_l2"] < 5e-3,
        })

    # A masked-out block must contribute nothing: perturbing V inside excluded
    # blocks may not move the output by a single bit.
    k = max(1, round(0.20 * n_blocks))
    mask = _mask_from_scores(padded[0], padded[1], KERNEL_BLOCK, KERNEL_BLOCK, scale, "bf16", k).unsqueeze(0)
    baseline = sparse_bf16(padded[0], padded[1], padded[2], mask, sizes)[:, :seq_len]
    union_selected = mask[0].any(dim=1).any(dim=0)
    tampered = padded[2].clone()
    for block in range(n_blocks):
        if not bool(union_selected[block]):
            tampered[:, block * KERNEL_BLOCK:(block + 1) * KERNEL_BLOCK] += 100.0
    tampered_out = sparse_bf16(padded[0], padded[1], tampered, mask, sizes)[:, :seq_len]
    results.append({
        "check": "masked_blocks_do_not_contribute",
        "never_selected_blocks": int((~union_selected).sum().item()),
        "bit_identical": bool(torch.equal(baseline, tampered_out)),
        "pass": bool(torch.equal(baseline, tampered_out)) or int((~union_selected).sum().item()) == 0,
    })


def check_dense_paths(device, results: list[dict]) -> None:
    """Dense BF16, native NVFP4 and the simulated NVFP4 stand-in at Wan's shape."""
    torch.manual_seed(11)
    scale = HEAD_DIM**-0.5
    assert_kernel_scale_matches(HEAD_DIM, scale)
    query = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

    reference = dense_bf16(query, key, value, scale)
    native = dense_nvfp4_native(query, key, value, scale)
    results.append({
        "check": "dense_nvfp4_native_vs_dense_bf16",
        "native_or_simulated": "native",
        **error_metrics(native, reference),
        "pass": True,
    })

    for arm in ("nvfp4", "nvfp4_sim"):
        route_q, _ = quantize_router_input(query, arm)
        route_k, _ = quantize_router_input(key, arm)
        simulated = dense_bf16(route_q.to(torch.bfloat16), route_k.to(torch.bfloat16), value, scale)
        metrics = error_metrics(simulated, reference)
        agreement = error_metrics(simulated, native)
        results.append({
            "check": f"dense_{arm}_dequant_simulation_vs_dense_bf16",
            "native_or_simulated": "simulated",
            **metrics,
            "rel_l2_vs_native_nvfp4": agreement["rel_l2"],
            "cosine_vs_native_nvfp4": agreement["cosine"],
            "pass": True,
        })


def check_random_perturbation(device, results: list[dict]) -> None:
    torch.manual_seed(3)
    n_q, n_k, heads = 64, 128, 4
    scores_ref = torch.randn(heads, n_q, n_k, device=device)
    scores_cand = scores_ref + 0.05 * torch.randn_like(scores_ref)
    k = 26
    reference_mask = topk_block_mask(scores_ref, k)
    candidate_mask = topk_block_mask(scores_cand, k)
    generator = torch.Generator(device=device).manual_seed(1234)
    perturbed = random_matched_mask(reference_mask, candidate_mask, generator)
    swaps_candidate = (reference_mask & ~candidate_mask).sum(-1)
    swaps_random = (reference_mask & ~perturbed).sum(-1)
    results.append({
        "check":
        "random_matched_mask_swap_count_and_budget",
        "budget_preserved":
        bool((perturbed.sum(-1) == k).all().item()),
        "swap_counts_match":
        bool((swaps_candidate == swaps_random).all().item()),
        "mean_swaps":
        float(swaps_candidate.float().mean().item()),
        "pass":
        bool((perturbed.sum(-1) == k).all().item() and (swaps_candidate == swaps_random).all().item()),
    })


def check_attention_mass(device, results: list[dict]) -> None:
    """Mass rows must be a probability distribution over key blocks."""
    torch.manual_seed(5)
    # 15 whole 64-token blocks plus a 40-token ragged tail: 16 kernel blocks, so
    # the 128-token query grid tiles the kernel's 64-row grid exactly.
    seq_len = 15 * KERNEL_BLOCK + 40
    query = torch.randn(1, seq_len, 2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, seq_len, 2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    padded_q, padded_k = pad_to_kernel_block(query), pad_to_kernel_block(key)
    geometry = raster_geometry(seq_len, 128, device)
    mass = block_attention_mass(padded_q, padded_k, 0, [0, 1, 4], geometry, HEAD_DIM**-0.5)
    totals = mass.sum(dim=-1)
    results.append({
        "check": "block_attention_mass_normalized",
        "rows": int(mass.shape[0]),
        "max_abs_deviation_from_one": float((totals - 1.0).abs().max().item()),
        "pass": bool((totals - 1.0).abs().max().item() < 1e-4),
    })


def check_128x64_geometry(device, results: list[dict]) -> None:
    """A 128-row router mask executed on the kernel's 64-row grid.

    Phase 1 scored at 128x64; the kernel's query grid is 64. Splitting each
    128-token query block into its two constituent kernel blocks must reproduce
    the intended mask exactly, so verify against the masked reference built from
    the *expanded* mask.
    """
    torch.manual_seed(13)
    seq_len = 25 * KERNEL_BLOCK + 56
    heads = 3
    scale = HEAD_DIM**-0.5
    query = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, seq_len, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    padded = [pad_to_kernel_block(t) for t in (query, key, value)]
    sizes = variable_block_sizes_for(seq_len, device)
    n_k_blocks = sizes.numel()
    valid = torch.arange(padded[0].shape[1], device=device) < seq_len
    k = max(1, round(0.20 * n_k_blocks))
    assert_query_grid_alignment(seq_len, 128)
    mask_128 = _mask_from_scores(padded[0], padded[1], 128, KERNEL_BLOCK, scale, "nvfp4", k)
    expanded = expand_query_axis(mask_128, 128 // KERNEL_BLOCK).unsqueeze(0)
    kernel_out = sparse_bf16(padded[0], padded[1], padded[2], expanded, sizes)[:, :seq_len]
    reference = masked_reference(padded[0], padded[1], padded[2], expanded, valid, scale)[:, :seq_len]
    metrics = error_metrics(kernel_out, reference)
    results.append({
        "check": "expanded_128x64_mask_matches_masked_reference",
        "seq_len": seq_len,
        "k": k,
        "n_k_blocks": n_k_blocks,
        "kernel_query_blocks": expanded.shape[-2],
        **metrics,
        "pass": metrics["rel_l2"] is not None and metrics["rel_l2"] < 5e-3,
    })


def check_score_dtype_ties(device, results: list[dict]) -> None:
    """Reproduce trap 8: fp32 block scores manufacture boundary ties, fp64 does not."""
    torch.manual_seed(17)
    seq_len = 64 * KERNEL_BLOCK
    scale = HEAD_DIM**-0.5
    # Wan Q/K carry a large common-mode component; that is what puts the scores
    # at ~5e5 while the discriminative spread is ~14.
    common = 6.0 * torch.randn(1, 1, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    query = (torch.randn(1, seq_len, 2, HEAD_DIM, device=device, dtype=torch.bfloat16) + common)
    key = (torch.randn(1, seq_len, 2, HEAD_DIM, device=device, dtype=torch.bfloat16) + common)
    row: dict[str, Any] = {"check": "fp32_vs_fp64_boundary_ties"}
    k = max(1, round(0.20 * (seq_len // KERNEL_BLOCK)))
    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        pooled_q = pool_blocks_1d(query, 128)
        pooled_k = pool_blocks_1d(key, KERNEL_BLOCK)
        scores = block_scores(pooled_q, pooled_k, scale, dtype)
        sorted_scores = torch.sort(scores, dim=-1, descending=True, stable=True).values
        ties = int((sorted_scores[..., k - 1] == sorted_scores[..., k]).sum().item())
        row[f"boundary_ties_{dtype_name}"] = ties
        row[f"score_magnitude_{dtype_name}"] = float(scores.abs().median().item())
        row[f"score_spread_{dtype_name}"] = float((sorted_scores[..., 0] - sorted_scores[..., -1]).median().item())
    row["pass"] = row["boundary_ties_float64"] <= row["boundary_ties_float32"]
    results.append(row)


def check_geometry_layout_roundtrip(device, results: list[dict]) -> None:
    """Layout invariants for all three geometries, at Wan's real token grid.

    The cube arm re-orders tokens *and* pads to 39936 slots; if the round-trip
    were not exact, or if a pad slot leaked into a pooled score, every cube number
    would be quietly wrong. So check: the round trip is bit-exact, pad slots hold
    zeros, no block is all-pad, and mean pooling over a padded layout equals the
    mean over that block's real tokens computed independently.
    """
    torch.manual_seed(23)
    geometries = [
        raster_geometry(WAN_SEQ_LEN, 128, device),
        raster_geometry(WAN_SEQ_LEN, KERNEL_BLOCK, device),
        cube_geometry(WAN_DIT_SEQ_SHAPE, device),
    ]
    x = torch.randn(1, WAN_SEQ_LEN, 2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    for geometry in geometries:
        laid_out = to_block_layout(x, geometry)
        restored = from_block_layout(laid_out, geometry)
        pad_slots = laid_out[0, ~geometry.valid]
        pooled = pool_geometry_blocks(laid_out, geometry.key_block_sizes, geometry.block_k)
        # Independent per-block mean over the real tokens only, no padded view.
        sizes = geometry.key_block_sizes.tolist()
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        checks = [0, len(sizes) // 2, len(sizes) - 1]
        worst = 0.0
        for block in checks:
            start = block * geometry.block_k
            slots = laid_out[0, start:start + geometry.block_k, 0].float()
            direct = slots[geometry.valid[start:start + geometry.block_k]].mean(dim=0)
            worst = max(worst, float((pooled[0, block] - direct).abs().max().item()))
        described = geometry.describe()
        results.append({
            "check":
            "geometry_layout_roundtrip",
            **described,
            "roundtrip_bit_identical":
            bool(torch.equal(restored, x)),
            "pad_slots_all_zero":
            bool(pad_slots.numel() == 0 or bool((pad_slots == 0).all().item())),
            "n_valid_tokens":
            int(geometry.valid.sum().item()),
            "pooled_vs_direct_mean_max_abs":
            worst,
            "pass": (bool(torch.equal(restored, x)) and (pad_slots.numel() == 0 or bool(
                (pad_slots == 0).all().item())) and int(geometry.valid.sum().item()) == WAN_SEQ_LEN
                     and described["all_pad_blocks"] == 0 and worst < 1e-5),
        })


def check_geometry_kernel_vs_reference(device, results: list[dict]) -> None:
    """The gate the geometry study turns on: does the kernel execute the mask?

    For each of the three geometries, a top-k mask derived in that geometry's own
    layout is run on the block-sparse kernel and compared against the independent
    fp32 masked-softmax reference over the *same* padded layout. Uses a reduced
    head count so the naive reference is affordable at Wan's 39936-slot cube
    layout. Also verifies that perturbing V inside never-selected blocks — and
    inside pad slots — cannot move the output.
    """
    torch.manual_seed(29)
    heads = 2
    scale = HEAD_DIM**-0.5
    query = torch.randn(1, WAN_SEQ_LEN, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, WAN_SEQ_LEN, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, WAN_SEQ_LEN, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    geometries = [
        raster_geometry(WAN_SEQ_LEN, 128, device),
        raster_geometry(WAN_SEQ_LEN, KERNEL_BLOCK, device),
        cube_geometry(WAN_DIT_SEQ_SHAPE, device),
    ]
    for geometry in geometries:
        laid_q = to_block_layout(query, geometry)
        laid_k = to_block_layout(key, geometry)
        laid_v = to_block_layout(value, geometry)
        for sparsity in (0.80, 0.95):
            k = max(1, min(math.ceil((1 - sparsity) * geometry.n_k_blocks), geometry.n_k_blocks))
            mask = _geometry_mask(query, key, geometry, scale, "nvfp4", k)
            kernel_mask = expand_query_axis(mask, geometry.query_expand).unsqueeze(0)
            kernel_out = sparse_bf16(laid_q, laid_k, laid_v, kernel_mask, geometry.key_block_sizes)
            reference = masked_reference(laid_q, laid_k, laid_v, kernel_mask, geometry.valid, scale)
            metrics = error_metrics(from_block_layout(kernel_out, geometry), from_block_layout(reference, geometry))
            results.append({
                "check": "geometry_mask_on_kernel_vs_masked_reference",
                "geometry": geometry.name,
                "token_order": geometry.token_order,
                "sparsity": sparsity,
                "k": k,
                "n_k_blocks": geometry.n_k_blocks,
                "retained_token_fraction": retained_token_fraction(mask, geometry.key_block_sizes),
                **metrics,
                "pass": metrics["rel_l2"] is not None and metrics["rel_l2"] < 5e-3,
            })

        k = max(1, min(math.ceil(0.20 * geometry.n_k_blocks), geometry.n_k_blocks))
        mask = _geometry_mask(query, key, geometry, scale, "bf16", k)
        kernel_mask = expand_query_axis(mask, geometry.query_expand).unsqueeze(0)
        baseline = sparse_bf16(laid_q, laid_k, laid_v, kernel_mask, geometry.key_block_sizes)
        never_selected = ~kernel_mask[0].any(dim=1).any(dim=0)
        tampered = laid_v.clone()
        for block in range(geometry.n_k_blocks):
            if bool(never_selected[block]):
                tampered[:, block * KERNEL_BLOCK:(block + 1) * KERNEL_BLOCK] += 100.0
        # Pad slots inside *selected* blocks must be inert too, which is what the
        # kernel's variable_block_sizes column mask is for.
        tampered[:, ~geometry.valid] += 100.0
        tampered_out = sparse_bf16(laid_q, laid_k, tampered, kernel_mask, geometry.key_block_sizes)
        identical = bool(torch.equal(from_block_layout(baseline, geometry), from_block_layout(tampered_out, geometry)))
        results.append({
            "check": "geometry_masked_and_pad_slots_do_not_contribute",
            "geometry": geometry.name,
            "never_selected_blocks": int(never_selected.sum().item()),
            "pad_slots_perturbed": int((~geometry.valid).sum().item()),
            "bit_identical": identical,
            "pass": identical,
        })


def check_geometry_attention_mass(device, results: list[dict]) -> None:
    """Mass rows must still be a probability distribution at every geometry."""
    torch.manual_seed(31)
    scale = HEAD_DIM**-0.5
    query = torch.randn(1, WAN_SEQ_LEN, 2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, WAN_SEQ_LEN, 2, HEAD_DIM, device=device, dtype=torch.bfloat16)
    for geometry in (raster_geometry(WAN_SEQ_LEN, KERNEL_BLOCK, device), cube_geometry(WAN_DIT_SEQ_SHAPE, device)):
        laid_q = to_block_layout(query, geometry)
        laid_k = to_block_layout(key, geometry)
        blocks = [0, geometry.n_q_blocks // 2, geometry.n_q_blocks - 1]
        mass = block_attention_mass(laid_q, laid_k, 0, blocks, geometry, scale)
        totals = mass.sum(dim=-1)
        deviation = float((totals - 1.0).abs().max().item())
        results.append({
            "check": "geometry_block_attention_mass_normalized",
            "geometry": geometry.name,
            "rows": int(mass.shape[0]),
            "max_abs_deviation_from_one": deviation,
            "pass": deviation < 1e-4,
        })


def check_tie_denominator_reconciliation(device, results: list[dict]) -> None:
    """Phase 1 vs Phase 2 tie counts: same data, two denominators, three geometries.

    Phase 1 reported ~110 boundary ties per cell at 128x64; Phase 2 reported
    ~1,400 at the same geometry. The candidate explanation is that Phase 1's
    ``boundary_ties`` was **per head** (256 query blocks) while Phase 2's was
    summed over all 12 heads (3,072 query blocks) — a factor of 12 — with the
    remainder from real Q/K having a larger common-mode offset than this synthetic
    stand-in. This check establishes the ratio arithmetic on identical scores so
    the reconciliation does not rest on reading two reports.
    """
    torch.manual_seed(37)
    scale = HEAD_DIM**-0.5
    common = 6.0 * torch.randn(1, 1, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) + common
    key = torch.randn(1, WAN_SEQ_LEN, WAN_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16) + common
    for geometry in (raster_geometry(WAN_SEQ_LEN, 128,
                                     device), raster_geometry(WAN_SEQ_LEN, KERNEL_BLOCK,
                                                              device), cube_geometry(WAN_DIT_SEQ_SHAPE, device)):
        laid_q = to_block_layout(query, geometry)
        laid_k = to_block_layout(key, geometry)
        pooled_q = pool_geometry_blocks(laid_q, geometry.query_block_sizes, geometry.block_q)
        pooled_k = pool_geometry_blocks(laid_k, geometry.key_block_sizes, geometry.block_k)
        row: dict[str, Any] = {
            "check": "tie_denominator_reconciliation",
            "geometry": geometry.name,
            "n_q_blocks": geometry.n_q_blocks,
            "n_heads": WAN_HEADS,
        }
        for name, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            scores = block_scores(pooled_q, pooled_k, scale, dtype)
            ordered = torch.sort(scores, dim=-1, descending=True, stable=True).values
            k = max(1, min(math.ceil(0.20 * geometry.n_k_blocks), geometry.n_k_blocks))
            ties = (ordered[..., k - 1] == ordered[..., k]).sum(dim=-1)
            row[f"ties_per_cell_{name}"] = int(ties.sum().item())
            row[f"ties_per_head_median_{name}"] = float(ties.to(torch.float64).median().item())
        row["per_cell_over_per_head_ratio"] = (None if not row["ties_per_head_median_fp32"] else
                                               row["ties_per_cell_fp32"] / row["ties_per_head_median_fp32"])
        row["pass"] = row["ties_per_cell_fp64"] <= row["ties_per_cell_fp32"]
        results.append(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry-only",
                        action="store_true",
                        help="run only the Phase 2B geometry gate, skipping the already-passed Phase 2 checks")
    args = parser.parse_args()
    device = torch.device("cuda")
    results: list[dict] = []
    if not args.geometry_only:
        check_small_sparse_vs_reference(device, results)
        check_128x64_geometry(device, results)
        check_score_dtype_ties(device, results)
        check_random_perturbation(device, results)
        check_attention_mass(device, results)
        check_dense_paths(device, results)
    check_geometry_layout_roundtrip(device, results)
    check_geometry_kernel_vs_reference(device, results)
    check_geometry_attention_mass(device, results)
    check_tie_denominator_reconciliation(device, results)

    failures = [row for row in results if not row.get("pass")]
    payload = {
        "verdict": "PASS" if not failures else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "checks": results,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
