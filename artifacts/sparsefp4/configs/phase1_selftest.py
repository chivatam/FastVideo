"""Unit checks for the Phase-1 routing-probe metric helpers.

Runs on GPU (the NVFP4 arm needs the flashinfer kernel) but uses tiny synthetic
tensors, so it is a seconds-long gate before any 50-step generation.
"""

from __future__ import annotations

import math

import torch

from fastvideo.attention.backends.routing_probe_attn import (E2M1_MAX, average_ranks, compare_masks, margins,
                                                             pool_blocks_1d, quantize_router_input,
                                                             sort_scores_descending, spearman_median)
from fastvideo.attention.backends.video_sparse_attn import compute_topk

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
    print(f"{'PASS' if condition else 'FAIL'}  {message}")


def check_ragged_masked_mean() -> None:
    seq_len, block = 32760, 128
    x = torch.ones(1, seq_len, 2, 4, device="cuda", dtype=torch.bfloat16)
    pooled = pool_blocks_1d(x, block)
    n_blocks = math.ceil(seq_len / block)
    check(pooled.shape == (2, n_blocks, 4), f"pool shape is [H, ceil(S/blk), D]: {tuple(pooled.shape)}")
    check(torch.allclose(pooled, torch.ones_like(pooled)),
          "masked mean of an all-ones tensor is 1.0 in every block including the ragged tail")
    check(n_blocks == 256 and seq_len - 255 * block == 120, f"ragged tail at 32760/128 is 120 tokens (n={n_blocks})")

    ramp = torch.arange(seq_len, device="cuda", dtype=torch.float32).view(1, seq_len, 1, 1).to(torch.bfloat16)
    pooled_ramp = pool_blocks_1d(ramp.expand(1, seq_len, 1, 4).contiguous(), block)
    tail_expected = float(torch.arange(255 * block, seq_len, device="cuda", dtype=torch.float32).mean())
    check(abs(float(pooled_ramp[0, -1, 0]) - tail_expected) / tail_expected < 2e-3,
          "ragged block pools over valid tokens only (no zero-pad shrinkage)")


def check_tie_break() -> None:
    scores = torch.tensor([[[1.0, 1.0, 1.0, 0.5]]], device="cuda")
    _, index = sort_scores_descending(scores)
    check(index[0, 0, :3].tolist() == [0, 1, 2], f"ties resolve to ascending key-block index: {index[0, 0].tolist()}")


def check_equal_budget_and_identity() -> None:
    torch.manual_seed(0)
    reference = torch.randn(3, 8, 64, device="cuda")
    candidate = reference.clone()
    candidate[..., 0] += 5.0
    ref_sorted = sort_scores_descending(reference)
    cand_sorted = sort_scores_descending(candidate)
    k = compute_topk(0.90, 64)
    check(k == 7, f"compute_topk(0.90, 64) == ceil(0.1*64) == 7, got {k}")

    self_stats = compare_masks(ref_sorted, ref_sorted, k, 64)
    check(all(value == k * 8 for value in self_stats["intersection"]),
          "null control: reference vs itself gives full intersection (jaccard 1.0)")

    stats = compare_masks(ref_sorted, cand_sorted, k, 64)
    for head in range(3):
        intersection = stats["intersection"][head]
        budget = k * 8
        union = 2 * budget - intersection
        recall = intersection / budget
        jaccard = intersection / union
        check(abs(jaccard - recall / (2 - recall)) < 1e-12,
              f"head {head}: jaccard == recall/(2-recall) identity holds")


def check_margin_null_at_full_budget() -> None:
    scores = torch.randn(2, 4, 16, device="cuda")
    values, _ = sort_scores_descending(scores)
    raw, norm = margins(values, 16)
    check(torch.isnan(raw).all() and torch.isnan(norm).all(),
          "k == n_key_blocks yields a null margin (no s_(k+1)), not 0 or -1")
    raw8, norm8 = margins(values, 8)
    check(bool((raw8 >= 0).all()) and bool(((norm8 >= 0) & (norm8 <= 1)).all()),
          "margins are non-negative and margin_norm is in [0, 1]")


def check_rank_correlation() -> None:
    scores = torch.tensor([[[3.0, 1.0, 1.0, 1.0, 5.0]]], device="cuda")
    ranks = average_ranks(scores)
    check(ranks[0, 0].tolist() == [4.0, 2.0, 2.0, 2.0, 5.0],
          f"average-tie-corrected ranks: {ranks[0, 0].tolist()}")
    rho_self = spearman_median(scores, scores)
    check(abs(rho_self[0] - 1.0) < 1e-6, f"spearman of a vector with itself is 1.0, got {rho_self[0]}")
    rho_flip = spearman_median(scores, -scores)
    check(abs(rho_flip[0] + 1.0) < 1e-6, f"spearman of a vector with its negation is -1.0, got {rho_flip[0]}")


def check_quantizers() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)

    identity, sat = quantize_router_input(x, "bf16")
    check(identity is x and sat == 0.0, "bf16 arm is the identity on the captured tensor")

    fp8, sat8 = quantize_router_input(x, "fp8_e4m3")
    fp8_err = ((fp8 - x.float()).norm() / x.float().norm()).item()
    check(0.0 < fp8_err < 0.05, f"fp8_e4m3 round-trip rel-L2 is small but non-zero: {fp8_err:.5f}")

    native, sat_native = quantize_router_input(x, "nvfp4")
    native_err = ((native - x.float()).norm() / x.float().norm()).item()
    check(0.02 < native_err < 0.25, f"native nvfp4 round-trip rel-L2 in expected band: {native_err:.5f}")
    check(native_err > fp8_err, "nvfp4 is coarser than fp8_e4m3, as the formats imply")

    simulated, _ = quantize_router_input(x, "nvfp4_sim")
    agreement = (simulated == native).float().mean().item()
    check(agreement > 0.98, f"simulated nvfp4 agrees with the native codes elementwise: {agreement:.4f}")

    scales = native[native != 0].abs()
    ratios = (scales.view(-1) / scales.view(-1)).unique()
    check(ratios.numel() >= 1, "native dequantized values are finite")
    check(bool(torch.isfinite(native).all()) and bool(torch.isfinite(simulated).all()),
          "no NaN/Inf from either nvfp4 path")
    check(0.0 <= sat_native <= 1.0 and 0.0 <= sat8 <= 1.0, "saturation fractions are in [0, 1]")

    zeros = torch.zeros(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    zero_native, _ = quantize_router_input(zeros, "nvfp4")
    zero_sim, _ = quantize_router_input(zeros, "nvfp4_sim")
    check(bool((zero_native == 0).all()) and bool((zero_sim == 0).all()),
          "an all-zero group quantizes to zero without dividing by zero")

    big = torch.full((1, 128, 1, 128), 8.0, device="cuda", dtype=torch.bfloat16)
    big_native, big_sat = quantize_router_input(big, "nvfp4")
    check(bool(torch.isfinite(big_native).all()), "saturating input clamps instead of emitting inf")
    check(big_sat > 0.5, f"a constant tensor saturates the E2M1 top code (max {E2M1_MAX}): {big_sat:.3f}")


def main() -> int:
    check_ragged_masked_mean()
    check_tie_break()
    check_equal_budget_and_identity()
    check_margin_null_at_full_budget()
    check_rank_correlation()
    check_quantizers()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
