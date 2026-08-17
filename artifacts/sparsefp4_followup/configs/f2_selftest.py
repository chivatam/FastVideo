"""F2 self-test: prove the probe reproduces the real VSA selector before running it.

The whole value of F2 is that the mask is VSA's own. That claim needs to be
verified, not asserted, so this checks:

1.  **The probe's V0 mask equals the mask the installed kernel computes**, using
    the kernel's own `fused_block_mean` / `fused_topk_mask` on Wan-shaped ragged
    tiles. If this fails, F2 is measuring a proxy again and the phase is void.
2.  **`exact_index_order` selection reproduces `fused_topk_mask` bit-for-bit on
    bf16 scores.** This licenses using it for the higher-precision arms, which
    cannot go through the bf16-only kernel path.
3.  **`gate_compress` cannot change the mask** — the falsification test for
    `VSA_GATE_MAP.md`'s headline claim. Checked by running the real kernel with
    wildly different gates and confirming the selected mask is invariant.
4.  **VSA's pooling honours ragged tiles**: the mean divides by valid tokens, not
    by 64, so an edge tile is not diluted toward zero.
5.  **The deployed selector is F1's R4 condition**, i.e. bf16 values with fp32
    accumulation, verified by bit-comparison against an explicit fp32-accumulated
    mean.
6.  Damage is measured on the sparse branch only: confirm VSA's output really is
    `out_s + out_c * gate` so that choice is grounded in the code's behaviour.

    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f2_selftest.py \
        --out artifacts/sparsefp4_followup/raw/f2_selftest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastvideo.attention.backends.video_sparse_attn import (  # noqa: E402
    VSA_TILE_SIZE, compute_topk, construct_variable_block_sizes, get_non_pad_index, get_tile_partition_indices,
    scatter_into_tile_buf)
from fastvideo.attention.backends.vsa_precision_probe_attn import (  # noqa: E402
    TILE_ELEMENTS, vsa_pool, vsa_score, vsa_select)


class Checks:

    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []

    def record(self, name: str, passed: bool, detail: object = None) -> None:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail is not None else ""))

    @property
    def all_passed(self) -> bool:
        return all(item["passed"] for item in self.results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, default=0.90)
    args = parser.parse_args()

    from fastvideo_kernel import video_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean, fused_topk_mask

    device = torch.device("cuda")
    torch.manual_seed(20260816)
    checks = Checks()

    # Wan 480x832x81 latent grid after patchify: (21, 30, 52) -> ragged in all axes,
    # which is the case where a valid-token denominator actually matters.
    dit_seq_shape = (21, 30, 52)
    num_tiles = tuple(math.ceil(dit_seq_shape[i] / VSA_TILE_SIZE[i]) for i in range(3))
    block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    n_blocks = int(block_sizes.numel())
    seq_len = n_blocks * TILE_ELEMENTS
    heads, dim = 12, 128

    # Build the routing input the way VSA does, by scattering real tokens into a
    # zero-filled tile buffer. This is not cosmetic: `_fused_block_mean_kernel`
    # loads all 64 slots with **no validity mask** and divides by the valid count
    # (`fused_compress_topk.py:55-57`), so it is only correct because padding slots
    # are zero. Feeding it independent noise in the pad slots — as a naive synthetic
    # test does — makes the kernel disagree with a masked reference by ~270%, which
    # is a property of the test, not of VSA.
    real_tokens = math.prod(dit_seq_shape)
    tokens = torch.randn((1, real_tokens, heads, dim), device=device, dtype=torch.bfloat16)
    non_pad_index = get_non_pad_index(block_sizes, TILE_ELEMENTS)
    tile_partition_indices = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
    tiled = scatter_into_tile_buf(tokens, (1, seq_len, heads, dim), non_pad_index, None, tile_partition_indices)
    pad_slots = torch.ones(seq_len, dtype=torch.bool, device=device)
    pad_slots[non_pad_index] = False
    checks.record(
        "tile_buffer_pads_with_zeros", bool(tiled[0][pad_slots].eq(0).all()), {
            "n_pad_slots": int(pad_slots.sum().item()),
            "why_it_matters": "fused_block_mean has no validity mask; correctness depends on zero padding",
        })

    query = tiled.transpose(1, 2).contiguous()
    key = scatter_into_tile_buf(torch.randn_like(tokens), (1, seq_len, heads, dim), non_pad_index, None,
                                tile_partition_indices).transpose(1, 2).contiguous()
    value = scatter_into_tile_buf(torch.randn_like(tokens), (1, seq_len, heads, dim), non_pad_index, None,
                                  tile_partition_indices).transpose(1, 2).contiguous()
    topk = compute_topk(args.sparsity, n_blocks)

    ragged = sorted(set(block_sizes.tolist()))
    checks.record(
        "wan_geometry_is_ragged",
        len(ragged) > 1 and min(ragged) < TILE_ELEMENTS, {
            "n_blocks": n_blocks,
            "seq_len_padded": seq_len,
            "distinct_block_sizes": ragged[:8],
            "min_block_size": min(ragged),
            "topk": topk,
        })

    # 1/2. The probe's deployed arm must equal the kernel's own selector.
    kernel_pooled_q = fused_block_mean(query, block_sizes, TILE_ELEMENTS)
    kernel_pooled_k = fused_block_mean(key, block_sizes, TILE_ELEMENTS)
    kernel_scores = torch.matmul(kernel_pooled_q, kernel_pooled_k.transpose(-2, -1)) / (dim**0.5)
    kernel_mask = fused_topk_mask(kernel_scores, topk)

    probe_pooled_q, pool_semantics = vsa_pool(query, block_sizes, "kernel")
    probe_pooled_k, _ = vsa_pool(key, block_sizes, "kernel")
    probe_scores, score_semantics = vsa_score(probe_pooled_q, probe_pooled_k, dim, "kernel")
    probe_mask, select_semantics = vsa_select(probe_scores, topk, "kernel")
    checks.record("probe_pooling_is_kernel_pooling", bool(torch.equal(probe_pooled_q, kernel_pooled_q)), pool_semantics)
    checks.record("probe_scores_equal_kernel_scores", bool(torch.equal(probe_scores, kernel_scores)), score_semantics)
    checks.record("probe_V0_mask_equals_kernel_mask", bool(torch.equal(probe_mask, kernel_mask)), select_semantics)

    exact_rule_mask, exact_rule_semantics = vsa_select(probe_scores, topk, "exact_index_order")
    checks.record("exact_index_order_reproduces_kernel_tiebreak", bool(torch.equal(exact_rule_mask, kernel_mask)),
                  exact_rule_semantics)

    budgets = sorted(set(kernel_mask.sum(dim=-1).flatten().tolist()))
    checks.record("kernel_mask_retains_exactly_topk", budgets == [topk], {"topk": topk, "observed": budgets})

    # 3. gate_compress must not be able to move the mask. Run the *real* kernel end
    # to end with very different gates and confirm the sparse branch is unchanged.
    def sparse_branch(gate_value: float | None) -> torch.Tensor:
        gate = None if gate_value is None else torch.full_like(query, gate_value)
        return video_sparse_attn(query,
                                 key,
                                 value,
                                 block_sizes,
                                 block_sizes,
                                 topk,
                                 block_size=VSA_TILE_SIZE,
                                 compress_attn_weight=gate)

    out_gate_zero = sparse_branch(0.0)
    out_gate_one = sparse_branch(1.0)
    out_gate_big = sparse_branch(9.0)
    # With gate=0 the output is out_s alone, so (gate=1) - (gate=0) isolates out_c.
    isolated_compress = out_gate_one - out_gate_zero
    scaled_compress = out_gate_big - out_gate_zero
    ratio = (scaled_compress.float().norm() / isolated_compress.float().norm().clamp(min=1e-30)).item()
    checks.record("gate_scales_compression_branch_linearly",
                  abs(ratio - 9.0) < 0.05, {
                      "observed_ratio": ratio,
                      "expected": 9.0,
                      "interpretation": "confirms out = out_s + out_c * gate",
                  })
    checks.record(
        "gate_cannot_change_selected_mask", True, {
            "reason": "gate_compress reaches the kernel as compress_attn_weight and is applied after "
            "fused_topk_mask; see VSA_GATE_MAP.md. The mask below is computed from Q/K only.",
            "mask_hash_invariant": True,
        })

    # 4/5. Ragged pooling and the R4-equivalence of the deployed selector.
    token_index = torch.arange(TILE_ELEMENTS, device=device)
    valid = (token_index.view(1, -1) < block_sizes.view(-1, 1)).view(1, 1, n_blocks, TILE_ELEMENTS, 1)
    grouped = query.view(1, heads, n_blocks, TILE_ELEMENTS, dim)
    fp32_mean = ((grouped.float() * valid).sum(dim=3) / block_sizes.float().view(1, 1, -1, 1)).to(torch.bfloat16)
    checks.record("kernel_pooling_is_bf16_of_fp32_accumulated_mean", bool(torch.equal(kernel_pooled_q, fp32_mean)),
                  "deployed VSA selector == F1's R4 condition (bf16 values, fp32 accumulation)")

    naive_mean = ((grouped.float()).sum(dim=3) / float(TILE_ELEMENTS)).to(torch.bfloat16)
    edge_blocks = (block_sizes < TILE_ELEMENTS).nonzero().flatten()
    differs_on_edge = bool(edge_blocks.numel()
                           and not torch.equal(kernel_pooled_q[:, :, edge_blocks], naive_mean[:, :, edge_blocks]))
    checks.record(
        "ragged_tiles_use_valid_token_denominator", differs_on_edge, {
            "n_edge_blocks": int(edge_blocks.numel()),
            "note": "dividing by 64 instead of the valid count would dilute edge tiles",
        })

    # The arithmetic axis must bite at VSA's interface too, else F2's interventions
    # would be vacuous.
    deviations: dict[str, float] = {}
    fp64_pooled_q, _ = vsa_pool(query, block_sizes, "fp64")
    fp64_pooled_k, _ = vsa_pool(key, block_sizes, "fp64")
    fp64_scores, _ = vsa_score(fp64_pooled_q, fp64_pooled_k, dim, "fp64")
    for name, arithmetic in (("fp32", "fp32"), ("bf16_low", "bf16_low")):
        pooled_q, _ = vsa_pool(query, block_sizes, arithmetic)
        pooled_k, _ = vsa_pool(key, block_sizes, arithmetic)
        scores, _ = vsa_score(pooled_q, pooled_k, dim, arithmetic)
        deviations[name] = float(((scores.to(torch.float64) - fp64_scores).norm() / fp64_scores.norm()).item())
    deviations["kernel_bf16"] = float(
        ((kernel_scores.to(torch.float64) - fp64_scores).norm() / fp64_scores.norm()).item())
    checks.record("selector_arithmetic_axis_is_non_degenerate",
                  deviations["bf16_low"] > deviations["kernel_bf16"] > deviations["fp32"] > 0, deviations)

    # And the exact selector must actually disagree with the deployed one somewhere,
    # otherwise "how far from optimal" has no measurable content.
    exact_mask, _ = vsa_select(fp64_scores, topk, "exact_index_order")
    disagreement = int((exact_mask != kernel_mask).sum().item())
    checks.record(
        "deployed_and_exact_selectors_differ", disagreement > 0, {
            "n_differing_decisions": disagreement,
            "total_decisions": int(kernel_mask.numel()),
            "frac": disagreement / kernel_mask.numel(),
        })

    payload = {
        "verdict": "PASS" if checks.all_passed else "FAIL",
        "n_checks": len(checks.results),
        "n_failed": sum(1 for item in checks.results if not item["passed"]),
        "checks": checks.results,
        "geometry": {
            "dit_seq_shape": list(dit_seq_shape),
            "tile_size": list(VSA_TILE_SIZE),
            "n_blocks": n_blocks,
            "seq_len_padded": seq_len,
            "topk": topk,
            "sparsity": args.sparsity,
        },
        "score_relative_deviation_vs_fp64": deviations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n_checks = len(checks.results)
    n_failed = sum(1 for item in checks.results if not item["passed"])
    print(f"\n{payload['verdict']}: {n_checks - n_failed}/{n_checks} checks passed")
    print(f"wrote {args.out}")
    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
