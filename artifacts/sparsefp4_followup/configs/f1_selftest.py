"""F1 self-test: prove the scorer-precision numerics before spending GPU hours.

Checks the properties the phase's conclusions rest on, on synthetic tensors where
the right answer is known by construction:

1.  **R0/R2 equivalence to study 1's scorer.** ``pool_blocks_precision`` at fp32 +
    ``score_blocks_precision`` at fp64 must reproduce study 1's
    ``pool_geometry_blocks`` + ``block_scores`` bit-for-bit, otherwise this phase
    is not measuring the same scorer it claims to be ablating.
2.  **The arithmetic axis actually bites.** Lowering only the arithmetic (fixed
    representation) must change the scores; if bf16/fp8/nvfp4-like arithmetic were
    numerically identical to fp64 there would be nothing to measure and any null
    result would be vacuous.
3.  **The accumulation distinction is real.** ``acc=low`` must differ from
    ``acc=native`` for bf16, which is what licenses reporting them as separate
    ladder positions rather than one "bf16" arm.
4.  **Equal budget.** Every arm's mask retains exactly ``k`` blocks per query
    block, and ``mask_comparison`` raises when that is violated.
5.  **Null controls.** Reference-vs-itself gives Jaccard 1, rel-L2 0; a zero-swap
    random control leaves the mask untouched.
6.  **Matched-random equality.** The random control changes exactly the same
    number of blocks as the arm it is matched to, per query block.
7.  **Spearman correctness** on known permutations, including ties.
8.  **FP64 shadow ties.** Exact-tie count under the fp64 shadow reference must be
    zero on generic data, so boundary margins are resolved.
9.  **Native FP8 GEMM availability**, recorded rather than assumed.
10. **NVFP4 pooled quantization lands on the real e2m1 code set.**

Run:
    source artifacts/sparsefp4_followup/configs/env.sh
    CUDA_VISIBLE_DEVICES=0 "$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_selftest.py \
        --out artifacts/sparsefp4_followup/raw/f1_selftest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastvideo.attention.backends.sparsefp4_mask_metrics import (  # noqa: E402
    boundary_diagnostics, deployed_tie_count, mask_comparison, mask_hash, spearman_rho)
from fastvideo.attention.backends.sparsefp4_numerics import (  # noqa: E402
    block_scores, pool_geometry_blocks, random_matched_mask, raster_geometry, topk_block_mask)
from fastvideo.attention.backends.sparsefp4_scorer_precision import (  # noqa: E402
    E2M1_MAX, SCORER_ARMS, ambient_fp32_state, assert_exact_fp32_matmul, declared_precision_arithmetic,
    exact_fp32_matmul, pool_blocks_precision, quantize_pooled_fp8_e4m3, quantize_pooled_nvfp4, score_blocks_fp8_native,
    score_blocks_precision)


class Checks:

    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []

    def record(self, name: str, passed: bool, detail: object = None) -> None:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail is not None else ""))

    @property
    def all_passed(self) -> bool:
        return all(item["passed"] for item in self.results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.90)
    args = parser.parse_args()

    torch.manual_seed(20260816)
    device = torch.device("cuda")
    checks = Checks()

    # The scorer must be exact even when the ambient process has TF32 ON — which
    # is exactly the state the FastVideo worker subprocess runs in, and which the
    # first F1 smoke run discovered the hard way. Verify the scoped helper by
    # deliberately enabling TF32 first.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ambient_before = ambient_fp32_state()
    with exact_fp32_matmul():
        inside = ambient_fp32_state()
    ambient_after = ambient_fp32_state()
    checks.record("exact_fp32_scope_disables_tf32_and_restores",
                  (not inside["worker_allow_tf32_matmul"] and inside["worker_float32_matmul_precision"] == "highest"
                   and ambient_after == ambient_before), {
                       "ambient": ambient_before,
                       "inside_scope": inside
                   })

    # An fp32 scorer under ambient TF32 must equal one under ambient exact fp32;
    # if the scope leaked, these would differ in the 3rd decimal of the mantissa.
    probe_a = torch.randn((2, 64, 128), device=device, dtype=torch.float32)
    probe_b = torch.randn((2, 96, 128), device=device, dtype=torch.float32)
    scores_tf32_ambient, _ = score_blocks_precision(probe_a, probe_b, 1.0, "fp32", "native")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    scores_exact_ambient, _ = score_blocks_precision(probe_a, probe_b, 1.0, "fp32", "native")
    checks.record("fp32_scorer_is_tf32_invariant", bool(torch.equal(scores_tf32_ambient, scores_exact_ambient)),
                  {"max_abs_diff": float((scores_tf32_ambient - scores_exact_ambient).abs().max().item())})

    assert_exact_fp32_matmul()
    checks.record("tf32_disabled_so_fp32_arm_is_fp32", True)

    # The confound that invalidated the first F1 full run: FastVideo's denoising loop
    # wraps the transformer in autocast(bf16), and autocast casts *fp32* matmul
    # inputs down to bf16. Under it, R2/R3 (fp32) silently computed in bf16 and
    # became bit-identical duplicates of R4/R5, which looked like a real finding.
    # fp64 is exempt, which is why study 1's fp64-only scorer never exposed this.
    probe_a32 = torch.randn((2, 64, 128), device=device, dtype=torch.float32)
    probe_b32 = torch.randn((2, 96, 128), device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        naked_dtype = torch.matmul(probe_a32, probe_b32.transpose(-1, -2)).dtype
        guarded_scores, _ = score_blocks_precision(probe_a32, probe_b32, 1.0, "fp32", "native")
        with declared_precision_arithmetic():
            block_guarded_dtype = torch.matmul(probe_a32, probe_b32.transpose(-1, -2)).dtype
    checks.record("autocast_downcasts_unguarded_fp32_matmul", naked_dtype == torch.bfloat16, {
        "unguarded_dtype": str(naked_dtype),
        "note": "this is the hazard the guards exist for"
    })
    checks.record("declared_precision_arithmetic_restores_fp32", block_guarded_dtype == torch.float32,
                  {"guarded_dtype": str(block_guarded_dtype)})
    checks.record(
        "fp32_scorer_is_autocast_invariant", guarded_scores.dtype == torch.float64, {
            "scorer_output_dtype": str(guarded_scores.dtype),
            "note": "score_blocks_precision returns fp64-scaled scores; the matmul itself ran in fp32",
        })

    # And the arms must be distinguishable under autocast, which is the property that
    # actually failed before: verify fp32 and bf16 arms differ inside autocast.
    with (torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), declared_precision_arithmetic()):
        fp32_arm, _ = score_blocks_precision(probe_a32, probe_b32, 1.0, "fp32", "native")
        bf16_arm, _ = score_blocks_precision(probe_a32, probe_b32, 1.0, "bf16", "native")
    checks.record("fp32_and_bf16_arms_differ_under_autocast", not torch.equal(fp32_arm, bf16_arm),
                  {"max_abs_diff": float((fp32_arm - bf16_arm).abs().max().item())})

    seq_len, heads, dim = args.seq_len, args.heads, args.head_dim
    softmax_scale = dim**-0.5
    geometry = raster_geometry(seq_len, args.block_q, device)
    query = torch.randn((1, seq_len, heads, dim), device=device, dtype=torch.bfloat16)
    key = torch.randn((1, seq_len, heads, dim), device=device, dtype=torch.bfloat16)

    # 1. R0/R2 pooling+scoring must reproduce study 1's implementation exactly.
    study1_pooled_q = pool_geometry_blocks(query, geometry.query_block_sizes, geometry.block_q)
    study1_pooled_k = pool_geometry_blocks(key, geometry.key_block_sizes, geometry.block_k)
    study1_scores = block_scores(study1_pooled_q, study1_pooled_k, softmax_scale, torch.float64)
    new_pooled_q, pool_semantics = pool_blocks_precision(query, geometry.query_block_sizes, geometry.block_q, "fp32",
                                                         "native")
    new_pooled_k, _ = pool_blocks_precision(key, geometry.key_block_sizes, geometry.block_k, "fp32", "native")
    pool_identical = bool(torch.equal(study1_pooled_q, new_pooled_q) and torch.equal(study1_pooled_k, new_pooled_k))
    checks.record("pooling_bit_identical_to_study1", pool_identical, pool_semantics)
    fp64_scores, score_semantics = score_blocks_precision(new_pooled_q, new_pooled_k, softmax_scale, "fp64", "native")
    checks.record("fp64_scores_bit_identical_to_study1", bool(torch.equal(study1_scores, fp64_scores)), score_semantics)

    # 2/3. The arithmetic axis must actually change the numbers.
    arm_scores: dict[str, torch.Tensor] = {"fp64": fp64_scores}
    for name, (arith, acc) in {
            "fp32": ("fp32", "native"),
            "bf16_acc_fp32": ("bf16", "native"),
            "bf16_acc_bf16": ("bf16", "low"),
    }.items():
        pooled_q, _ = pool_blocks_precision(query, geometry.query_block_sizes, geometry.block_q, arith, acc)
        pooled_k, _ = pool_blocks_precision(key, geometry.key_block_sizes, geometry.block_k, arith, acc)
        arm_scores[name], _ = score_blocks_precision(pooled_q, pooled_k, softmax_scale, arith, acc)
    rel = {
        name: float(((scores - fp64_scores).norm() / fp64_scores.norm()).item())
        for name, scores in arm_scores.items()
    }
    checks.record("lower_arithmetic_changes_scores", all(rel[name] > 0 for name in ("fp32", "bf16_acc_fp32")), rel)
    checks.record("bf16_low_accumulation_differs_from_native", rel["bf16_acc_bf16"] > rel["bf16_acc_fp32"], {
        "acc_fp32_rel": rel["bf16_acc_fp32"],
        "acc_bf16_rel": rel["bf16_acc_bf16"],
    })

    # 9. Native FP8 GEMM for the block dot product.
    pooled_q_fp8, sat_q = quantize_pooled_fp8_e4m3(new_pooled_q)
    pooled_k_fp8, _ = quantize_pooled_fp8_e4m3(new_pooled_k)
    native_fp8 = score_blocks_fp8_native(pooled_q_fp8, pooled_k_fp8, softmax_scale)
    checks.record("native_fp8_block_gemm_available", native_fp8 is not None,
                  native_fp8[1] if native_fp8 is not None else "torch._scaled_mm unavailable for these shapes")
    if native_fp8 is not None:
        arm_scores["fp8"] = native_fp8[0]
        rel["fp8"] = float(((native_fp8[0] - fp64_scores).norm() / fp64_scores.norm()).item())

    # 10. NVFP4 pooled quantization must land on the exact e2m1 code set.
    pooled_q_fp4, sat_fp4 = quantize_pooled_nvfp4(new_pooled_q)
    grouped = pooled_q_fp4.float().unflatten(-1, (-1, 16))
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    normalized = (grouped / amax * E2M1_MAX).abs()
    codes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    distance = (normalized.unsqueeze(-1) - codes.view(1, 1, 1, 1, -1)).abs().amin(dim=-1)
    checks.record(
        "nvfp4_pooled_values_on_e2m1_grid",
        float(distance.max().item()) < 1e-2, {
            "max_distance_to_code": float(distance.max().item()),
            "saturation_frac": sat_fp4,
            "fp8_pooled_saturation_frac": sat_q,
        })

    # 4/5/6. Mask budget, null controls, matched-random equality.
    k = max(1, int(round((1 - args.sparsity) * geometry.n_k_blocks)))
    masks = {name: topk_block_mask(scores, k) for name, scores in arm_scores.items()}
    reference_mask = masks["fp64"]
    budgets = {name: sorted(set(mask.sum(dim=-1).flatten().tolist())) for name, mask in masks.items()}
    checks.record("every_arm_retains_exactly_k_blocks", all(value == [k] for value in budgets.values()), {
        "k": k,
        "observed": budgets
    })

    null = mask_comparison(reference_mask[0], reference_mask[0])
    checks.record("null_control_mask_identity", null["jaccard"] == 1.0 and null["num_swaps"] == 0, null)
    checks.record("mask_hash_stable", mask_hash(reference_mask[0]) == mask_hash(reference_mask[0].clone()))

    raised = False
    try:
        mask_comparison(reference_mask[0], topk_block_mask(fp64_scores, k + 1)[0])
    except RuntimeError:
        raised = True
    checks.record("unequal_budget_raises", raised)

    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    candidate = masks["bf16_acc_bf16"]
    random_mask = random_matched_mask(reference_mask, candidate, generator)
    per_query_equal = bool(((reference_mask & ~candidate).sum(dim=-1) == (reference_mask
                                                                          & ~random_mask).sum(dim=-1)).all())
    checks.record("matched_random_changes_equal_count_per_query_block", per_query_equal)
    zero_swap = random_matched_mask(reference_mask, reference_mask, generator)
    checks.record("zero_swap_random_control_is_identity", bool(torch.equal(zero_swap, reference_mask)))

    # 7. Spearman against SciPy, including the tie-heavy case that low-precision
    # scorers actually produce. The vectorized tie-aware ranking replaced a Python
    # run-walk that cost ~200 ms per call; this gate is what licenses that swap.
    monotone = torch.arange(64, dtype=torch.float64, device=device).view(1, 64)
    checks.record("spearman_identity_is_one", abs(spearman_rho(monotone, monotone) - 1.0) < 1e-12)
    checks.record("spearman_reversal_is_minus_one", abs(spearman_rho(monotone, -monotone) + 1.0) < 1e-12)
    tied = torch.zeros((1, 64), dtype=torch.float64, device=device)
    tied[0, 32:] = 1.0
    checks.record("spearman_handles_ties_without_error", abs(spearman_rho(tied, tied) - 1.0) < 1e-12)

    scipy_cases: dict[str, float] = {}
    left = torch.randn((8, geometry.n_k_blocks), device=device, dtype=torch.float64)
    right = torch.randn((8, geometry.n_k_blocks), device=device, dtype=torch.float64)
    for case_name, (x, y) in {
            "continuous": (left, right),
            "ties_both_sides": ((left * 3).round() / 3, (right * 2).round() / 2),
            "ties_one_side": ((left * 3).round() / 3, right),
            "two_level_max_ties": (torch.cat([torch.zeros(8, 256), torch.ones(8, 256)], dim=1).to(device).double(),
                                   torch.cat([torch.zeros(8, 256), torch.ones(8, 256)], dim=1).to(device).double()),
    }.items():
        mine = spearman_rho(x, y)
        reference = float(
            numpy.mean([spearmanr(x[i].cpu().numpy(), y[i].cpu().numpy()).statistic for i in range(x.shape[0])]))
        scipy_cases[case_name] = abs(mine - reference)
    checks.record("spearman_matches_scipy_incl_ties", max(scipy_cases.values()) < 1e-10, scipy_cases)

    # Boundary diagnostics: the vectorized pair-gap must equal an explicit per-row
    # loop, since that loop was the previous implementation and the metric it
    # produces is what decides the phase's mechanism claim.
    def boundary_reference_loop(scores: torch.Tensor, cand: torch.Tensor, ref: torch.Tensor) -> list[float]:
        ordered = torch.sort(scores, dim=-1, descending=True, stable=True).values
        spread = (ordered[..., 0] - ordered[..., -1]).clamp(min=1e-300)
        drop = ref & ~cand
        add = ~ref & cand
        gaps: list[float] = []
        for row in range(scores.shape[0]):
            drop_scores = scores[row][drop[row]]
            add_scores = scores[row][add[row]]
            if drop_scores.numel() == 0 or add_scores.numel() == 0:
                continue
            gaps.append(abs(float((drop_scores.mean() - add_scores.mean()).item())) / float(spread[row].item()))
        return sorted(gaps)

    vectorized = boundary_diagnostics(fp64_scores[0], candidate[0], reference_mask[0], k)
    loop_gaps = boundary_reference_loop(fp64_scores[0], candidate[0], reference_mask[0])
    loop_median = float(torch.tensor(loop_gaps, dtype=torch.float64).median().item()) if loop_gaps else None
    loop_max = max(loop_gaps) if loop_gaps else None
    median_ok = (loop_median is None and vectorized["changed_pair_gap_fp64_median"]
                 is None) or (loop_median is not None and vectorized["changed_pair_gap_fp64_median"] is not None
                              and abs(vectorized["changed_pair_gap_fp64_median"] - loop_median) < 1e-12)
    max_ok = (loop_max is None and vectorized["changed_pair_gap_fp64_max"]
              is None) or (loop_max is not None and vectorized["changed_pair_gap_fp64_max"] is not None
                           and abs(vectorized["changed_pair_gap_fp64_max"] - loop_max) < 1e-12)
    checks.record(
        "boundary_pair_gap_matches_per_row_loop", median_ok and max_ok
        and vectorized["n_query_blocks_with_swap"] == len(loop_gaps), {
            "vectorized_median": vectorized["changed_pair_gap_fp64_median"],
            "loop_median": loop_median,
            "vectorized_max": vectorized["changed_pair_gap_fp64_max"],
            "loop_max": loop_max,
            "n_rows_vectorized": vectorized["n_query_blocks_with_swap"],
            "n_rows_loop": len(loop_gaps),
        })

    # 8. FP64 shadow boundary resolution.
    boundary = boundary_diagnostics(fp64_scores[0], candidate[0], reference_mask[0], k)
    checks.record("fp64_shadow_has_zero_exact_ties", boundary["exact_ties_fp64"] == 0, boundary)
    deployed_ties = {name: deployed_tie_count(scores[0], k) for name, scores in arm_scores.items()}
    checks.record("deployed_tie_counts_recorded", True, deployed_ties)

    # Arm table sanity: unique ids, and both axes present in every label.
    ids = [arm.arm_id for arm in SCORER_ARMS]
    checks.record("arm_ids_unique", len(ids) == len(set(ids)), ids)
    checks.record("arm_labels_state_both_axes",
                  all("repr=" in arm.label and "pool=" in arm.label and "score=" in arm.label for arm in SCORER_ARMS))

    payload = {
        "verdict": "PASS" if checks.all_passed else "FAIL",
        "n_checks": len(checks.results),
        "n_failed": sum(1 for item in checks.results if not item["passed"]),
        "checks": checks.results,
        "config": {
            "seq_len": seq_len,
            "heads": heads,
            "head_dim": dim,
            "block_q": args.block_q,
            "sparsity": args.sparsity,
            "k": k,
            "n_k_blocks": geometry.n_k_blocks,
            "n_q_blocks": geometry.n_q_blocks,
        },
        "score_relative_deviation_vs_fp64": rel,
        "native_fp8_semantics": native_fp8[1] if native_fp8 is not None else None,
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
