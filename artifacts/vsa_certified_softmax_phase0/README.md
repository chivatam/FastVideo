# Phase 0 — Certified Fixed Softmax Reference (Center-Radius Bound): NO-GO

**Hypothesis:** replace the sm_100a kernel's online-softmax running max /
alpha rescale / correction handshake with a precomputed certified upper
bound `U(q) = max_b [q·k_bar_b + ||q||·rho_b]/sqrt(D)` over the selected
VSA blocks, exponentiating `exp(score − U)` from the start.

**Verdict: NO-GO, on two independent grounds.**

## Workload

FastWan2.1-T2V-1.3B, bf16, B200; VSA (4,4,4)/0.9, head_dim 128. Real Q/K/V +
selected indices captured opt-in (`FASTVIDEO_VSA_CAPTURE_QK`) from real
inference: 3 prompts × 3 steps × 6 layers (0,3,14,15,27,29) at 720p + 1
prompt at 480p → 71 calls, **108,000 real query rows** (12 heads × 8 query
blocks × 16 rows each). Block semantics match `fused_block_mean` exactly
(valid-token means; padding excluded from rho; verified by tests).

## Result 1 — the bound holds but is too loose

Zero violations in 108K rows (min slack +0.51). Slack `delta = U − m_true`
(score units, production 1/sqrt(D) scale):

| P50 | P75 | P90 | P95 | **P99** | P99.9 | max |
|---|---|---|---|---|---|---|
| 13.4 | 20.3 | 29.6 | 37.4 | **235** | 327 | 371 |

The tail is catastrophic and layer-structured (late layers 27/29 reach
P99 = 144–314 vs 13.5 at layer 0 — a few heads have huge-norm outlier keys).
In the kernel's exp2 domain (arg = −delta·log2e): **2.56% of rows flush
their TOP-scoring token to exactly 0** under FTZ/bf16 (exp arg < −126),
3.7% fall below 2^−60. Measured fixed-U attention output error vs the
stable reference: **max-abs 7.9** (garbage rows from 0/0-like collapse) vs
online reference 3.7e-3 (bf16) — NO-GO criterion "severe exponent collapse"
met. Tighter box (per-dim min/max, 64x metadata) still leaves P99 ≈ 63:
also unusable (`results/tighter_bound_box.json`).

## Result 2 — the online softmax barely rescales anyway

Kernel-semantics simulation (128-token softmax streams, RESCALE_THRESHOLD=8,
same selected order) over all rows: median 72 tiles/row, median 5 raw max
updates, but **median 0 nontrivial rescales (mean 0.41, P99 = 3); 70% of
rows never trigger a single O rescale**. The kernel's existing threshold
trick already eliminated the correction work this idea targets — Phase 4's
correction-warp idling is handshake latency, not rescale compute. Even a
perfect U would remove almost nothing.

## Cost side (for completeness)

Bound QK is exactly 1/64 of fine QK; `k_bar` is already materialized by the
selector (`k_c`, bf16); rho would need one extra fused reduction, 69 KB per
call. Cheap — but pointless given Results 1–2.

## Files

`capture_or_extract.py` (Q/K/V capture runner), `certified_bound.py` (bound,
references, kernel-semantics rescale simulation), `analyze_bound_slack.py`,
`tests/test_bounds.py` (11 tests: synthetic Cauchy-Schwarz, valid-token
summaries, boundary blocks, zero-radius, adversarial q, translation
invariance, scaling consistency, loose-U underflow demo, online==exact,
rescale sanity), `results/`, `plots/`, `summary.json`.

Capture uses the Phase-1 opt-in instrumentation pattern (env-gated,
execution unchanged; bit-identity established in Phase 1).
