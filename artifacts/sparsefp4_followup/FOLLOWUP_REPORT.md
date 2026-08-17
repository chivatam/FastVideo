# Follow-up validation report

Commit `6e886a9c`, single 8xB200 host, CUDA 13.0. 4,017,600 paired measurement records
across four runs, every one validated with a complete lattice and zero holes.

## 1. Executive verdict

**VALIDATED WITH NARROWER SCOPE.**

The mechanism survives every test that could have falsified it, including the two the
original study could not run: the deployed VSA selector, and the scorer-arithmetic axis.
It survives across three seeds and a 2.3x token-count change. The scope narrows in three
specific ways, all of which make the paper harder to dismiss rather than weaker:

1. **The "high-precision routing rescues quality" direction is contradicted.** Routing
   VSA at fp64 is marginally *worse* than the shipped bf16 selector (+0.13% of
   sparsification error at sparsity 0.90; better in only 9.9% of cells). Any remedial
   framing must be removed. The correct claim is stronger and more interesting: routing
   precision is not the binding constraint.
2. **Damage exceeds the 0.1% "survives strongly" bound** for every arm below fp32, so the
   verdict is partial support rather than strong survival. No arm reaches the 1% revision
   threshold.
3. **One decision threshold is genuinely unresolved.** One F2 arm's isolation ratio sits
   at 9.04 / 9.97 / 10.49 against a hard 10x criterion, and its bootstrap interval spans
   10x at all three sparsities. Reported as indeterminate rather than decided.

Not "CURRENT CLAIM CONTRADICTED": the central negative-result mechanism — that
low-precision routing perturbs selection at a cost small relative to the sparsification
error already accepted, and that the perturbation is a decision-boundary effect rather
than generic mask damage — holds on the real selector, at every sparsity, on every seed,
at both token counts. Only the remedial corollary is contradicted.

## 2. What was already known

Study 1 (`artifacts/sparsefp4/REPORT.md`, frozen into `baseline_snapshot.json` before any
follow-up run) established, on a controlled mean-pooled proxy scorer with Wan2.1-1.3B at
480x832:

| Quantity (sparsity 0.90) | Value | n |
|---|---|---|
| Mask Jaccard, NVFP4 routing | 0.973756 | 72,000 |
| Wrong-mask excess (D−C) | 3.326e-05 | 20,400 |
| Sparsification error (C) | 0.176219 | 20,400 |
| Random-mask excess | 7.205e-04 | 20,400 |
| Random / quantization ratio | 21.66 | 20,400 |
| Wrong-mask share of total error | 1.564e-04 | 20,400 |

Three limitations were explicit: a single seed, a single configuration, and a proxy
scorer rather than a deployed selector. A fourth was implicit and more serious — block
scores were computed in **fp64**, an arithmetic precision no kernel uses, so the study
had measured only the representation axis.

## 3. Scorer arithmetic

Full detail: `SCORER_PRECISION.md`. Run `20260816-215059-6e886a9c-f1-full`, 1,555,200
records, 11/11 checks PASS.

**Setup.** A 2 (Q/K representation: exact BF16, NVFP4) x 6 (scorer arithmetic: fp64,
fp32, bf16 with fp32 accumulation, bf16 with bf16 accumulation, FP8-E4M3 native dot,
NVFP4-like) factorial, measured as a side channel on one shared dense-BF16 trajectory so
all 12 arms are exactly paired within each cell. 10 prompts x 30 layers x 6 timesteps x
2 CFG branches x 12 heads x 3 sparsities.

**Exact arms.** R0/R1 reproduce study 1's reference and NVFP4 conditions. R2/R3 are fp32
with TF32 disabled. R4/R5 use `torch.matmul` on bf16 (which accumulates in fp32 on tensor
cores); R4L/R5L force genuine bf16 accumulation, the worst case for a kernel without an
fp32 accumulator. R6/R7 use a **native** `torch._scaled_mm` FP8 dot. R8/R9 are simulated
NVFP4 — no native NVFP4 block-dot exists — and are labelled as such in every record.

**Results (sparsity 0.90).** fp32 scorer arithmetic is **exactly free**: R2's masks are
bit-identical to fp64's, damage exactly zero. Damage then grows monotonically down the
ladder — bf16/fp32-acc 7.05e-04, bf16/bf16-acc 1.94e-03, native FP8 1.66e-03, NVFP4-like
6.52e-03 (exact Q/K) — with the worst arm at 7.00e-03 of sparsification error. Matched-
random isolation holds at 12.2x–490x with every bootstrap interval entirely above 10x.
Study 1 reproduces independently at all three sparsities (Jaccard 0.9812/0.9751/0.9646 vs
frozen 0.9807/0.9738/0.9611).

**Decision.** `PARTIAL_SUPPORT` at all three sparsities. No arm reaches the 1% revision
threshold; no arm below fp32 meets the 0.1% strong-survival bound.

A confound was found and fixed mid-phase: `torch.autocast(bfloat16)` in the denoising loop
silently downcast the fp32 arms, which had produced a spurious "fp32 and bf16 masks are
bit-identical" result. Guarded, self-tested, affected runs quarantined, lesson written.

## 4. Actual VSA selector

Full detail: `VSA_GATE.md`, implementation map in `VSA_GATE_MAP.md`. Run
`20260816-215059-6e886a9c-f2-full`, 1,166,400 records, 17/17 checks PASS.

**Implementation map.** `gate_compress` is **not** the selector — it is a multiplicative
weight on the compression-branch output. The real selector is `fused_block_mean` (bf16 in,
fp32 accumulate, bf16 out) → bf16 matmul with fp32 accumulation → `fused_topk_mask`
(32-iteration fp32 bisection threshold, ties toward the lower key-block index), over VSA's
4x4x4 cube tiles of 64 tokens.

**Intervention.** The probe subclasses `VideoSparseAttentionImpl`, so the model follows a
genuine `VIDEO_SPARSE_ATTN` trajectory and the measurement uses VSA's own kernels. Nine
arms: routing-representation quantization (FP8, NVFP4), selector-arithmetic changes (fp32,
fp64, genuine bf16 accumulation), their combination, a `gate_compress` invariant test, and
a tie-break contrast. Higher-precision arms use a stable-sort selection rule proven
bit-identical to the kernel on bf16 scores, because pushing fp64 scores through a bf16
kernel would have silently nullified the intervention.

**Results (sparsity 0.90).** NVFP4 routing: Jaccard 0.9687, damage 1.118e-03 of VSA's own
sparsification error, isolation 225x [124, 707]. FP8 routing: 0.9869, 4.530e-04, 310x.
Degraded selector arithmetic: 0.9622, 8.752e-04, 47.7x. The `gate_compress` invariant
holds in **129,600 / 129,600 records** — quantizing the gate cannot move the mask, exactly
as the map predicts. And the fp64 rescue is negative at every sparsity.

**Decision.** `INDETERMINATE_ISOLATION_THRESHOLD` at 0.80 and 0.90, `PARTIAL_SUPPORT` at
0.95 — the difference between those verdicts is one arm's ratio crossing 10x inside its own
confidence interval, so the aggregator declines to assert a side.

**Incidental correctness finding.** `fused_topk_mask` returns `topk + 1` blocks when the
k-th and (k+1)-th scores tie exactly: the bisection approaches the k-th value from below
and never lands on it, so the tie-fill branch is skipped. ~1 row in 7,488 at affected
cells, scale-invariant, and present on the **shipped** path. Reproduced deterministically;
recorded per row rather than treated as an error, since VSA's selector is the measurand.

## 5. Robustness and generalization

Full detail: `GENERALIZATION.md`.

**Seeds.** Run `20260816-223631-6e886a9c-f3a-seeds`, 1,036,800 new records. Seeds 1234,
2026, 3407, pre-declared in `configs/f3a_seeds.json` before execution; the analyzer refuses
any seed not on that list. Seed is the unit of replication and the verdict is a conjunction
over seeds. Worst-arm damage share 7.000e-03 / 6.991e-03 / 6.959e-03; minimum isolation
12.18 / 11.88 / 11.62; identical direction signature for all three. Decision:
**`SEED_ROBUST`** — all four criteria hold per seed, with three-significant-figure
agreement.

**Additional configuration.** Run `20260816-225059-6e886a9c-f3b-720p`, 259,200 records.
Priority-1 choice: the same model at 720x1280, 32,760 → 75,600 tokens (**2.3077x**), key
blocks 512 → 1,182, compared against the same 5 prompts in the baseline. Damage share
*decreases* at every arm (ratios 0.56–0.82) and Jaccard slightly improves. Decision:
**`GENERALIZES_ON_TOKEN_COUNT`**, labelled **proxy generalization** because it uses the
controlled scorer, not VSA. At this block count `torch._scaled_mm` has no native kernel, so
the FP8 arms' dot is emulated — recorded in the data, not silently substituted.

## 6. Numerical and statistical gates

`raw/f4_gates.json`: 16 checks, **0 hard failures**.

**Lattice (F4.1).** All four runs report observed = expected with zero holes: 1,555,200 /
1,166,400 / 1,036,800 / 259,200. Because a record exists only when the probe's `forward`
actually executes, a complete lattice is positive proof that every DiT self-attention layer
resolved onto the probe and none silently fell back.

**Pairing (F4.2).** Zero duplicate cells. The cell key is
(arm, prompt, seed, layer, timestep, CFG branch, head, sparsity) — `prompt_id` and `seed`
both belong in it, since each (prompt, seed) pair is an independent trajectory whose
reference mask legitimately differs. An earlier key omitted them and produced false
duplicate/inconsistent-reference failures; fixed, and both runs re-validated.

**Nulls (F4.3).** Reference-vs-itself Jaccard exactly 1.0 and excess exactly 0.0 in all
129,600 reference records per phase, with matching mask hashes. The matched-random control
changes exactly the same number of blocks as the arm it is paired with, in every record.

**Simulation fidelity (F4.5).** Native NVFP4 vs simulated, on 30 Q/K tensors captured from
the real model at layers 0/1/15/28/29 and timesteps 0/25/48: relative disagreement median
9.80e-03, p90 1.07e-02, max 1.10e-02; **99.95%** of elements bit-identical; saturation
fractions agree to 0.02%. After block pooling — the form the scorer actually consumes —
disagreement falls to median 1.39e-03, max 2.84e-03. Simulated arms are barred from any
latency claim.

**Scorer-resolution diagnostics (F4.4).** The fp64 shadow reference produced **zero** exact
ties across 398,131,200 (F1) and 727,833,600 (F2) query blocks, so the scientific boundary
reference is fully resolved and decision margins are meaningful.

**Uncertainty.** Isolation ratios are ratios of medians with no closed form, so intervals
are 95% percentile bootstraps (4000 resamples, seed 20260816) resampling **prompts** —
cells within a prompt share a trajectory and are not independent. One shared implementation
(`configs/sparsefp4_stats.py`) serves both the gates and the figures, so a figure cannot
disagree with the gate that validated it. Every damage-share interval falls entirely below
1%. Three isolation intervals straddle 10x and are reported as unresolved.

## 7. Paper impact

Full protocol with exact edits: `PAPER_UPDATE.md`. Claim ledger:
`tables/f5/tableB_claim_before_after.md`. Boundary matrix:
`tables/f5/tableA_claim_boundary_matrix.md`.

**Claim before.** Low-precision routing perturbs block-sparse attention selection; the
damage is small relative to sparsification error; a higher-precision router recovers it.
Established on one seed, one configuration, a proxy scorer, and an fp64 scorer.

**Claim after.** Low-precision routing perturbs selection on the *deployed* VSA selector
too, at a cost of 0.11% of the sparsification error the method already accepts, with
matched-random perturbations 225x more damaging. The scorer's own arithmetic can be reduced
to fp32 with *bit-identical* masks and to NVFP4-like precision for under 0.7% of
sparsification error. Stable across three seeds and 2.3x token count. **But a
higher-precision router does not recover anything** — at VSA's operating point it is
marginally worse than the shipped bf16 selector — so routing precision is not the binding
constraint.

**Recommended sentence for the abstract/conclusion:**

> On Wan2.1's deployed Video Sparse Attention selector, reducing routing precision to
> NVFP4 changes the selected block set by 3.1% (Jaccard 0.969) yet accounts for only 0.11%
> of the sparsification error the method already accepts — while an equal-count random
> mask perturbation is 225x more damaging — and restoring the selector to fp64 does not
> reduce output error but slightly increases it; routing precision is therefore not the
> binding constraint on block-sparse video attention quality, and the scorer's arithmetic
> can be reduced to fp32 with bit-identical routing decisions.

## 8. Remaining untested question

> Does native sparse tile skipping compose efficiently with native NVFP4 attention on
> Blackwell?

This follow-up measured no latency and makes no speed claim. Answering it requires a fused
kernel that skips tiles *and* computes NVFP4 attention on the surviving tiles, which does
not exist in this codebase — FlashAttention-4's NVFP4 path is dense, and VSA's sparse path
is BF16. The NVFP4 scorer arms here are simulated, so they cannot stand in for a timing
measurement even indirectly.

Secondary questions this study leaves open:

- The one unresolved isolation threshold (needs more prompts, not more cells).
- VSA generalization across token counts (F3B used the proxy scorer).
- End-to-end perceptual video quality — all damage figures are attention-output relative-L2
  against a dense reference. The escalation rule did not trigger a video sweep: the internal
  effect is bounded and shrinks with token count.
- A second model family.

No kernel project was started.

## Appendix — artifact index

| Artifact | Path |
|---|---|
| Study-1 baseline, frozen | `baseline_snapshot.json` |
| Environment capture | `env.json` |
| F1 phase report | `SCORER_PRECISION.md` |
| F2 implementation map | `VSA_GATE_MAP.md` |
| F2 phase report | `VSA_GATE.md` |
| F3 phase report | `GENERALIZATION.md` |
| Paper update protocol | `PAPER_UPDATE.md` |
| This report | `FOLLOWUP_REPORT.md` |
| Phase status | `STATUS.md` |
| Validations | `raw/f1_full_validation.json`, `raw/f2_full_validation.json`, `raw/f3a_validation.json`, `raw/f3b_validation.json` |
| Statistical gates | `raw/f4_gates.json`, `raw/f4_representation_fidelity.json` |
| Tables | `tables/f1_full/`, `tables/f2_full/`, `tables/f3a/`, `tables/f3b/`, `tables/f5/` |
| Figures | `figures/figureA_scorer_arithmetic.*`, `figureB_vsa_selector.*`, `figureC_generalization.*` |
| Compact caches | `raw/cache/*.npz` |
| Raw records (preserved) | `/mnt/nvme/scratch/sparsefp4_followup/<run-id>/*.jsonl`, 13 GB across 4 runs |
| Lessons | `.agents/lessons/autocast-downcasts-fp32-in-precision-ablations.md`, `.agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md`, `.agents/lessons/vsa-block-mean-assumes-zero-padding.md` |
