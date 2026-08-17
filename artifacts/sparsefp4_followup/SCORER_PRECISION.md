# F1 — Scorer arithmetic precision

Run ID `20260816-215059-6e886a9c-f1-full`, commit `6e886a9c`, 1,555,200 records, all 11
validator checks PASS (`raw/f1_full_validation.json`).

## Why this phase exists

Study 1 varied the *representation* of Q/K entering the router while computing block
scores in fp64. No deployed kernel does that. So study 1's conclusion — that
block-sparse routing tolerates low-precision inputs — was silent on the axis a real
implementation would actually reduce: the arithmetic of the scorer itself. This phase
crosses the two axes so "representation is what breaks routing" and "arithmetic is what
breaks routing" become separable claims.

## Setup

`SCORER_PRECISION_ATTN` (`fastvideo/attention/backends/scorer_precision_attn.py`)
measures as a **side channel** on the model's real trajectory: the model consumes dense
BF16 attention, and every arm scores the same captured Q/K, so all arms are exactly
paired within a cell (same prompt, layer, timestep, CFG branch, head, sparsity).

- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a…`, 480x832, 81 frames, 50 steps.
- Lattice: 10 prompts x 30 layers x 6 timesteps (0, 1, 10, 25, 40, 48) x 2 CFG branches
  x 12 heads x 3 sparsities (0.80/0.90/0.95) x 12 arms = 1,555,200 records, zero holes.
- Geometry: 128x64 raster blocks, ragged final block averaged over its valid tokens only.
- Every arm carries an **fp64 shadow** re-scoring of its own pooled vectors, so
  score-matmul precision is separable from pooling precision. Zero fp64 ties in
  398,131,200 query blocks, so the scientific boundary reference is fully resolved.

## The 12 arms

A 2 (representation) x 6 (arithmetic) factorial. R0/R1 reproduce study 1's reference and
NVFP4 conditions, so this phase re-derives its own baseline rather than trusting a
remembered number.

| Arm | Q/K representation | Pool | Score matmul | Native? |
|---|---|---|---|---|
| R0 | exact BF16 | fp64 | fp64 | native (reference) |
| R1 | NVFP4 | fp64 | fp64 | native (study 1's condition) |
| R2 | exact BF16 | fp32 | fp32, TF32 disabled | native |
| R3 | NVFP4 | fp32 | fp32, TF32 disabled | native |
| R4 | exact BF16 | bf16 | bf16 values, library fp32 accumulation | native |
| R5 | NVFP4 | bf16 | bf16 values, library fp32 accumulation | native |
| R4L | exact BF16 | bf16, bf16 acc | sequential rank-1, bf16 accumulation | native |
| R5L | NVFP4 | bf16, bf16 acc | sequential rank-1, bf16 accumulation | native |
| R6 | exact BF16 | fp32 → FP8-E4M3 | `torch._scaled_mm`, fp32 accumulate | **native FP8** |
| R7 | NVFP4 | fp32 → FP8-E4M3 | `torch._scaled_mm`, fp32 accumulate | **native FP8** |
| R8 | exact BF16 | fp32 → NVFP4 | fp32 | **simulated** |
| R9 | NVFP4 | fp32 → NVFP4 | fp32 | **simulated** |

R4/R5 vs R4L/R5L is deliberate: `torch.matmul` on bf16 inputs accumulates in fp32 on
tensor cores, so "bf16" alone is ambiguous. The L arms force genuine bf16 accumulation,
which is the worst case a kernel without an fp32 accumulator would produce.

## Results at sparsity 0.90

| Arm | Ladder | Jaccard | Damage share | Isolation [95% CI] | vs R1 |
|---|---|---|---|---|---|
| R2 | fp32 | **1.000000** | **exactly 0** | n/a | 0 |
| R3 | fp32 | 0.97507 | 1.374e-03 | 20.9 [15.7, 28.3] | 1.00x |
| R4 | bf16 (fp32 acc) | 0.99073 | 7.049e-04 | 104 | 0.08x |
| R5 | bf16 (fp32 acc) | 0.97288 | 1.606e-03 | 23.5 | 0.97x |
| R4L | bf16 (bf16 acc) | 0.95678 | 1.939e-03 | 490 | 0.08x |
| R5L | bf16 (bf16 acc) | 0.94905 | 2.447e-03 | 45.9 | 1.03x |
| R6 | fp8 native | 0.95981 | 1.660e-03 | 67.0 [53.1, 85.7] | 0.55x |
| R7 | fp8 native | 0.95291 | 2.144e-03 | 25.0 [19.8, 31.6] | 1.67x |
| R8 | NVFP4-like | 0.88742 | 6.523e-03 | 14.3 [12.1, 16.8] | 7.79x |
| R9 | NVFP4-like | 0.88488 | 7.000e-03 | 12.2 [10.4, 14.5] | 9.35x |

"Damage share" is `median(|wrong_mask_excess|) / median(sparsification_error)`, computed
from paired within-cell differences. Bootstrap intervals resample prompts, 4000 draws.

**fp32 scorer arithmetic is exactly free.** R2's masks are bit-identical to R0's — not
approximately equal. An implementer can drop the scorer to fp32 with provably zero
routing change. This is the phase's most directly actionable result.

**Damage is ordered by precision and bounded.** It grows monotonically down the ladder,
but the worst arm reaches 0.70% of sparsification error: below the 1% revision
threshold, above the 0.1% strong-survival bound. Hence `PARTIAL_SUPPORT`, not strong
survival.

**Isolation holds everywhere.** 12.2x–490x, with every bootstrap interval entirely above
10x, so the effect remains a decision-boundary effect rather than generic mask damage.

**Study 1 reproduces** (`tables/f1_full/table6_r1_vs_study1_baseline.csv`): Jaccard
0.9812/0.9751/0.9646 vs frozen 0.9807/0.9738/0.9611; isolation 29.5/21.1/14.4 vs
27.0/21.7/10.1, at sparsities 0.80/0.90/0.95.

## Decision

`PARTIAL_SUPPORT` at all three sparsities. The mechanism survives the new axis: no arm
reaches the 1% revision threshold and isolation never fails, but the sub-0.1% bound
does not hold below fp32, so the paper cannot claim strong survival.

## The confound that invalidated the first pass

The first diagnostic reported R2 (fp32) and R4 (bf16, fp32 accumulate) as producing
**bit-identical** masks — which should be impossible if the arithmetic differs. Cause:
the denoising loop runs under `torch.autocast(bfloat16)`, which downcasts fp32 matmul
inputs, so the "fp32" arm was silently bf16 and the ladder had collapsed.

Fixed by `declared_precision_arithmetic()` (disables autocast for all side-channel
measurement) and `exact_fp32_matmul()` (also disables TF32). Self-test checks now gate
against both. Affected runs are quarantined as `DISCARDED-autocast-confound-*`. Lesson:
`.agents/lessons/autocast-downcasts-fp32-in-precision-ablations.md`.

TF32 is deliberately *not* mutated globally — that would move the model's own matmuls off
the trajectory study 1 measured. Each record carries the worker's ambient TF32 and
autocast state so the distinction is auditable rather than assumed.

## Artifacts

- Tables: `tables/f1_full/` (6 tables: headline, by region, by timestep, by CFG branch,
  by layer, and the study-1 comparison)
- Validation: `raw/f1_full_validation.json`
- Self-test: `raw/f1_selftest.json`; micro-profile: `raw/f1_profile.json`
- Figure: `figures/figureA_scorer_arithmetic.{png,pdf}`
- Cache for statistics: `raw/cache/f1_full.npz` (19 MB from 4.7 GB of JSONL)
