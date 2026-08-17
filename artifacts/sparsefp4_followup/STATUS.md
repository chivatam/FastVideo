# Follow-up validation study — status

**COMPLETE.** All phases F0–F7 executed. Verdict: **VALIDATED WITH NARROWER SCOPE**
(`FOLLOWUP_REPORT.md`).

Commit `6e886a9c`, one 8xB200 host, CUDA 13.0. 4,017,600 paired records, every run
validated with a complete lattice and zero holes. Baseline for comparison is study 1,
frozen in `baseline_snapshot.json` before any follow-up run.

| Phase | Run ID | Records | Gates | Verdict |
|---|---|---|---|---|
| F1 scorer arithmetic | `20260816-215059-6e886a9c-f1-full` | 1,555,200 | 11/11 | `PARTIAL_SUPPORT` (all sparsities) |
| F2 real VSA selector | `20260816-215059-6e886a9c-f2-full` | 1,166,400 | 17/17 | `INDETERMINATE` @0.8/0.9, `PARTIAL_SUPPORT` @0.95 |
| F3A seeds 2026/3407 | `20260816-223631-6e886a9c-f3a-seeds` | 1,036,800 | 11/11 | `SEED_ROBUST` |
| F3B 720p (2.31x tokens) | `20260816-225059-6e886a9c-f3b-720p` | 259,200 | 11/11 | `GENERALIZES_ON_TOKEN_COUNT` |
| F4 statistical gates | — | — | 16 checks, 0 hard failures | `PASS` |

## Headline findings

1. **fp32 scorer arithmetic is exactly free** — bit-identical masks to fp64, so an
   implementer can drop the scorer to fp32 with provably zero routing change.
2. **The result transfers to the deployed selector.** On a genuine `VIDEO_SPARSE_ATTN`
   trajectory, NVFP4 routing costs 0.11% of VSA's own sparsification error with
   matched-random 225x more damaging.
3. **The high-precision rescue is negative.** Routing at fp64 is *worse* than the shipped
   bf16 selector (+0.13% at sparsity 0.90, better in only 9.9% of cells). Routing
   precision is not the binding constraint — this contradicts any remedial framing and is
   the single most consequential result.
4. **`gate_compress` is provably outside the selection path** — 129,600/129,600 masks
   bit-identical under genuine NVFP4 quantization of the gate.
5. **Stable** across three pre-declared seeds (three significant figures) and 2.31x token
   count (damage *decreases*).
6. **One threshold is unresolved**: an F2 arm's isolation ratio (9.04/9.97/10.49) straddles
   the 10x criterion in its bootstrap interval at all three sparsities, so it is reported
   as indeterminate rather than decided in either direction.

## Deliverables

| Required by skill | Path | Status |
|---|---|---|
| `baseline_snapshot.json` | `baseline_snapshot.json` | done |
| `SCORER_PRECISION.md` | `SCORER_PRECISION.md` | done |
| `VSA_GATE_MAP.md` | `VSA_GATE_MAP.md` | done |
| `VSA_GATE.md` | `VSA_GATE.md` | done |
| `GENERALIZATION.md` | `GENERALIZATION.md` | done |
| `FOLLOWUP_REPORT.md` | `FOLLOWUP_REPORT.md` | done |
| `PAPER_UPDATE.md` | `PAPER_UPDATE.md` | done |

Plus figures A/B/C (`figures/`), Tables A/B (`tables/f5/`), per-phase tables
(`tables/f{1_full,2_full,3a,3b}/`), validations and gates (`raw/`), and compact
statistics caches (`raw/cache/`).

## Two bugs found, root-caused and documented

**Autocast confound (ours).** The denoising loop runs under `torch.autocast(bfloat16)`,
which silently downcast the fp32 scorer arms — producing a spurious "fp32 and bf16 masks
are bit-identical" result in the first diagnostic. Fixed with explicit
`declared_precision_arithmetic()` / `exact_fp32_matmul()` guards plus self-test checks;
affected runs quarantined as `DISCARDED-autocast-confound-*`. Lesson:
`.agents/lessons/autocast-downcasts-fp32-in-precision-ablations.md`.

**Upstream VSA top-k bug (theirs).** `fused_topk_mask` returns `topk + 1` blocks when the
k-th and (k+1)-th block scores tie exactly: its 32-iteration fp32 bisection converges
toward the k-th value from below and never lands on it, so the tie-fill branch is skipped.
~1 row in 7,488 at affected cells, scale-invariant, and present on the **shipped** `V0`
path. Reproduced deterministically by `configs/f2_kernel_topk_bug.py`. Recorded per row
rather than treated as an error, because VSA's selector is F2's measurand. Lesson:
`.agents/lessons/vsa-fused-topk-mask-can-overselect-on-ties.md`.

## Reproducing

```bash
source artifacts/sparsefp4_followup/configs/env.sh

# One phase, sharded over GPUs (SEEDS defaults to 1234)
artifacts/sparsefp4_followup/configs/launch_sweep.sh f1 <run-id> "0 1 2 3" 10 \
    --sparsities 0.80 0.90 0.95
SEEDS="2026 3407" artifacts/sparsefp4_followup/configs/launch_sweep.sh f1 <run-id> "0 1" 10 \
    --sparsities 0.90

# Validate, then aggregate — never aggregate an unvalidated run
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_validate.py --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
    --out artifacts/sparsefp4_followup/raw/f1_full_validation.json
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/f1_aggregate.py --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
    --tables artifacts/sparsefp4_followup/tables/f1_full --stage F1-full

# Statistics run off compact caches, not the raw JSONL (32 MB vs 8.2 GB)
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/build_stats_cache.py --shard "$FV_RAW_ROOT/<run-id>"/*.jsonl \
    --out artifacts/sparsefp4_followup/raw/cache/f1_full.npz
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/f4_gates.py \
    --f1-cache artifacts/sparsefp4_followup/raw/cache/f1_full.npz \
    --f2-cache artifacts/sparsefp4_followup/raw/cache/f2_full.npz \
    --out artifacts/sparsefp4_followup/raw/f4_gates.json
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/f5_figures.py
"$FV_PYTHON" artifacts/sparsefp4_followup/configs/f5_tables.py
```

## Discarded and debug runs

| Directory | Why |
|---|---|
| `DISCARDED-autocast-confound-f1-diag` | fp32 arm silently autocast to bf16 |
| `DISCARDED-autocast-confound-f1-full` | same confound, 10 shards |
| `debug-f2-*`, `smoke-*` | targeted reproductions and smoke tests, not study data |
