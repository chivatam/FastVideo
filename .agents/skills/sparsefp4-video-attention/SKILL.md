---
name: sparsefp4-video-attention
description: Run and report a go/no-go research study on dynamic block-sparse attention combined with NVFP4 attention for video diffusion/DiT models, especially Wan2.1 on NVIDIA Blackwell using FastVideo, VSA/SpargeAttention, and FlashAttention-4. Use when asked to reproduce, benchmark, analyze, or extend the SparseFP4 experiment, measure BF16-vs-NVFP4 sparse-mask stability, test precision-decoupled routing, or prepare evidence for a short workshop paper.
---

# SparseFP4 Video Attention Experiment Skill

## Mission

Act as an autonomous research engineer. Execute a **small, falsifiable study** answering:

> Does NVFP4 perturb dynamic sparse-attention routing in Video DiTs, and can a higher-precision low-cost router recover quality while the expensive attention compute remains low precision?

The intended output is evidence suitable for a **4-page workshop paper**, not a production library.

Primary baseline:
- Framework: FastVideo
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Hardware target: NVIDIA B200/B300
- Dense low-precision attention: FastVideo / FlashAttention-4 NVFP4 path when available
- Sparse execution: FastVideo VSA or SpargeAttention-style block sparse attention
- Default resolution: 480x832
- Default frames: 81
- Preserve the framework/model's normal inference defaults unless the experiment config overrides them.

Read `references/EXPERIMENT_SPEC.md` before editing code.
Use `assets/experiment_config.yaml` as the default experiment matrix.
Use `references/REPORT_TEMPLATE.md` for the final report.

## Research hypotheses

Test, do not assume, the following:

- **H1 — Routing instability:** NVFP4 Q/K changes the top block selection relative to BF16, especially at high sparsity.
- **H2 — Localized sensitivity:** routing instability is concentrated in particular heads, layers, and/or diffusion timesteps rather than uniform.
- **H3 — Precision-decoupled routing:** BF16 or FP8 routing with low-precision attention compute reduces sparse-attention error versus NVFP4 routing at the same sparsity.
- **H4 — Systems composability:** if a native sparse-NVFP4 kernel is implemented, sparsity provides measurable wall-clock benefit beyond dense NVFP4 on Blackwell.

H4 is a stretch goal. H1–H3 are sufficient for a useful short empirical paper if the effect is strong and reproducible.

## Scientific integrity rules

These are mandatory.

1. **Never call simulated/fake-quantized sparse attention a native NVFP4 kernel.**
2. **Never report a theoretical FLOP reduction as measured latency speedup.**
3. **Never claim end-to-end sparse-NVFP4 speedup unless the sparse tile skipping happens inside the native low-precision attention kernel and has been benchmarked.**
4. Clearly label each result as one of:
   - native BF16
   - native FP8
   - native NVFP4
   - fake/simulated FP8
   - fake/simulated NVFP4
5. Keep BF16 dense output as the primary numerical reference.
6. Use identical prompts, seeds, model checkpoint, scheduler, steps, dimensions, and compilation settings for pairwise comparisons.
7. Save raw metrics before aggregating them.
8. Record code revision, environment, GPU, dependency versions, and exact commands.
9. Do not silently discard failed runs or outliers. Record exclusions and reasons.
10. Do not optimize the method after looking at the test set without reporting that the set was used for development.

## Autonomous execution policy

Do not ask the user for routine implementation choices. Inspect the repository and use the closest current API.

Only stop for user input if blocked by something the agent cannot fix, such as:
- no compatible NVIDIA GPU and no acceptable simulation path,
- model access/authentication failure,
- disk quota preventing even the minimal run,
- unrecoverable build/toolchain failure after reasonable attempts.

Otherwise:
- make minimal changes,
- run the smallest diagnostic first,
- update `artifacts/sparsefp4/STATUS.md`,
- continue based on the go/no-go rules below.

Do **not** spend substantial time writing a fused kernel before Phase 2 establishes that the research effect exists.

## Phase 0 — Repository and environment preflight

### 0.1 Locate the implementation

If already inside a FastVideo checkout, use it.

Otherwise, if network access is available, obtain the current FastVideo source using the repository's documented installation path. Do not assume stale file paths from this skill; search the repository for:
- `nvfp4_fa4`
- attention backend registries
- `VIDEO_SPARSE_ATTN`
- VSA
- FlashAttention-4 / FA4
- Q/K/V projection and attention call sites

When external repositories are needed, prefer pinned commits once the experiment begins.

### 0.2 Protect the repository

Before edits:
- run `git status --short`
- do not overwrite unrelated user changes
- create a dedicated branch if safe, e.g. `exp/sparsefp4-mask-stability`
- record `git rev-parse HEAD`

### 0.3 Capture the environment

Run:

```bash
python scripts/check_env.py --output artifacts/sparsefp4/env.json
```

Also save:
- `nvidia-smi -q`
- `nvcc --version`
- `pip freeze`
- FastVideo commit
- optional sparse-attention dependency commits
- model identifier/checkpoint revision

### 0.4 Smoke-test dense baselines

Run one tiny or normal short inference for:
1. dense BF16
2. dense NVFP4 FA4 if supported on the machine

Do not proceed to large sweeps until both paths produce usable outputs. If native NVFP4 is unavailable, continue H1–H3 with clearly labeled fake quantization and mark H4 unavailable.

## Phase 1 — Instrument Q/K and measure sparse-mask stability

### Goal

Determine whether the same attention blocks are selected when routing is computed from BF16 Q/K versus low-precision Q/K.

### 1.1 Instrument at the correct point

Capture or analyze the **same Q and K tensors that are passed into the attention backend**, after all model-specific preprocessing required for the actual attention computation, such as:
- Q/K normalization,
- rotary/position embedding transforms,
- layout/transposes.

Do not compare an earlier BF16 tensor to a later transformed low-precision tensor.

Avoid writing full Q/K activations for all layers/steps unless storage is small. Prefer online metric computation.

### 1.2 Block scorer

Implement a research-only block scorer independent of the expensive full attention matrix.

Default block geometry:
- query block: 128 tokens
- key block: 64 tokens

Also support 64x64 and 128x128 for ablation.

For each head, pool each Q and K block and compute a coarse score. Default:

```python
q_block = q.float().mean(dim=token_axis)
k_block = k.float().mean(dim=token_axis)
score = q_block @ k_block.transpose(-1, -2)
```

Use the actual candidate sparse method's scoring rule when integrating VSA/SpargeAttention, but preserve this simple scorer as a controlled diagnostic.

### 1.3 Precision variants for routing

Evaluate at least:
- BF16 routing
- FP8 routing if available
- NVFP4 routing

Prefer the framework's real quantizer/dequantizer or quantization helper.

If native quantized tensors cannot be inspected before the kernel, implement deterministic fake quantization for **routing diagnostics only**. Label those results simulated.

### 1.4 Sparsity sweep

Default sparsities:

```text
50%, 70%, 80%, 90%, 95%
```

Equivalent retained fractions:

```text
50%, 30%, 20%, 10%, 5%
```

At each head/layer/timestep compare the selected block sets.

Required metrics:
- top-k recall against BF16 mask
- Jaccard overlap
- optional score rank correlation
- top-k decision margin: score[k] - score[k+1]
- fraction of routing decisions changed

For equal-sized top-k masks, precision equals recall; do not present them as independent evidence.

Use `scripts/analyze_masks.py` where convenient.

### 1.5 Required outputs

Save raw records in JSONL/Parquet/CSV with at least:

```text
prompt_id
seed
layer
head
timestep
block_q
block_k
sparsity
routing_precision
reference_precision
intersection
union
selected_reference
selected_candidate
recall
jaccard
decision_margin_reference
decision_margin_candidate
```

Generate:

1. **Mask Jaccard vs sparsity** for BF16↔FP8 and BF16↔NVFP4.
2. **Layer × timestep heatmap** of BF16↔NVFP4 mask Jaccard at 80% and 90% sparsity.
3. Optional head-level distribution/boxplot.

## Phase 2 — Decompose numerical error

### Goal

Separate:
- quantization error,
- sparsification error,
- wrong-mask error caused by quantized routing.

For representative prompts/layers/timesteps compute:

```text
A. dense BF16
B. dense low precision
C. sparse BF16 compute + BF16-derived mask
D. sparse BF16 compute + NVFP4-derived mask
E. sparse low-precision compute + NVFP4-derived mask
F. sparse low-precision compute + FP8/BF16-derived mask
```

If E/F cannot use a native sparse low-precision kernel, simulate them and mark them **numerical-only; no native latency claim**.

Required output metrics against dense BF16:
- relative L2 error
- cosine similarity
- max absolute error if numerically stable
- optional downstream hidden-state error after the residual update

Use the exact same retained fraction across C–F.

### Main test of H3

At the same sparse compute budget compare:

```text
NVFP4 router -> sparse low-precision attention
vs
FP8 router   -> sparse low-precision attention
vs
BF16 router  -> sparse low-precision attention
```

The main method is justified only if higher-precision routing produces a meaningful and repeatable recovery.

## Go / no-go checkpoint

After Phases 1–2, write `artifacts/sparsefp4/GO_NO_GO.md`.

### GO: continue to full video evaluation if either pattern is present

A. Routing instability is material:
- substantial layer/head/timestep regions have visibly lower overlap at high sparsity, e.g. Jaccard < 0.8 at 90% sparsity, **or**
- the aggregate effect is smaller but strongly structured and reproducible.

B. Precision-decoupled routing matters:
- FP8/BF16 routing reduces relative attention-output error by roughly 20% or more versus NVFP4 routing in affected regions, **or**
- provides a clear downstream video-quality recovery at equal sparsity.

These are heuristics, not statistical thresholds. Report the actual distributions.

### PIVOT / NO-GO

If BF16 and NVFP4 masks are nearly identical almost everywhere (e.g. Jaccard > 0.95 at 80–90% sparsity) **and** higher-precision routing does not improve output error, do not force the hypothesis.

Pivot the paper question to one of:
1. whether sparse tile skipping composes with native NVFP4 in wall-clock performance,
2. how block size and Blackwell tile geometry affect sparse-NVFP4 efficiency,
3. a negative result: NVFP4 routing is unexpectedly stable, allowing fully low-precision sparse routing.

A strong negative result is preferable to manufacturing a method.

## Phase 3 — Sparse execution integration

Do this only after the checkpoint.

### 3.1 First use an existing sparse backend

Prefer the minimum-intrusion route:
- FastVideo VSA / `VIDEO_SPARSE_ATTN`, or
- SpargeAttention / another backend that accepts a user-provided block mask.

Add a research backend or wrapper such as:

```text
PRECISION_SPARSE_ATTN
```

Conceptually:

```python
def precision_sparse_attention(q, k, v, sparsity, route_precision):
    route_q, route_k = quantize_for_router(q, k, route_precision)
    block_scores = compute_block_scores(route_q, route_k)
    sparse_mask = select_top_blocks(block_scores, sparsity=sparsity)
    return sparse_attention(q, k, v, sparse_mask)
```

Keep the router precision separate from the attention compute precision.

### 3.2 Do not conflate router precision and compute precision

Model the experiment using two independent axes:

```text
router_precision in {bf16, fp8, nvfp4}
attention_precision in {bf16, fp8, nvfp4}
```

A higher-precision router is allowed to be very cheap relative to the attention kernel.

### 3.3 Main end-to-end configurations

Run at least:

| ID | Sparsity | Attention compute | Router |
|---|---|---|---|
| DENSE-BF16 | none | BF16 | n/a |
| DENSE-FP4 | none | NVFP4 | n/a |
| SPARSE-BF16 | selected | BF16 | BF16 |
| SPARSE-FP4-NAIVE | selected | NVFP4 | NVFP4 |
| SPARSE-FP4-ROUTE8 | selected | NVFP4 | FP8 |
| SPARSE-FP4-ROUTE16 | selected | NVFP4 | BF16 |

If the sparse FP4 compute is simulated, keep `SPARSE-FP4-*` out of native latency tables until Phase 4.

## Phase 4 — Optional native sparse-NVFP4 FlashAttention-4 kernel

This is a stretch goal and must not block the paper.

### Objective

Modify the existing Blackwell FP4 FlashAttention-4 path so that masked K/V tiles are skipped while preserving correct online softmax across all retained tiles.

### Kernel constraint

Conceptually transform:

```python
for q_tile in q_tiles:
    for k_tile in all_k_tiles:
        process_fp4_tile(q_tile, k_tile)
```

into:

```python
for q_tile in q_tiles:
    for k_tile in selected_k_tiles[q_tile]:
        process_fp4_tile(q_tile, k_tile)
```

Do **not** compute softmax independently per selected block. Preserve the running max and normalization state across retained blocks.

Start numerical development in this order:
1. NVFP4 QK + BF16 PV
2. NVFP4 QK + FP8 PV
3. only then consider fully low-precision PV if supported and useful

### Correctness gate before benchmarking

Compare native sparse kernel output against a trusted reference using:
- identical Q/K/V,
- identical sparse mask,
- multiple sequence lengths,
- multiple sparsities,
- multiple heads.

Do not report speed until numerical error is acceptable and no masked-out block contributes to the result.

## Phase 5 — End-to-end video evaluation

### Development set

Start small:
- 10 prompts
- 1 seed for activation study
- optionally 3 seeds for confirming the effect

Use `assets/prompts.txt`.

### Paper-scale set

If results justify it:
- 50–100 prompts from a recognized video benchmark / VBench-compatible set
- fixed seed for primary paired comparison
- optionally 3 seeds on a smaller subset

Do not claim benchmark-wide superiority from the 10-prompt development set.

### Required quality metrics

At minimum use available VBench dimensions relevant to temporal generation, plus an aggregate score when the benchmark supports it.

Also retain:
- sample videos for qualitative inspection,
- paired grids or contact sheets,
- failure examples.

### Performance metrics

Measure separately:
1. attention-kernel latency
2. transformer/DiT step latency
3. end-to-end generation latency
4. peak GPU memory

Benchmark protocol:
- warm up compilation/caches first
- use multiple measured repetitions
- synchronize CUDA around microbenchmarks
- report median and dispersion
- keep batch/shape identical
- record whether CUDA graphs/compile modes are on
- avoid comparing first-run compile time against warmed execution

## Default experiment matrix

Use `assets/experiment_config.yaml`.

Do not run the full Cartesian product immediately.

Order:
1. one prompt × one seed × all layers/timesteps, 80% and 90% sparsity
2. 10 prompts × one seed, sparsity sweep
3. error decomposition on representative sensitive and insensitive regions
4. only then end-to-end video runs
5. only then native kernel work

## Artifact layout

Write all results under:

```text
artifacts/sparsefp4/
```

Recommended structure:

```text
artifacts/sparsefp4/
├── STATUS.md
├── GO_NO_GO.md
├── env.json
├── env/
├── configs/
├── raw/
├── tables/
├── figures/
├── videos/
├── logs/
└── REPORT.md
```

Never overwrite a completed run. Use timestamped/run-ID subdirectories where needed.

## STATUS.md behavior

Update `STATUS.md` after every major phase with:
- what ran,
- what passed/failed,
- exact next action,
- strongest current result,
- current blocker,
- whether H1/H2/H3/H4 is supported, unsupported, or untested.

Keep it concise and factual.

## Final reporting

The final deliverable is `artifacts/sparsefp4/REPORT.md` following `references/REPORT_TEMPLATE.md`.

The report must contain:

1. Executive conclusion in 3–6 bullets.
2. Hardware/software/model setup.
3. Exact code changes by file.
4. Exact commands used.
5. Hypotheses table: supported / unsupported / inconclusive.
6. Mask-stability results.
7. Numerical-error decomposition.
8. End-to-end quality results if run.
9. Native performance results only if truly measured.
10. Failed approaches and debugging notes.
11. Limitations/confounders.
12. 4-page-paper viability decision.
13. The smallest defensible paper claim.
14. Recommended next experiment.
15. Paths to raw data, plots, and sample videos.

Include paper-ready figures if possible:
- Figure 1: mask overlap vs sparsity
- Figure 2: layer × timestep routing-sensitivity heatmap
- Figure 3: quality/latency Pareto if native performance exists
- Table 1: main comparison across dense/sparse and routing precision

## Paper viability rubric

At the end, assign one:

### STRONG GO
Use when:
- routing sensitivity is reproducible,
- precision-decoupled routing materially helps,
- video quality confirms the numerical result,
- and/or a native sparse-NVFP4 path gives clear measured speed benefit.

### GO
Use when:
- H1/H2/H3 provide a clean empirical story even without a fused kernel,
- enough evidence exists for a focused 4-page workshop submission.

### BORDERLINE
Use when:
- effect exists but is small, inconsistent, or only visible in internal attention metrics.

### NO-GO / PIVOT
Use when:
- low-precision routing is already stable,
- quality does not recover with higher-precision routing,
- or confounders prevent a defensible conclusion.

If NO-GO, explicitly propose the best pivot rather than continuing expensive experiments.

## Expected claim boundaries

Potential claim if supported:

> Dynamic sparse routing in FP4 Video DiTs is more precision-sensitive than the expensive attention computation; retaining FP8/BF16 precision only in the low-cost routing stage recovers sparse-attention accuracy while preserving low-precision attention compute.

Do not claim:
- first sparse quantized video attention,
- first low-bit sparse attention,
- first NVFP4 video attention,
- native sparse-NVFP4 speedup without a measured native kernel.

## Completion criterion

The task is complete when:
- Phase 0 passes,
- Phases 1–2 are fully analyzed,
- GO/NO-GO is written,
- all justified follow-on experiments have been run,
- `REPORT.md` is complete,
- plots/tables/raw results are linked from the report,
- and every major claim can be traced to a saved measurement.
