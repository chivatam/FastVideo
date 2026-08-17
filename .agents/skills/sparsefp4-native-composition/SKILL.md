---
name: sparsefp4-native-composition
description: Course-correct the SparseFP4 project to directly test native NVFP4 x block-sparse attention for video diffusion using established evaluation patterns from FPSAttention, QuantSparse, VSA, and FP4-attention work. Primary proof is a native joint kernel plus standard quality and latency baselines; routing/Jaccard/random-mask analyses are optional diagnostics only.
---

# SparseFP4 Native Composition — Course-Corrected Research Skill

## Mission

Answer the actual research question:

> Can native NVFP4 attention compute and block-sparse video attention be composed on NVIDIA Blackwell to obtain compounded inference efficiency without unacceptable video-quality degradation?

This is **not primarily a routing-precision study**.

The central experimental pattern is:

1. dense BF16,
2. dense native NVFP4,
3. sparse BF16,
4. **native sparse NVFP4**,
5. standard attention/video quality metrics,
6. measured kernel, DiT-step, and end-to-end latency.

The task is incomplete until arm **D0: native sparse NVFP4** truly exists and is measured.

Read:
- `references/LITERATURE_ALIGNMENT.md`
- `references/EXPERIMENT_PROTOCOL.md`
- `assets/experiment_config.yaml`
- existing reports in `artifacts/sparsefp4/` and `artifacts/sparsefp4_followup/`

Old routing experiments are debugging/appendix material only.

## Non-negotiable native definition

A result may be called **native sparse NVFP4** only if:

1. retained Q/K tiles are represented in native Blackwell NVFP4 form,
2. retained QK products execute with native NVFP4-capable MMA,
3. unselected K/V tiles are actually skipped,
4. Q/K are not materialized back to BF16/FP16 before sparse QK matmul,
5. online softmax is correct across all retained tiles,
6. source + runtime/profiler evidence proves this path,
7. speed is wall-clock measured.

These do **not** count:
- NVFP4 quantize -> dequantize -> BF16 sparse attention,
- fake FP4 -> FP32/BF16 matmul,
- dense NVFP4 followed by masking,
- theoretical sparse FLOP reduction.

## Track 1 — controlled 2x2 operator composition

All four arms see identical captured Q/K/V.

C0 and D0 use the **same frozen sparse mask**.

| Arm | Sparse | QK compute | PV | Mask |
|---|---|---|---|---|
| A0 DENSE-BF16 | no | BF16 | BF16 | none |
| B0 DENSE-NVFP4 | no | native NVFP4 | BF16 | none |
| C0 SPARSE-BF16 | yes | BF16 | BF16 | frozen VSA mask |
| D0 SPARSE-NVFP4 | yes | **native NVFP4** | BF16 | **same as C0** |

Do not vary routing precision between C0 and D0.

Preferred mask:
- actual deployed VSA selector,
- generated once per captured cell,
- frozen/reused byte-for-byte.

## Track 2 — production FastVideo path

| Arm | System |
|---|---|
| P0 | dense BF16 FastVideo |
| P1 | dense native NVFP4 FastVideo |
| P2 | deployed VSA sparse BF16 |
| P3 | deployed VSA selector/coarse branch + **native sparse-NVFP4 fine branch** |

P2/P3 must share:
- selector,
- top-k,
- coarse branch,
- tile geometry,
- model/generation config.

Only fine sparse attention compute changes.

## Primary metrics

### Attention/operator
Against A0:
- MSE,
- relative L2,
- cosine similarity,
- SNR if straightforward.

Always show raw A0/B0/C0/D0 values.

### Video quality
Paired generation with identical prompt/seed/scheduler/steps/guidance/checkpoint/resolution/frames:
- VBench supported dimensions/aggregate,
- PSNR,
- SSIM,
- LPIPS.

Optional if already supported:
- VQA/text-video alignment,
- fixed qualitative grids.

Do not use pixel MAE as the sole quality metric.

### Performance
Measure:
1. attention-kernel latency,
2. DiT/transformer-step latency,
3. E2E generation latency,
4. peak memory,
5. throughput where meaningful.

Never infer speedup from FLOPs.

## Secondary diagnostics

Do **not** run these by default:
- Jaccard,
- top-k recall,
- FP64 scorer ladders,
- matched-random masks,
- damage-share ratios.

Only use them if D0 is unexpectedly bad and standard diagnostics cannot localize the failure.
If used, label them custom/exploratory and keep them out of the headline result.

# C0 — Code-path audit

Before editing:
```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Write `artifacts/sparsefp4_native/CODE_PATH_AUDIT.md` answering:

1. Where does dense native NVFP4 QK execute?
2. Which function owns packed FP4 Q/K and scales?
3. Where does VSA produce selected block indices/masks?
4. Which kernel computes VSA fine sparse attention?
5. Is there already a block-sparse FA4/CuTe Blackwell path?
6. Which dtypes does it support?
7. Smallest modification for retained sparse QK tiles to execute natively in NVFP4?
8. Does the path preserve online softmax across non-contiguous retained tiles?

Do not start large experiments before this map exists.

# C1 — Canonical baseline reproduction

Run/save receipts for:
- A0/P0 dense BF16,
- B0/P1 dense native NVFP4,
- C0 controlled sparse BF16 using frozen VSA mask,
- P2 deployed `VIDEO_SPARSE_ATTN`.

Gate:
- identical paired configs,
- finite outputs,
- dense NVFP4 native receipt,
- sparse BF16 realized-sparsity receipt.

# C2 — Implement native sparse NVFP4

Core target:

```text
selected sparse tiles
+ native NVFP4 QK MMA
+ correct online softmax
+ BF16 PV
```

Preferred implementation order:
1. extend existing FastVideo/fastvideo-kernel block-sparse Blackwell path if possible,
2. adapt existing dense FA4/CuTe NVFP4 kernel to consume sparse K-tile indices,
3. dedicated CuTeDSL/CUDA kernel only if necessary.

Do not reimplement everything if a minimal extension exists.

Conceptually:
```python
for q_tile in q_tiles:
    state = init_online_softmax()
    for k_tile in selected_k_tiles[q_tile]:
        qk = native_nvfp4_mma(q_tile, k_tile)
        state = online_softmax_update(state, qk, v_tile)
    out[q_tile] = finalize(state)
```

Never softmax each selected tile independently.

Start with:
```text
QK = native NVFP4
PV = BF16
```

Only add FP8 PV after correctness.

# C3 — Native-proof gate

Write `artifacts/sparsefp4_native/NATIVE_PROOF.md`.

Required:

### Source proof
Show exact functions for:
- packed NVFP4 Q/K,
- scale-factor use,
- sparse-index consumption,
- skipped unselected K tiles,
- low-precision QK MMA,
- online softmax.

### Runtime proof
Record:
- backend,
- kernel symbol,
- GPU arch,
- Q/K storage/packing,
- scale dtype,
- selected block count/sparsity,
- absence of BF16/FP16 Q/K materialization before QK MMA.

### Profiler proof
Use Nsight Compute/Systems, PyTorch profiler with kernel symbols, or an equivalent kernel receipt.

### Work-scaling sanity
Benchmark retained fractions:
```text
100%, 50%, 25%, 10%
```
Runtime need not scale linearly, but source/profiler must show skipped work.

If compute is still dense, D0 is invalid.

# C4 — Kernel correctness

Compare D0 to a trusted slow reference with:
- identical Q/K/V,
- identical mask,
- identical NVFP4 quantization/scales.

The reference may dequantize because it is only a correctness oracle.

Test multiple:
- layers,
- early/mid/late timesteps,
- heads,
- sparsities,
- sequence lengths.

Report:
- MSE,
- rel-L2,
- cosine,
- max abs,
- finiteness.

Set tolerances using the dense native NVFP4 kernel's deviation from its own dequantized reference.

# C5 — Controlled 2x2 experiment

Run A0/B0/C0/D0 on exactly paired tensors.

First model:
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

Primary:
- 480x832, 81 frames,
- deployed VSA sparsity.

Secondary if cheap:
- 720x1280,
- sparsities 0.80 / 0.90 / 0.95.

Required table:

| Arm | Sparse | QK | Native | MSE | rel-L2 | cosine | SNR |
|---|---|---|---|---|---|---|---|

Primary evidence = raw four-arm table.

Describe:
- quant-only error = B0 vs A0,
- sparse-only error = C0 vs A0,
- joint error = D0 vs A0.

Do not assume errors are additive and do not invent a headline ratio.

# C6 — Failure diagnosis only if joint composition fails

Trigger only if D0 is materially worse than B0/C0.

First:
- MSE/SNR by denoising timestep,
- error by sparsity,
- error by resolution,
- QK vs PV precision,
- tile/scaling granularity.

Only afterward use old custom routing diagnostics if still necessary.

Possible later pivots:
- step-aware precision/sparsity schedule (FPSAttention-style),
- QAT/fine-tuning (Attn-QAT/SLA2-style),
- tile/granularity co-design.

Do not choose a remedy before locating the failure.

# C7 — End-to-end video quality

Development:
- 10 prompts x fixed seed for debugging only.

Paper:
- follow the official/repo-integrated VBench-compatible protocol,
- same prompts/seeds/config across P0/P1/P2/P3.

Required table:

| System | VBench | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| P0 | | | | |
| P1 | | | | |
| P2 | | | | |
| P3 | | | | |

Do not silently drop unfavorable metrics.

For paired tests:
- pair by prompt/seed,
- declare comparison family,
- correct multiple comparisons if testing many dimensions.

# C8 — Performance

Kernel:
- warmup/JIT first,
- identical shape/mask,
- CUDA synchronize,
- multiple repetitions,
- median + dispersion.

Measure at 480p-class and 720p-class sequence lengths if supported.

DiT step:
- identical model/compile/graph settings.

E2E:
- P0/P1/P2/P3,
- repeated steady-state generations,
- report compile separately.

Required table:

| System | Attn ms | Attn speedup | DiT-step ms | E2E s | E2E speedup | Peak mem |
|---|---|---|---|---|---|---|

Never put simulated arms in this table.

# C9 — Established-style ablations

Only after the four-arm result exists:
- sparsity,
- resolution/sequence length,
- denoising timestep,
- QK BF16 vs native NVFP4,
- PV BF16 then optional FP8,
- tile geometry only if materially relevant.

# C10 — Decision

Write `artifacts/sparsefp4_native/RESULTS_DECISION.md`.

Choose:
- STRONG POSITIVE
- POSITIVE
- NEGATIVE BUT USEFUL
- SYSTEMS NO-GO
- INVALID / INCOMPLETE

**INVALID / INCOMPLETE** if no truly native D0/P3 exists.

# Paper framing

The main paper must answer:
1. Does the native joint path work numerically?
2. Does it preserve video quality?
3. Is it actually faster?
4. How does it compare with quant-only and sparse-only?
5. If it fails, what standard error analysis explains it?

Main figures:
1. system diagram of A/B/C/D,
2. quality-vs-E2E-latency Pareto,
3. timestep/sparsity error only if explanatory.

Main table:
**four-arm quality + performance matrix.**

That table is the paper.

# Artifact layout

```text
artifacts/sparsefp4_native/
├── STATUS.md
├── CODE_PATH_AUDIT.md
├── NATIVE_PROOF.md
├── RESULTS_DECISION.md
├── env/
├── configs/
├── raw/operator/
├── raw/quality/
├── raw/performance/
├── tables/
├── figures/
├── videos/
├── logs/
├── REPORT.md
└── PAPER_UPDATE.md
```

# Autonomous policy

Do not ask routine implementation questions.
Preserve old artifacts.
Stop only for hard blockers.

If native sparse NVFP4 is infeasible in the current stack, **do not substitute simulation**. Record the blocker and return INVALID/INCOMPLETE.

# Completion criterion

Complete only when:
1. A0/B0/C0/D0 exist,
2. D0 is proven native,
3. P0/P1/P2/P3 exist,
4. P3 is proven native,
5. standard operator metrics exist,
6. standard video-quality metrics exist,
7. native kernel/DiT/E2E timing exists,
8. four-arm matrix is complete,
9. custom routing diagnostics are not substitute evidence,
10. REPORT.md and PAPER_UPDATE.md map every claim to saved measurements.

**No native D0/P3 = no SparseFP4 composition claim.**
