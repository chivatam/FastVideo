# SparseFP4 Video Attention — REPORT.md Template

Copy this file to `artifacts/sparsefp4/REPORT.md` and fill it in. Keep the
section order — it is the order `SKILL.md` requires.

**Rules for filling this in:**

1. Every `<!-- FILL: ... -->` marker must be replaced or explicitly answered with
   `not run` / `not measured` / `unknown`. Do not delete a marker to avoid a
   question.
2. **Every number must be traceable to a raw-data path.** A number without a path
   into `artifacts/sparsefp4/raw/<run_id>/` (or a table/figure CSV derived from
   it) is not allowed in this report — delete it or go measure it. Tables below
   include a `source` column for exactly this reason.
3. Every result row states `native` or `simulated`. Simulated arms are written as
   "fake/simulated NVFP4" / "fake/simulated FP8" and never appear in a latency
   table.
4. Every aggregate carries `n=`.
5. Theoretical FLOP reduction is never presented as measured speedup.

---

## 1. Executive conclusion

3–6 bullets. Lead with the answer, not the method.

<!-- FILL: bullet 1 — did NVFP4 perturb routing? at what sparsity, how much, n= -->
<!-- FILL: bullet 2 — was the instability localized (heads/layers/timesteps)? -->
<!-- FILL: bullet 3 — did higher-precision routing recover error? by how much, paired n= -->
<!-- FILL: bullet 4 — was anything native, or is this all numerical/simulated? -->
<!-- FILL: bullet 5 — paper verdict (STRONG GO / GO / BORDERLINE / NO-GO-PIVOT) -->
<!-- FILL: bullet 6 (optional) — the single most surprising observation -->

## 2. Setup

| Item | Value | Source |
|---|---|---|
| GPU(s) | <!-- FILL --> | `env/nvidia-smi.txt` |
| Driver / CUDA / nvcc | <!-- FILL --> | `env/nvcc.txt`, `env.json` |
| Torch version | <!-- FILL --> | `env.json` |
| FastVideo commit | <!-- FILL: 40-char sha --> | `env.json` |
| Sparse dep commits (VSA / SpargeAttention / FA4) | <!-- FILL or "n/a" --> | `env.json` |
| Model id | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `configs/` |
| Model revision (pinned) | <!-- FILL --> | `configs/` |
| Scheduler / steps / guidance | <!-- FILL: actual resolved values, not "default" --> | `configs/` |
| Resolution / frames | 480x832 / 81 | `configs/` |
| `seq_len` / layers / heads / head_dim | <!-- FILL --> | `configs/` |
| Attention backend(s) exercised | <!-- FILL --> | `configs/` |
| Compile / CUDA graphs | <!-- FILL: on/off per arm --> | `configs/` |
| Determinism flags | <!-- FILL: TF32, cudnn, deterministic algorithms --> | `env.json` |
| Prompt set | `assets/prompts.txt` (10-prompt **development** set) | — |
| Seeds | <!-- FILL --> | `configs/` |
| Native NVFP4 available? | <!-- FILL: yes/no + evidence --> | `logs/` |

<!-- FILL: one paragraph on anything unusual about the environment -->

## 3. Code changes by file

| File | Change | Why | Kept or reverted? |
|---|---|---|---|
| <!-- FILL --> | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |

<!-- FILL: note any research-only code that must not be shipped as a feature -->

## 4. Exact commands

Copy-pasteable, in execution order, one block per stage.

```bash
# FILL: env capture
# FILL: stage 1 (1 prompt x 1 seed, sparsity 0.80/0.90)
# FILL: stage 2 (10 prompts, sparsity sweep)
# FILL: stage 3 (error decomposition A-F)
# FILL: analysis
#   python .agents/skills/sparsefp4-video-attention/scripts/analyze_masks.py \
#     --raw artifacts/sparsefp4/raw --out-tables artifacts/sparsefp4/tables \
#     --out-figures artifacts/sparsefp4/figures
# FILL: stage 4 (end-to-end video), stage 5 (native kernel) if run
```

## 5. Hypotheses

| ID | Hypothesis | Verdict | Key evidence (number + n) | Raw path |
|---|---|---|---|---|
| H1 | NVFP4 Q/K changes top-block selection vs BF16 | supported / unsupported / inconclusive / untested | <!-- FILL --> | <!-- FILL --> |
| H2 | Instability is localized by head/layer/timestep | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| H3 | Higher-precision routing reduces sparse-attention error | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| H4 | Native sparse-NVFP4 gives wall-clock benefit | <!-- FILL: usually "untested" --> | <!-- FILL --> | <!-- FILL --> |

## 6. Mask-stability results (Phase 1)

Definitions used: `sparsity`, `retained_fraction`, `k`, tie-breaking, force
retention — as in `references/EXPERIMENT_SPEC.md` §1. Note that precision ==
recall for equal-sized top-k masks, so recall and Jaccard are **one**
measurement, not two.

### 6.1 Mask overlap vs sparsity

<!-- FILL: reference figures/fig1_mask_overlap_vs_sparsity.png -->

**Figure 1 — mask overlap vs sparsity**, one line per routing precision, with
dispersion bands (median, IQR).
Values: `figures/fig1_mask_overlap_vs_sparsity.csv`

| sparsity | retained | routing_precision | native/simulated | jaccard median | jaccard IQR | recall median | n |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> | <!-- FILL --> |
| 0.70 | 0.30 | | | | | | |
| 0.80 | 0.20 | | | | | | |
| 0.90 | 0.10 | | | | | | |
| 0.95 | 0.05 | | | | | | |

Source: `tables/agg_by_sparsity_precision.csv`

### 6.2 Localization (H2)

<!-- FILL: reference figures/fig2_layer_timestep_jaccard_s0.80.png and _s0.90.png -->

**Figure 2 — layer x timestep heatmap** of BF16↔NVFP4 mask Jaccard, one panel per
sparsity (default 0.80 and 0.90).
Values: `figures/fig2_layer_timestep_jaccard_s0.80.csv`, `..._s0.90.csv`

| Region | sparsity | jaccard median | IQR | n | affected? (median < 0.90) |
|---|---|---|---|---|---|
| most affected `(layer, timestep)` | <!-- FILL --> | | | | |
| least affected `(layer, timestep)` | <!-- FILL --> | | | | |

Head-level distribution (optional Figure): `figures/fig3_head_jaccard_box.png`,
values `figures/fig3_head_jaccard_box.csv`

<!-- FILL: is the structure reproducible across prompts/seeds? cite n -->

### 6.3 Decision margins and mechanism

<!-- FILL: relationship between reference decision margin and fraction of
     decisions changed; boundary tie counts; saturation fractions -->

### 6.4 Null control

<!-- FILL: BF16-vs-BF16 self-comparison gave jaccard == 1.0 for n= ... records -->

## 7. Numerical-error decomposition (Phase 2)

All errors are against **A (dense BF16)**, the sole numerical reference. C–F use
an identical retained fraction. Differences between rows are *attributions*, not
an exact additive decomposition.

| ID | Sparse? | Attention compute | Mask source | native/simulated | rel_L2 median | rel_L2 IQR | cosine median | max_abs | n | Raw path |
|---|---|---|---|---|---|---|---|---|---|---|
| A | no | BF16 | n/a | native | 0 (reference) | — | 1 | 0 | — | <!-- FILL --> |
| B | no | <!-- FILL --> | n/a | <!-- FILL --> | | | | | | |
| C | yes | BF16 | BF16 | <!-- FILL --> | | | | | | |
| D | yes | BF16 | NVFP4 | <!-- FILL --> | | | | | | |
| E | yes | NVFP4 | NVFP4 | <!-- FILL --> | | | | | | |
| F | yes | NVFP4 | FP8/BF16 | <!-- FILL --> | | | | | | |

Attributions:

- quantization error ≈ `err(B)` = <!-- FILL -->
- sparsification error ≈ `err(C)` = <!-- FILL -->
- wrong-mask error ≈ `err(D) - err(C)` = <!-- FILL -->
- router-recoverable error ≈ `err(E) - err(F)` = <!-- FILL -->

Post-residual hidden-state error (optional): <!-- FILL or "not measured" -->

### 7.1 H3 test at equal budget

| Router precision | Attention compute | sparsity | rel_L2 p10 / p50 / p90 | reduction vs NVFP4 router | frac cells improved | n | native/simulated |
|---|---|---|---|---|---|---|---|
| NVFP4 | NVFP4 | <!-- FILL --> | | — | — | | |
| FP8 | NVFP4 | | | <!-- FILL: % --> | <!-- FILL --> | | |
| BF16 | NVFP4 | | | <!-- FILL: % --> | <!-- FILL --> | | |

Pre-registered support threshold: **≥ 20% relative reduction in median `rel_l2`**
in affected regions, with the full paired distribution reported.

- Affected regions: <!-- FILL: reduction %, n -->
- Unaffected regions (control): <!-- FILL: reduction %, n -->
- Distribution figure/CSV: <!-- FILL -->
- Verdict: <!-- FILL: supported / borderline / unsupported -->

## 8. End-to-end quality results

<!-- FILL or state "not run" -->

| ID | Sparsity | Attention compute | Router | native/simulated | VBench dims | Aggregate | Sample videos | Raw path |
|---|---|---|---|---|---|---|---|---|
| DENSE-BF16 | none | BF16 | n/a | | | | | |
| DENSE-FP4 | none | NVFP4 | n/a | | | | | |
| SPARSE-BF16 | <!-- FILL --> | BF16 | BF16 | | | | | |
| SPARSE-FP4-NAIVE | | NVFP4 | NVFP4 | | | | | |
| SPARSE-FP4-ROUTE8 | | NVFP4 | FP8 | | | | | |
| SPARSE-FP4-ROUTE16 | | NVFP4 | BF16 | | | | | |

Prompt set: 10-prompt **development** set (`assets/prompts.txt`) —
<!-- FILL: state explicitly that no benchmark-wide superiority is claimed -->

<!-- FILL: qualitative notes, contact sheets, failure examples with paths -->

## 9. Native performance results

Only rows with a **measured native kernel** may appear here. If nothing native
was benchmarked, write "no native sparse-NVFP4 kernel was built; no latency
claim" and leave the table empty.

| ID | native? | attention-kernel latency (median, IQR) | DiT step latency | e2e latency | peak mem | warmup iters | reps | compile/CUDA graphs | Raw path |
|---|---|---|---|---|---|---|---|---|---|
| <!-- FILL --> | | | | | | | | | |

<!-- FILL: confirm no simulated arm appears above; confirm no FLOP-derived number
     is presented as a speedup -->

**Table 1 — main comparison across dense/sparse and routing precision** (numerical
columns always; latency columns only for native arms):
<!-- FILL: path to tables/table1_main_comparison.csv -->

**Figure 3 — quality/latency Pareto**, only if native performance exists:
<!-- FILL or "not applicable: no native sparse-NVFP4 measurement" -->

## 10. Failed approaches and debugging notes

<!-- FILL: what broke, what was abandoned and why, dead ends worth documenting -->

| Attempt | Outcome | Why abandoned | Evidence |
|---|---|---|---|
| <!-- FILL --> | | | |

Excluded runs / outliers (never silently dropped):

| run_id / filter | count excluded | reason |
|---|---|---|
| <!-- FILL --> | | |

## 11. Limitations and confounders

Address each of these explicitly; "not applicable" is an acceptable answer only
with a reason.

- Simulated vs native quantization, and what it means for the conclusions: <!-- FILL -->
- Single prompt / single seed stages: <!-- FILL -->
- Diagnostic scorer vs the real sparse method's scoring rule: <!-- FILL -->
- Token ordering / patchification effects: <!-- FILL -->
- Ragged final block: <!-- FILL -->
- Prompt-length / `seq_len` variation: <!-- FILL -->
- Timestep sampling and scheduler coupling: <!-- FILL -->
- Head-dim scaling and where the attention scale is applied vs quantization: <!-- FILL -->
- Force-retained diagonal/local blocks diluting overlap metrics: <!-- FILL -->
- Development-set usage (this set was used for development, per `SKILL.md` §10): <!-- FILL -->

## 12. 4-page-paper viability decision

Assign exactly one. Criteria inlined so no re-reading is needed.

- **STRONG GO** — routing sensitivity is reproducible; precision-decoupled
  routing materially helps; video quality confirms the numerical result; and/or a
  native sparse-NVFP4 path gives a clear measured speed benefit.
- **GO** — H1/H2/H3 form a clean empirical story even without a fused kernel;
  enough evidence for a focused 4-page workshop submission.
- **BORDERLINE** — the effect exists but is small, inconsistent, or visible only
  in internal attention metrics.
- **NO-GO / PIVOT** — low-precision routing is already stable; quality does not
  recover with higher-precision routing; or confounders prevent a defensible
  conclusion.

Supporting go/no-go heuristics (not statistical thresholds):

- GO pattern A: substantial layer/head/timestep regions with clearly lower
  overlap at high sparsity (e.g. Jaccard < 0.8 at 90% sparsity), **or** a smaller
  but strongly structured and reproducible aggregate effect.
- GO pattern B: FP8/BF16 routing reduces relative attention-output error by
  roughly ≥ 20% versus NVFP4 routing in affected regions, **or** gives a clear
  downstream video-quality recovery at equal sparsity.
- PIVOT if BF16 and NVFP4 masks are nearly identical almost everywhere (e.g.
  Jaccard > 0.95 at 80–90% sparsity) **and** higher-precision routing does not
  improve output error.

**Verdict:** <!-- FILL: STRONG GO | GO | BORDERLINE | NO-GO / PIVOT -->

**Justification (cite numbers + n):** <!-- FILL -->

**If NO-GO / PIVOT, the chosen pivot:** <!-- FILL: one of (1) does sparse tile
skipping compose with native NVFP4 in wall-clock, (2) block size / Blackwell tile
geometry effects on sparse-NVFP4 efficiency, (3) the negative result that NVFP4
routing is unexpectedly stable, enabling fully low-precision sparse routing -->

## 13. Smallest defensible paper claim

One sentence, scoped to what was actually measured (model, sparsity range,
native vs simulated, development set).

<!-- FILL -->

### Expected claim boundaries

The one potential claim, if the evidence supports it:

> Dynamic sparse routing in FP4 Video DiTs is more precision-sensitive than the
> expensive attention computation; retaining FP8/BF16 precision only in the
> low-cost routing stage recovers sparse-attention accuracy while preserving
> low-precision attention compute.

Forbidden claims — do not write any of these, in any phrasing:

1. first sparse quantized video attention
2. first low-bit sparse attention
3. first NVFP4 video attention
4. native sparse-NVFP4 speedup without a measured native kernel

<!-- FILL: confirm none of the four appear anywhere in this report -->

## 14. Recommended next experiment

The single highest-information next run, with its cost.

<!-- FILL: what, why it discriminates between hypotheses, estimated GPU-hours -->

## 15. Paths to raw data, plots, and sample videos

| Artifact | Path |
|---|---|
| Status log | `artifacts/sparsefp4/STATUS.md` |
| Go/no-go | `artifacts/sparsefp4/GO_NO_GO.md` |
| Environment | `artifacts/sparsefp4/env.json`, `artifacts/sparsefp4/env/` |
| Resolved configs | `artifacts/sparsefp4/configs/` |
| Raw records (per run) | `artifacts/sparsefp4/raw/<run_id>/*.jsonl` |
| Codebase map | `artifacts/sparsefp4/CODEBASE_MAP.md` |
| Tables | `artifacts/sparsefp4/tables/` |
| Figures (+ per-figure value CSVs) | `artifacts/sparsefp4/figures/` |
| Videos | `artifacts/sparsefp4/videos/` |
| Logs | `artifacts/sparsefp4/logs/` |
| Analysis summary | `artifacts/sparsefp4/tables/summary.md` |

<!-- FILL: list the run_ids that back this report, in order -->

### Final self-check before publishing this report

- [ ] Every number has a raw-data path.
- [ ] Every aggregate has `n=`.
- [ ] Every arm is labeled native or fake/simulated.
- [ ] No simulated arm appears in a latency table.
- [ ] No FLOP reduction is described as a speedup.
- [ ] Recall and Jaccard are not presented as independent evidence.
- [ ] All exclusions are recorded with reasons.
- [ ] Development-set usage is disclosed.
- [ ] No forbidden claim appears.
- [ ] Every `<!-- FILL: ... -->` marker is resolved or explicitly answered.
