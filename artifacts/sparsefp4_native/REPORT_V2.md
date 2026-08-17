# REPORT_V2 — SparseFP4 Native Composition (post re-audit)

All numbers trace to receipts under `raw/`, `tables/`, `logs/`; environment
in `env/`. Model Wan2.1-T2V-1.3B, VSA sparsity 0.90, seed 1234, 50 steps,
8x B200 (sm_100), CUDA 13.0. Supersedes REPORT.md where they differ.

## 1. Exact research question

Can native NVFP4 attention compute and block-sparse video attention be
composed on Blackwell to obtain real speedup without unacceptable quality
loss?

Answer in one line: they compose *numerically* without interaction penalty
and the native kernel is real and fast — but at the production operating
point FP4's incremental speed over sparse BF16 is small, integration costs
invert it end-to-end, and at paper scale NVFP4 carries a measurable
no-reference quality cost; the larger systems win is geometry alignment,
which is precision-independent.

## 2. Native SparseFP4 implementation

`NATIVE_PROOF.md` (unchanged by re-audit): packed `float4_e2m1fn_x2` Q/K +
per-16 E4M3 SFs, `flash_fwd_sm100_fp4` block-scaled MMA over
retained-tile-only load/MMA/softmax loops, BF16 PV, no BF16 Q/K
materialization; profiler + work-scaling receipts. Enabled by repairing 7
latent version-skew bugs in the flash-attention-fp4 fork
(`configs/fa4-fork-sparse-fp4-repair.patch`); no kernel math changed.
Envelope limit: fully-empty Q rows deadlock multi-wave grids (unreachable
under VSA topk>=1).

## 3. Canonical exact-10% A0/B0/C0/D0 table

`tables/c5_matrix_vsa256_exact10.md` — VSA256/FA4-aligned selector, 1:1 mask
mapping (median keep 0.1006, zero coarsening), 25 cells per resolution,
byte-identical C0/D0 masks, rel-L2 medians vs A0:

| Resolution | quant-only B0 | sparse-only C0_256 | joint D0_256 | conditional D0 vs C0 |
|---|---|---|---|---|
| 480x832x81 | 0.0951 | 0.1918 | 0.2885 | **0.0957** |
| 720x1280x81 | 0.1018 | 0.2175 | 0.3005 | **0.0918** |

Conditioned on the same mask, NVFP4 on retained QK tiles adds the same
perturbation magnitude as dense NVFP4 — at both resolutions, no additivity
assumed, raw values reported. (The earlier 24%-coarsened table is demoted to
appendix context.)

## 4. Incremental FP4 speed vs sparse BF16 across retention

Kernel-only, Wan 480p shape, plain sparse lists (`c3_native_proof.json`):

| retained | BF16 ms | FP4 ms | FP4 increment |
|---|---|---|---|
| dense kernel | 7.414 | 6.017 | 1.26x |
| 100% | 7.591 | 6.010 | 1.26x |
| 50% | 3.815 | 3.074 | 1.24x |
| 25% | 1.814 | 1.654 | 1.10x |
| 10% | 0.830 | 0.800 | **1.04x** |

At 720p geometry/10% with plain lists: 3.98 vs 3.16 ms (1.26x). **The 9.3x
headline vs dense BF16 is a sparsity speedup.** FP4's increment shrinks as
QK MMA stops dominating sparse-kernel time — the central regime-change
observation.

## 5. P4 performance root cause and optimization

`P4_PERF_ROOT_CAUSE.md`. The 250.9 s (0.59x) 720p E2E was: (a) dominant —
CUDA caching-allocator thrash from ~200 MB/call transient FP4 buffers
(fixed: `expandable_segments`, 250.9 -> 111.7 s; P0/P4G controls unchanged);
(b) vbs mask_mod validity predicate: +42% on the softmax-bound FP4 kernel,
free on BF16 (~4 s/video); (c) quantization ~0.5-1 s/video. After (a), the
residual P4-P4G gap (5.7 s) is fully attributed to (b)+(c). Native P4 does
**not** beat its BF16 twin E2E at the target shape; it does at kernel level
without the validity predicate. Untried paths: tile-aligned padding to
eliminate the predicate; preallocated quantize workspace; FP8/NVFP4 PV
(blocked in this fork build: `MmaF8F6F4Op` under dsl 4.5.3).

## 6. Paper-scale P0/P1/P2/P4G/P4 quality

`tables/paper_scale_quality.md` — 326 official VBench prompts (all prompts
of the 7 repo-scorable dimensions), paired seed/config, dimension-routed
scoring, prompt-level bootstrap:

Key rows (VBench means):

| Dim | P0 | P1 | P2 | P4G | P4 |
|---|---|---|---|---|---|
| subject_consistency | 0.931 | 0.940 | 0.869 | 0.869 | 0.863 |
| imaging_quality | 0.650 | 0.506 | 0.583 | 0.585 | 0.484 |
| dynamic_degree | 0.764 | 0.458 | 0.819 | 0.958 | 0.708 |
| aesthetic_quality | 0.597 | 0.542 | 0.352 | 0.322 | 0.316 |

**P4 - P4G paired (load-bearing):** significant negatives on
imaging_quality (-0.101, CI [-0.112, -0.090]) and dynamic_degree (-0.250,
CI [-0.361, -0.153]); significant small positives on temporal_flickering
(+0.009) and motion_smoothness (+0.016); subject/background/aesthetic ~n.s.
The same signature appears in dense NVFP4 (P1 vs P0), so the penalty is
NVFP4's own, consistent with §3's no-amplification result. **"NVFP4 quality
is the same" is not supported**; sparsity-family costs (subject/aesthetic
drops vs dense) are shared by P2/P4G/P4 alike. Geometry note: P4G matches
deployed VSA (P2) throughout — the 256-tile selector is quality-neutral.

## 7. Kernel / DiT / E2E performance

`tables/c8_performance.md` + `P4_PERF_ROOT_CAUSE.md`, all arms under
identical allocator config at 720p:

| System | 480p E2E s (x) | 720p E2E s (x) |
|---|---|---|
| P0 dense BF16 | 46.9 (1.00) | 148.7 (1.00) |
| P1 dense NVFP4 | 44.4 (1.06) | 135.8 (1.10) |
| P2 deployed VSA | 50.0 (0.94) | 131.7 (1.13) |
| P4G VSA256-FA4 BF16 (10%) | 45.6 (1.03) | **106.0 (1.40)** |
| P4 VSA256-FA4 NVFP4 (10%) | 47.3 (0.99) | 111.7 (1.33) |

(P2G/P3, the coarsened-geometry arms, demoted: 48.7/53.1 s at 480p.)
Peak memory flat across arms. DiT-step numbers in the table file. No
FLOP-derived figures anywhere.

## 8. Positive-vs-negative systems verdict

**Direction B** (`RESULTS_DECISION_V2.md`): native SparseFP4 is correct and
kernel-positive, but arithmetic acceleration and sparsification are not
multiplicative — at 90% sparsity FP4's MMA advantage (~1.04-1.26x) is
smaller than the integration costs it introduces (validity predicate in the
softmax-bound pipeline, quantization, allocator pressure), and NVFP4 carries
its own no-reference quality cost. The positive, defensible systems result
is **geometry-aligned sparse attention**: matching selector tiles to the
kernel's sparse granularity is worth 1.40x E2E at 720p in BF16 at
deployed-baseline quality.

## 9. Limitations

Single model (Wan2.1-1.3B), single sparsity operating point (0.90), one
seed for paper-scale quality (paired design mitigates but does not replace
the official 5-sample protocol); 7 of 16 VBench dimensions scorable in this
environment (GRiT/ViCLIP dims need uninstalled deps); FP8/NVFP4 PV untested
(fork build limitation); QAT recovery evidence is feasibility-scale on a
motion-poor 47-video dataset (dynamic_degree collapsed) — supplementary
only; the mask_mod-free FP4 configuration (tile-aligned padding) is designed
but not implemented; fork repairs are a patch on a moving research codebase.

## 10. Exact defensible paper claims

1. Native block-sparse NVFP4 attention (packed E2M1 QK, per-16 E4M3 SF,
   retained-tile-only execution, BF16 PV) exists and is proven on B200,
   with kernel-level wins over sparse BF16 at matched masks (up to 1.26x)
   and exactness at the FP4 arithmetic floor vs a dequantized oracle.
2. Conditioned on identical masks at the exact deployment geometry, NVFP4
   on retained tiles perturbs attention outputs by the same magnitude as
   dense NVFP4 (rel-L2 0.092-0.096 vs 0.095-0.102) — no composition
   amplification, both resolutions.
3. FP4's incremental kernel speedup decays from 1.26x (dense) to ~1.04x at
   90% sparsity; sparse scheduling/predicate/quantization overheads can
   invert it end-to-end (111.7 vs 106.0 s) — sparsity and low-precision
   arithmetic do not automatically compound without sparse-specific kernel
   co-design.
4. Selector-geometry alignment to the kernel's sparse granularity is the
   dominant lever: 1.40x E2E vs dense at 720p (BF16), 1.24x vs deployed
   VSA, at statistically indistinguishable quality from the deployed sparse
   baseline.
5. At paper scale, NVFP4 QK reduces no-reference imaging quality and motion
   dynamism by similar margins in dense and sparse settings (paired CIs
   exclude zero); temporal-stability metrics slightly improve; QAT-style
   recovery is indicated (feasibility evidence supplementary).
