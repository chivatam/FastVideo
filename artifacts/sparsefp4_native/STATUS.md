# SparseFP4 Native Composition — STATUS

_Last updated: 2026-08-17 15:10 ET — V3 (paper-validation pass) COMPLETE.
V2 notes below are historical; V3 receipts supersede where they differ._

## V3 pass (this session) — all deliverables landed

1. **Canonical performance V2** — `tables/c8_performance_v2.md`: all 5 arms
   x both resolutions rerun in fresh processes under one allocator config
   (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); receipts
   `raw/performance/perf_v2/`, logs `logs/perf_v2/`. P4 720p = 112.6 s
   (1.33x); P4G = 106.2 s (1.40x). The stale `tables/c8_performance.md`
   (P4 720p 250.9 s pre-fix) is historical/root-cause evidence only.
2. **Unified statistics** (`configs/paired_stats_v2.py`, 10k prompt-level
   bootstrap, Holm across 7 dims; JSONs in `raw/statistics/`):
   - P1-P0 (`tables/p1_vs_p0_quality_bootstrap.md`): imaging -0.144,
     dynamic -0.306, aesthetic -0.055 (sig); background +0.005 (sig).
   - P4-P4G (`tables/p4_vs_p4g_quality_bootstrap.md`): imaging -0.101,
     dynamic -0.250 (sig); temporal +0.009, motion +0.016 (sig).
   - Interaction (P4-P4G)-(P1-P0)
     (`tables/nvfp4_sparsity_interaction.md`): imaging +0.043 and
     aesthetic +0.049 POSITIVE (penalty smaller under sparsity), subject
     -0.014 negative, dynamic no detectable interaction at this n.
   - P4G-P2 (`tables/p4g_vs_p2_quality_bootstrap.md`): NOT
     indistinguishable — aesthetic -0.030, motion -0.018, background
     -0.008, dynamic +0.139 (all sig). "Comparable with trade-offs".
3. **Literature audit** — `SOTA_RECOVERY_LIT_REVIEW.md` (12 papers,
   primary-source-only; priority-claim verdict: "native block-sparse NVFP4
   attention" unclaimed in the verified record, guard wording required).
4. **DQ-VSA recovery design + smoke** — `TRAINING_RECOVERY_PLAN.md`;
   new training backend `SPARSEFP4_QAT_VSA256_ATTN`
   (`fastvideo/attention/backends/sparsefp4_qat_vsa256.py`) and Stage-2
   velocity-distillation pipeline
   (`fastvideo/training/wan_dqvsa_distillation_pipeline.py`). 20-step smoke
   PASSED (`logs/dqvsa_smoke.log`): teacher(BF16 fine)/student(fake-quant
   NVFP4 fine) asymmetry verified (30/30 impls), gradients finite 20/20,
   checkpoint saved, trained weights served through the native P4 path
   (`DQVSA_SERVE_RC=0`). No large training run started.
5. **Claim cleanup** — `REPORT_V3.md`, `PAPER_UPDATE_V3.md` (binding
   wording); superseded-notes added to REPORT_V2/PAPER_UPDATE_V2.

## Verdict (V3)

Direction B refined: native SparseFP4 correct and kernel-positive; E2E it
trails its BF16 twin for fully-attributed integration reasons; NVFP4 QK
carries a real Holm-significant quality cost (imaging, dynamism) present
dense and sparse alike, NOT amplified by sparsity on those dimensions.
Positive systems result: geometry-aligned sparse attention (P4G 1.40x,
comparable quality with small significant trade-offs vs deployed VSA).
Recovery path (DQ-VSA) designed, literature-anchored, smoke-tested.

---

# Historical (V2) status below

_Last updated: 2026-08-17 09:25 ET — STUDY COMPLETE (V2). Re-audit (V2) in progress (older incremental notes
superseded; history in git)._

## Verdict (V2, final)

**Direction B** (`RESULTS_DECISION_V2.md`): native SparseFP4 is real,
numerically clean (exact-geometry receipts), and kernel-positive — but FP4's
increment over sparse BF16 decays to ~1.04x at 90% sparsity, integration
costs invert it E2E (111.7 vs 106.0 s @720p), and paper-scale paired
evaluation shows NVFP4's own quality cost (imaging -0.101, dynamic -0.25).
The positive systems result is geometry alignment: P4G (BF16) 1.40x vs
dense at deployed-baseline quality.

## Live right now — RE-AUDIT (V2) in progress

- **P4 720p root cause cornered:** kernel-only FP4 BEATS BF16 (3.16 vs
  3.98 ms) at 720p geometry; mask_mod predicate costs FP4 +42% (BF16 free);
  synced in-model per-layer walls are 8.00 (FP4) vs 6.05 ms (BF16) — the
  145 s E2E gap only exists UNSYNCED -> CUDA caching-allocator thrash from
  ~200 MB/call transient FP4 buffers. **CONFIRMED + FIXED: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  takes P4 720p E2E 250.9 -> 111.7 s (2.25x recovery).** Fair-twin control: P0 dense under the same allocator config = 148.7 s
  (unchanged from 148.8) -> the fix is FP4-specific, not a global speedup.
  P4G rerun in flight; durable in-backend fix
  (preallocated quantize workspace) still worth adding.
- **Priority 2:** VSA256 exact-10% captures running (480p 5/25+, 720p
  queued, GPU1); offline A0/B0/C0_256/D0_256 runner staged.
- **Priority 3: generation COMPLETE** — 1,630 videos (326 official VBench
  prompts x 5 arms, paired, unified allocator config; one transient shard
  failure backfilled). Scoring (VBench dims + paired metrics + P4-P4G
  bootstrap) running on all 8 GPUs.
- Claim-language corrections (1.40x is P4G/BF16; 9.3x is sparsity not FP4;
  FP4 increment over sparse BF16 ~1.04x at 10%; no QuantSparse-contradiction
  claim; QAT supplementary-only) will land in REPORT_V2/RESULTS_DECISION_V2/
  PAPER_UPDATE_V2.

## Results at a glance

### Operator 2x2 (25 genuine-VSA cells, frozen masks; median rel-L2 vs A0)

| | value |
|---|---|
| B0 dense NVFP4 (quant-only) | 0.098 |
| C0 sparse BF16 (24.2% kept) | 0.128 |
| D0 native sparse NVFP4 (same mask) | 0.204 |
| **D0 vs C0 (quant cost on sparse)** | **0.096 = no amplification** |
| D0 vs dequantized oracle | 0.0017 (kernel exact to FP4 floor) |

### E2E (median steady-state; 50 steps)

| System | 480p s (x) | 720p s (x) |
|---|---|---|
| P0 dense BF16 | 46.9 (1.00) | 148.8 (1.00) |
| P1 dense NVFP4 | 44.4 (1.06) | 135.8 (1.10) |
| P2 deployed VSA@0.9 | 50.0 (0.94) | 131.7 (1.13) |
| P2G VSA sel + FA4 BF16 fine (24%) | 48.7 (0.96) | — |
| P3 VSA sel + NVFP4 fine (24%) | 53.1 (0.88) | — |
| **P4G VSA256-FA4 BF16 fine (10%)** | **45.6 (1.03)** | **106.1 (1.40)** |
| P4 VSA256-FA4 NVFP4 fine (10%) | 47.3 (0.99) | 250.9 (0.59) — open perf item, see below |

Kernel-only (Wan shape, 10% retained): native sparse NVFP4 0.80 ms vs dense
BF16 7.41 ms (9.3x); vs sparse BF16 0.83 ms.

### Quality (10 dev prompts; VBench means)

- Sparse arms cost subject-consistency vs dense (0.976 -> 0.85-0.90);
  **NVFP4 adds no consistent extra penalty over BF16 twins**; the 256-tile
  selector (P4/P4G) matches deployed VSA's quality.
- **QAT recovery (400 steps, 47 videos, 1 GPU, ~47 min):** P3 subject
  consistency 0.846 -> **0.974** (dense 0.976), background 0.918 -> **0.977**
  (= dense), aesthetic 0.410 -> 0.540. Caveat: dynamic_degree collapsed on
  the static-camera mini-dataset — use a motion-diverse corpus at scale.
- **B3 receipt:** NVFP4 round-trip of *selector* inputs leaves the deployed
  mask 99.55% identical — selector stays BF16 by design, costs nothing.

## Open engineering items (documented in REPORT/PAPER_UPDATE; not blockers)

1. **P4 (NVFP4 fine) at 720p is 250.9 s — NOT yet fixed, only diagnosed.**
   The 7 fork bugs we fixed were correctness/crash bugs; this one is a
   *performance* regression inside the (unmodified) FP4 kernel: at 720p tile
   geometry (360 tiles, 92160 tokens) the sparse-FP4 kernel runs 6.09 ms vs
   its BF16 twin's 2.95 ms — the same kernel that wins at 480p geometry
   (1.44 vs 1.65 ms incl. quantize). Root cause: `flash_fwd_sm100_fp4`'s
   scheduler/tiling was tuned for dense shapes; large sparse row-counts hit a
   bad regime. Candidate fixes (untried): non-persistent scheduling for
   sparse FP4, m/n block-size retune, SF-load pipelining depth.
2. 480p P3/P4 in-model FP4 fine path trails its own microbenchmark —
   suspected allocator churn from per-call quantize buffers; candidate fix:
   preallocated/cached quantize workspace.
3. BF16 sparse+mask_mod kernel variants re-JIT ~13 min in every fresh
   process (mask_mod hash not stable across processes); FP4 variants do not.
   Candidate fix: stable mask_mod hashing or persistent compile cache key.

## Phase ledger (all skill criteria met)

| Phase | State |
|---|---|
| C0 audit | DONE (`CODE_PATH_AUDIT.md`) |
| C2 native sparse NVFP4 | DONE — fork repair patch (7 commits) `configs/fa4-fork-sparse-fp4-repair.patch` |
| C3 native proof | DONE (`NATIVE_PROOF.md`) |
| C4 kernel correctness | DONE (32 cells, oracle-exact) |
| C5 capture + 2x2 matrix | DONE (`tables/c5_matrix_s090.md`) |
| C7 quality P0-P4 | DONE (paired + VBench, `raw/quality/`) |
| C8 performance | DONE (`tables/c8_performance.md`, 480p+720p) |
| C10 decision/report/paper | DONE (`RESULTS_DECISION.md`, `REPORT.md`, `PAPER_UPDATE.md`) |
| Track D QAT recovery | DONE (400 steps; consistency restored to dense level) |
| B3 selector receipt | DONE (99.55% agreement) |
| B1/B2 PV + MXFP8 ladder | DONE (MXFP8~=NVFP4; FP8-PV blocked in build) |

## Provenance

Repo branch `exp/sparsefp4-paper-validation`, pushed to
`chivatam/FastVideo` through `31f2f491`. FA4 fork branch
`sparsefp4-native-composition` at `/mnt/nvme/scratch/fa4-fork` (7 commits
over pin `940bf7e5`; exported as patch in `configs/`). Env receipts `env/`;
venv `/mnt/nvme/scratch/fv-venv` (torch 2.12.0+cu130, 8x B200, CUDA 13.0).
