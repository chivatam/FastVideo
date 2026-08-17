# P4_PERF_ROOT_CAUSE — native sparse-NVFP4 E2E performance investigation

Question: why did P4 (VSA256 + native NVFP4 fine branch) run 250.9 s at 720p
— 0.59x vs dense and 2.4x slower than its BF16 twin P4G — when the FP4
kernel wins microbenchmarks?

## Evidence chain (all receipts under `raw/performance/`, `logs/`)

### Step 1 — persistent-scheduler hypothesis: DISPROVEN
`FA4_FP4_DISABLE_PERSISTENT` knob added to the fork; A/B at 480p/720p
geometry, 10% keep (`p4_persistence_fix.json`):

| geometry | FP4 persistent | FP4 non-persistent | BF16 |
|---|---|---|---|
| 156 tiles (480p) | 0.593 ms | 0.619 ms | 0.726 ms |
| 360 tiles (720p) | 3.106 ms | 3.189 ms | 3.955 ms |

Persistence is within noise, and with plain sparse lists **FP4 beats BF16 at
both geometries** — the kernel itself is not the 720p problem.

### Step 2 — vbs mask_mod predicate: an FP4-SPECIFIC kernel cost
4-way A/B at 720p geometry with realistic partial-tile validity
(`p4_maskmod_ab.json`):

| | no mask_mod | with mask_mod |
|---|---|---|
| FP4 | 3.16 ms | **4.48 ms (+42%)** |
| BF16 | 3.98 ms | 3.89 ms (free) |

The per-element validity predicate runs in the softmax warp group; the FP4
kernel is softmax-unit-bound (FA4 paper's asymmetric-scaling analysis), so
the predicate costs it disproportionately, inverting the FP4-vs-BF16
ordering. Contribution: ~1.3 ms/call = ~4 s/video — real but small vs the
145 s gap.

### Step 3 — in-model synced walls: the gap VANISHES under synchronization
Env-gated per-forward `cuda.synchronize` timing inside the real 720p model
(`FASTVIDEO_SPARSEFP4_TIMING=1`, 2 steps x 2 CFG x 30 layers = 120 samples):

| fine branch | median fwd wall | p90 |
|---|---|---|
| nvfp4 | 8.00 ms | 8.69 |
| bf16 | 6.05 ms | 6.18 |

Synced, the whole attention forward accounts for only ~6 s/video of
difference — yet the unsynced steady-state E2E gap was 145 s. A gap that
only exists without synchronization and is not kernel time is allocator
behavior: the FP4 path allocates ~200 MB of transient packed-FP4/SF buffers
per call at 92160 tokens under ~19 GB model pressure; with a deep async
queue, the CUDA caching allocator hits repeated
alloc-failure -> cudaFree/synchronize cycles (fragmentation thrash).

### Step 4 — allocator fix: CONFIRMED, 2.25x recovery
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, all arms rerun under the
identical config (`p_runs/perf720-fix/`):

| arm | 720p E2E before | after | x vs P0 |
|---|---|---|---|
| P0 dense BF16 | 148.8 | 148.7 s | 1.00 (control: unchanged) |
| P4G BF16 fine | 106.1 | 106.0 s | 1.40 (control: unchanged) |
| **P4 NVFP4 fine** | **250.9** | **111.7 s** | **1.33** |

The control arms are unchanged — the setting specifically cures the FP4
path's transient-buffer thrash, confirming the diagnosis. Residual P4-P4G
gap: 5.7 s/video ~= the synced per-layer delta (1.95 ms x 3000 calls =
5.9 s), i.e. fully explained by Step 2's mask_mod cost plus quantization
(0.17 ms/call measured, `quant_overhead.json`).

## Current answer to "does native FP4 beat the BF16 sparse twin?"

- **Kernel level: yes** at every retained fraction and both geometries
  *without* the vbs mask_mod (e.g. 3.16 vs 3.98 ms at 720p/10%), and at 480p
  even with quantization included (1.44 vs 1.65 ms wall).
- **E2E: not yet** — 111.7 vs 106.0 s at 720p. The remaining deficit is
  precisely attributable: (a) mask_mod predicate in the softmax-bound FP4
  kernel (~4 s), (b) per-call quantize (~1 s), (c) small residual
  integration overhead (<1 s).
- Incremental-FP4-vs-sparsity context (kernel, no mask_mod): dense 1.26x ->
  50% 1.24x -> 25% 1.10x -> 10% 1.04-1.26x depending on geometry — FP4's
  advantage shrinks as QK MMA stops dominating sparse-kernel time.

## Paths to closing the E2E gap (ranked, not yet implemented)

1. **Eliminate the in-kernel validity predicate** (expected ~4 s): pad the
   latent grid so all 256-token tiles are full (video dims already nearly
   tile-aligned), or precompute per-block K validity into the sparse lists
   with boundary blocks handled by a BF16 epilogue pass, or move the
   predicate out of the softmax warp group (kernel change).
2. **Preallocated quantize workspace** (expected ~1 s + robustness):
   removes the dependency on the global allocator flag.
3. FP8/NVFP4 PV (blocked in this fork build: `MmaF8F6F4Op` unsupported
   under dsl 4.5.3) — would raise FP4's kernel margin so the integration
   overheads no longer flip the ordering.

## Honest bottom line

The 0.59x number was an integration artifact, now fixed and controlled
(1.33x). At the target production shape today, native sparse FP4 is
kernel-faster but E2E-slightly-slower (111.7 vs 106.0 s) than its BF16
twin; the deficit is fully accounted for and dominated by a predicate that
is architectural (softmax-unit pressure), not numerical. Unless item 1
lands, the paper should report the composition as numerically clean and
kernel-positive with an explicitly characterized E2E integration gap —
the Direction-B framing of the re-audit brief.
