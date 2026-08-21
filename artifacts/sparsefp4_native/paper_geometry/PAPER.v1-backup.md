# Geometry-Aligned Sparse Attention: Matching Selector Tiles to Kernel Granularity Is the Dominant Speed Lever for Video Diffusion on Blackwell

*Spin-out paper covering only the geometry-alignment (P4G) component of the SparseFP4 study. Every number traces to receipts under `artifacts/sparsefp4_native/` (canonical sources per `PAPER_ARTIFACT_MAP.md`). No NVFP4 result is claimed here; the 1.40x speedup reported below belongs to the BF16 configuration.*

---

## Abstract

Block-sparse attention promises large speedups for video diffusion transformers, whose attention cost grows quadratically with token count. In practice, deployed dynamic block-sparse attention often fails to convert its theoretical FLOP reduction into wall-clock speedup: FastVideo's deployed Video Sparse Attention (VSA) at 90% sparsity yields only 1.13x end-to-end (E2E) speedup at 720p on an NVIDIA B200, and is a net *slowdown* (0.94x) at 480p, despite skipping 90% of attention work. We identify the cause as a **tile-geometry mismatch** between the sparsity selector and the fastest available attention kernel: VSA's (4,4,4)=64-token cubes cannot be consumed by the FlashAttention-4 (FA4) Blackwell block-sparse kernel, whose sparse granularity is 256 query tokens x 128 key tokens. This mismatch forces the fine branch onto a slower Triton kernel, and any post-hoc mask coarsening onto FA4 granularity inflates retention by ~2.4x, destroying the sparsity budget. We propose a minimal fix: change the *selector geometry itself* to (4,8,8)=256-token tiles so the top-k mask maps 1:1 onto FA4's native sparse granularity with zero mask inflation, keeping the VSA algorithm family (pooled-mean scoring, fused top-k, gated coarse branch) otherwise unchanged. On Wan2.1-T2V-1.3B at 720x1280x81 with exact 10% retention, geometry alignment delivers **1.40x E2E speedup over dense BF16** and **1.24x over deployed VSA**, with an 8.9x attention-kernel speedup, flat peak memory, and — under a 326-prompt paired VBench protocol with Holm correction — quality comparable to deployed VSA on measured dimensions, with small significant trade-offs (aesthetic quality -0.030, motion smoothness -0.018, background consistency -0.008) and one significant gain (dynamic degree +0.139). The result is precision-independent and requires no training, no new kernel, and no change to the sparsity budget: aligning the selector's tile geometry to the kernel's sparse granularity is the dominant lever for converting attention sparsity into real speedup.

---

## 1. Introduction

Video diffusion transformers (DiTs) spend a large and growing fraction of inference time in 3D full self-attention: a 5-second 720p Wan2.1 generation attends over ~75k tokens per layer, and attention cost grows quadratically as resolution and duration scale. Trainable block-sparse attention methods such as Video Sparse Attention (VSA) exploit the strong spatiotemporal locality of video to skip 80-95% of attention blocks with modest quality impact, and report large *kernel-level* speedups.

Kernel-level speedup, however, is not deployment speedup. When we measured FastVideo's deployed VSA path (90% sparsity) on an NVIDIA B200 (sm_100) against a dense FlashAttention-4 (FA4) baseline, the E2E picture was sobering:

| System (720x1280x81) | E2E | Speedup vs dense |
|---|---|---|
| Dense BF16 (FA4) | 149.1 s | 1.00x |
| Deployed VSA @ 0.90 sparsity | 131.5 s | **1.13x** |

At 480x832x81, deployed VSA is a net slowdown (49.7 s vs 46.6 s dense, 0.94x). Ninety percent of attention work is skipped; almost none of it is recovered as wall-clock time.

**The cause is geometric, not algorithmic.** VSA selects blocks at (4,4,4)=64-token cube granularity. On sm_100 this has two consequences:

1. **Kernel downgrade.** The 64-token tile size cannot be consumed by the fastest available attention kernel on the platform — the FA4 Blackwell block-sparse path, whose sparse granularity is 256 query tokens x 128 key tokens. FastVideo's deployed VSA fine branch therefore dispatches to a Triton kernel (the sm_90 ThunderKittens CUDA extension does not exist on sm_100), leaving the tensor-core pipeline of the platform's flagship kernel unused.
2. **Coarsening inflation.** Mapping a 64-token-tile mask onto FA4's 256x128 granularity after selection requires an any-pool: a 256x128 super-block must be retained if *any* of its constituent 64-token tile pairs (4 query tiles x 2 key tiles) is selected. At 90% sparsity this coarsening inflates effective retention by ~2.4x, converting a 10%-retention mask into a ~24%-retention mask and forfeiting most of the sparsity budget before the kernel runs.

Our fix is deliberately minimal: **change the selector's tile geometry instead of the mask or the kernel.** We pool and select at (4,8,8)=256-token tiles, so that one selector tile equals exactly one FA4 sparse query row (256 tokens) and exactly two FA4 key blocks (2x128 tokens). The top-k mask then maps 1:1 onto the kernel's native sparse structure — zero inflation, exact retention — and the fine branch runs on the FA4 SM100 block-sparse kernel. Everything else in the VSA algorithm family is preserved: mean-pooled coarse scores, fused top-k selection at the same sparsity target, and the gated coarse-attention compression branch.

![Figure 1: selector-tile / kernel-granularity mismatch and its repair](figures/fig1_geometry_schematic.png)

*Figure 1. The mismatch and its repair (illustrative masks; the inflation factor is annotated from measurement). (a) The deployed VSA selector decides sparsity on 64-token tiles (blue) that straddle the FA4 kernel's 256x128 super-blocks (dark grid), keeping 10%. (b) Pooling that mask onto the kernel's blocks retains every super-block touched by any selected tile — retention inflates to ~25% (measured ~2.4x) and the sparsity budget is forfeited before the kernel runs. (c) Selecting directly on 256-token tiles (ours) maps 1:1 onto kernel granularity, keeping exactly 10%.*

**Contributions.**

1. We identify selector-tile/kernel-granularity mismatch as the dominant reason a deployed dynamic block-sparse video attention system converts a 10x attention-FLOP reduction into only 1.13x (720p) or 0.94x (480p) E2E speedup on Blackwell, and quantify both mismatch mechanisms (kernel downgrade; ~2.4x retention inflation under any-pool coarsening).
2. We propose and implement **geometry-aligned VSA (VSA256-FA4)**: a (4,8,8)=256-token selector whose top-k mask maps 1:1 onto FA4's 256x128 sparse granularity, with variable-block-size handling for non-tile-aligned video shapes. The change is selector-side only; no new kernel is written and the sparsity budget is unchanged (median realized keep 0.1006 at a 0.90 sparsity target).
3. On Wan2.1-T2V-1.3B we measure **1.40x E2E at 720p vs dense BF16** (1.24x vs deployed VSA) and 8.9x attention-kernel speedup at 10% retention, with flat peak memory, no compilation, and no training.
4. Under a 326-prompt paired VBench protocol (prompt-level bootstrap, Holm correction), geometry-aligned VSA is **comparable to deployed VSA on measured dimensions**, with small significant trade-offs (aesthetic -0.030, motion smoothness -0.018, background consistency -0.008) and significantly higher dynamic degree (+0.139). We report all deltas; we do not claim statistical indistinguishability or non-inferiority.

We emphasize a claim boundary: all speedups reported here are for **BF16** execution. The companion SparseFP4 study composes this geometry with NVFP4 attention arithmetic; none of its precision results are needed for, or claimed by, this paper.

---

## 2. Background and Related Work

**Video Sparse Attention (VSA)** (arXiv:2505.13389) is a trainable, hierarchical two-stage sparse attention for video DiTs. A coarse stage mean-pools (4,4,4) cubes (tile size B=64) of the video latent into cube-level Q_c/K_c/V_c, computes dense cube-to-cube attention, and produces a coarse output O_c; a fine stage computes token-level block-sparse attention only inside the top-K cubes selected by the coarse scores. The final output combines both branches through learned gates. Notably, VSA's own tile-size ablation prefers B=64: at B=256 ((4,8,8) cubes — exactly our geometry) pretraining loss worsens from 0.13162 to 0.13375 (VSA §3.1, Table 1(d)). VSA's design point therefore optimizes selection resolution for *training quality*, not for the sparse granularity of the fastest inference kernel on a given platform. Our results quantify the other side of that trade at inference time on Blackwell.

**Sliding/static-pattern sparse attention.** STA-style sliding-tile attention, used by FPSAttention (arXiv:2506.04648), restricts each query tile to a local 3D window and executes via FlexAttention mask_mod plus Triton kernels on Hopper — a static pattern co-designed with FP8 quantization at the kernel level. FPSAttention's central lesson (quantization and sparsity granularity must match the kernel's tile primitives) is the same lesson we draw for *dynamic* selection geometry, in pure BF16.

**Dynamic block-sparse selection.** SpargeAttention2 (arXiv:2602.13515) selects blocks from pooled attention scores at b_q=128, b_kv=64 with CUDA kernels; SLA/SLA2 (arXiv:2509.24006, arXiv:2602.12675) split attention into sparse and linear branches with learnable routing at b_q=128, b_kv=64 blocks. These systems choose block shapes matched to *their own* kernels. Our contribution is complementary: when the selector and the platform's best kernel are developed separately (VSA's 64-token cubes vs FA4's 256x128 granularity), the alignment itself — not any new kernel or selector — is worth more E2E than the choice of either component.

**FlashAttention-4 on Blackwell.** The FA4 SM100 kernel family (CuTe-DSL; tcgen05 block-scaled MMA, TMEM, 2-CTA pipelining; described second-hand in arXiv:2603.00040 §2.5/App. B — the FA4 paper itself has no located arXiv ID) includes a block-sparse path that consumes packed per-row retained-block index lists at a granularity of one 256-token query row x 128-token key blocks, with online softmax iterated over the gathered block list. This granularity is a hardware-pipeline consequence (q_stage x m_block = 2x128 query tokens per softmax row-state), and it is what any selector must produce to use the kernel losslessly.

**Positioning.** We do not propose a new sparsity algorithm, kernel, or training method. We change one integer triple in the selector — (4,4,4) to (4,8,8) — and show that this single alignment decision dominates the deployed system's E2E speedup at 720p. To our knowledge, prior video sparse-attention work reports kernel or FLOP reductions at the selector's native granularity, and does not isolate selector-geometry/kernel-granularity alignment as a measured E2E lever.

---

## 3. Method: Geometry-Aligned VSA (VSA256-FA4)

### 3.1 Preserved algorithm family

The backend (`fastvideo/attention/backends/sparsefp4_vsa256_fa4.py`, enum `SPARSEFP4_VSA256_FA4_ATTN`) keeps the deployed VSA algorithm intact and changes only the tile geometry and the fine-branch kernel:

1. **Tiling.** The video latent (t, h, w) is partitioned into (4,8,8)=256-token tiles (deployed VSA: (4,4,4)=64). Boundary tiles at non-divisible dimensions are padded; per-tile valid-token counts (`variable_block_sizes`) are tracked exactly as in deployed VSA.
2. **Coarse branch.** Q, K, V are mean-pooled per tile (`fused_block_mean`), coarse scores are computed as q̄k̄ᵀ/√d, and a dense tile-level attention produces the compression output, broadcast back to tokens and weighted by the model's learned `gate_compress` — the same gated compression branch as deployed VSA.
3. **Selection.** `fused_topk_mask` retains the top-k key tiles per (batch, head, query tile), with k derived from the same sparsity target (0.90) via `compute_topk`. Precision of selection arithmetic is unchanged. The sparsity *budget* is identical; only the tile size over which it is expressed changes.

### 3.2 Exact 1:1 mapping onto FA4 sparse geometry

The FA4 SM100 block-sparse interface consumes, per 256-token query row, packed lists of retained 128-token key blocks (`BlockSparseTensorsTorch`: full-block and partial-block count+index lists). With 256-token selector tiles the mapping is exact:

- **Query side:** one selector tile == one FA4 sparse query row (q_stage x m_block = 256 tokens). No pooling across query tiles.
- **Key side:** one selector tile == exactly two 128-token FA4 key blocks (`repeat_interleave(2)` on the mask's key axis). Retaining a tile retains its two halves; nothing else is dragged in.

Retention is therefore *exact*: a 10% tile mask becomes a 10% kernel-block mask (measured median keep 0.1006 at the 0.90 sparsity target; zero coarsening). Contrast the deployed geometry: mapping a 64-token-tile mask onto 256x128 super-blocks requires `any`-pooling over 4 query tiles x 2 key columns, which measured ~2.4x retention inflation — a 10% mask becomes ~24% before the kernel sees it.

- **Boundary handling.** Per-128-block valid-token counts split retained blocks into *full* (128 valid tokens; fast path, no predicate) and *partial* (boundary; handled by a `mask_mod` predicate that trims padded tokens). On the BF16 kernel this predicate is measured to be free (3.98 ms vs 3.89 ms with/without at 720p geometry; `P4_PERF_ROOT_CAUSE.md` Step 2).
- **Softmax correctness.** The FA4 sparse path runs the ordinary online-softmax state machine (running row-max/row-sum, output-accumulator rescale) once per *retained* block index over the gathered list — identical math to dense, restricted to retained blocks (`CODE_PATH_AUDIT.md` §8).

### 3.3 What is explicitly not changed

No training or fine-tuning; no new kernel (the FA4 block-sparse path pre-exists); no change to the sparsity target, selection scoring, gating, scheduler, or model weights. The configuration is a drop-in attention-backend swap selected by `FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_VSA256_FA4_ATTN` with BF16 fine precision (`FASTVIDEO_SPARSEFP4_FINE=bf16`) — the arm designated **P4G** throughout.

---

## 4. Experimental Setup

- **Model/config:** Wan-AI/Wan2.1-T2V-1.3B-Diffusers; 50 denoising steps; CFG; seed 1234; VSA sparsity 0.90; resolutions 480x832x81 and 720x1280x81.
- **Hardware/software:** NVIDIA B200 (sm_100, one GPU of an 8x node), CUDA 13.0, torch 2.12.0+cu130, fastvideo-kernel 0.3.2, flash-attn-4 fork `hao-ai-lab/flash-attention-fp4@940bf7e5`; environment receipts in `env/`.
- **Arms:** **P0** dense BF16 (FA4); **P2** deployed VSA @0.90 (64-token tiles, Triton fine kernel); **P4G** geometry-aligned VSA256-FA4, BF16 fine, exact 10% kept. (P1/P4 are NVFP4 arms of the companion study; shown only for context, never as claims of this paper.)
- **Timing protocol:** E2E medians of 3 steady-state generations of one prompt per arm (first generation excluded as warmup/JIT; dispersion across reps <0.5%); one process per arm; identical `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in every arm; no torch.compile, no CUDA graphs. Kernel latencies via CUDA events, median of 50, pre-quantized inputs. Nothing is inferred from FLOPs.
- **Quality protocol:** 326 official VBench prompts (all prompts of the 7 dimensions scorable in this environment), paired by prompt/seed/config across arms; prompt-level percentile bootstrap (10,000 resamples), two-sided bootstrap p-values, Holm correction across the 7 VBench dimensions per contrast. Pixel metrics (PSNR/SSIM/LPIPS) computed against the dense P0 reference as similarity-to-P0, paired.

---

## 5. Results

### 5.1 Attention-kernel latency

CUDA-event medians of 50, Wan attention shape B=1, S=39936, H=12, D=128, BF16 (`tables/c8_performance_v2.md`):

| Configuration | Retained | Median ms | Speedup vs dense |
|---|---|---|---|
| Dense BF16 (FA4) | 1.00 (dense loop) | 7.414 | 1.00x |
| Sparse BF16 (FA4 block-sparse) | 1.00 | 7.591 | 0.98x |
| Sparse BF16 | 0.50 | 3.815 | 1.94x |
| Sparse BF16 | 0.25 | 1.814 | 4.09x |
| Sparse BF16 | 0.10 | 0.830 | **8.93x** |

Two observations. First, the sparse machinery itself costs ~2% at full retention (7.591 vs 7.414 ms) — the block-list iteration is nearly free. Second, at the deployment operating point (10% retained) the kernel realizes 8.93x of the theoretical 10x, i.e., the aligned mapping converts the sparsity budget into kernel time almost losslessly.

![Figure 2: kernel latency vs retained fraction](figures/fig2_kernel_scaling.png)

*Figure 2. Attention-kernel time vs fraction of attention kept, under the aligned 1:1 mapping (log-log; left-to-right reads as increasing sparsity). The measured FA4 block-sparse BF16 kernel (green) tracks the ideal dense-times-retention line (dashed grey) across the whole range — sparsity decided at kernel granularity is converted into kernel time almost losslessly, reaching 8.9x at the 10% deployment operating point. CUDA-event medians of 50, B=1 S=39936 H=12 D=128; per-point values in §5.1.*

### 5.2 End-to-end performance

Medians of 3 steady-state generations, identical allocator config in all arms (`tables/c8_performance_v2.md`):

**720x1280x81:**

| System | E2E s | Speedup vs P0 | DiT s | Peak MB |
|---|---|---|---|---|
| P0 dense BF16 (FA4) | 149.13 | 1.000x | 144.82 | 19022 |
| P2 deployed VSA @0.9 (Triton fine) | 131.49 | 1.134x | 127.23 | 19028 |
| **P4G VSA256-FA4 BF16 (10% exact)** | **106.20** | **1.404x** | 101.93 | 19028 |

**480x832x81:**

| System | E2E s | Speedup vs P0 | DiT s | Peak MB |
|---|---|---|---|---|
| P0 dense BF16 (FA4) | 46.62 | 1.000x | 44.18 | 8888 |
| P2 deployed VSA @0.9 (Triton fine) | 49.71 | 0.938x | 47.22 | 8893 |
| **P4G VSA256-FA4 BF16 (10% exact)** | **45.62** | **1.022x** | 43.24 | 8893 |

At 720p, geometry alignment yields **1.40x vs dense** and **1.24x vs deployed VSA** (131.49/106.20), at identical peak memory and with the identical sparsity budget as P2. At 480p, P4G converts deployed VSA's net *slowdown* (0.938x) into a small net win (1.022x); the remaining headroom is bounded by attention's smaller share of total time at 33k tokens (§6.3).

![Figure 3: end-to-end latency by system and resolution](figures/fig3_e2e_performance.png)

*Figure 3. End-to-end generation time (bar length; shorter is faster) at (a) 480p and (b) 720p for dense BF16, deployed VSA, and the geometry-aligned configuration (ours, green); labels give speedup vs dense. At 480p the deployed geometry loses time outright (0.94x) while alignment restores a small win (1.02x); at 720p alignment delivers 1.40x vs dense — 1.24x vs deployed VSA — at identical sparsity budget and flat peak memory. Wan2.1-T2V-1.3B, 50 steps, sparsity 0.90; medians of 3 steady-state repetitions (dispersion <0.5%), identical allocator configuration, no compilation; exact values in §5.2.*

### 5.3 Quality: geometry contrast at paper scale

Paired P4G - P2 deltas over 326 VBench prompts, Holm-corrected (`tables/p4g_vs_p2_quality_bootstrap.md`); positive = P4G scores higher:

| VBench dimension | n | Mean Δ | 95% CI | Holm-sig @0.05 |
|---|---|---|---|---|
| subject_consistency | 72 | +0.0006 | [-0.0063, +0.0075] | no |
| background_consistency | 86 | -0.0078 | [-0.0107, -0.0052] | **yes** |
| temporal_flickering | 75 | -0.0029 | [-0.0058, -0.0000] | no (Holm p=0.14) |
| motion_smoothness | 72 | -0.0175 | [-0.0203, -0.0147] | **yes** |
| dynamic_degree | 72 | **+0.1389** | [+0.0694, +0.2222] | **yes** |
| imaging_quality | 93 | +0.0021 | [-0.0113, +0.0152] | no |
| aesthetic_quality | 93 | -0.0298 | [-0.0400, -0.0203] | **yes** |

Pixel similarity to the dense P0 reference (paired Δ of similarity-to-P0, n=326): P4G is slightly farther from P0 than P2 (ΔPSNR -0.25 [-0.33, -0.17], ΔSSIM -0.006, ΔLPIPS +0.023; all p=0.0002). We report this honestly and do not interpret pixel proximity to the dense reference as quality.

![Figure 4: forest plot of paired VBench deltas, P4G vs deployed VSA](figures/fig4_quality_forest.png)

*Figure 4. Paired VBench score change (P4G − P2) with 95% prompt-level bootstrap CIs, Holm-corrected across the 7 dimensions; per-dimension n shown with each label (unit of replication: prompt). Color encodes the corrected result: orange = significantly worse, green = significantly better, grey = not separable. Three small significant negatives (aesthetic, motion smoothness, background consistency), one large significant positive (dynamic degree), three dimensions not separable; exact estimates and CIs in the table above. The geometry change is comparable on measured dimensions with these enumerated trade-offs; no non-inferiority claim is made.*

**Licensed interpretation (binding wording):** P4G is *comparable to deployed VSA on measured dimensions, with small significant trade-offs* — it gives up a little aesthetic quality (-0.030), motion smoothness (-0.018), and background consistency (-0.008) while producing markedly more motion (dynamic degree +0.139); subject consistency, imaging quality, and temporal flickering do not separate after Holm correction. No non-inferiority margin was pre-registered, so no non-inferiority claim is made, and we do not describe the arms as statistically indistinguishable.

Two context notes. (i) This contrast isolates the *geometry* effect: both arms are BF16, both use the same sparsity budget and the same selection algorithm family; only tile geometry and fine kernel differ. The quality cost of sparsification itself (either sparse arm vs dense P0 at 0.90 sparsity, applied at inference) is a property of the sparsity family shared by both arms and is outside this paper's claims. (ii) The direction of the trade-off is consistent with VSA's own tile ablation (B=256 worsens pretraining loss, §2): coarser selection measurably costs a little on smoothness/aesthetic dimensions — but at inference time on this platform the cost is small and buys 1.24x wall-clock over the deployed geometry.

### 5.4 Operator-level sparsification error at the aligned geometry

For completeness, the controlled operator matrix (25 cells per resolution — 5 layers x 5 timesteps captured from a genuine P4G trajectory; `tables/c5_matrix_vsa256_exact10.md`) puts the aligned-geometry sparsification error at rel-L2 0.192 (480p) / 0.218 (720p) vs dense attention outputs at exact 10% retention (median keep 0.1006), with cosine similarity 0.985/0.979. These are attention-output perturbations, not end-video quality; the end-to-end consequences are the VBench results of §5.3.

---

## 6. Analysis: Where the 1.4x Comes From

### 6.1 Decomposing the win over deployed VSA

P4G's 25.3 s advantage over P2 at 720p (131.49 -> 106.20 s) comes from two coupled mechanisms that the geometry change unlocks simultaneously:

1. **Kernel class.** The 64-token tile geometry strands the deployed fine branch on a Triton kernel (sm_100 has no ThunderKittens build); the 256-token geometry makes the mask consumable by the FA4 SM100 block-sparse path — the same kernel family as the dense baseline, with its full tensor-core pipeline (2-CTA scheduling, TMEM, TMA block-gathered loads).
2. **Retention fidelity.** The alternative route to the FA4 kernel — keep the 64-token selector and any-pool the mask to 256x128 — was measured in the companion study's demoted coarsened arms: retention inflates ~2.4x and the 480p E2E lands at 48.7-53.1 s, i.e., *worse than doing nothing*. Alignment is what makes the fast kernel and the true 10% budget available at the same time.

The 8.93x kernel speedup (§5.1) against a 1.40x E2E speedup also quantifies the Amdahl envelope: attention is the dominant but not the only cost at 720p, and the non-attention remainder (projections, FFN, norms, scheduler, VAE) bounds what any attention-side intervention can deliver.

### 6.2 Why the result is precision-independent

The claim boundary from the companion study is worth restating as a positive finding: the 1.40x is achieved in plain BF16. Composing this geometry with native NVFP4 attention arithmetic (arm P4) yields 1.33x — *slower* than the BF16 twin — because at 10% retention the QK MMA is no longer the kernel bottleneck and FP4-specific integration costs (a validity predicate on the softmax-bound FP4 pipeline, per-call quantization) exceed FP4's residual arithmetic advantage (`P4_PERF_ROOT_CAUSE.md`). Geometry alignment, by contrast, attacks the part of the cost that sparsification actually leaves behind: it needs no arithmetic-format change, and it is the *dominant* lever at this operating point.

### 6.3 Resolution scaling

The speedup grows with token count, as expected for a quadratic-cost target: at 33k tokens (480p) attention is a modest share of DiT time and P4G is roughly performance-neutral (1.02x); at 75k tokens (720p) it is 1.40x. The mechanism predicts larger gains at higher resolutions and longer durations, but we measured only these two points and make no extrapolated claim.

---

## 7. Limitations

- **Single model, single sparsity point, single seed.** All results are on Wan2.1-T2V-1.3B at VSA sparsity 0.90; paper-scale quality uses one seed with a paired design (which mitigates but does not replace the official multi-sample VBench protocol). Generalization to other models, sparsity targets, and seeds is untested.
- **Training-free selector-geometry swap.** The 256-token selector is applied at inference without retraining the gates or the model at the new geometry. VSA's ablation indicates B=64 is the better *training* geometry; whether fine-tuning at B=256 would remove the small aesthetic/smoothness trade-offs (or whether the deployed gates are mildly mismatched to the coarser tiles) is unknown.
- **Quality coverage.** 7 of 16 VBench dimensions are scorable in this environment (the remainder require uninstalled dependencies); dynamic_degree is a near-binary metric with wide CIs at n=72.
- **Timing scope.** E2E medians come from 3 steady-state repetitions of one prompt per arm (dispersion <0.5% across reps); no torch.compile or CUDA graphs anywhere, so all arms could shift under compilation (a supplementary observation in the companion study measured 1.34-1.39x forward-block gains from torch.compile, orthogonal to and not composed with these numbers).
- **Platform specificity.** The 256x128 granularity is a property of the FA4 SM100 kernel; the *principle* (align selector tiles to the target kernel's sparse granularity) should transfer, but the specific tile triple and the measured magnitudes are Blackwell-specific.
- **Two resolutions only.** No measurements above 720p or beyond 81 frames.

---

## 8. Conclusion

A deployed dynamic block-sparse attention system that skips 90% of attention work delivered only 1.13x end-to-end at 720p — and lost time outright at 480p — because its selection tiles could not be consumed by the platform's fastest attention kernel without either a kernel downgrade or a ~2.4x retention inflation. Changing one design constant — the selector tile from (4,4,4) to (4,8,8), so the top-k mask maps 1:1 onto FlashAttention-4's 256x128 sparse granularity — recovers 8.9x at the kernel and **1.40x end-to-end vs dense (1.24x vs deployed VSA) at 720p**, with unchanged sparsity budget, flat memory, no training, no new kernel, and quality comparable to the deployed baseline on measured dimensions with small, fully-reported trade-offs.

The general lesson for sparse-attention deployment is that **selector geometry is a first-class systems parameter**: the granularity at which sparsity is *decided* must match the granularity at which the kernel can *skip*, or most of the theoretical budget is forfeited at the boundary between the two. In our measurements this single alignment decision was worth more wall-clock time than the choice of attention arithmetic precision, and it composes with — rather than competes against — every other component of the stack.

---

## References

1. VSA: Faster Video Diffusion with Trainable Sparse Attention. arXiv:2505.13389.
2. FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Video Diffusion. arXiv:2506.04648.
3. SpargeAttention2. arXiv:2602.13515.
4. SLA: Sparse-Linear Attention. arXiv:2509.24006.
5. SLA2: Sparse-Linear Attention with Learnable Routing and QAT. arXiv:2602.12675.
6. Attn-QAT: 4-Bit Attention With Quantization-Aware Training. arXiv:2603.00040. (Source for second-hand FA4 SM100 kernel characterization; the FA4 paper itself has no located arXiv ID.)
7. QuantSparse. arXiv:2509.23681.
8. SageAttention3: Microscaling FP4 Attention for Inference. arXiv:2505.11594.
9. FastVideo and fastvideo-kernel (VSA deployment). https://github.com/hao-ai-lab/FastVideo
10. flash-attention-fp4 fork (FA4 SM100 block-sparse interface). hao-ai-lab/flash-attention-fp4@940bf7e5.

All arXiv-cited claims were verified against primary sources on 2026-08-17 (`SOTA_RECOVERY_LIT_REVIEW.md`, verification appendix).

---

## Appendix A. Reproducibility and Artifact Map

All receipts live under `artifacts/sparsefp4_native/` in the FastVideo checkout (branch `exp/sparsefp4-paper-validation`).

| Paper section | Canonical source |
|---|---|
| §5.1, §5.2 performance | `tables/c8_performance_v2.md` (raw: `raw/performance/perf_v2/`, logs: `logs/perf_v2/`) |
| §5.3 quality | `tables/p4g_vs_p2_quality_bootstrap.md` (stats: `configs/paired_stats_v2.py`, `raw/statistics/`) |
| §5.4 operator matrix | `tables/c5_matrix_vsa256_exact10.md` (raw: `raw/operator/c5b_exact10.jsonl`) |
| §6.1-6.2 root cause / predicate / allocator | `P4_PERF_ROOT_CAUSE.md` (`p4_maskmod_ab.json`, `quant_overhead.json`) |
| Kernel/interface audit | `CODE_PATH_AUDIT.md` |
| Environment | `env/` (GPU, driver, CUDA, torch, pip freeze, model revision) |
| Implementation | `fastvideo/attention/backends/sparsefp4_vsa256_fa4.py` |
| Figures | `paper_geometry/figures/` — contract-first pipeline synthesized from the nature-figure, ARIS paper-figure, K-Dense scientific-visualization, and publication-chart skills: `FIGURE_CONTRACT.md` (claim/evidence-role/archetype per figure, written before plotting), `data/*.csv` (source data transcribed verbatim from the canonical tables; scripts hardcode nothing), `make_figures.py` (serif face matching body text, Okabe-Ito accents, panel letters, no in-figure titles, column-width physical sizing), `provenance_manifest.json` (per-figure source-data provenance), PNG 300 dpi + vector PDF with Type-42 fonts. Figure 1's mask panels are illustrative; its ~2.4x inflation annotation is the measured value. |

Reproduction outline: (1) install the pinned environment per `env/`; (2) select the backend with `FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_VSA256_FA4_ATTN` and `FASTVIDEO_SPARSEFP4_FINE=bf16`; (3) run the standard FastVideo Wan2.1-T2V-1.3B generation config (50 steps, seed 1234, VSA sparsity 0.90) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; (4) compare against `VIDEO_SPARSE_ATTN` (P2) and the dense FA4 baseline (P0) under the identical protocol of §4.

## Appendix B. Claim Boundaries Inherited From the Parent Study

- The 1.40x E2E speedup belongs to the **BF16** P4G configuration; it must never be attributed to FP4 or any quantized arm.
- "Comparable quality" is bounded to the 7 measured VBench dimensions with the enumerated significant trade-offs of §5.3; "statistically indistinguishable" and non-inferiority claims are not licensed.
- The 8.93x kernel figure is a *sparsity* speedup vs dense BF16 at matched shape; it is not an E2E claim.
- E2E numbers are medians of 3 steady-state repetitions of one prompt per arm under one allocator configuration; nothing is inferred from FLOPs.
