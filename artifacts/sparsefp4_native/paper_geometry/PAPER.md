# Geometry-Aligned Sparse Attention: Matching Selector Tiles to Kernel Granularity for Video Diffusion on Blackwell

*Spin-out paper covering only the geometry-alignment (P4G) component of the SparseFP4 study. Every quantitative claim maps to a row in `CLAIM_LEDGER.md` and a receipt under `artifacts/sparsefp4_native/`. The 1.40x speedup belongs to the BF16 configuration; no NVFP4 result is claimed here.*

---

## Abstract

We show that re-aligning a dynamic sparse-attention selector's tile geometry to the attention kernel's native sparse granularity converts a deployed 90%-sparse video diffusion transformer from 1.13x to 1.40x end-to-end speedup on an NVIDIA B200, with no training, no new kernel, and no change to the sparsity budget. Realizing this speedup is hard because the selector and the kernel are designed separately: Video Sparse Attention (VSA) decides sparsity on (4,4,4)=64-token cubes, while the FlashAttention-4 (FA4) Blackwell block-sparse kernel can only skip work at 256x128-token granularity — so the deployed system either falls back to a slower Triton kernel or inflates retention ~2.4x when the mask is pooled across the boundary. Our fix changes one constant: the selector pools and selects on (4,8,8)=256-token tiles, so the top-k mask maps 1:1 onto kernel granularity with exact retention, and the fine branch runs on FA4. On Wan2.1-T2V-1.3B at 720x1280x81, the aligned configuration reaches an 8.9x attention-kernel speedup and 1.40x end-to-end vs dense BF16 (1.24x vs deployed VSA) at flat peak memory; under a 326-prompt paired VBench protocol with Holm correction, quality is comparable to deployed VSA with three small significant losses (aesthetic -0.030, motion smoothness -0.018, background consistency -0.008) and one significant gain (dynamic degree +0.139). In the same study, the identical geometry with native NVFP4 attention arithmetic ran *slower* end-to-end than BF16 — at this operating point, selector-kernel geometry alignment is worth more wall-clock time than attention arithmetic precision.

---

## 1. Introduction

A 5-second 720p generation with Wan2.1 attends over ~75k tokens per layer in every one of 30 transformer blocks and 50 denoising steps, and this quadratic cost grows with resolution and duration. Dynamic block-sparse attention methods such as Video Sparse Attention (VSA) exploit the spatiotemporal locality of video to skip 90% of attention blocks, which should yield close to a 10x reduction in attention time.

The deployed system does not deliver it. Measured on an NVIDIA B200 (sm_100) against a dense FlashAttention-4 (FA4) baseline, FastVideo's deployed VSA path at 0.90 sparsity yields 1.13x end-to-end at 720p — and at 480p it *loses* time (0.94x). Ninety percent of attention work is skipped; almost none of it returns as wall-clock speedup.

The loss occurs at the boundary between two independently designed components. VSA decides sparsity on (4,4,4)=64-token cubes, a granularity its authors selected for training quality. The fastest attention kernel on the platform — the FA4 SM100 block-sparse path — skips work at a granularity of one 256-token query row by 128-token key blocks, a shape fixed by its hardware pipeline. Every route across this mismatch forfeits the budget:

- **Kernel downgrade.** A 64-token-tile mask cannot feed the FA4 sparse path, so the deployed fine branch dispatches to a Triton kernel (the sm_90 ThunderKittens extension does not exist on sm_100), leaving the platform's flagship tensor-core pipeline unused.
- **Retention inflation.** Pooling the 64-tile mask up to 256x128 super-blocks after selection retains every super-block that any selected tile touches. Measured at 0.90 sparsity, this inflates retention ~2.4x — a 10% mask becomes ~24% before the kernel runs — and the coarsened arms measured *worse than dense* at 480p (48.7-53.1 s vs 46.6 s).

Our fix changes the selector rather than the mask or the kernel: pool and select on (4,8,8)=256-token tiles, so one selector tile equals exactly one FA4 sparse query row and exactly two FA4 key blocks. The top-k mask then maps 1:1 onto kernel granularity with exact retention (measured median keep 0.1006 at the 0.90 target), and the fine branch runs on FA4. The VSA algorithm family — mean-pooled coarse scores, fused top-k at the same sparsity target, gated compression branch — is otherwise unchanged, as are the model weights.

This paper makes four contributions:

1. **Diagnosis (C1-C3).** We identify selector-tile/kernel-granularity mismatch as the reason a deployed dynamic block-sparse video attention system converts a 10x attention-FLOP reduction into 1.13x (720p) or 0.94x (480p) end-to-end, and quantify both failure routes: kernel downgrade and ~2.4x any-pool retention inflation.
2. **Method (C4).** Geometry-aligned VSA (VSA256-FA4): a one-constant selector change — (4,4,4) to (4,8,8) — that maps the top-k mask 1:1 onto FA4's 256x128 sparse granularity, with variable-block-size handling for non-tile-aligned video shapes.
3. **Performance (C5-C7).** 8.93x attention-kernel speedup at 10% retention, 1.40x end-to-end at 720p vs dense BF16 (1.24x vs deployed VSA), flat peak memory — all in BF16. The identical geometry with native NVFP4 arithmetic is slower end-to-end, so the gain is precision-independent and is the dominant lever at this operating point.
4. **Quality accounting (C8-C10).** Under a 326-prompt paired VBench protocol with Holm correction, the aligned configuration is comparable to deployed VSA on measured dimensions, with three small significant losses (aesthetic -0.030, motion smoothness -0.018, background consistency -0.008) and one significant gain (dynamic degree +0.139). We report every delta and claim neither statistical indistinguishability nor non-inferiority.

The general lesson is that selector geometry is a first-class systems parameter: the granularity at which sparsity is decided must match the granularity at which the kernel can skip, or the budget is spent at the boundary between them.

---

## 2. Background and Related Work

**Trainable dynamic block-sparse attention.** VSA (arXiv:2505.13389) selects blocks per query from mean-pooled cube scores, combining a coarse cube-level attention branch with a fine block-sparse branch through learned gates; SpargeAttention2 (arXiv:2602.13515) selects from pooled attention maps at b_q=128, b_kv=64 with its own CUDA kernels; SLA and SLA2 (arXiv:2509.24006, arXiv:2602.12675) route attention mass between sparse and linear branches at b_q=128, b_kv=64 blocks. Each system chooses a block shape matched to its own kernel. VSA's tile-size ablation prefers B=64 over B=256 for pretraining loss (0.13162 vs 0.13375), which optimizes selection resolution for training quality rather than for the sparse granularity of a later platform's fastest kernel. Our work measures the inference-side cost of that choice on Blackwell and shows the trade is worth reversing at deployment.

**Static and sliding sparse patterns.** FPSAttention (arXiv:2506.04648) co-designs FP8 quantization with sliding-tile attention at the kernel's tile granularity on Hopper, executing through FlexAttention mask_mod and Triton. Its central premise — sparsity granularity must match kernel tile primitives — is the premise we test for *dynamic*, per-input selection, in pure BF16 and without retraining.

**Kernel-side sparse granularity.** The FA4 SM100 kernel family (CuTe-DSL, tcgen05 MMA, TMEM, 2-CTA pipelining; characterized second-hand in arXiv:2603.00040 §2.5/App. B — the FA4 paper itself has no located arXiv ID) exposes a block-sparse path that consumes per-row packed lists of retained 128-token key blocks at 256-token query rows, running online softmax over the gathered list. This granularity is a consequence of the hardware pipeline (q_stage x m_block = 2x128 query tokens per softmax row-state). Any selector that wants this kernel losslessly must produce masks at exactly this shape — the constraint our method satisfies by construction.

**Positioning.** We propose no new sparsity algorithm, kernel, or training method. The contribution is the measured demonstration that when the selector and the platform's best kernel are designed separately, aligning their geometries — one integer triple — is worth more end-to-end than the choice of either component's arithmetic precision. Prior video sparse-attention work reports kernel or FLOP reductions at the selector's native granularity; to our knowledge none isolates selector/kernel geometry alignment as a measured end-to-end lever.

---

## 3. Method: Geometry-Aligned VSA (VSA256-FA4)

### 3.1 Preserved algorithm family

The method keeps the deployed VSA algorithm and changes only tile geometry and the fine-branch kernel (implementation: `fastvideo/attention/backends/sparsefp4_vsa256_fa4.py`, backend enum `SPARSEFP4_VSA256_FA4_ATTN`):

1. **Tiling.** The video latent (t, h, w) is partitioned into (4,8,8)=256-token tiles (deployed VSA: (4,4,4)=64). Boundary tiles at non-divisible dimensions are padded, and per-tile valid-token counts (`variable_block_sizes`) are tracked exactly as in deployed VSA.
2. **Coarse branch.** Q, K, V are mean-pooled per tile (`fused_block_mean`); coarse scores q̄k̄ᵀ/√d drive a dense tile-level attention whose output, broadcast back to tokens and weighted by the model's learned `gate_compress`, forms the same gated compression branch as deployed VSA.
3. **Selection.** `fused_topk_mask` retains the top-k key tiles per (batch, head, query tile), with k derived from the same 0.90 sparsity target via `compute_topk`. The sparsity budget is identical; only the tile size over which it is expressed changes.

![Figure 1: selector-tile / kernel-granularity mismatch and its repair](figures/fig1_geometry_schematic.png)

*Figure 1. The mismatch and its repair (illustrative masks; the inflation factor is annotated from measurement). (a) The deployed VSA selector decides sparsity on 64-token tiles (blue) that straddle the FA4 kernel's 256x128 super-blocks (dark grid), keeping 10%. (b) Pooling that mask onto the kernel's blocks retains every super-block touched by any selected tile — retention inflates to ~25% (measured ~2.4x) and the sparsity budget is forfeited before the kernel runs. (c) Selecting directly on 256-token tiles (ours) maps 1:1 onto kernel granularity, keeping exactly 10%.*

### 3.2 Exact 1:1 mapping onto FA4 sparse geometry

The FA4 SM100 block-sparse interface consumes, per 256-token query row, packed count+index lists of retained 128-token key blocks (`BlockSparseTensorsTorch`, split into full and partial blocks). With 256-token selector tiles the mapping is exact:

- **Query side.** One selector tile equals one FA4 sparse query row (q_stage x m_block = 256 tokens); no pooling across query tiles.
- **Key side.** One selector tile equals exactly two 128-token FA4 key blocks (`repeat_interleave(2)` on the mask's key axis); retaining a tile retains its two halves and nothing else.

Retention is therefore exact: a 10% tile mask becomes a 10% kernel-block mask (measured median keep 0.1006, zero coarsening). The deployed geometry has no such route — mapping its 64-token-tile mask onto 256x128 super-blocks requires `any`-pooling over 4 query tiles x 2 key columns, which measured ~2.4x retention inflation.

Two details preserve correctness at real video shapes. Per-128-block valid-token counts split retained blocks into *full* blocks (128 valid tokens, fast path) and *partial* boundary blocks handled by a `mask_mod` predicate that trims padded tokens; on the BF16 kernel this predicate costs nothing measurable (3.89 vs 3.98 ms with/without, 720p geometry). The FA4 sparse path runs its ordinary online-softmax state machine — running row-max/row-sum with output-accumulator rescale — once per retained block over the gathered list, so the math is identical to dense attention restricted to retained blocks.

### 3.3 What is explicitly not changed

No training or fine-tuning; no new kernel (the FA4 block-sparse path pre-exists); no change to the sparsity target, selection scoring, gating, scheduler, or model weights. The configuration is a drop-in attention-backend swap (`FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_VSA256_FA4_ATTN`, `FASTVIDEO_SPARSEFP4_FINE=bf16`), designated **P4G** throughout.

---

## 4. Experimental Setup

- **Model and configuration.** Wan-AI/Wan2.1-T2V-1.3B-Diffusers; 50 denoising steps; CFG; seed 1234; VSA sparsity 0.90; resolutions 480x832x81 and 720x1280x81.
- **Hardware and software.** NVIDIA B200 (sm_100, one GPU of an 8x node), CUDA 13.0, torch 2.12.0+cu130, fastvideo-kernel 0.3.2, flash-attn-4 fork `hao-ai-lab/flash-attention-fp4@940bf7e5`; full environment receipts in `env/`.
- **Systems compared.** **P0** dense BF16 (FA4); **P2** deployed VSA at 0.90 sparsity (64-token tiles, Triton fine kernel); **P4G** geometry-aligned VSA256-FA4, BF16 fine branch, exact 10% kept. The parent study's NVFP4 arms (P1, P4) appear only in §6.2 as evidence of precision independence.
- **Timing protocol.** End-to-end medians of 3 steady-state generations of one prompt per arm (first generation excluded as warmup/JIT; dispersion across repetitions <0.5%); one process per arm; identical `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in every arm; no torch.compile, no CUDA graphs. Kernel latencies use CUDA events, medians of 50, pre-quantized inputs. No number is inferred from FLOPs.
- **Quality protocol.** 326 official VBench prompts (all prompts of the 7 dimensions scorable in this environment), paired by prompt/seed/config across arms; prompt-level percentile bootstrap (10,000 resamples), two-sided bootstrap p-values, Holm correction across the 7 VBench dimensions per contrast; unit of replication is the prompt. Pixel metrics (PSNR/SSIM/LPIPS) are computed as paired similarity-to-P0.

---

## 5. Results

### 5.1 The aligned mapping converts retention into kernel time almost losslessly

This experiment tests claim C5: that a 1:1 selector-to-kernel mask mapping realizes the sparsity budget at the kernel level. We sweep retained fraction at fixed shape (B=1, S=39936, H=12, D=128, BF16) and compare the FA4 block-sparse kernel against its dense counterpart (`tables/c8_performance_v2.md`):

| Configuration | Retained | Median ms | Speedup vs dense |
|---|---|---|---|
| Dense BF16 (FA4) | dense loop | 7.414 | 1.00x |
| Sparse BF16 (FA4 block-sparse) | 1.00 | 7.591 | 0.98x |
| Sparse BF16 | 0.50 | 3.815 | 1.94x |
| Sparse BF16 | 0.25 | 1.814 | 4.09x |
| Sparse BF16 | 0.10 | 0.830 | **8.93x** |

Two observations. The sparse machinery costs ~2% at full retention (7.591 vs 7.414 ms), so block-list iteration is close to free. At the 10% deployment operating point the kernel realizes 8.93x of the theoretical 10x — the aligned mapping loses almost nothing between decided sparsity and executed sparsity.

![Figure 2: kernel latency vs retained fraction](figures/fig2_kernel_scaling.png)

*Figure 2. Attention-kernel time vs fraction of attention kept, under the aligned 1:1 mapping (log-log; left-to-right reads as increasing sparsity). The measured FA4 block-sparse BF16 kernel (green) tracks the ideal dense-times-retention line (dashed grey) across the whole range, reaching 8.9x at the 10% deployment operating point. CUDA-event medians of 50, B=1 S=39936 H=12 D=128; per-point values in the table above.*

### 5.2 Geometry alignment delivers 1.40x end-to-end at 720p

This experiment tests claim C6: that the kernel-level gain survives to end-to-end generation. All arms share checkpoint, scheduler, steps, seed, guidance, and allocator configuration (`tables/c8_performance_v2.md`):

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

At 720p, alignment yields 1.40x vs dense and 1.24x vs deployed VSA (131.49/106.20), at identical peak memory and identical sparsity budget. At 480p, alignment repairs the deployed slowdown (0.938x becomes 1.022x); attention's smaller share of total time at 33k tokens bounds the remaining headroom (§6.3).

![Figure 3: end-to-end latency by system and resolution](figures/fig3_e2e_performance.png)

*Figure 3. End-to-end generation time (bar length; shorter is faster) at (a) 480p and (b) 720p for dense BF16, deployed VSA, and the geometry-aligned configuration (ours, green); labels give speedup vs dense. Wan2.1-T2V-1.3B, 50 steps, sparsity 0.90; medians of 3 steady-state repetitions (dispersion <0.5%), identical allocator configuration, no compilation; exact values in the tables above.*

### 5.3 Quality is comparable to deployed VSA, with enumerated trade-offs

This experiment tests claim C8: what the coarser selection geometry costs. Both arms are BF16 with the same sparsity budget and selection algorithm family; only tile geometry and fine kernel differ, so the paired contrast isolates the geometry effect. Paired P4G - P2 deltas over 326 VBench prompts, Holm-corrected (`tables/p4g_vs_p2_quality_bootstrap.md`); positive means P4G scores higher:

| VBench dimension | n | Mean Δ | 95% CI | Holm-sig @0.05 |
|---|---|---|---|---|
| subject_consistency | 72 | +0.0006 | [-0.0063, +0.0075] | no |
| background_consistency | 86 | -0.0078 | [-0.0107, -0.0052] | **yes** |
| temporal_flickering | 75 | -0.0029 | [-0.0058, -0.0000] | no (Holm p=0.14) |
| motion_smoothness | 72 | -0.0175 | [-0.0203, -0.0147] | **yes** |
| dynamic_degree | 72 | **+0.1389** | [+0.0694, +0.2222] | **yes** |
| imaging_quality | 93 | +0.0021 | [-0.0113, +0.0152] | no |
| aesthetic_quality | 93 | -0.0298 | [-0.0400, -0.0203] | **yes** |

The licensed interpretation is that P4G is comparable to deployed VSA on measured dimensions, with small significant trade-offs: it gives up a little aesthetic quality (-0.030), motion smoothness (-0.018), and background consistency (-0.008) while producing markedly more motion (dynamic degree +0.139); subject consistency, imaging quality, and temporal flickering do not separate after Holm correction. No non-inferiority margin was pre-registered, so no non-inferiority claim is made, and the arms are not described as statistically indistinguishable.

On pixel metrics against the dense reference (paired similarity-to-P0, n=326), P4G sits slightly farther from P0 than P2 (ΔPSNR -0.25 [-0.33, -0.17], ΔSSIM -0.006, ΔLPIPS +0.023; all p=0.0002). We report this without interpreting pixel proximity to the dense reference as quality (claim C9). The quality cost of sparsification itself — either sparse arm against dense P0 at 0.90 sparsity applied training-free — is shared by both arms and lies outside this paper's claims.

![Figure 4: forest plot of paired VBench deltas, P4G vs deployed VSA](figures/fig4_quality_forest.png)

*Figure 4. Paired VBench score change (P4G − P2) with 95% prompt-level bootstrap CIs, Holm-corrected across the 7 dimensions; per-dimension n shown with each label (unit of replication: prompt). Color encodes the corrected result: orange = significantly worse, green = significantly better, grey = not separable. Exact estimates and CIs in the table above; no non-inferiority claim is made.*

### 5.4 Operator-level sparsification error at the aligned geometry

For completeness, the controlled operator matrix (25 cells per resolution — 5 layers x 5 timesteps captured from a genuine P4G trajectory; `tables/c5_matrix_vsa256_exact10.md`) places the aligned-geometry sparsification error at rel-L2 0.192 (480p) and 0.218 (720p) against dense attention outputs at exact 10% retention (median keep 0.1006), with cosine similarity 0.985/0.979 (claim C10). These are attention-output perturbations, not end-video quality; §5.3 carries the end-to-end consequences.

---

## 6. Analysis

### 6.1 Where the 25 seconds come from

P4G's 25.3 s advantage over deployed VSA at 720p (131.49 to 106.20 s) rests on two mechanisms the geometry change unlocks at once. First, the kernel class changes: the 64-token geometry strands the deployed fine branch on Triton, while the 256-token geometry feeds the FA4 SM100 block-sparse path — the same kernel family as the dense baseline, with its full pipeline (2-CTA scheduling, TMEM, TMA block-gathered loads). Second, retention fidelity: the alternative route to FA4 — keep the 64-token selector, any-pool the mask — was measured in the parent study's coarsened arms, where retention inflated ~2.4x and 480p end-to-end landed at 48.7-53.1 s, worse than running dense. Only alignment makes the fast kernel and the true 10% budget available simultaneously.

The gap between the 8.93x kernel speedup and the 1.40x end-to-end speedup measures the Amdahl envelope: attention dominates but does not exhaust 720p inference, and the non-attention remainder (projections, FFN, norms, scheduler, VAE) bounds what any attention-side change can deliver.

### 6.2 The gain is precision-independent

The parent study composed this geometry with native NVFP4 attention arithmetic (arm P4) and measured 1.33x — slower than its BF16 twin (112.58 vs 106.20 s at 720p). At 10% retention the QK MMA no longer dominates kernel time, and FP4-specific integration costs — a validity predicate on the softmax-bound FP4 pipeline, per-call quantization — exceed FP4's residual arithmetic advantage (`P4_PERF_ROOT_CAUSE.md`). Geometry alignment attacks the cost that sparsification actually leaves behind, needs no arithmetic-format change, and at this operating point outweighs the choice of attention precision (claim C7).

### 6.3 Resolution scaling

The speedup grows with token count, as expected for a quadratic-cost target: 1.02x at 33k tokens (480p), 1.40x at 75k tokens (720p). The mechanism predicts larger gains at higher resolutions and longer durations, but we measured only these two points; the prediction is untested (H1).

---

## 7. Limitations

- **Single model, sparsity point, and seed.** All results use Wan2.1-T2V-1.3B at VSA sparsity 0.90; paper-scale quality uses one seed with a paired design, which mitigates but does not replace a multi-sample protocol. Other models, sparsity targets, and seeds are untested.
- **Training-free geometry swap.** The 256-token selector runs at inference without retraining gates or weights at the new geometry. VSA's own ablation indicates B=64 is the better training geometry (pretraining loss 0.13162 vs 0.13375 at B=256), so the small aesthetic/smoothness losses in §5.3 may reflect gates mildly mismatched to coarser tiles; whether fine-tuning at B=256 removes them is untested (H2).
- **Quality coverage.** 7 of 16 VBench dimensions are scorable in this environment; dynamic_degree is a near-binary metric with wide CIs at n=72.
- **Timing scope.** End-to-end medians come from 3 steady-state repetitions of one prompt per arm (dispersion <0.5%); no torch.compile or CUDA graphs anywhere, so all arms could shift under compilation. A supplementary observation in the parent study measured 1.34-1.39x forward-block gains from torch.compile, orthogonal to and not composed with these numbers.
- **Platform specificity.** The 256x128 granularity belongs to the FA4 SM100 kernel. The principle — align selector tiles to the target kernel's sparse granularity — should transfer, but the specific tile triple and measured magnitudes are Blackwell-specific (H3).
- **Two resolutions.** No measurements above 720p or beyond 81 frames.

---

## 8. Conclusion

A deployed dynamic block-sparse attention system skipped 90% of attention work yet returned 1.13x end-to-end at 720p and lost time at 480p, because its selection tiles could not reach the platform's fastest kernel without a kernel downgrade or a ~2.4x retention inflation. Changing one design constant — the selector tile, from (4,4,4) to (4,8,8), so the top-k mask lands 1:1 on FlashAttention-4's 256x128 sparse granularity — recovers 8.9x at the kernel and 1.40x end-to-end against dense (1.24x against deployed VSA), with the sparsity budget, memory footprint, kernel code, and model weights all unchanged, and with quality comparable to the deployed baseline up to three small, fully reported trade-offs.

For sparse-attention deployment the finding generalizes into a design rule: the granularity at which sparsity is *decided* must match the granularity at which the kernel can *skip*. In our measurements this single alignment decision returned more wall-clock time than the choice of attention arithmetic precision, and it composes with — rather than competes against — every other part of the inference stack.

---

## References

1. VSA: Faster Video Diffusion with Trainable Sparse Attention. arXiv:2505.13389.
2. FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Video Diffusion. arXiv:2506.04648.
3. SpargeAttention2. arXiv:2602.13515.
4. SLA: Sparse-Linear Attention. arXiv:2509.24006.
5. SLA2: Sparse-Linear Attention with Learnable Routing and QAT. arXiv:2602.12675.
6. Attn-QAT: 4-Bit Attention With Quantization-Aware Training. arXiv:2603.00040. (Source for the second-hand FA4 SM100 kernel characterization; the FA4 paper itself has no located arXiv ID.)
7. QuantSparse. arXiv:2509.23681.
8. SageAttention3: Microscaling FP4 Attention for Inference. arXiv:2505.11594.
9. FastVideo and fastvideo-kernel (VSA deployment). https://github.com/hao-ai-lab/FastVideo
10. flash-attention-fp4 fork (FA4 SM100 block-sparse interface). hao-ai-lab/flash-attention-fp4@940bf7e5.

All arXiv-cited claims were verified against primary sources on 2026-08-17 (`SOTA_RECOVERY_LIT_REVIEW.md`, verification appendix). No entry was generated from memory.

---

## Appendix A. Reproducibility and Artifact Map

All receipts live under `artifacts/sparsefp4_native/` in the FastVideo checkout (branch `exp/sparsefp4-paper-validation`).

| Paper section | Canonical source |
|---|---|
| Claims and wording boundaries | `paper_geometry/CLAIM_LEDGER.md` |
| Story and outline | `paper_geometry/PAPER_PLAN.md` |
| §5.1, §5.2 performance | `tables/c8_performance_v2.md` (raw: `raw/performance/perf_v2/`, logs: `logs/perf_v2/`) |
| §5.3 quality | `tables/p4g_vs_p2_quality_bootstrap.md` (stats: `configs/paired_stats_v2.py`, `raw/statistics/`) |
| §5.4 operator matrix | `tables/c5_matrix_vsa256_exact10.md` (raw: `raw/operator/c5b_exact10.jsonl`) |
| §6.1-6.2 root cause / predicate / allocator | `P4_PERF_ROOT_CAUSE.md` (`p4_maskmod_ab.json`, `quant_overhead.json`) |
| Kernel/interface audit | `CODE_PATH_AUDIT.md` |
| Environment | `env/` (GPU, driver, CUDA, torch, pip freeze, model revision) |
| Implementation | `fastvideo/attention/backends/sparsefp4_vsa256_fa4.py` |
| Figures | `paper_geometry/figures/` — `FIGURE_CONTRACT.md` (claim/evidence-role/archetype per figure), `data/*.csv` (source data transcribed verbatim from canonical tables; scripts hardcode nothing), `make_figures.py`, `provenance_manifest.json`, PNG 300 dpi + vector PDF (Type-42 fonts). Figure 1's mask panels are illustrative; its ~2.4x inflation annotation is the measured value. |

Reproduction outline: (1) install the pinned environment per `env/`; (2) select the backend with `FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_VSA256_FA4_ATTN` and `FASTVIDEO_SPARSEFP4_FINE=bf16`; (3) run the standard FastVideo Wan2.1-T2V-1.3B generation config (50 steps, seed 1234, VSA sparsity 0.90) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; (4) compare against `VIDEO_SPARSE_ATTN` (P2) and the dense FA4 baseline (P0) under the protocol of §4.

## Appendix B. Claim Boundaries Inherited From the Parent Study

- The 1.40x end-to-end speedup belongs to the **BF16** P4G configuration; it must never be attributed to FP4 or any quantized arm.
- "Comparable quality" is bounded to the 7 measured VBench dimensions with the enumerated significant trade-offs of §5.3; "statistically indistinguishable" and non-inferiority claims are not licensed.
- The 8.93x kernel figure is a *sparsity* speedup vs dense BF16 at matched shape; it is not an end-to-end claim.
- End-to-end numbers are medians of 3 steady-state repetitions of one prompt per arm under one allocator configuration; nothing is inferred from FLOPs.
