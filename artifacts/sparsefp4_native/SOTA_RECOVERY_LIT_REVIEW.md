# SOTA Recovery Literature Review: Training-Based Recovery for Native Sparse NVFP4 Attention ("DQ-VSA")

**Date:** 2026-08-17
**Purpose:** Primary-source-only review of training-based recovery (distillation / QAT) methods relevant to
recovering the NVFP4-QK degradation observed in our P4 operator (VSA256 block-sparse attention, 10% retained
blocks, 256-token tiles, BF16 selector, native NVFP4 packed-E2M1 Q/K with per-16 E4M3 scales, BF16 PV) on
Wan2.1-T2V-1.3B / NVIDIA B200. Paired 326-prompt VBench showed NVFP4 QK vs. its BF16 twin (P4 vs. P4G):
imaging_quality −0.101 (CI [−0.112, −0.090]), dynamic_degree −0.250 (CI [−0.347, −0.153]); same signature
for dense NVFP4 (P1 vs P0): imaging_quality −0.144, dynamic_degree −0.306, aesthetic −0.055.

**Verification policy:** Every paper below was fetched from its arXiv abstract/HTML page on 2026-08-17.
Claims are cited as `(arXiv:ID, §section)`. Items that could not be fetched are explicitly marked
**COULD NOT VERIFY / NOT FOUND**. All 8 required papers were successfully fetched and verified; 4
supplementary papers were also fetched and verified (SageAttention3, SLA, NVIDIA NVFP4 pretraining, SageBwd).

---

## Summary Table

| Paper (arXiv) | Attention | Sparsity | Quant format | Training? | Teacher/student | Loss | Model(s) | Recovery result |
|---|---|---|---|---|---|---|---|---|
| VSA (2505.13389) | Block-sparse, 64-tok tiles, coarse+fine, learned gates | Dynamic top-K (K=32), trainable, ~87.5–91.2% | None (BF16) | Fine-tune (4k steps) + DMD2 distill | DMD2: full-attn teacher, sparse few-step student | Flow-matching; DMD2 losses unchanged | Wan2.1 1.3B/14B; 60M–1.4B pretrain | VBench 82.77 vs 83.63 full f.t.; 50.9× distill speedup |
| FPSAttention (2506.04648) | STA sliding-tile local window, 3D tiles | Static local window, timestep-scheduled W(t) | FP8 (E4M3): per-3D-tile QK, per-channel V, per-tensor P | QAT (full fine-tune, ~7 days ×512 H20) | None (task loss only) | Flow-matching (rflow) with quant+sparse in fwd | Wan2.1 1.3B/14B | VBench total 0.8160 vs 0.8019 base; +1.8% |
| QuantSparse (2509.23681) | Full attn + SVG static sparse mask at inference | Static (SVG), 15–40% density | INT W4A8/W6A6 linear layers (not attention math) | PTQ calibration + attention distillation (no weight training) | FP model attention maps → quantized model | L_quant + λ·MSE(pooled attn) + λ·MSE(salient-query attn) | Wan2.1 1.3B/14B, HunyuanVideo 13B | 20.88 vs 16.85 PSNR (Q-VDiT); near-lossless VBench |
| SLA2 (2602.12675) | Sparse (128×64 blocks) + linear branch, learned router R + learned mix α | Dynamic learnable top-k (3–5%), 97% sparsity | INT8/FP8 QK+PV in sparse branch (SageAttention2++ scheme), QAT | 2-stage: router init (MSE vs full attn) + e2e diffusion fine-tune, 500 steps | None (diffusion loss) | Stage1: MSE(FullAttn, SLA2); Stage2: diffusion loss | Wan2.1 1.3B/14B | ≥ full-attention VBench at 97% sparsity; 18.6× attn speedup |
| SpargeAttention2 (2602.13515) | Block-sparse (128×64), pooled-QK masker | Dynamic hybrid Top-k ∪ Top-p, 95% | None (BF16) | Velocity distillation fine-tune, 500 steps | Frozen full-attn teacher → sparse student (same init) | L_VD = ‖u_sparse − u_full‖² on identical (x_t,c,t) | Wan2.1 1.3B/14B | Beats full attn on IQ/AQ at 95% sparsity; 16.2× attn speedup |
| Attn-QAT (2603.00040) | Dense FlashAttention (block-sparse supported in B200 kernel) | None (dense) | NVFP4 (E2M1, per-16 E4M3 scales): QK and PV fake-quant; STE | QAT fine-tune (3–4k steps) | None (task loss) | Rectified flow-matching (diffusion); CE (LLM) | Wan2.1 1.3B/14B; Qwen3-14B, Llama-3.1-70B | VBench 0.8252 vs 0.8267 BF16 (1.3B); recovers IQ fully, DD partially |
| NVFP4 QAD (2601.20088) | N/A (linear-layer GEMM quant; attention kept BF16) | None | NVFP4 weights+activations, all GEMMs | Distillation (QAD) vs QAT | BF16 teacher → NVFP4 student, KL divergence | KL(p_teacher ‖ p_student), T=1 | Nemotron family LLMs/VLMs 7–49B | Near-BF16; QAD ≫ QAT esp. for RL-trained models |
| 6Bit-Diffusion (2603.18742) | Full attention (linear layers quantized, not attn math) | Block-skip caching (TDC) | Dynamic NVFP4/INT8 mixed per-layer routing (W4A6 avg) | Training-free (PTQ + linear predictor calibration) | None | None | CogVideoX 2B/5B | W4A6 ≈ FP16 VBench; 1.92× speedup, 3.32× memory |
| SageAttention3 (2505.11594) † | Dense FlashAttention | None | NVFP4 QK+PV, two-level P scaling, Q/K smoothing | Training-free (inference); SageBwd = INT8 trainable | None | N/A | CogVideoX, HunyuanVideo, Mochi, Flux, SD3.5 | "Almost no" loss claimed; Attn-QAT (2603.00040) measured 0.7834 vs 0.8267 VBench on Wan1.3B |
| SLA (2509.24006) † | Sparse (64×64) + linear branch, heuristic top-k split | Dynamic top-k_h=5% + skip bottom k_l=10%, 95% | None (BF16) | Fine-tune 2000 steps, bs 64 | None | Diffusion loss | Wan2.1-1.3B, LightningDiT | ≈ full attention at 95% sparsity; 13.7× attn speedup |
| NVFP4 pretraining (2509.25149) † | Attention kept BF16/FP32 (only linear GEMMs in NVFP4) | None | NVFP4 all 3 GEMMs; RHT, 2D weight scaling, stochastic rounding | Native FP4 pretraining from scratch | None | CE | 12B hybrid Mamba-Transformer, 10T tokens | Matches FP8 loss within ~1.5%; MMLU-pro 62.58 vs 62.62 |
| SageBwd (2603.02170) † | Dense FlashAttention | None | INT8 fwd+bwd (6 of 7 matmuls), dP kept FP16 | Trainable low-bit attention (pretraining study) | None | CE | LLM pretraining | Matches FPA in pretraining with QK-norm + reduced tokens/step |

† = supplementary paper (not on the required list) verified from primary source.

---

## 1. VSA: Faster Video Diffusion with Trainable Sparse Attention — arXiv:2505.13389 [VERIFIED from primary source]

*Zhang, Chen, Huang, Lin, Liu, Stoica, Xing, Zhang (UCSD/MBZUAI/Berkeley). This is the attention our P4/P4G operator generalizes (we use 256-token tiles instead of 64).*

1. **Attention mechanism.** Hierarchical two-stage sparse attention. A coarse stage mean-pools (4,4,4) cubes (tile size B=64) of the video latent into cube-level Q_c, K_c, V_c, computes dense cube-to-cube attention, and outputs O_c; a fine stage computes token-level block-sparse attention only inside top-K selected cubes. Final output O = O_c⊙G_c + O_f⊙G_f with learned linear gate projections G (arXiv:2505.13389, §2.2). Tile ablation: B=64 optimal; B=256 ((4,8,8) cubes — our geometry) loses quality: pretraining loss 0.13375 (B=256) vs 0.13162 (B=64) (§3.1, Table 1(d)).
2. **Sparse mechanism.** Dynamic, data-dependent, trainable. Row-wise Top-K on the coarse attention matrix A_c selects K=32 KV tiles per query tile (~87.5% sparsity at 16K tokens; 91.2% on Wan-1.3B 480P). Selection is not differentiable itself but the coarse stage is trained end-to-end because its output O_c contributes to the final output through the gate (§2.2). Data-dependent selection beats fixed local patterns (Table 1(b)).
3. **Quantization.** None — everything in BF16/FP16. VSA is a pure sparsity method.
4. **Training required?** Yes. Either pretraining from scratch or sparse-adaptation fine-tuning of a full-attention checkpoint; direct replacement without adaptation is unstable (§2.3).
5. **Teacher/student.** Only in the Sparse-Distill pilot: DMD2 distillation where the *student* is a few-step generator with VSA (80% sparsity) and the real/fake score models (teacher side) remain full attention (§2.3, §C.6).
6. **Losses.** Sparse adaptation: standard flow-matching loss (LogitNormal(0,1) timestep sampler) (§C, Table 2(b)). Sparse-Distill: unchanged DMD2 losses ("we preserve the original distillation loss and all hyperparameters") (§2.3).
7. **Sparsity curriculum.** Yes, explicit annealing: initialize K = L/B (equivalent to full attention), coarse gate G_c initialized to zero, fine gate removed (=1). For Wan-1.3B adaptation: 50 steps full attention first, then decrease attended cubes by 10 (Top-K by 4) every 50 steps until Top-K=32 (§2.3, §C.5). "Progressive decay schedule enables a smooth transition and mitigates training instability" (§C.5).
8. **QAT semantics.** N/A (no quantization). Kernel implements block-sparse forward *and backward* in ThunderKittens, retaining 85% of FA3 MFU (§2.4).
9. **Steps/compute.** Wan-1.3B adaptation: 4,000 steps on 32 H200 (DDP, per-GPU bs 1, grad-accum 2). Wan-14B: 4,000 steps on 64 H200, global bs 64. Ablations/scaling: ~90k H200-hours total (§C.5, §1).
10. **LR/optimizer.** Adaptation LR 1e-5; pretraining AdamW β=(0.9,0.95), wd 1e-2, LR 6e-4 (§C.5, Table 2(b)).
11. **Dataset.** Wan-1.3B: 80,000 synthetic videos generated by Wan-14B, 448×832×61f. Wan-14B: 200,000 synthetic 768×1280×77f videos (§C.5). Pretraining ablations: Vchitect-T2V-Dataverse (§3.1).
12. **Models.** Wan2.1 1.3B and 14B; scratch DiTs 60M–1.4B.
13. **Recovery.** VBench: Ori-Wan 82.56, Full-attn fine-tune 83.63, VSA fine-tune 82.77 total (§3.3, Table 3a). Human eval: VSA ≥ SVG at higher sparsity; 14B human-pref parity with full attention. Sparse-Distill: 50.9× denoising speedup, no quality drop (§3.3).
14. **Relevance to DQ-VSA.** This is our exact sparsity substrate. Key takeaways: (a) synthetic teacher-generated data (Wan-14B outputs) works well and even *boosted* quality — directly applicable to motion-diverse distillation data; (b) 4k steps @ LR 1e-5 with flow-matching loss suffices to adapt Wan-1.3B to a drastically changed attention operator; (c) sparsity annealing exists but is only needed when *changing* geometry — since we freeze VSA256 geometry (already trained), no curriculum on sparsity should be needed, only on precision; (d) VSA itself found 256-tile quality inferior to 64-tile at pretraining, consistent with our P4G being a mild quality compromise even before quantization; (e) VSA is compatible with DMD2 distillation without loss changes — evidence that a distillation objective tolerates an approximate attention operator in the student.

---

## 2. FPSAttention: Training-Aware FP8 and Sparsity Co-Design — arXiv:2506.04648 [VERIFIED from primary source]

*Liu, Zhang, et al. (Monash/DAMO/ZJU). Page dated August 11, 2026 (updated version). The closest prior work in spirit: joint quantization+sparsity QAT for Wan, but FP8 (not FP4) and static local-window sparsity (not dynamic top-k).*

1. **Attention mechanism.** 3D bidirectional attention with STA (Sliding Tile Attention): each query tile attends only to key tiles within a local 3D window W(u) (arXiv:2506.04648, §3.1, Eq. 2–3). Executed via FlexAttention mask_mod/score_mod + Triton fused kernels on Hopper (§3.4, App. A).
2. **Sparse mechanism.** Static, local, non-trainable pattern (window sizes are hyperparameters), but *timestep-scheduled*: window size W(t) varies across denoising steps (§3.3). Best window (6,6,1) gives 5.16× kernel speedup (§4.2, Table 4). Not top-k, not learned.
3. **Quantization.** FP8 (E4M3-style; max-scaled). Per-3D-tile scales for Q and K (s = max|X|/M_FP8max per tile, tile up to (24,32,32) and scheduled g(t)); per-channel scales for V ("keeping fine granularity for V is critical"); fixed scalar 1/448 for P following SageAttention2 (§3.2, Eq. 4–5). Output dequantized to BF16. Weights: the training config lists "Data Type: fp8, Quantization: True" for training (App. F, Table 8) — attention QK/PV activation quantization is the paper's focus.
4. **Training required?** Yes — QAT ("training-aware co-design"). Explicit motivation: training-free FP8+STA combination collapses (VBench total 0.6325 vs baseline 0.8019, −21.1%; catastrophic on semantic dims: Human Action 0.02, Multiple Objects 0.0) while FPSAttention reaches 0.8160 (+1.8%) (App. C, Table 5).
5. **Teacher/student.** None. Straight fine-tuning with the quantized+sparse operator in the forward pass.
6. **Losses.** Standard rectified-flow diffusion loss (scheduler "rflow-wanx", 1000 timesteps, logit-normal sampling), no distillation term (App. F, Table 8). Training loss starts ~15% higher than BF16 baseline, converges to <2% gap after 2,000 steps "through adaptive learning rate scheduling and gradient accumulation" (§4.2, Fig. 7).
7. **Sparsity curriculum.** Not a progressive anneal, but a *denoising-timestep schedule*: S(t)=[g(t),W(t)] piecewise over three regimes with thresholds α₁D, α₂D — early steps [g_coarse, W_sparse], mid steps [g_fine, W_dense], late steps [g_intermediate, W_medium] — because mid-denoising steps are most error-sensitive (§3.3, Eq. 6, Fig. 5). The same schedule is used during both training and inference to keep train/test consistent.
8. **QAT semantics.** Quantization is applied in the forward during training (the model "adaptively compensates for joint quantization-sparsity errors" during training; §3.3). The paper does not detail STE/backward-precision mechanics or FlashAttention-backward handling (implemented via FlexAttention). No explicit fake-quant/backward-recomputation discussion.
9. **Steps/compute.** Wan2.1-14B: 64 nodes × 8 H20 for 7 days; 1.3B: 16 nodes × 8 H20, ~7 days (§4 Implementation; App. B). Loss converges to baseline trend by ~2,000 steps (§4.2).
10. **LR/optimizer.** LR 5e-6, weight decay 1e-4, grad clip 1.0, warmup 200, EMA 0.99, Adam ε=1e-15 (App. F, Table 8).
11. **Dataset.** Curated high-quality video: subtitle removal, black-border cropping, monochrome exclusion, Q-Align>3.5, Aesthetic>2.0, optical-flow magnitude 0.05–2.0 filtering, dedup; 480p, 16fps, 5s clips (App. B "Dataset").
12. **Models.** Wan2.1-1.3B and 14B.
13. **Recovery.** Wan-1.3B VBench total: baseline 0.8019 → training-free FP8+sparse 0.6325 → FPSAttention 0.8160. Notably Dynamic Degree *improves* (0.3014 → 0.4195) and Imaging Quality improves (0.6708 → 0.7103) after QAT (App. D, Table 6). Kernel 7.09×, E2E 4.96× at 720p on 14B (§1, Table 1).
14. **Relevance to DQ-VSA.** The strongest existing precedent that (a) quantization error in attention interacts multiplicatively with sparsity ("sparsity prioritizes high-magnitude scores; quantization disproportionately errs on exactly those values", §1) — matching our P4-vs-P4G finding that quantized-sparse loses more than dense-quantized on some axes; (b) plain diffusion-loss QAT with the exact inference operator in the forward pass recovers and even exceeds baseline, *including dynamic_degree* — the dimension we lose most; (c) their optical-flow-filtered training data (flow magnitude 0.05–2.0) is explicit dataset guidance for preserving motion; (d) timestep-aware precision scheduling is empirically motivated (mid-denoising most sensitive) and transferable to NVFP4-QK scheduling. Caveats: FP8 not FP4 (much larger dynamic range), static local sparsity not VSA top-k, Hopper not Blackwell, and it changes the sparsity geometry during training (our constraint is frozen geometry).

## 3. QuantSparse — arXiv:2509.23681 [VERIFIED from primary source]

*Feng, Yang, Qin, et al. (ICT-CAS/ETH/SJTU). Unified INT-quantization + sparse-attention compression via calibration-time attention distillation and inference-time residual correction.*

1. **Attention mechanism.** Full softmax attention; sparsification applied via SparseVideoGen (SVG) static per-head spatial-temporal masks at inference (chosen as the most robust baseline sparsifier, App. D). Attention math itself stays high precision (they separately show compatibility with SageAttention INT8 attention, App. J).
2. **Sparse mechanism.** Static (SVG pattern), not trainable, 15–40% attention density. Plus a novel inference-time correction: Second-Order Sparse Attention Reparameterization (SSAR) caches Δ_quant (full-minus-sparse attention residual) and its second-order difference, projected by SVD onto top-r=16 temporally stable components, refreshed every 5 timesteps (arXiv:2509.23681, §3.3, Eq. 14–16).
3. **Quantization.** INT quantization of *linear layers* (weights/activations): W4A8 and W6A6; channel-wise weight scales, dynamic token-wise activation scales, learnable rotations à la QuaRot/FlatQuant (§4.1, App. C). Not FP4 and not attention-QK quantization.
4. **Training required?** Calibration only — block-wise PTQ where quantization parameters (channel-wise scale, rotation matrix, quant scale) are *learned* against distillation losses, but model weights are frozen. Extremely cheap: 0.64 h for Wan-1.3B, 2.6 h for Wan-14B on one A800 (App. I, Table 14).
5. **Teacher/student.** FP model is the teacher supplying attention-map targets; the quantized model is the student. Only quantization parameters are optimized.
6. **Losses.** Multi-Scale Salient Attention Distillation (MSAD): L_distill = L_quant + λ_g·MSE(A_global^FP ‖ A_global^quant) + λ_l·MSE(A_local^FP ‖ A_local^quant), where A_global is the softmax attention over average-pooled Q,K (stride s=128, O(L²/s⁴) memory) and A_local is high-resolution attention restricted to top-k=256 salient query tokens ranked by aggregate received attention s_j = Σ_{h,i} A_{h,i,j} (§3.2, Eq. 6–9). λ chosen to match loss magnitudes (1e-4 for Wan) (App. C).
7. **Sparsity curriculum.** None (static mask, no schedule).
8. **QAT semantics.** No STE / weight training; gradient descent only on quant params (AdamW, cosine schedule) with 15 epochs per transformer block, block-wise reconstruction (App. C). SSAR is purely inference-time.
9. **Steps/compute.** 20 random calibration samples, 15 epochs per block; ≤0.5–1.6% overhead vs naive PTQ (App. I).
10. **LR/optimizer.** AdamW; LR 5e-3 for channel-scale/rotation, 5e-2 for quant scales; cosine schedule (App. C).
11. **Dataset.** 20 randomly generated calibration samples; evaluation on OpenSORA prompt sets and 8 VBench dimensions (§4.1, App. B).
12. **Models.** Wan2.1-1.3B/14B, HunyuanVideo-13B, Hunyuan-DiT (image).
13. **Recovery.** HunyuanVideo W4A8 @15% density: PSNR 20.88 vs Q-VDiT 16.85; VQA 81.19 ≈ FP 81.23; 3.68× storage, 1.88× E2E speedup (§4.2, Table 2). Ablation: attention distillation lifts PSNR 14.35→18.72; MSAD ≈ full-attention-map distillation at 40× less memory (App. H, Table 13). Key measured fact: joint quant+sparse attention shift (0.685 MSE) ≫ sum of individual shifts (0.216 + 0.134) — the "amplified attention shift" (App. F, Table 8).
14. **Relevance to DQ-VSA.** (a) Formal + empirical case that sparsity amplifies quantization-induced attention shift super-additively (Prop 3.1, Eq. 5: extra O(‖ε‖_F·‖M‖₀) term) — the theoretical frame for our P4 result; (b) attention-map distillation (pooled global + salient-local MSE) is a memory-feasible auxiliary loss we could add on top of output-level distillation, supervising the NVFP4 QK logits directly against BF16 logits inside retained blocks; (c) shows meaningful recovery is possible *without touching model weights* — a cheap first rung on the recovery ladder before full QAT; (d) their pooled-attention loss operates at exactly the granularity of a VSA selector, suggesting a "selector-consistency" loss variant. Caveat: their quantization is INT linear-layer W/A, not FP4 attention-QK, so recovery magnitudes don't transfer directly.

---

## 4. SLA2: Sparse-Linear Attention with Learnable Routing and QAT — arXiv:2602.12675 [VERIFIED from primary source]

*Zhang, Wang, Jiang, Zheng, Jiang, Stoica, Chen, Zhu, Gonzalez (Tsinghua/Berkeley). The first paper to combine trainable sparse attention with attention-QAT for video diffusion.*

1. **Attention mechanism.** Hybrid: O = α⊙O_s + (1−α)⊙O_l with learnable per-row α∈[0,1]; O_s is block-sparse softmax attention, O_l is linear attention (φ=softmax feature map) over the complementary blocks; both fused in one FlashAttention-style kernel (arXiv:2602.12675, §3, Eq. 13–14; Alg. 2). Blocks b_q=128, b_kv=64 (§9.1).
2. **Sparse mechanism.** Dynamic, **learnable router** R: mean-pool Q,K per block, apply learned projections proj_q, proj_k (d×d), compute P_c = proj_q(Q̄)proj_k(K̄)ᵀ, then row-wise Top-k (k%=3–5%) (§4, Eq. 15–16). During Stage-1 training Top-k is replaced by differentiable SoftTop-k: σ(P_c/τ + λ_i) with λ_i solved by binary search so each row sums to k%·N/b_k; gradients via reparameterization (§6, Eq. 17; τ=0.1). Inference uses hard Top-k (§7).
3. **Quantization.** Sparse branch computed in low-bit (INT8 or FP8) following SageAttention2++: quantize Q,K → dequant(Q̂K̂ᵀ), softmax, quantize P,V → dequant(P̂V̂); K smoothing (K−colmean(K)) retained (§5; Alg. 2 lines 2, 13, 17). Low-bit adds ~1.3× kernel speedup on top of sparsity (§9.4). **Not FP4.**
4. **Training required?** Yes — quantization-aware fine-tuning. Ablation: without QAT (fine-tune in high precision, quantize at inference) quality drops: IQ 65.28, AQ 61.85, VR 0.0850 vs SLA2 66.64/64.62/0.1039 (§9.4, Table 2).
5. **Teacher/student.** None; Stage-1 uses full-attention outputs as regression targets for router init, Stage-2 is plain end-to-end diffusion loss (Alg. 1).
6. **Losses.** Stage 1: L = MSE(FullAttn(Q,K,V), SLA2(Q,K,V,k%,R,α)) trained on sampled Q,K,V tensors from every layer/timestep, for k%∈{5,4,3} (Alg. 1). Stage 2: end-to-end diffusion loss over all model params Θ and α (router R frozen), with hard Top-k routing matching inference (§6, §8 Insight 2).
7. **Sparsity curriculum.** Multi-sparsity Stage-1 training (5%→4%→3% variants) but no in-training anneal reported; two-stage design is for router-initialization stability and train/inference consistency (§8).
8. **QAT semantics.** Explicit: "during training, we use low-bit attention only in the forward pass, while the backward pass remains fully in FP16" (§5). Forward quantizes Q,K then P,V within the FlashAttention loop; backward uses original FP16 inputs and forward output O_s: dQ,dK,dV = backward(dO_s, O_s, Q, K, V) (§5). (Note: unlike Attn-QAT below, no low-precision recomputation of P in backward and no separate high-precision O — SLA2's INT8/FP8 error is small enough that this simpler recipe suffices; Attn-QAT shows it fails at FP4.)
9. **Steps/compute.** 500 fine-tuning steps; batch size 64 (1.3B) / 15 (14B) (§9.1).
10. **LR/optimizer.** Not stated in the fetched text.
11. **Dataset.** Private 3,000 videos (~5 s each) from public sources; captions by Qwen3-VL-Flash (§9.1).
12. **Models.** Wan2.1-T2V-1.3B-480P and 14B-720P.
13. **Recovery.** At 97% sparsity + low-bit sparse branch: 1.3B IQ 66.64 / OC 21.42 / AQ 64.62 / VR 0.1039 vs Full-Attention 63.67/20.27/64.41/0.1084 — exceeds full attention on IQ/OC (§9.2, Table 1). 18.6–18.7× attention speedup on RTX5090, 2.30×/4.35× E2E (§9.3). Learnable router ≫ heuristic Top-k router (Table 2).
14. **Relevance to DQ-VSA.** Closest published *combination* of trainable dynamic-top-k sparsity + attention QAT for Wan: proves the two compose under a 500-step fine-tune with quantized-forward/FP16-backward semantics. Differences from our target: INT8/FP8 (not NVFP4), selector is retrained (ours frozen), adds a linear branch we don't have, block 128×64 not 256×256. For DQ-VSA it supports: (a) short (≈500-step) QAT fine-tunes can suffice when the operator error is moderate; (b) BF16-backward + quantized-forward is a viable default to try before the heavier Attn-QAT recipe; (c) freezing the router during Stage-2 (their choice) parallels our frozen-selector constraint.

## 5. SpargeAttention2 — arXiv:2602.13515 [VERIFIED from primary source]

*Zhang, Jiang, Xiang, Feng, Hu, Xi, Chen, Zhu (Tsinghua/Berkeley). Trainable block-sparse attention with hybrid Top-k+Top-p masking and a velocity-distillation objective. No quantization.*

1. **Attention mechanism.** Block-sparse FlashAttention (b_q=128, b_kv=64); block mask from pooled attention P̄ = Softmax(pool(Q)pool(K)ᵀ/√d) (arXiv:2602.13515, §2.2, §4.3, Alg. 1). CUDA fwd+bwd kernels.
2. **Sparse mechanism.** Dynamic, per-row hybrid mask: keep block (i,j) if j ∈ Top-k(P̄_i, k%) ∪ Top-p(P̄_i, p%) (§4.1, Eq. 9). Rationale: Top-k fails on near-uniform rows (fixed count captures little mass), Top-p fails on skewed rows (attention sinks satisfy threshold with too few blocks) (§3.2, Case 1, Tables 1). Calibrated to ~95% sparsity: 1.3B k=0.03/p=0.2; 14B k=0.03/p=0.16 (App. A). Masker is heuristic (not learned); the *model* is trained to the mask.
3. **Quantization.** None (BF16).
4. **Training required?** Yes — training-free variant collapses (1.3B: IQ 53.18, VQA 20.40 vs trained 67.68/86.73) (§5.4, Table 6).
5. **Teacher/student.** Yes: frozen full-attention model = teacher; sparse-attention model = student; **identical initialization**, differing only in the attention operator; identical inputs (x_t, t, c_txt) per step (§4.2, Alg. 2).
6. **Losses.** Velocity distillation: L_VD = E[‖u_sparse(x_t,c,t) − u_full(x_t,c,t)‖²]. Standard diffusion loss is *not* used; fine-tuning data serve only to construct noisy inputs x_t (§4.2). Motivation (Case 3, §3.2): with mismatched fine-tuning data, even *full-attention* diffusion-loss fine-tuning degrades the model (1.3B AQ 0.6441→0.6183, VQA-a 81.28→75.45; 14B similar, Table 3) because the diffusion loss forces fitting the (lower-quality) fine-tune distribution. Ablation "−VD" (diffusion loss instead): IQ 67.23→ but AQ 63.34 vs 65.05 and VQA 85.05 vs 86.73 — distillation wins consistently (Table 6).
7. **Sparsity curriculum.** None reported; fixed k%,p% from step 0.
8. **QAT semantics.** N/A (no quantization). Block-sparse fwd/bwd in CUDA on FlashAttention (§4.3).
9. **Steps/compute.** 500 steps (main results); 100 steps for 14B ablations (App. A).
10. **LR/optimizer.** Not stated in fetched text. Batch size 64 (1.3B/480p), 16 (14B/720p) (App. A).
11. **Dataset.** Same private 3,000-video corpus as SLA2 (5 s clips, public sources, Qwen3-VL-Flash captions) (§5.1).
12. **Models.** Wan2.1-1.3B-480p, Wan2.1-14B-720p.
13. **Recovery.** 95% sparsity: 1.3B IQ 67.68 vs full 63.67, AQ 65.05 vs 64.41, VR 0.1010 vs 0.1084, VQA-a 83.86 vs 81.28; 16.2× attention speedup, 2.3×/4.7× E2E; beats SLA, VSA, VMoBA at equal-or-higher sparsity (§5.2–5.3, Tables 4–5).
14. **Relevance to DQ-VSA.** The cleanest evidence for the *loss choice* in DQ-VSA: a same-initialization frozen-BF16-teacher → quantized-operator-student velocity/flow-matching-output MSE distillation (i) preserves the original model's behavior, (ii) is robust to fine-tuning-data distribution mismatch (we don't have Wan's pretraining data either), and (iii) empirically beats plain diffusion-loss fine-tuning. Our student would keep VSA256+NVFP4-QK in the forward; the teacher runs BF16 dense (or BF16 P4G if we want to isolate quantization recovery from sparsity effects). Also validates that 500-step-scale adaptation budgets can be enough when only the attention operator changes.

---

## 6. Attn-QAT: 4-Bit Attention With Quantization-Aware Training — arXiv:2603.00040 [VERIFIED from primary source]

*Zhang, Noto, Tan, Jiang, Lin, Zhou, Zhang (UCSD/hao-ai-lab — same group as VSA/FastVideo). The single most relevant paper: NVFP4 attention QAT on Wan2.1, with a FlashAttention-4 CuTe-DSL B200/B300 kernel. The FP4 degradation signature they measure matches ours.*

1. **Attention mechanism.** Dense FlashAttention (Triton training kernels; SageAttention3-derived CUDA inference kernel on RTX5090; CuTe-DSL FA4-based kernel on B200/B300). The B200/B300 kernel "supports block-sparse attention but not paged attention" (arXiv:2603.00040, §5) — sparse support exists in the kernel but no sparse experiments are reported.
2. **Sparse mechanism.** None evaluated (dense). Future-work statement: "We are also working on FP4 attention with quantization-aware distillation (QAD) and sparse attention" (§5) — i.e., our exact DQ-VSA target is their declared future work, not a delivered result.
3. **Quantization.** NVFP4: E2M1 elements, per-1×16 blocks, E4M3 scale s=max|X|/6 (§2.1, Eq. 1). Both matmuls quantized: QK (Q̂,K̂ NVFP4) and PV (P̂,V̂ NVFP4) in the plain variant (Alg. 1). No outlier-mitigation heuristics (no Q/K smoothing, no two-level P) — QAT is shown to make them redundant (§3.2, Exp. 4–6: adding SmoothK or two-level-P to QAT changes VBench only marginally). On B200 they find quantizing PV is *slower* than BF16 PV due to the softmax-bound critical path, so the deployed B200/B300 config is NVFP4 QK + FP8 or BF16 PV (§2.5, App. B.2) — exactly our P4 configuration (NVFP4 QK, BF16 PV).
4. **Training required?** Yes — QAT. Training-free NVFP4 on Wan-1.3B: VBench overall 0.7785 vs BF16 0.8267; SageAttention3 0.7834. Attn-QAT: 0.8252 (§3.2, Table 2).
5. **Teacher/student.** None — plain task-loss QAT (QAD explicitly future work).
6. **Losses.** Standard rectified flow-matching loss for diffusion (App. C.1); cross-entropy for LLM continued pretraining/SFT (§3.3).
7. **Sparsity curriculum.** N/A.
8. **QAT semantics — the paper's core contribution.** (i) Forward (training): fake-quantize Q,K,V once (φ⁻¹(φ(·))), run BF16 matmuls; fake-quantize P̃ per tile before PV (Alg. 2). (ii) Backward uses STE (§2.2, Eq. 7). Two stability requirements identified: **(R1)** the P recomputed in backward from the stored logsumexp L must be fake-quantized to the *same low precision* as forward (Alg. 3 line 11) — omitting it keeps final VBench similar but makes gradient norms much noisier (Exp. 8, Fig. 3a); **(R2)** FlashAttention's identity P_iᵀdP_i = dO_iᵀO_i breaks when forward O uses quantized P; they compute an extra *high-precision* output O′_i = Σ_j P_ij V^F_j in forward, stored solely for D = rowsum(dO⊙O′) in backward (Alg. 2 line 13; §2.3, Eq. 9). Omitting O′ ⇒ exploding gradients and VBench collapse to 0.7185 (Exp. 7, Fig. 3a–b). Naive FP4-forward + reused BF16 FA backward ⇒ consistently exploding gradients (§3.2). Softmax itself stays FP32. dS uses high-precision P (not P^F). Overhead: +~50% forward FLOPs (O′), ~25% attention-level memory, 1.2–2× wall-clock during QAT only (App. C.1).
9. **Steps/compute.** Wan-1.3B: 16 B200 (GB200 NVL72), 4,000 steps (~12.5 h), best checkpoint at 3,000 steps; global bs 16, full grad checkpointing. Wan-14B: 64 H200, 400 steps (1 day), HSDP 8×8, Ulysses SP=2, global bs 32 (App. C.1). LLMs: ≤4k steps on 4 B200 (App. C.2).
10. **LR/optimizer.** AdamW (β₁=0.9, β₂=0.999), **LR 1e-6**, wd 0.01, bf16 mixed precision (App. C.1). LLM: 5e-6.
11. **Dataset.** Synthetic latents generated by Wan-2.1-14B: 81K examples @480P for the 1.3B run; 13K @720P for the 14B run (§3.1). LLM: C4 (10% shard), Dolci-Instruct.
12. **Models.** Wan2.1-1.3B/14B; Qwen3-14B, Llama-3.1-70B.
13. **Recovery.** Wan-1.3B (Table 2): FP4 no-training IQ 0.5592 / DD 0.1160 vs BF16 0.6728/0.3923 → Attn-QAT IQ 0.6775 (fully recovered, slightly above BF16), AQ 0.6764 (>BF16), subject/background/motion ≥ BF16, **Dynamic Degree 0.3039 — recovered from 0.1160 but still below BF16's 0.3923**. Wan-14B (Table 1): FP4 DD 0.2983 → QAT 0.3646 vs BF16 0.5193 (partial); IQ 0.6745 vs 0.6869 (near-full); overall 0.8279 vs 0.8335; blind human eval on 99 prompts: parity with BF16 (Fig. 2). Kernels: 1.1–1.5× over SageAttention3 (RTX5090); B200 NVFP4-QK+FP8-PV 2026 TFLOPS = 1.31× BF16 FA4; B300 1.74× (§3.4, Fig. 6).
14. **Relevance to DQ-VSA.** This *is* the QAT recipe for our operator, minus sparsity and minus distillation: (a) confirms the same FP4 signature we measure — imaging_quality and especially dynamic_degree collapse under training-free NVFP4 QK on Wan (their dense FP4 DD −0.276 on 1.3B; ours −0.306 dense, −0.250 sparse); (b) shows plain QAT restores IQ fully but DD only partially — motivating adding distillation (their own future work) and/or motion-weighted data for DQ-VSA; (c) gives the exact fwd/bwd numerics we must respect if we backprop through the NVFP4 fine path under a block-sparse mask: fake-quant P in backward recomputation + high-precision O′ for the softmax-Jacobian scalar; the sparse mask restricts which (i,j) tiles are visited but does not change these identities; (d) provides the B200 FA4 CuTe kernel lineage our fork extends and the hyperparameters (LR 1e-6, 3–4k steps, Wan-14B-synthesized data) as a starting recipe; (e) verifies fake-quant Triton training vs. real-FP4 CUDA inference produce visually indistinguishable outputs (Fig. 4) — supporting a fake-quant training / native-kernel inference split for DQ-VSA.

## 7. Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery — arXiv:2601.20088 [VERIFIED from primary source]

*Xin, Priyadarshi, …, Mao et al. (NVIDIA technical report). QAD best practices for NVFP4 LLMs/VLMs. Not attention quantization (attention kept high precision), but the definitive study of distillation-vs-QAT loss choice for NVFP4 recovery.*

1. **Attention mechanism.** N/A — attention math is *kept in high precision*; for Nemotron Nano 9B V2 the attention layers of Transformer blocks stay BF16; for Nemotron 3 Nano the 6 self-attention layers + preceding Mamba-2 layers stay BF16 (arXiv:2601.20088, §3.4 "Quantization Configuration").
2. **Sparse mechanism.** None.
3. **Quantization.** NVFP4 (block-16 E2M1, E4M3 block scales, second-level FP32 tensor scale) on GEMM weights+activations; KV cache FP8 for Nemotron 3 Nano (§2.1, §3.4). Forward-only quantization (QAT/QAD graph): Wgrad/Dgrad stay high precision (App. D, Fig. 2).
4. **Training required?** Yes — QAD, a short post-training stage. PTQ alone insufficient for small models (§2.1; large ≥253B models are fine with PTQ, App. C).
5. **Teacher/student.** Original BF16 model = teacher; NVFP4-quantized same model = student. Using a *larger* teacher (12B for a 9B student) is *worse* than the original-size teacher (§4.3, Table 9) — recover-the-distribution, not transfer-knowledge.
6. **Losses.** L_QAD = D_KL(p_teacher ‖ p_student) over the vocabulary, softmax temperature T=1 for both (§3.1, Eq. 1). KL > MSE-on-logits across benchmarks (§4.3, Table 8). Diagnostic: QAT matches teacher *cross-entropy* but has KL-to-teacher 0.311; QAD achieves KL 0.004 — QAT silently rewrites the output distribution ("effectively acting as an additional post-training stage"), QAD preserves it (§3.1, Table 1).
7. **Sparsity curriculum.** N/A.
8. **QAT/QAD semantics.** Standard forward fake-quant of weights+activations; gradients high precision; QAD differs from QAT *only* in the loss (§3.1, Fig. 1; App. D).
9. **Steps/compute.** Data-to-convergence: 0.3B tokens (Llama Nemotron Super 49B), 0.8B (AceReason 7B), 2.5B (Nemotron 3 Nano 30B-A3B), 6B (Nano 9B V2) — orders of magnitude below original post-training (§3.4).
10. **LR/optimizer.** LR 1e-6 (SFT-heavy models: at/below original SFT LR) to 1e-5 (RL-heavy models benefit from higher LR); recommend 1e-6–1e-5 overall (§3.4, §4.2, Tables 6–7).
11. **Dataset.** Remarkably robust: SFT data ≈ teacher-generated data from RL prompts ≈ BOS-token-seeded self-generations; *including incorrect generations helps*; even random tokens don't break the model (§4.1, Table 5); partial-domain data (code-only) recovers math via cross-domain transfer through teacher soft labels (§3.3, Table 4).
12. **Models.** Llama Nemotron Super V1 49B, Nemotron Nano 9B V2, Nemotron 3 Nano 30B-A3B, AceReason 7B, Nemotron Nano 12B V2 VL.
13. **Recovery.** QAD → near-BF16 across the board; QAT *degrades below PTQ* on RL-heavy models (AceReason AIME25: BF16 63.5 / PTQ 58.7 / QAT 46.1 / QAD 62.0) (§3.2, Tables 2–3).
14. **Relevance to DQ-VSA.** Strongest evidence that for NVFP4 recovery specifically, **distillation from the original model beats task-loss QAT**, chiefly because QAT drifts the output distribution while QAD pins it — precisely what we want when the only intended change is the attention operator's numerics. Also: (a) teacher = the *same* model (P4G/BF16 twin), not a bigger one; (b) synthetic teacher-generated data suffices (matches VSA and Attn-QAT practice of Wan-14B-synthesized latents); (c) budgets are small (≲ a few B tokens equivalent; for video, thousands of steps); (d) LR 1e-6–1e-5 window. The diffusion analogue of the KL-on-vocabulary loss is the velocity-MSE of SpargeAttention2 (§4.2 there), since a flow-matching DiT's "output distribution" is its velocity field.

---

## 8. 6Bit-Diffusion — arXiv:2603.18742 [VERIFIED from primary source]

*Su, Zhang, Yuan, Duanmu, Chen, Zhu (Fudan/Tsinghua/SJTU). Training-free inference-time NVFP4/INT8 mixed-precision routing for video DiT linear layers + delta caching. Included for NVFP4-on-video context; it is not an attention-quantization or training method.*

1. **Attention mechanism.** Unmodified full attention; quantization targets the linear layers (attention Q/K/V/O projections and FFN) (arXiv:2603.18742, §4.1). They note the speedup "is naturally bounded by the attention mechanism, which consumes over half of the total execution time" (§5.3).
2. **Sparse mechanism.** No attention sparsity; instead Temporal Delta Cache (TDC) skips whole transformer blocks when the residual delta Δ_t^l is temporally stable, with error-guided cache switching (E_acc threshold τ, max consecutive skips N_max=2) (§4.2, Eq. 9–11).
3. **Quantization.** Weights all NVFP4 offline; activations routed per-layer per-timestep between NVFP4 (block-16, shared FP8 scale, max/6.0) and INT8 based on a linear predictor E_rel = α·Γ_{t−1} + β, where Γ is the block's input-output relative L1 change at the previous timestep (§3, §4.1, Eq. 3–7). Online block Hadamard transform (B=128) for outlier smoothing (§4.1). Average effective activation width ≈6.0 bits (W4A6) (§5.2, Table 2 note).
4. **Training required?** No — training-free PTQ; only offline linear-regression fitting of (α,β) on 100 calibration prompts (§5.1).
5–7. **Teacher/student, losses, curriculum.** None. (The timestep dimension appears as *dynamic precision routing* rather than a training schedule.)
8. **QAT semantics.** N/A. Purified Delta Refresh (PDR): layers with outlier ratio max|X|/mean|X| > 25 fall back to FP16 when refreshing the cache, because cached quantization noise is linearly amplified by skip count N (Eq. 12; §4.3).
9–11. **Compute/data.** Single RTX-5090; CogVideoX configs, DDIM 50 steps, CFG 6.0; calibration = 100 EvalCrafter prompts (§5.1).
12. **Models.** CogVideoX-2B and 5B.
13. **Recovery.** W4A6 DMPQ ≈ FP16 on VBench (2B Aesthetic 0.5437 vs 0.5464; 5B 0.5724 vs 0.5922, far above static ViDiT-Q W4A6 0.4433); full framework 1.92× E2E speedup, 3.32× memory (§5.2, Tables 1–2).
14. **Relevance to DQ-VSA.** (a) Independent confirmation that uniform NVFP4 activations on video DiTs are lossy training-free and that *temporal volatility predicts quantization sensitivity* — a diagnostic we can apply to decide which timesteps/layers of the NVFP4-QK path most need protection or distillation weight; (b) supports timestep-adaptive precision (kin to FPSAttention's schedule) as an orthogonal, training-free complement to DQ-VSA; (c) does not touch attention QK quantization, so it neither anticipates nor blocks a native-sparse-NVFP4-attention claim.

## 9. Supplementary verified papers

### 9a. SageAttention3: Microscaling FP4 Attention for Inference — arXiv:2505.11594 [VERIFIED from primary source]

*Zhang, Wei, Wang, Zhang, Xu, Huang, Jiang, Chen, Zhu (Tsinghua). First FP4 (NVFP4) attention kernel; training-free; plus SageBwd, the first trainable (INT8) attention.*

- **Attention/quant.** Dense FlashAttention with NVFP4 microscaling on both QKᵀ and PV; chooses NVFP4 over MXFP4 (attention CosSim 99.52% vs 98.37% on CogVideoX tensors) (arXiv:2505.11594, §3.1, Table 1a). Outlier mitigation: K smoothing (K−mean(K)) and Q smoothing per SageAttention2 (Alg. 1). Two-level P quantization: per-row rescale of P̃ into [0, 448×6] before per-16 FP4 quantization so the E4M3 scale range is used (raw scales would sit in [0, 0.167]) — direct quantization CosSim 93.32% → 99.52% two-level (§3.2, Eq. 5, Table 1b).
- **Training.** SageAttention3 itself is training-free inference. SageBwd (same paper) is INT8 trainable attention: 6 of 7 attention matmuls INT8, keeping dOVᵀ in FP16 because its error accumulates into dS→dQ,dK along the FlashAttention backward recurrence (§4.2, Table 1c: dQ CosSim 99.77% FP16 vs 97.47% INT8). Lossless in fine-tuning; *slower convergence in pretraining* (§5.2, Fig. 8).
- **Results.** 1038 TOPS on RTX5090 (5× FA2); claims "almost no end-to-end quality loss" on CogVideoX/HunyuanVideo/Mochi/Flux/SD3.5 by CLIPSIM/VQA/FScore (§5.2, Table 2). Note: on Wan (not in their eval set), Attn-QAT independently measured SageAttention3 at VBench 0.7834 vs BF16 0.8267 (arXiv:2603.00040, Table 2) — i.e., FP4 PTQ attention *does* degrade Wan noticeably.
- **Relevance.** Defines the NVFP4-attention baseline our P4 fine path descends from (packed E2M1 + per-16 E4M3 is the same format), and its two-level P trick matters only if we ever quantize PV (we keep PV BF16). Training-free heuristics demonstrably do not close the Wan gap → training-based recovery is warranted.

### 9b. SLA: Sparse–Linear Attention — arXiv:2509.24006 [VERIFIED from primary source]

- Trainable hybrid: attention weights split into critical (top k_h=5%, sparse FlashAttention), marginal (linear attention), negligible (bottom k_l=10%, skipped); O = O_s + Proj(O_l); fused fwd+bwd kernel; 64×64 blocks (arXiv:2509.24006, §4, Alg. 1–2). Rationale: the non-top attention mass is extremely low-rank (§3.2, Eq. 1).
- Fine-tune 2,000 steps, bs 64, plain diffusion loss on 20,000 480p 5-s videos (Pexels/CommonCrawl); "less than 0.1% of pretraining cost" (§6.1, §6.3). No quantization (that arrives in SLA2).
- 95% sparsity ≈ full-attention quality on Wan2.1-1.3B (VA 76.96 vs 76.78); 13.7× attention speedup, 2.2× E2E (§6.2–6.3, Table 1).
- **Relevance.** Baseline for SLA2; shows plain diffusion-loss fine-tuning *can* adapt Wan to a radically different attention operator in 2k steps when data quality is adequate — but SpargeAttention2 (§3.2 Case 3) later showed this is data-sensitive, and their velocity distillation is safer.

### 9c. Pretraining Large Language Models with NVFP4 — arXiv:2509.25149 [VERIFIED from primary source]

- NVIDIA. Native NVFP4 *pretraining* of a 12B hybrid Mamba-Transformer on 10T tokens matching FP8 (loss gap <1–1.5%; MMLU-pro 62.58 vs 62.62) (arXiv:2509.25149, §3, Table 2).
- Method: keep ~15% of linear layers (mostly last blocks) high precision; Random Hadamard transforms (16×16) on Wgrad inputs only; 2D 16×16 block scaling for weights (consistent fwd/bwd quantized representation — chain-rule consistency); stochastic rounding on gradients only (round-to-nearest for weights/activations, SR in forward is harmful); each component necessary at 12B/10T scale (§4, Fig. 4).
- **Attention is excluded**: "we retain the original precision … for … attention components, including softmax and the query-key and attention score-value batched GEMMs" (§4.1); extending NVFP4 "to attention" is listed as future work (§6).
- **Relevance.** (a) Proves NVFP4 numerics are trainable-through at scale for GEMMs, and catalogs the stabilizers (SR on grads, representation consistency) that a *native* (not fake-quant) FP4 training path would need; (b) explicitly leaves FP4 attention open — supporting novelty space for FP4-attention work generally; (c) its "consistent quantized representations across forward/backward" principle is the pretraining-scale cousin of Attn-QAT's requirement R1.

### 9d. SageBwd: A Trainable Low-bit Attention — arXiv:2603.02170 [VERIFIED from primary source]

- Follow-up to SageAttention3's SageBwd: closes the INT8-attention *pretraining* gap. Findings: dS (softmax-gradient) is the dominant error source in the low-bit backward due to systematically small magnitude; QK-norm is necessary for stability at large tokens/step; reducing tokens per optimization step lets INT8 attention match full-precision attention pretraining; K-smoothing essential, Q-smoothing marginal (arXiv:2603.02170, Abstract, §4).
- **Relevance.** If DQ-VSA ever moves from fake-quant QAT to *native* low-bit backward, dS fragility and QK-norm (already present in Wan as RMS QK-norm) are the failure modes to watch. For our forward-only-quantized QAT, it corroborates Attn-QAT's finding that backward-path precision handling is the crux.

### 9e. Items cited but NOT independently verified

- **FlashAttention-4** ("Dao et al. 2026, FlashAttention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling") — cited by Attn-QAT (arXiv:2603.00040, refs) together with the `Dao-AILab/flash-attention` SM100 CuTe source file and the `hao-ai-lab/flash-attention-fp4` repo (Zhang et al. 2026). No arXiv ID is given in the citing paper and I did not locate/fetch a primary arXiv page for FA4 itself → the FA4 *paper* is **COULD NOT VERIFY from arXiv** (its existence is attested only via citations and named GitHub repos in verified papers). FA4 kernel behavior on B200/B300 (softmax-bound, MUFU.EX2 throughput analysis) is reported second-hand in arXiv:2603.00040 §2.5/App. B, which I verified.
- **TurboDiffusion (arXiv:2512.16093), VMoBA (2506.23858), DSV (2502.07590), SVG (2502.01776), Bidirectional Sparse Attention (2509.01085), TetraJet-v2 (2510.27527), Quartet (2505.14669), "FP4 All the Way" (2505.19115)** — referenced within verified papers; not fetched; any statements above about them come only from the citing papers' text.

---

## 10. Synthesis

### (a) Best-supported training-loss and QAT-mechanics choices for recovering NVFP4-QK degradation with frozen sparsity geometry

**Loss: distillation from the BF16 twin, not plain task loss.**
- NVIDIA QAD (2601.20088, §3.1–3.2): for NVFP4 recovery, KL-to-teacher ≫ task-loss QAT; QAT can silently rewrite the output distribution (KL 0.311 vs 0.004) and, on heavily post-trained models, degrade below PTQ. Teacher should be the *same* model in BF16; T=1; KL > logit-MSE.
- SpargeAttention2 (2602.13515, §4.2, Table 6): the diffusion-model analogue — velocity distillation L_VD = ‖u_sparse − u_full‖² on identical (x_t, t, c) beats diffusion-loss fine-tuning on every quality metric and is robust to fine-tuning-data mismatch (we lack Wan's pretraining data).
- Counterpoint: FPSAttention (2506.04648) and Attn-QAT (2603.00040) both recovered most quality with *plain* diffusion-loss QAT — so task-loss QAT is a viable fallback, but Attn-QAT's residual dynamic_degree gap (0.3039 vs 0.3923 on 1.3B) plus QAD's distribution-drift evidence argue for distillation as the primary loss for DQ-VSA. A practical composite: L = ‖u_student − u_teacher‖² (+ optional QuantSparse-style pooled/salient attention-map MSE inside retained VSA blocks (2509.23681, §3.2) to directly supervise the NVFP4 QK logits).

**QAT forward/backward mechanics (from the only systematic FP4-attention study, 2603.00040 §2.2–2.3):**
1. Fake-quantize Q,K to NVFP4 in forward (we keep PV BF16, which sidesteps their P̃ quantization and, per their App. B.2, is also the *faster* B200 configuration).
2. STE backward in BF16, **but**: any P recomputed in the backward from stored logsumexp must be re-fake-quantized to match forward precision (stabilizes gradient norms), and the FlashAttention softmax-Jacobian scalar must use a high-precision output O′ computed alongside the quantized O in the forward — omitting O′ ⇒ exploding gradients. With NVFP4 restricted to QK only (P,V in BF16), O = O′ and requirement R2 vanishes; R1 reduces to recomputing S from the *fake-quantized* Q,K in the backward — cheap and exact under the frozen block mask.
3. Freeze the selector path (coarse BF16 stage) entirely; SLA2 (2602.12675, §6) likewise freezes its router during end-to-end fine-tuning, and our geometry-freeze constraint requires it. Gradients then flow only through the fine-path Q,K,V (and the rest of the network) under the fixed mask — VSA's block-sparse backward (2505.13389, §2.4) composes with Attn-QAT's fake-quant insertions tile-by-tile.
4. Hyperparameter anchor: LR 1e-6 (AdamW β=(0.9,0.999), wd 0.01), 3–4k steps, bs ~16, best-checkpoint selection around 3k steps (2603.00040, App. C.1); QAD suggests up to 1e-5 is safe when the student must move further (2601.20088, §4.2). Outlier heuristics (K-smoothing, two-level P) are unnecessary once training is in the loop (2603.00040, Table 2 Exp. 4–6).

### (b) Sparsity-curriculum evidence

- VSA (2505.13389, §2.3, §C.5): the only explicit anneal — start at full attention (Top-K=L/B), decay Top-K every 50 steps to target; needed when *introducing* sparsity into a dense checkpoint.
- FPSAttention (2506.04648, §3.3): no anneal; instead a *denoising-timestep* schedule of window size/quant granularity, fixed across training and inference.
- SLA2 (2602.12675, §6): multi-sparsity router pretraining (5/4/3%) then fixed sparsity; SLA/SpargeAttention2: fixed sparsity from step 0, no curriculum, 500–2000 steps suffice.
- **Implication for DQ-VSA:** since our checkpoint is already VSA-adapted and geometry is frozen, the literature gives no reason for a sparsity curriculum. If anything, a *precision* curriculum (e.g., FP8→NVFP4 QK, or NVFP4 on a growing fraction of timesteps) is the analogous knob, but no paper tests precision annealing for attention; FPSAttention's timestep schedule is the nearest verified relative.

### (c) Does any paper already demonstrate NATIVE sparse NVFP4 attention? (priority-claim audit — strict)

**No verified paper demonstrates native sparse NVFP4 attention.** The closest, precisely characterized:
- **Attn-QAT (2603.00040)**: native NVFP4 attention (real packed-FP4 tcgen05 GEMMs via SageAttention3-CUDA/RTX5090 and FA4-CuTe/B200-B300) — but **dense**; its B200 kernel "supports block-sparse attention" as an implementation capability (§5) with no sparse experiment, no sparse quality data, and QAD+sparse explicitly named future work. Same research group as VSA/FastVideo.
- **SLA2 (2602.12675)**: trainable **sparse + low-bit** attention with QAT — but the low-bit sparse branch is **INT8/FP8** (SageAttention2++ scheme, §5), not FP4/NVFP4.
- **FPSAttention (2506.04648)**: training-aware **sparse + FP8** (E4M3) tile-quantized QK — FP8 not FP4, static local windows not top-k, Hopper not Blackwell.
- **SageAttention3 (2505.11594)**: native NVFP4 attention kernels — dense, training-free.
- **QuantSparse (2509.23681)**: sparse attention + INT4/8 *linear-layer* weights — attention math not quantized to FP4.
- **NVFP4 pretraining (2509.25149)**: NVFP4 GEMMs everywhere *except* attention (§4.1).
**Verdict:** a claim of "first **native** (packed E2M1 + per-16 E4M3 hardware GEMM) **block-sparse** NVFP4 **attention** for video diffusion, with paired quality evaluation" is defensible against everything verified here, provided it is worded with all four qualifiers (native / sparse / NVFP4 / attention). Risks: (i) the FA4 paper (unverifiable here) or the `flash-attention-fp4` repo could add block-sparse FP4 results at any time — Attn-QAT's §5 shows the capability exists in-kernel and that the same group intends to publish "FP4 attention with QAD and sparse attention"; (ii) claim should therefore emphasize the *measured sparse-FP4 quality interaction* (P4 vs P4G paired VBench) and the *training-based recovery under frozen geometry*, which no paper provides. A "first to train/recover" claim (DQ-VSA) is currently uncontested in the verified record.

### (d) Timestep-aware precision scheduling evidence

- FPSAttention (2506.04648, §3.3, Fig. 5): direct measurement that joint quant+sparse error tolerance is U-shaped over denoising — early/late steps tolerate coarse quantization + high sparsity; **mid-denoising steps need finest granularity and densest attention**; encoded as a 3-regime piecewise schedule S(t)=[g(t),W(t)] used in training *and* inference (train/test consistency emphasized).
- 6Bit-Diffusion (2603.18742, §3–4, Fig. 1): layer sensitivity to (NVFP4 vs INT8) activations fluctuates strongly across timesteps; previous-step block input–output change Γ_{t−1} linearly predicts current quantization error (E_rel = αΓ+β) — an online sensitivity signal.
- QuantSparse (2509.23681, §3.3): quantization noise makes the sparse-attention residual timestep-*unstable* (first-order residual invariance breaks; second-order stabilizes) — more evidence that quantization error is timestep-structured.
- VSA (2505.13389, §3.5): selector accuracy increases monotonically with timestep — early (high-noise) steps have the least accurate block selection, compounding with quantization there.
- **Implication:** a DQ-VSA variant that keeps BF16 QK (or FP8) on the sensitive mid-denoising band and NVFP4 elsewhere is well supported; if used, apply the same schedule during QAT/distillation (FPSAttention's consistency principle).

### (e) Dataset guidance for motion-diverse distillation

- FPSAttention (2506.04648, App. B): the only paper with explicit motion filtering — optical-flow magnitude 0.05–2.0 (plus Q-Align>3.5, aesthetic>2.0, dedup, 480p/16fps/5s); its QAT *raised* dynamic_degree above baseline (0.3014→0.4195), the only method here to do so.
- Attn-QAT (2603.00040, §3.1) and VSA (2505.13389, §C.5): Wan-14B-*synthesized* latents (81K@480P / 80K–200K videos) — convenient, distribution-matched, and sufficient for IQ recovery, but Attn-QAT's dynamic_degree stayed below BF16 — consistent with synthetic Wan data under-representing large motion (Wan-1.3B's own DD baseline is only ~0.39).
- NVIDIA QAD (2601.20088, §4.1): distillation is robust to data source/quality — teacher-generated data ≈ real SFT data; *unfiltered* generations slightly beat correctness-filtered ones; even random inputs don't break the student. Cross-domain transfer through teacher soft targets (§3.3).
- SpargeAttention2 (2602.13515, §3.2 Case 3) + SLA2 (2602.12675, §9.2): small (3,000-video) real corpora suffice *if* the loss is distillation; with plain diffusion loss, low-quality data actively harms.
- **Implication:** for DQ-VSA, distillation loosens data-quality requirements (per QAD), but our target deficit is *dynamic_degree*, so bias the x_t construction corpus toward high-motion content: real videos filtered by optical-flow magnitude (FPSAttention recipe) and/or teacher generations from motion-heavy prompts; a few×10³–10⁴ clips at 480p appears sufficient (3k–80k range across verified papers). Evaluate with paired P4-vs-teacher VBench dynamic_degree as the primary recovery metric.

---

## Verification appendix

| arXiv ID | Fetched | Status |
|---|---|---|
| 2505.13389 (VSA) | abs page, full HTML text | VERIFIED |
| 2506.04648 (FPSAttention) | abs page, full HTML text (v. dated 2026-08-11) | VERIFIED |
| 2509.23681 (QuantSparse) | abs page, full HTML text | VERIFIED |
| 2602.12675 (SLA2) | abs page, full HTML text | VERIFIED |
| 2602.13515 (SpargeAttention2) | abs page, full HTML text | VERIFIED |
| 2603.00040 (Attn-QAT) | abs page, full HTML text | VERIFIED |
| 2601.20088 (NVFP4 QAD) | abs page, full HTML text | VERIFIED |
| 2603.18742 (6Bit-Diffusion) | abs page, full HTML text | VERIFIED |
| 2505.11594 (SageAttention3) | abs page, full HTML text | VERIFIED (supplementary) |
| 2509.24006 (SLA) | abs page, full HTML text | VERIFIED (supplementary) |
| 2509.25149 (NVFP4 pretraining) | abs page, full HTML text | VERIFIED (supplementary) |
| 2603.02170 (SageBwd) | abs page, full HTML text | VERIFIED (supplementary) |
| FlashAttention-4 paper | not fetched (no arXiv ID located) | COULD NOT VERIFY — known only via citations in 2603.00040 |

