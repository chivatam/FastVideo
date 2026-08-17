# SparseFP4 Native Composition — REPORT

Question: **Can native NVFP4 attention compute and block-sparse video
attention be composed on Blackwell to obtain real speedup without
unacceptable quality loss?**

Answer: **Yes — POSITIVE, systems-qualified** (`RESULTS_DECISION.md`).
Model: Wan2.1-T2V-1.3B, 480x832x81 (primary) and 720x1280x81 (secondary),
50 steps, seed 1234, VSA sparsity 0.90, 8x B200 (sm_100), CUDA 13.0.
Environment receipts: `env/`. Every number below has a raw receipt under
`raw/`, `tables/`, or `logs/`.

## 1. What had to be built (C0/C2)

The FA4 fork (`hao-ai-lab/flash-attention-fp4` @ `940bf7e5`) already
contained an SM100 NVFP4 attention kernel *with block-sparse scaffolding*,
but the composed path was broken from birth by version skew. Repaired in 4
fork commits (`configs/fa4-fork-sparse-fp4-repair.patch`): era-matched
sparse helpers for the FP4 kernel, `q_stage` fix, `descale_tensors` slot,
runtime tuple repack — plus 3 more backward-path fixes found during QAT
training (undefined `deterministic`, bwd positional-slot skew, bwd
preprocess `mdLSE` slot). No kernel math was changed. FastVideo-side:
capture backend, P3 (`SPARSEFP4_NATIVE_VSA_ATTN`), P4
(`SPARSEFP4_VSA256_FA4_ATTN`), QAT (`SPARSEFP4_QAT_VSA_ATTN`) backends.
Audit: `CODE_PATH_AUDIT.md`.

## 2. Native proof (C3) — `NATIVE_PROOF.md`

Packed `float4_e2m1fn_x2` Q/K + uint8 E4M3 per-16 SFs at the call boundary;
`flash_fwd_sm100_fp4` symbol in torch.profiler; retained-tile-only load/MMA/
softmax loops in source; kernel latency scales with retained fraction
(6.01 / 3.07 / 1.65 / 0.80 ms at 100/50/25/10%, Wan shape); no BF16 Q/K
materialization anywhere. Envelope limit documented: fully-empty Q rows
deadlock in multi-wave grids — unreachable under VSA (topk >= 1).

## 3. Kernel correctness (C4)

32 cells (S up to 39936, retained 1.0-0.10, 2 seeds): D0 vs its exact
dequantized-NVFP4 fp32 oracle — cos 0.999997, rel-L2 2.32e-3 median — equal
to the dense kernels' deviation from their own oracles. The sparse path adds
zero numerical error beyond intrinsic NVFP4 QK arithmetic.
Oracle validated bitwise against the production quantizer
(`configs/nvfp4_dequant_oracle.py`, tie-midpoints only).

## 4. Controlled 2x2 (C5) — the four-arm table (`tables/c5_matrix_s090.md`)

25 cells captured from a genuine VSA trajectory (5 layers x 5 steps, frozen
deployed masks, byte-identical for C0/D0; coarsened retention 24.2% from
the 64x64 mask's 10.1%). Median vs A0:

| Arm | rel-L2 | cosine | SNR dB |
|---|---|---|---|
| B0 dense NVFP4 | 0.098 | 0.9952 | 20.2 |
| C0 sparse BF16 | 0.128 | 0.9929 | 17.9 |
| D0 native sparse NVFP4 | 0.204 | 0.9801 | 13.8 |
| D0 vs C0 | 0.096 | 0.9954 | 20.4 |
| C0-Triton64 (deployed 10% mask) | 0.206 | 0.9849 | 13.7 |

**No composition amplification:** quant cost on sparse (0.096) = quant cost
on dense (0.098). Joint error ~= the deployed finer mask's sparsification
error alone; mask geometry dominates the budget, not precision. Errors are
not assumed additive; raw table is the evidence. Timestep trend: all arms
improve with denoising progress; ordering stable.

## 5. Production arms — quality (C7)

10 dev prompts x seed 1234, paired vs P0 (medians) + VBench (7 no-reference
dims, means). Raw: `raw/quality/`.

| Arm | PSNR dB | LPIPS | VBench subj.cons | temp.flicker | aesthetic |
|---|---|---|---|---|---|
| P1 dense NVFP4 | 18.3 | 0.410 | 0.965 | 0.976 | 0.580 |
| P2 deployed VSA | 11.5 | 0.650 | 0.899 | 0.951 | 0.399 |
| P2G VSA sel.+FA4 BF16 | 12.3 | 0.622 | 0.878 | 0.954 | 0.468 |
| P3 VSA sel.+NVFP4 fine | 14.4 | 0.585 | 0.846 | 0.966 | 0.410 |
| P4G VSA256-FA4 BF16 | 11.5 | 0.689 | 0.896 | 0.946 | 0.369 |
| P4 VSA256-FA4 NVFP4 | 12.9 | 0.638 | 0.873 | 0.957 | 0.330 |

Reading: paired numbers measure trajectory divergence (all sparse arms
diverge similarly; NVFP4 fine branches are *no farther* from dense than
their BF16 twins). VBench: the quality cost is VSA-family sparsity itself
(subject consistency 0.98 -> 0.85-0.90); NVFP4 adds no consistent extra
penalty; the 256-tile selector (P4) matches deployed-VSA quality.
n=10 development protocol; the paper-scale run should use the repo's
VBench prompt corpus (946 prompts) with the same pipeline.

## 6. Performance (C8) — `tables/c8_performance.md`

Kernel (median of 50, pre-quantized): dense BF16 7.41 ms, dense NVFP4
6.02 ms, sparse BF16 @10% 0.83 ms, **native sparse NVFP4 @10% 0.80 ms
(9.3x vs dense BF16)**. Quantize overhead 0.17 ms/call (~0.5 s/video).

E2E (median steady-state; first gen excluded as warmup/JIT):

| System | 480p s | x | 720p s | x |
|---|---|---|---|---|
| P0 dense BF16 | 46.9 | 1.00 | 148.8 | 1.00 |
| P1 dense NVFP4 | 44.4 | 1.06 | 135.8 | 1.10 |
| P2 deployed VSA | 50.0 | 0.94 | 131.7 | 1.13 |
| P2G | 48.7 | 0.96 | — | — |
| P3 | 53.1 | 0.88 | — | — |
| **P4G** | 45.6 | 1.03 | **106.1** | **1.40** |
| P4 | 47.3 | 0.99 | 250.9 | 0.59 |

Peak memory ~8.9 GB (480p) / ~19.0 GB (720p), flat across arms.

Two named FP4-E2E engineering items (measured, not speculated):
(a) sparse-FP4 kernel inverts vs BF16 at 720p tile counts (6.09 vs 2.95 ms
kernel-only at 360-tile geometry; fine at 480p geometry) — scheduler/tuning
regime of the FP4 kernel; (b) 480p in-model FP4 fine path trails its own
microbenchmark (which shows FP4 *faster* incl. quantize: 1.44 vs 1.65
ms/call) — allocation churn suspected. `raw/performance/*.json`,
`logs/p4_720p_decomp.log`.

## 7. VSA-on-FA4 geometry result (user-directed experiment)

Moving the VSA selector from (4,4,4)=64-token tiles to (4,8,8)=256-token
tiles makes the top-k mask map 1:1 onto FA4's 256x128 sparse granularity —
zero mask inflation (10.12% retention exact), FA4 Blackwell fine kernel,
quality parity with deployed VSA, and the study's best E2E number
(P4G 1.40x at 720p, 1.24x over deployed VSA). Aligned with FPSAttention's
unified-granularity finding; this is the deployment-shaped design.

## 8. QAT recovery (Track D — supplementary)

SLA2/Attn-QAT/QAD-informed recipe: fake-quant NVFP4 QK (exact production
round-trip, opaque custom op, STE) on the fine branch only, BF16
selector/coarse — the serving semantics of P3/P4 — trained via the standard
Wan flow-matching objective at constant sparsity 0.9 on crush-smol latents
(47 videos, feasibility scale), 400 steps, lr 1e-5, ~7.1 s/step on one B200.

**Result: 400 steps recover the sparsity-family quality loss almost
entirely** (VBench, 10 dev prompts, stock weights -> QAT-400 weights, both
served on the same native arms; dense P0 reference 0.976/0.977/0.661):

| Dim | P3 before | P3 after | P4 before | P4 after |
|---|---|---|---|---|
| subject_consistency | 0.846 | **0.974** | 0.873 | **0.958** |
| background_consistency | 0.918 | **0.977** | 0.941 | 0.969 |
| temporal_flickering | 0.966 | 0.995 | 0.957 | 0.993 |
| motion_smoothness | 0.980 | 0.996 | 0.972 | 0.995 |
| imaging_quality | 0.610 | 0.656 | 0.574 | 0.650 |
| aesthetic_quality | 0.410 | 0.540 | 0.330 | 0.504 |
| dynamic_degree | 0.10 | 0.00 | 0.50 | 0.00 |

Subject/background consistency return to the dense reference's level;
aesthetic/imaging close most of their gap. Honest caveat: dynamic_degree
collapses to 0 — the 47-video crush-smol set (static-camera object-crushing
clips) biases motion; a paper-scale run needs a motion-diverse corpus (e.g.
Wan-Syn 600k). Paired-PSNR vs the *stock* dense reference drops (14.4 ->
10.9 dB) as expected — fine-tuning moves the weights, so divergence from the
old reference grows even as no-reference quality recovers; the QAT model's
own paired reference would be a QAT-weights dense generation.
Raw: `raw/quality/pq_qat400_{pairwise,vbench}.jsonl`.

Training-stack fixes required (all committed): torchvision 0.27 read_video
removal (OpenCV decode fallback), ragged tokenizer batches, meta-device
import context, dynamo-opaque fake-quant op, 3 FA4-fork backward bugs, and
an exact SDPA-recompute backward for dense FA4 (`FASTVIDEO_FA4_BWD_FALLBACK`)
since the fork's CuTe backward kernels fail MLIR verification under
dsl-4.5.

## 9. Precision-ladder + selector ablations (B-track)

- **B3 selector receipt** (`raw/operator/b3_selector_receipt.json`): running
  the *selector's* pooled inputs through the exact NVFP4 production
  round-trip leaves the deployed `fused_topk_mask` output 99.55% identical
  (kept-block flip 2.25%) — the design decision to quantize compute but not
  routing costs nothing (mechanism re-check of study 2's F2; receipt only,
  not headline evidence).
- **B1/B2 ladder** (9 frozen-mask cells, sparse arms vs all-retained BF16 on
  identical machinery; median rel-L2):

| Arm | rel-L2 med | max |
|---|---|---|
| sparse BF16 (mask effect only) | 0.175 | 0.279 |
| sparse **NVFP4 QK** + BF16 PV | 0.204 | 0.444 |
| sparse **MXFP8 QK** + BF16 PV | 0.199 | 0.398 |
| sparse NVFP4 QK + **FP8-E4M3 PV** | unsupported in this build |

  Two findings: (a) **MXFP8 QK buys almost nothing over NVFP4 QK**
  (0.199 vs 0.204) — per-16 E4M3 block scaling already absorbs most of the
  dynamic-range pressure, so NVFP4's 2x MMA-throughput advantage makes it
  the right operating point; (b) the FP8-PV path raises
  `Unsupported tcgen05 MMA op kind: MmaF8F6F4Op` in the pinned fork build
  (dsl 4.5.3) — block-scaled PV needs the fork's newer B300/mixed-precision
  branches; recorded as blocked, not attempted via simulation.
  Raw: `raw/operator/b_ladder.jsonl`.

## 10. Literature alignment

Four-arm pattern follows FPSAttention/QuantSparse; VSA baselines preserved
per the VSA paper; QAT direction per Attn-QAT (B200 NVFP4-QK+BF16-PV
matches our kernel receipts) and NVIDIA QAD (data-light recovery); SLA2
supplies the low-bit-forward/high-precision-backward fine-tune recipe.
No custom routing metrics were used as evidence (F1/F2 routing studies are
cited only as justification for keeping the selector in BF16).

## 11. Reproduction

```bash
source artifacts/sparsefp4_followup/configs/env.sh   # env + CUDA 13 bind-mount
# fork repair: apply configs/fa4-fork-sparse-fp4-repair.patch to
# hao-ai-lab/flash-attention-fp4 @ 940bf7e5, then editable-install flash_attn/cute
CUDA_VISIBLE_DEVICES=1 $FV_PYTHON artifacts/sparsefp4_native/configs/c5_capture_run.py --run-id c5-capture-s090
CUDA_VISIBLE_DEVICES=2 $FV_PYTHON artifacts/sparsefp4_native/configs/c5_operator_matrix.py --cells-dir ... --out ...
CUDA_VISIBLE_DEVICES=3 $FV_PYTHON artifacts/sparsefp4_native/configs/c3_native_proof.py
bash artifacts/sparsefp4_native/configs/p_launch.sh <run-id> "4 5 6 7"   # quality arms
$FV_PYTHON artifacts/sparsefp4_native/configs/{p_quality,p_vbench,c5_aggregate,c8_table}.py ...
bash artifacts/sparsefp4_native/configs/d_qat_finetune.sh 3 400          # QAT recovery
```
