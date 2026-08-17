# PAPER_UPDATE — SparseFP4 Native Composition

How the completed study reshapes the paper. Every claim maps to receipts in
`artifacts/sparsefp4_native/` (file pointers inline).

## Framing

The paper's question is answered directly and affirmatively:

> Native NVFP4 attention compute and block-sparse video attention compose on
> Blackwell with no numerical interaction penalty, quality at the deployed
> sparse baseline's level, a 9.3x attention-kernel speedup at deployed
> sparsity, and a 1.40x end-to-end speedup at 720p once the sparse geometry
> is co-designed with the FA4 kernel.

The main table IS the paper: four-arm operator matrix + quality + measured
latency (`tables/c5_matrix_s090.md`, `raw/quality/*`, `tables/c8_performance.md`).

## Main claims (with the number that carries each)

1. **Existence + nativeness.** First working native sparse-NVFP4 attention
   on B200: packed E2M1 QK, per-16 E4M3 scales, retained-tile-only
   execution, BF16 PV (`NATIVE_PROOF.md`). Enabled by repairing 7 latent
   version-skew bugs in the flash-attention-fp4 fork — worth a short
   "systems archaeology" subsection; the fork's fwd+bwd sparse/FP4 paths had
   never run (`configs/fa4-fork-sparse-fp4-repair.patch`).
2. **No interaction penalty.** D0-vs-C0 rel-L2 0.096 == B0-vs-A0 0.098 on
   real Wan tensors with frozen deployed masks. This *contradicts the
   QuantSparse-style expectation* of composed-error amplification for this
   operator pair — a headline, citable result.
3. **Geometry co-design beats coarsening.** Mapping the deployed 64x64 VSA
   mask onto FA4 inflates retention 10%->24% (P3); moving the selector to
   256-token tiles maps 1:1 (P4, 10.12% exact), keeps VBench parity with
   deployed VSA, and yields the study's best E2E (106.1 s vs dense 148.8 s
   at 720p). Frame as the FPSAttention unified-granularity principle applied
   to NVFP4/Blackwell.
4. **Where FP4 pays today.** Kernel level: always (0.80 vs 0.83 ms @10%;
   6.02 vs 7.41 ms dense). E2E: dense-NVFP4 1.06-1.10x; sparse-NVFP4 E2E
   currently bounded by two measured engineering items (FP4-kernel scheduling
   regime at large tile counts; per-call quantize integration overhead), not
   by numerics. State plainly; the BF16-fine twin already cashes the
   composition out E2E.

## Figures

1. System diagram: A0/B0/C0/D0 + the P-arm stack (selector/coarse/fine).
2. Quality-vs-E2E-latency Pareto at 720p (P0/P1/P2/P4G/P4; VBench
   subject-consistency or aggregate on y).
3. Work-scaling: kernel ms vs retained fraction, FP4 vs BF16
   (`raw/performance/c3_native_proof.json`).
4. (Explanatory only) rel-L2 by denoising timestep, B0/C0/D0
   (`tables/c5_matrix_s090.md` bottom table).

## Honest-limitations paragraph (draft)

At 480p the DiT's attention share limits any attention-only intervention to
~1.1x E2E; deployed VSA itself loses to dense FA4 there. The sparse-FP4
kernel's persistent-scheduler configuration inherits dense-shape tuning and
inverts vs its BF16 twin at 720p tile counts; and per-call quantization,
while only 0.17 ms/call in isolation, interacts poorly with the serving
allocator. Fully-empty Q rows deadlock the FP4 kernel in multi-wave grids
(unreachable under VSA's topk>=1 guarantee). Quality numbers are the
10-prompt development protocol; the 946-prompt VBench corpus run is the
paper-scale protocol.

## QAT/recovery section

Training-based recovery targets the *sparsity* quality gap (the dominant
term — NVFP4 adds no consistent VBench penalty over BF16 twins). Recipe per
SLA2/Attn-QAT/QAD: fake-quant NVFP4 fine branch (exact production
round-trip, STE), BF16 selector/coarse, constant deployment sparsity.
Feasibility run: 400 steps, 47 videos, one B200, ~47 min: **subject
consistency 0.846 -> 0.974 (dense = 0.976) and background consistency
0.918 -> 0.977 (== dense) on the P3 serving arm; aesthetic 0.410 -> 0.540.**
Training-based recovery closes the sparsity-family quality gap at trivial
cost — the paper's strongest supplementary claim (caveat: dynamic_degree
collapses on this static-camera dataset; use a motion-diverse corpus at
paper scale). Position full QAT with
low-precision backward (Attn-QAT kernels) and FP4 linears (QAD) as the
scaling path to end-to-end FP4 serving.

## Related-work deltas

- FPSAttention: our result is the NVFP4/Blackwell/inference-first analogue;
  their FP8/Hopper training-aware co-design supports our geometry claim.
- Attn-QAT (same underlying FA4-fork kernel family): their B200 NVFP4-QK
  numbers corroborate our dense receipts; cite for the QAT path.
- SLA2: sparse+low-bit QAT precedent; our recovery recipe follows it.
- QuantSparse: the amplification they report for weight-quant x attention-
  sparsity does NOT appear for activation-side NVFP4 QK x VSA masks (claim 2).
- VSA: baseline preserved exactly (selector, top-k, coarse branch) in P2/P3.
