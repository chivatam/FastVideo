# PAPER_UPDATE_V5 — final framing (experiments closed)

Supersedes PAPER_UPDATE_V4 by addition. All V3/V4 guardrails stand.

## Title direction (final)

"FLOPs Are Not Latency: Composing Sparsity, FP4 Attention, and FP4 GEMMs
in Video Diffusion on Blackwell"
(alt: keep V4's "Geometry, Bottlenecks, and Recovery" and fold the W4A4
gate into the bottleneck section)

## Final narrative arc

1. Native block-sparse NVFP4 attention on B200 (guarded first-ness).
2. Operator-level composition is numerically clean (exact-geometry 2x2).
3. **Bottleneck migration #1**: FP4 attention's arithmetic win is consumed
   by softmax predicates/quantization/allocator E2E; geometry alignment is
   the dominant speed lever (P4G 1.40x).
4. Quality: NVFP4 QK costs imaging/dynamics (Holm-significant), sparsity
   does not amplify it (factorial interaction).
5. **DQ-VSA recovers the quality** (V4): <=500-step teacher-preserving
   velocity distillation, serving operator unchanged; task-loss QAT fails
   by teacher drift; naive backward suffices vs Attn-QAT backward
   (QK-only/BF16-PV regime).
6. **Bottleneck migration #2 (the W4A4 gate, V5)**: linears are 63-78% of
   FLOPs but 11% of time (22-24% after compile); native W4A4 with real
   1.5-2x FFN kernels is E2E-negative; Amdahl panel + gate decision =
   principled STOP. FLOPs-vs-time table + Amdahl panel are main-paper
   material (`w4a4_gate/FULL_DQVSA_GATE_DECISION.md`).
7. Thesis: arithmetic-intensity reduction does not guarantee proportional
   E2E acceleration; optimization changes the bottleneck composition. The
   measured winning levers: geometry alignment, compilation, distillation.

## Main figures/tables (V5 additions)

- T6: FLOP share vs eager time vs optimized time (three-column component
  table, both resolutions).
- T7/F4: Amdahl panel — theoretical ceiling vs realistic ceiling vs
  measured W4A4 E2E (the measured points sit BELOW 1.0x).
- Sidebar: W4A4 engineering receipts (backend layout matrix, PDL deadlock)
  as reproducibility notes.

## Wording guardrails (V5)

- "W4A4 is slower" is a claim about THIS serving stack + flashinfer-0.6.17
  cudnn path with unfused activation quant; state the caveat and the
  infinite-GEMM ceiling (1.28-1.31x at D1) that bounds the opportunity.
- Do not say "fully FP4" anywhere: the shipped system is
  "VSA256 sparsity + native NVFP4 QK attention (BF16 selector/PV) + BF16
  linears"; W4A4 linears were gated out. Precision graph belongs in the
  paper (attention QK: NVFP4; everything else BF16/FP32).
- No quality claims for W4A4 PTQ (triage intentionally skipped; per-GEMM
  rel-L2 0.134 only).
- The two bottleneck-migration examples must each cite their own profiler
  receipts; no generalization beyond video DiTs of this class.

## Paper writing order (next work, no experiments)

1. Results section from V4+V5 tables (all exist).
2. Method section: native kernel + DQ-VSA (exists in REPORT_V2..V4).
3. Related work from `SOTA_RECOVERY_LIT_REVIEW.md`.
4. Limitations: REPORT_V4 §5 + V5 §4 caveats.
