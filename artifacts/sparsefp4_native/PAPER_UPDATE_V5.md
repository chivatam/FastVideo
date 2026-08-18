# PAPER_UPDATE_V5 — supplementary exploratory W4A4 study

> **STATUS: This file does NOT supersede PAPER_UPDATE_V4.**
> **PAPER_UPDATE_V4 is the canonical paper framing (paper frozen at V4).**
> V5 records an exploratory extension that was stopped after the systems
> gate failed. This experiment is excluded from the main contribution set
> and should appear, if space allows, as an appendix or a short discussion
> paragraph (wording in
> `supplementary/w4a4_gate/W4A4_EXPLORATORY_STUDY.md`).
> Do not source title, abstract, thesis, or contribution bullets from this
> file.

## Supplementary framing only

The W4A4 material below is appendix/discussion material. The former V5
title proposal ("FLOPs Are Not Latency: …") is WITHDRAWN from canonical
title recommendations — title direction returns to PAPER_UPDATE_V4
("Geometry, Bottlenecks, and Recovery: Composing Block-Sparse and NVFP4
Attention for Video Diffusion on Blackwell"). W4A4 GEMMs must not appear
in the title.

## Supplementary appendix content (NOT main-paper narrative)

The study shows, for this serving stack only:

1. Linears are 63-78% of nominal FLOPs but 11% of measured GPU time
   (22-24% after torch.compile).
2. Native W4A4 with real 1.5-2x FFN kernel wins is E2E-negative in every
   configuration; Amdahl panel + gate audit -> principled STOP
   (`supplementary/w4a4_gate/W4A4_NEGATIVE_RESULT.md`,
   `W4A4_AMDAHL_ANALYSIS.md`).
3. Maximum main-paper exposure: the single Discussion/Limitations
   paragraph in `supplementary/w4a4_gate/W4A4_EXPLORATORY_STUDY.md`.

The MAIN paper's bottleneck claim stays narrowly scoped to the
sparse-attention composition study (PAPER_UPDATE_V4 wording); the W4A4
study must not be used to broaden it into universal "FLOPs are not
latency" claims.

## Appendix figures/tables (if space allows)

- A1: FLOP share vs eager time vs optimized time (three-column component
  table, both resolutions).
- A2: Amdahl panel — theoretical ceiling vs realistic ceiling vs measured
  W4A4 E2E (the measured points sit BELOW 1.0x).
- A3 (optional): W4A4 engineering receipts (backend layout matrix, PDL
  deadlock) as reproducibility notes.

## Wording guardrails (supplementary scope)

- "W4A4 is slower" is a claim about THIS serving stack + flashinfer-0.6.17
  cudnn path with unfused activation quant; state the caveat and the
  infinite-GEMM ceiling (1.28-1.31x at D1) that bounds the opportunity.
- Do not say "fully FP4" anywhere: the shipped system is
  "VSA256 sparsity + native NVFP4 QK attention (BF16 selector/PV) + BF16
  linears"; W4A4 linears were gated out. Precision graph belongs in the
  paper (attention QK: NVFP4; everything else BF16/FP32).
- No quality claims for W4A4 PTQ (triage intentionally skipped; per-GEMM
  rel-L2 0.134 only).
- No generalization beyond video DiTs of this class; each profiler claim
  cites its own receipts.

## Paper writing (see PAPER_UPDATE_V4 + PAPER_ARTIFACT_MAP.md)

Canonical sources: `PAPER_CLAIMS_FINAL.md` (claims),
`PAPER_ARTIFACT_MAP.md` (section -> source). This file feeds the appendix
only.
