# REPORT_V5 — HISTORICAL / EXPLORATORY — NOT CANONICAL

> **This report is NOT the canonical paper state and does NOT supersede
> REPORT_V4.md.** The paper is frozen at V4: see `REPORT_V4.md`,
> `RESULTS_DECISION_V4.md`, `PAPER_UPDATE_V4.md` (and
> `REPORT_CANONICAL.md` for the artifact map). This file records the
> supplementary W4A4 gate exploration only
> (`supplementary/w4a4_gate/`). Its former framing of a "second
> centerpiece" / broadened central thesis is WITHDRAWN; the main paper's
> bottleneck claim stays narrowly scoped to the sparse-attention
> composition study (V4 wording).

V4's DQ-VSA recovery results are unchanged and canonical. Tags: [E]
experimental, [S] statistical, [L] literature, [H] hypothesis. Receipts:
`w4a4_gate/`, `full_dqvsa/`.

## 1. What this exploration was

The gate experiment asked whether extending the system to W4A4 NVFP4
linear GEMMs ("Full-DQ-VSA") is justified. The answer is a clean,
quantified NO for this serving stack. This is supplementary boundary
evidence; it is not a paper contribution.

## 2. Findings [E]

1. **FLOPs != latency, measured precisely.** Linear GEMMs are 78.2%/62.7%
   of nominal forward FLOPs (480p/720p) but only 11.3%/10.8% of measured
   GPU time in the eager stack — verified with two independent methods
   (module-level CUDA events; kernel-trace categorization), stable across
   batch 1/2/4 (`w4a4_gate/RUNTIME_BREAKDOWN_EAGER.md`).
2. **Reasonable serving optimization (D1)**: torch.compile on the blocks
   gives 1.39x/1.34x forward speedup with equivalent outputs (cosine
   0.9994) and raises the GEMM share to 23.7%/21.9%
   (`SERVING_OPTIMIZATIONS.md`, `RUNTIME_BREAKDOWN_OPTIMIZED.md`). The
   dominant residual is sparse-attention integration machinery
   (tiling scatter/gather, layout copies, quantize; ~38-40%).
3. **Native W4A4 works and is honest about its costs** [E]: production
   NVFP4 weights+activations through `flashinfer.mm_fp4` (cudnn backend;
   trtllm's PDL deadlocks against the persistent FA4 kernel in-model;
   cutlass JIT impractical). Kernel level: FFN 1.5-2.0x incl. activation
   quant; QKV 1.07x; o_proj 0.39x (slower — K=1536 memory-bound).
4. **In-model, W4A4 is a slowdown everywhere** [E]: eager +2.4%..+19%,
   compiled +4.5%..+33% across the W1 (FFN) / W2 (+o_proj) / W3 (+QKV)
   ladder at both resolutions; peak memory grows (transient quant
   buffers); no throughput or capacity benefit (shares flat in batch;
   compute-bound, not memory-bound).
5. **Gate decision** (`FULL_DQVSA_GATE_DECISION.md`): no gate (A latency /
   B bottleneck-shift / C throughput / D memory) triggers. **STOP
   FULL-DQ-VSA — AMDAHL-LIMITED IN THIS SERVING STACK.** No W4A4
   QAT/distillation, no 326-prompt W4A4 evaluation.

## 3. Relation to the main paper's systems claim (do not broaden)

The MAIN paper's bottleneck claim remains narrowly about the
sparse-attention composition study (REPORT_V4/PAPER_UPDATE_V4 wording):
sparsification changes the attention kernel's bottleneck composition, so
FP4's arithmetic advantage does not automatically compound with sparsity
at E2E. The W4A4 observation (GEMMs dominate FLOPs but not measured time
in this stack) is CONSISTENT supporting context for a Discussion paragraph
but must not be promoted into a universal "FLOPs are not latency" thesis —
it does not support claims beyond this serving stack.

## 4. State of claims

- V4 claims stand unchanged and canonical (native SparseFP4, composition,
  geometry, quality cost + interaction, DQ-VSA recovery with B0=T3-c500,
  backward ablation).
- Supplementary [E]: the FLOP-vs-time decomposition and the
  measured-negative W4A4 result above (appendix material).
- [H] recorded for completeness (not proposed work): fused
  norm->NVFP4-quant epilogues + CUDA-graph-safe sparse attention could
  shift the calculus; attention integration machinery (~40% of time) is
  the largest single optimization target in this stack.

## 5. Program status

All experiments are closed. The paper is frozen at V4
(`REPORT_CANONICAL.md`); this exploration feeds at most one appendix
section (`supplementary/w4a4_gate/`).
