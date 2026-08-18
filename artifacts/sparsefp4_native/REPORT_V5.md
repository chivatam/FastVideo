# REPORT_V5 — final: the W4A4 gate study closes the experimental program

Supersedes REPORT_V4.md only by addition; V4's DQ-VSA recovery results are
unchanged. Tags: [E] experimental, [S] statistical, [L] literature, [H]
hypothesis. Receipts: `w4a4_gate/`, `full_dqvsa/`.

## 1. What V5 adds

The final gate experiment asked whether extending the system to W4A4 NVFP4
linear GEMMs ("Full-DQ-VSA") is justified. The answer is a clean,
quantified NO for this serving stack — and it yields the paper's second
bottleneck-migration result.

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

## 3. The paper's central systems thesis, now doubly evidenced [E+S]

Two independent bottleneck migrations in one system:

1. **Sparse + FP4 attention** (V2-V3): FP4's QK MMA advantage (1.26x dense)
   shrinks to ~1.04x at 10% retention and is consumed E2E by softmax-unit
   predicates, quantization, and allocator behavior.
2. **W4A4 linears** (V5): GEMMs dominate FLOPs but not time; real 1.5-2x
   FFN kernels produce negative E2E once activation-quant and integration
   overheads are paid; even infinite GEMM speed caps at 1.28-1.31x on the
   optimized stack.

Conclusion (binding wording): *arithmetic-intensity reduction does not
guarantee proportional end-to-end acceleration in video DiTs, because each
optimization changes the bottleneck composition.* The productive levers in
this system were, in measured order: selector-geometry alignment (1.40x),
serving-stack compilation (1.34-1.39x forward), and teacher-preserving
distillation for quality (V4) — not further arithmetic compression.

## 4. Final state of claims

- V4 claims 1-6 stand (native SparseFP4, composition, geometry, quality
  cost + interaction, DQ-VSA recovery with B0=T3-c500, backward ablation).
- New [E]: the FLOP-vs-time decomposition and the measured-negative W4A4
  result above.
- [H] left explicitly open (future work, not run): fused norm->NVFP4-quant
  epilogues + CUDA-graph-safe sparse attention could shift the calculus;
  attention integration machinery (~40% of time) is the largest single
  optimization target.

## 5. Program status

All experiments are closed per the stop conditions (V4 recovery study
complete; V5 gate negative). Remaining work is paper writing
(`PAPER_UPDATE_V5.md` skeleton).
