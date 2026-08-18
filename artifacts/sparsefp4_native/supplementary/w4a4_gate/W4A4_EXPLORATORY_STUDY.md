# W4A4_EXPLORATORY_STUDY — supplementary negative systems study

> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** The canonical paper
> state is REPORT_V4.md / RESULTS_DECISION_V4.md / PAPER_UPDATE_V4.md.
> This study is boundary evidence: it explains why the proposed system
> stops at NVFP4 attention and does not extend to W4A4 linear GEMMs.

## What was asked

After the DQ-VSA recovery result (V4), we explored whether the system
should be extended with W4A4 NVFP4 linear GEMMs ("Full-DQ-VSA"), gated on
measured serving economics rather than FLOP counts.

## What was measured (all receipts under `../../w4a4_gate/`)

1. **FLOP-vs-time decomposition** (two independent methods, batch 1/2/4,
   480p+720p): linear GEMMs are 78.2%/62.7% of nominal forward FLOPs but
   only 11.3%/10.8% of measured GPU time in the eager stack
   (`RUNTIME_BREAKDOWN_EAGER.md`).
2. **Reasonable serving optimization** (torch.compile on blocks, outputs
   equivalent, 1.34-1.39x forward): GEMM share rises to 23.7%/21.9%
   (`SERVING_OPTIMIZATIONS.md`, `RUNTIME_BREAKDOWN_OPTIMIZED.md`).
3. **Native W4A4 PTQ ladder** (flashinfer `mm_fp4`, production NVFP4
   weights+activations, layout-verified): FFN GEMM kernels 1.5-2.0x faster
   including activation quant, but **in-model E2E is slower in every
   configuration** (eager +2.4..19%, compiled +4.5..33%)
   (`W4A4_PTQ_PERFORMANCE.md`).
4. **Gate decision**: no gate (latency / bottleneck-shift / throughput /
   memory) triggered -> W4A4 QAT/distillation correctly not run
   (`FULL_DQVSA_GATE_DECISION.md`, summarized in
   `W4A4_NEGATIVE_RESULT.md`).

## How this may appear in the paper (maximum exposure)

One short Discussion/Limitations paragraph:

> We additionally explored extending NVFP4 to the model's linear GEMMs.
> Although these GEMMs account for 63-78% of nominal FLOPs, they
> represented only 11% of eager GPU time (22-24% after compilation) in our
> serving stack. Native W4A4 FFN kernels improved locally, but activation
> quantization/integration overhead produced no E2E gain. We therefore
> gated off further W4A4 training. We report this exploratory result in
> the appendix.

Everything else (tables, Amdahl panel, engineering receipts) is appendix
material only. This study must not appear in the title, abstract headline,
introduction's first paragraph, main contribution bullets, or the primary
system diagram.
