# W4A4_NEGATIVE_RESULT — supplementary gate outcome

> **SUPPLEMENTARY / NOT PART OF MAIN PAPER CLAIMS.** Canonical paper state:
> REPORT_V4.md. This records why the W4A4 extension was gated off.

## Measured ladder (B0 weights, native P4 attention, median of 5 forwards)

Eager stack:

| Config | 480p ms (Δ vs W0) | 720p ms (Δ) |
|---|---|---|
| W0 all-BF16 linears | 447.9 | 1076.1 |
| W1 +W4A4 FFN | 476.7 (+6.4%) | 1102.2 (+2.4%) |
| W2 +o_proj | 490.7 (+9.6%) | 1122.2 (+4.3%) |
| W3 +QKV | 533.0 (+19.0%) | 1167.1 (+8.4%) |

Compiled (D1) stack: W1 +15.1%/+4.5%, W3 +33.2%/+12.4% slower. Peak memory
increased in all W-arms (transient quant buffers). Kernel-level receipts:
FFN GEMMs 1.5-2.0x faster incl. quant; QKV 1.07x; o_proj 0.39-0.49x
(slower). Native path proven (production NVFP4 packing + `mm_fp4`
block-scaled GEMM; backend layout verified against BF16 reference at the
NVFP4 arithmetic floor, rel-L2 0.134/GEMM).

## Gate audit

- Gate A (>=10% E2E latency): FAILED (sign negative everywhere).
- Gate B (GEMMs >=20-25% of D1 time AND credible >=10% E2E path): first
  clause borderline (21.9-23.7%); second clause FAILED (measured path is
  negative; realistic 2x-GEMM ceiling ~1.12-1.13x).
- Gate C (>=15-20% throughput/concurrency): FAILED (shares flat in batch;
  throughput tracks the worsened latency).
- Gate D (memory/capacity economics): FAILED (peak grew; capacity is
  compute-bound at 28 GB / 183 GB).

**Decision: STOP FULL-DQ-VSA — AMDAHL-LIMITED IN THIS SERVING STACK.**
No W4A4 QAT/distillation, no 326-prompt W4A4 evaluation, no quality triage
(moot: gate fails on performance alone).

## Value of the negative result

It confirms, with kernel receipts, that the V4 system's boundary is
correctly drawn at NVFP4 attention: further arithmetic compression of the
linears does not pay in this serving regime, whereas the measured winning
levers were geometry alignment (main paper), compilation (supplementary
observation, 1.34-1.39x), and distillation-based quality recovery (main
paper).
