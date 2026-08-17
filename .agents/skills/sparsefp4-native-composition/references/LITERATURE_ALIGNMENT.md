# Literature Alignment

This skill intentionally follows experimental patterns already used in closely related work rather than inventing new headline metrics.

## FPSAttention — arXiv:2506.04648, NeurIPS 2025 Spotlight
Closest experimental template for quantization + sparsity in video attention.

Patterns to follow:
- quantization-only, sparsity-only, and joint quantization+sparsity comparisons,
- native hardware-friendly kernel,
- standard video quality evaluation including VBench and fidelity metrics,
- kernel-level and end-to-end speed,
- denoising-step sensitivity only as an explanatory ablation.

Our difference:
- target native **NVFP4 on Blackwell**, not their FP8/Hopper design.

## QuantSparse — arXiv:2509.23681
Directly studies model quantization + attention sparsification and reports that naive composition can amplify attention shifts.

Patterns to follow:
- show quant-only / sparse-only / joint raw results,
- measure attention error directly,
- use standard quality metrics and end-to-end speed,
- only propose a remedy after showing the naive joint baseline fails.

## VSA — arXiv:2505.13389, NeurIPS 2025
The sparse production baseline in FastVideo.

Patterns to follow:
- compare full vs deployed sparse attention,
- report actual attention and end-to-end latency,
- use sparsity/tile ablations when relevant,
- preserve the real VSA selector/coarse branch for production comparisons.

## Attn-QAT — arXiv:2603.00040
Systematic 4-bit attention study with FP4 inference kernels and QAT.

Use only as a later pivot if native sparse FP4 quality is unacceptable.
Do not introduce training before the naive native joint baseline is measured.

## SLA2 — arXiv:2602.12675
Sparse + low-bit attention with QAT.

Use as evidence that training-based remediation is legitimate if needed, not as a replacement for the direct native composition baseline.

# Evaluation principle

The primary experiment should look deliberately conventional:

```text
Dense BF16
Dense low precision
Sparse BF16
Sparse + low precision
```

followed by:

```text
standard attention error
standard video quality
actual latency/speedup
```

Custom routing metrics are allowed only for diagnosis after a surprising joint result.
