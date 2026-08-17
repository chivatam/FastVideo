# Factorial interaction: does sparsity amplify the NVFP4 quality effect?

I = (P4 - P4G) - (P1 - P0) per prompt (difference-in-differences). I < 0 means NVFP4 hurts MORE under VSA256 sparsity than under dense attention; I ~ 0 means the NVFP4 effect is the same in both regimes. Defined on VBench dimensions only (pixel metrics use P0 as reference, so the dense NVFP4 term is degenerate).

Prompt-level percentile bootstrap, 10000 resamples, seed 20260817; two-sided bootstrap p; Holm correction across the 7 VBench dimensions.

## VBench dimensions (paired Δ)

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0144 | -0.0091 | [-0.0241, -0.0049] | 0.0032 | 0.0096 | yes |
| background_consistency | 86 | -0.0011 | -0.0034 | [-0.0054, +0.0033] | 0.5890 | 1.0000 | no |
| temporal_flickering | 75 | +0.0123 | +0.0125 | [+0.0101, +0.0146] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0143 | +0.0143 | [+0.0116, +0.0171] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | +0.0556 | +0.0000 | [-0.0833, +0.1944] | 0.5148 | 1.0000 | no |
| imaging_quality | 93 | +0.0426 | +0.0347 | [+0.0163, +0.0690] | 0.0024 | 0.0096 | yes |
| aesthetic_quality | 93 | +0.0490 | +0.0422 | [+0.0312, +0.0673] | 0.0002 | 0.0014 | yes |
