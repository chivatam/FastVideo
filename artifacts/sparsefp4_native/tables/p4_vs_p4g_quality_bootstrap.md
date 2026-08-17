# P4 - P4G: sparse NVFP4 quality effect (paired, 326-prompt protocol)

Δ_sparse = P4 - P4G per prompt (identical VSA256 geometry, only fine QK precision differs); positive = NVFP4 scores higher. Pixel Δ compares each arm's similarity-to-P0.

Prompt-level percentile bootstrap, 10000 resamples, seed 20260817; two-sided bootstrap p; Holm correction across the 7 VBench dimensions.

## VBench dimensions (paired Δ)

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0060 | -0.0043 | [-0.0135, +0.0011] | 0.0976 | 0.1952 | no |
| background_consistency | 86 | +0.0033 | +0.0033 | [+0.0004, +0.0064] | 0.0218 | 0.0654 | no |
| temporal_flickering | 75 | +0.0087 | +0.0084 | [+0.0060, +0.0114] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0162 | +0.0156 | [+0.0138, +0.0188] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | -0.2500 | +0.0000 | [-0.3472, -0.1528] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | -0.1009 | -0.1065 | [-0.1120, -0.0899] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | -0.0063 | -0.0048 | [-0.0135, +0.0010] | 0.1010 | 0.1952 | no |

## Pixel metrics vs P0 reference (paired Δ of similarity-to-P0)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | +0.9688 | +0.9599 | [+0.8785, +1.0589] | 0.0002 |
| ssim | 326 | +0.0169 | +0.0159 | [+0.0137, +0.0201] | 0.0002 |
| lpips | 326 | -0.0454 | -0.0429 | [-0.0489, -0.0419] | 0.0002 |
