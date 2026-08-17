# P4G - P2: geometry-aligned VSA256/FA4 vs deployed VSA (paired, BF16)

Δ_geometry = P4G - P2 per prompt (both BF16 sparse; selector tile geometry and fine kernel differ); positive = VSA256/FA4 scores higher. Pixel Δ compares each arm's similarity-to-P0. No non-inferiority margin is asserted; observed effects + CIs only.

Prompt-level percentile bootstrap, 10000 resamples, seed 20260817; two-sided bootstrap p; Holm correction across the 7 VBench dimensions.

## VBench dimensions (paired Δ)

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | +0.0006 | -0.0008 | [-0.0063, +0.0075] | 0.8794 | 1.0000 | no |
| background_consistency | 86 | -0.0078 | -0.0078 | [-0.0107, -0.0052] | 0.0002 | 0.0014 | yes |
| temporal_flickering | 75 | -0.0029 | -0.0030 | [-0.0058, -0.0000] | 0.0476 | 0.1428 | no |
| motion_smoothness | 72 | -0.0175 | -0.0159 | [-0.0203, -0.0147] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | +0.1389 | +0.0000 | [+0.0694, +0.2222] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | +0.0021 | +0.0004 | [-0.0113, +0.0152] | 0.7384 | 1.0000 | no |
| aesthetic_quality | 93 | -0.0298 | -0.0248 | [-0.0400, -0.0203] | 0.0002 | 0.0014 | yes |

## Pixel metrics vs P0 reference (paired Δ of similarity-to-P0)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | -0.2515 | -0.2179 | [-0.3341, -0.1712] | 0.0002 |
| ssim | 326 | -0.0061 | -0.0034 | [-0.0093, -0.0030] | 0.0002 |
| lpips | 326 | +0.0235 | +0.0186 | [+0.0198, +0.0272] | 0.0002 |
