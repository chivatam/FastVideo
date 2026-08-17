# P1 - P0: dense NVFP4 quality effect (paired, 326-prompt protocol)

Δ_dense = P1 - P0 per prompt; positive = NVFP4 scores higher. Pixel metrics are similarity of P1 to the P0 reference, so no paired pixel difference exists for this contrast; levels below.

Prompt-level percentile bootstrap, 10000 resamples, seed 20260817; two-sided bootstrap p; Holm correction across the 7 VBench dimensions.

## VBench dimensions (paired Δ)

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | +0.0084 | -0.0003 | [+0.0009, +0.0165] | 0.0262 | 0.0524 | no |
| background_consistency | 86 | +0.0045 | +0.0050 | [+0.0014, +0.0074] | 0.0040 | 0.0120 | yes |
| temporal_flickering | 75 | -0.0036 | -0.0027 | [-0.0052, -0.0021] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0020 | -0.0002 | [-0.0003, +0.0043] | 0.0844 | 0.0844 | no |
| dynamic_degree | 72 | -0.3056 | +0.0000 | [-0.4167, -0.1944] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | -0.1435 | -0.1359 | [-0.1681, -0.1186] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | -0.0553 | -0.0460 | [-0.0717, -0.0388] | 0.0002 | 0.0014 | yes |

## P1 similarity to P0 (levels, bootstrap CI) — no null hypothesis; describes how far dense NVFP4 output is from dense BF16

| Metric | n | mean | median | 95% CI |
|---|---|---|---|---|
| psnr | 326 | +15.7816 | +15.4192 | [+15.5644, +16.0146] |
| ssim | 326 | +0.3815 | +0.3563 | [+0.3665, +0.3963] |
| lpips | 326 | +0.6139 | +0.6101 | [+0.5991, +0.6289] |
