# Paper-scale quality (326 VBench prompts, 7 dims, paired seed 1234)

## VBench (dimension-routed, mean)

| Dim | n | P0 | P1 | P2 | P4G | P4 |
|---|---|---|---|---|---|---|
| subject_consistency | 72 | 0.9311 | 0.9395 | 0.8685 | 0.8691 | 0.8631 |
| background_consistency | 86 | 0.9700 | 0.9745 | 0.9665 | 0.9586 | 0.9620 |
| temporal_flickering | 75 | 0.9736 | 0.9700 | 0.9463 | 0.9434 | 0.9521 |
| motion_smoothness | 72 | 0.9713 | 0.9733 | 0.9504 | 0.9329 | 0.9491 |
| dynamic_degree | 72 | 0.7639 | 0.4583 | 0.8194 | 0.9583 | 0.7083 |
| imaging_quality | 93 | 0.6499 | 0.5064 | 0.5826 | 0.5847 | 0.4837 |
| aesthetic_quality | 93 | 0.5974 | 0.5421 | 0.3517 | 0.3218 | 0.3156 |

## Paired vs P0 (median)

| Arm | n | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| P1 | 326 | 15.42 | 0.3563 | 0.6101 |
| P2 | 326 | 9.04 | 0.1916 | 0.6965 |
| P4G | 326 | 8.74 | 0.1864 | 0.7265 |
| P4 | 326 | 9.83 | 0.2046 | 0.6804 |

## P4 - P4G paired differences (VBench, prompt-level bootstrap 10k, 95% CI)

| Dim | mean diff | CI low | CI high | significant |
|---|---|---|---|---|
| subject_consistency | -0.0060 | -0.0134 | +0.0010 | no |
| background_consistency | +0.0033 | +0.0004 | +0.0064 | yes |
| temporal_flickering | +0.0087 | +0.0060 | +0.0113 | yes |
| motion_smoothness | +0.0162 | +0.0139 | +0.0187 | yes |
| dynamic_degree | -0.2500 | -0.3611 | -0.1528 | yes |
| imaging_quality | -0.1009 | -0.1119 | -0.0897 | yes |
| aesthetic_quality | -0.0063 | -0.0132 | +0.0012 | no |
