# DQ-VSA recovery — paper-scale paired bootstrap (326-prompt protocol)

All candidates served through the NATIVE P4 path (VSA256/FA4, exact 10% retention, BF16 selector, native NVFP4 QK, BF16 PV; serving receipts in `DQVSA_NATIVE_SERVING_PROOF.md`).

- **T2** = velocity distillation from frozen P4G teacher + fake-quant NVFP4 forward + **naive/high-precision-style attention backward**
- **T3** = velocity distillation from frozen P4G teacher + fake-quant NVFP4 forward + **Attn-QAT-consistent backward semantics**
- (T1 = task-loss flow-matching QAT; excluded from paper-scale eval after dev-gate motion collapse, `t_matrix_gates.md`.)

Unified statistics: 10k prompt-level bootstrap, 95% CI, two-sided bootstrap p, Holm across the 7 VBench dimensions per contrast. Pixel metrics reported but NOT used for winner selection.

## Descriptive recovery fractions (pre-declared targets; NOT a significance statistic)

| Candidate | Dimension | n | trained mean | P4 mean | P4G mean | recovery |
|---|---|---|---|---|---|---|
| T2c250 | imaging_quality | 93 | 0.5693 | 0.4837 | 0.5847 | 85% |
| T2c250 | dynamic_degree | 72 | 0.9583 | 0.7083 | 0.9583 | 100% |
| T3c250 | imaging_quality | 93 | 0.5011 | 0.4837 | 0.5847 | 17% |
| T3c250 | dynamic_degree | 72 | 0.9583 | 0.7083 | 0.9583 | 100% |
| T3c500 | imaging_quality | 93 | 0.5398 | 0.4837 | 0.5847 | 56% |
| T3c500 | dynamic_degree | 72 | 0.8472 | 0.7083 | 0.9583 | 56% |

## T2c250 - P4G

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0120 | -0.0093 | [-0.0193, -0.0050] | 0.0008 | 0.0040 | yes |
| background_consistency | 86 | -0.0046 | -0.0057 | [-0.0071, -0.0021] | 0.0008 | 0.0040 | yes |
| temporal_flickering | 75 | -0.0258 | -0.0256 | [-0.0293, -0.0223] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | -0.0023 | -0.0031 | [-0.0058, +0.0012] | 0.1930 | 0.3860 | no |
| dynamic_degree | 72 | +0.0000 | +0.0000 | [-0.0556, +0.0556] | 1.0000 | 1.0000 | no |
| imaging_quality | 93 | -0.0154 | -0.0182 | [-0.0302, -0.0001] | 0.0490 | 0.1470 | no |
| aesthetic_quality | 93 | +0.0216 | +0.0213 | [+0.0137, +0.0297] | 0.0002 | 0.0014 | yes |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | -0.3446 | -0.3300 | [-0.4691, -0.2180] | 0.0002 |
| ssim | 326 | -0.0053 | -0.0012 | [-0.0097, -0.0011] | 0.0134 |
| lpips | 326 | -0.0034 | -0.0026 | [-0.0076, +0.0008] | 0.1156 |

## T2c250 - P4

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0060 | -0.0042 | [-0.0132, +0.0010] | 0.0928 | 0.0928 | no |
| background_consistency | 86 | -0.0080 | -0.0091 | [-0.0102, -0.0055] | 0.0002 | 0.0014 | yes |
| temporal_flickering | 75 | -0.0345 | -0.0352 | [-0.0377, -0.0312] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | -0.0185 | -0.0167 | [-0.0220, -0.0151] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | +0.2500 | +0.0000 | [+0.1528, +0.3472] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | +0.0855 | +0.0791 | [+0.0683, +0.1022] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | +0.0279 | +0.0267 | [+0.0192, +0.0367] | 0.0002 | 0.0014 | yes |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | -1.3134 | -1.3017 | [-1.4548, -1.1735] | 0.0002 |
| ssim | 326 | -0.0222 | -0.0183 | [-0.0266, -0.0182] | 0.0002 |
| lpips | 326 | +0.0420 | +0.0393 | [+0.0372, +0.0469] | 0.0002 |

## T3c250 - P4G

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0136 | -0.0126 | [-0.0218, -0.0055] | 0.0004 | 0.0014 | yes |
| background_consistency | 86 | -0.0108 | -0.0103 | [-0.0143, -0.0074] | 0.0002 | 0.0014 | yes |
| temporal_flickering | 75 | -0.0229 | -0.0227 | [-0.0264, -0.0192] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0125 | +0.0097 | [+0.0085, +0.0167] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | +0.0000 | +0.0000 | [-0.0417, +0.0417] | 1.0000 | 1.0000 | no |
| imaging_quality | 93 | -0.0836 | -0.0896 | [-0.0985, -0.0683] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | -0.0009 | -0.0036 | [-0.0072, +0.0057] | 0.7882 | 1.0000 | no |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | +1.2902 | +1.2456 | [+1.1421, +1.4404] | 0.0002 |
| ssim | 326 | +0.0200 | +0.0183 | [+0.0154, +0.0245] | 0.0002 |
| lpips | 326 | -0.0458 | -0.0417 | [-0.0513, -0.0405] | 0.0002 |

## T3c250 - P4

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0076 | -0.0104 | [-0.0141, -0.0006] | 0.0332 | 0.1168 | no |
| background_consistency | 86 | -0.0142 | -0.0121 | [-0.0171, -0.0113] | 0.0002 | 0.0014 | yes |
| temporal_flickering | 75 | -0.0315 | -0.0308 | [-0.0348, -0.0283] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | -0.0037 | -0.0044 | [-0.0068, -0.0004] | 0.0306 | 0.1168 | no |
| dynamic_degree | 72 | +0.2500 | +0.0000 | [+0.1528, +0.3611] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | +0.0174 | +0.0208 | [+0.0016, +0.0335] | 0.0292 | 0.1168 | no |
| aesthetic_quality | 93 | +0.0054 | +0.0005 | [-0.0023, +0.0133] | 0.1666 | 0.1666 | no |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | +0.3214 | +0.1905 | [+0.1806, +0.4664] | 0.0002 |
| ssim | 326 | +0.0031 | +0.0023 | [-0.0013, +0.0076] | 0.1696 |
| lpips | 326 | -0.0004 | +0.0024 | [-0.0048, +0.0040] | 0.8570 |

## T3c500 - P4G

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | +0.0257 | +0.0225 | [+0.0188, +0.0329] | 0.0002 | 0.0014 | yes |
| background_consistency | 86 | +0.0067 | +0.0062 | [+0.0041, +0.0095] | 0.0002 | 0.0014 | yes |
| temporal_flickering | 75 | +0.0072 | +0.0079 | [+0.0044, +0.0099] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0203 | +0.0184 | [+0.0174, +0.0233] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | -0.1111 | +0.0000 | [-0.1944, -0.0278] | 0.0072 | 0.0144 | yes |
| imaging_quality | 93 | -0.0448 | -0.0559 | [-0.0579, -0.0315] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | -0.0055 | -0.0038 | [-0.0133, +0.0021] | 0.1518 | 0.1518 | no |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | +0.8342 | +0.7892 | [+0.7042, +0.9628] | 0.0002 |
| ssim | 326 | +0.0157 | +0.0198 | [+0.0118, +0.0193] | 0.0002 |
| lpips | 326 | -0.0108 | -0.0115 | [-0.0154, -0.0064] | 0.0002 |

## T3c500 - P4

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | +0.0317 | +0.0291 | [+0.0249, +0.0386] | 0.0002 | 0.0014 | yes |
| background_consistency | 86 | +0.0034 | +0.0013 | [+0.0013, +0.0057] | 0.0020 | 0.0060 | yes |
| temporal_flickering | 75 | -0.0015 | -0.0017 | [-0.0038, +0.0008] | 0.2026 | 0.4052 | no |
| motion_smoothness | 72 | +0.0041 | +0.0027 | [+0.0022, +0.0060] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | +0.1389 | +0.0000 | [+0.0694, +0.2222] | 0.0004 | 0.0016 | yes |
| imaging_quality | 93 | +0.0561 | +0.0589 | [+0.0401, +0.0719] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | +0.0008 | +0.0067 | [-0.0075, +0.0087] | 0.8446 | 0.8446 | no |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | -0.1346 | -0.1671 | [-0.2914, +0.0182] | 0.0828 |
| ssim | 326 | -0.0012 | +0.0064 | [-0.0053, +0.0029] | 0.5888 |
| lpips | 326 | +0.0346 | +0.0334 | [+0.0301, +0.0391] | 0.0002 |

## P4 - P4G

### VBench

| Dimension | n | mean Δ | median Δ | 95% CI | p (boot) | p (Holm) | sig@0.05 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 72 | -0.0060 | -0.0043 | [-0.0134, +0.0011] | 0.0978 | 0.1932 | no |
| background_consistency | 86 | +0.0033 | +0.0033 | [+0.0004, +0.0063] | 0.0224 | 0.0672 | no |
| temporal_flickering | 75 | +0.0087 | +0.0084 | [+0.0060, +0.0113] | 0.0002 | 0.0014 | yes |
| motion_smoothness | 72 | +0.0162 | +0.0156 | [+0.0139, +0.0187] | 0.0002 | 0.0014 | yes |
| dynamic_degree | 72 | -0.2500 | +0.0000 | [-0.3472, -0.1528] | 0.0002 | 0.0014 | yes |
| imaging_quality | 93 | -0.1009 | -0.1065 | [-0.1119, -0.0894] | 0.0002 | 0.0014 | yes |
| aesthetic_quality | 93 | -0.0063 | -0.0048 | [-0.0138, +0.0012] | 0.0966 | 0.1932 | no |

### Pixel similarity-to-P0 (paired Δ; descriptive only)

| Metric | n | mean Δ | median Δ | 95% CI | p (boot) |
|---|---|---|---|---|---|
| psnr | 326 | +0.9688 | +0.9599 | [+0.8777, +1.0588] | 0.0002 |
| ssim | 326 | +0.0169 | +0.0159 | [+0.0137, +0.0201] | 0.0002 |
| lpips | 326 | -0.0454 | -0.0429 | [-0.0489, -0.0420] | 0.0002 |
