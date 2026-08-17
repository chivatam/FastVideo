| seed | arm | arm_label | n_cells | jaccard_median | wrong_mask_excess_median | sparsification_error_median | abs_wrong_mask_over_sparsification_median | isolation_ratio_abs | frac_cells_arm_worse_than_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1234 | R1 | repr=nvfp4, pool=fp64/native, score=fp64/native | 43200 | 0.9751 | 4.057e-05 | 0.1786 | 0.001373 | 21.05 | 0.5635 |
| 1234 | R3 | repr=nvfp4, pool=fp32/native, score=fp32/native | 43200 | 0.9751 | 4.057e-05 | 0.1786 | 0.001374 | 20.88 | 0.5635 |
| 1234 | R6 | repr=bf16, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9598 | 2.233e-05 | 0.1786 | 0.00166 | 67.04 | 0.5316 |
| 1234 | R7 | repr=nvfp4, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9529 | 6.79e-05 | 0.1786 | 0.002144 | 24.98 | 0.5684 |
| 1234 | R8 | repr=bf16, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8874 | 0.000316 | 0.1786 | 0.006523 | 14.28 | 0.6072 |
| 1234 | R9 | repr=nvfp4, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8849 | 0.0003794 | 0.1786 | 0.007 | 12.18 | 0.6197 |
| 2026 | R1 | repr=nvfp4, pool=fp64/native, score=fp64/native | 43200 | 0.9761 | 4.095e-05 | 0.1788 | 0.001403 | 18.74 | 0.5609 |
| 2026 | R3 | repr=nvfp4, pool=fp32/native, score=fp32/native | 43200 | 0.9761 | 4.095e-05 | 0.1788 | 0.001404 | 19.07 | 0.5609 |
| 2026 | R6 | repr=bf16, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9608 | 2.006e-05 | 0.1788 | 0.00178 | 66.89 | 0.5243 |
| 2026 | R7 | repr=nvfp4, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9545 | 5.762e-05 | 0.1788 | 0.002206 | 26.14 | 0.555 |
| 2026 | R8 | repr=bf16, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8902 | 0.0003026 | 0.1788 | 0.006592 | 13.29 | 0.5977 |
| 2026 | R9 | repr=nvfp4, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8878 | 0.0003449 | 0.1788 | 0.006991 | 11.88 | 0.6059 |
| 3407 | R1 | repr=nvfp4, pool=fp64/native, score=fp64/native | 43200 | 0.976 | 4.21e-05 | 0.1731 | 0.001375 | 17.97 | 0.5674 |
| 3407 | R3 | repr=nvfp4, pool=fp32/native, score=fp32/native | 43200 | 0.976 | 4.202e-05 | 0.1731 | 0.001375 | 18.01 | 0.5674 |
| 3407 | R6 | repr=bf16, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9608 | 1.178e-05 | 0.1731 | 0.001756 | 113.6 | 0.517 |
| 3407 | R7 | repr=nvfp4, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 43200 | 0.9542 | 5.784e-05 | 0.1731 | 0.002229 | 26.08 | 0.557 |
| 3407 | R8 | repr=bf16, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8901 | 0.0002944 | 0.1731 | 0.006632 | 13.44 | 0.6018 |
| 3407 | R9 | repr=nvfp4, pool=fp32+nvfp4/native, score=fp32/native | 43200 | 0.8877 | 0.0003413 | 0.1731 | 0.006959 | 11.62 | 0.6087 |
| pooled | R1 | repr=nvfp4, pool=fp64/native, score=fp64/native | 129600 | 0.9757 | 4.114e-05 | 0.177 | 0.001384 | 19.28 | 0.5639 |
| pooled | R3 | repr=nvfp4, pool=fp32/native, score=fp32/native | 129600 | 0.9757 | 4.112e-05 | 0.177 | 0.001384 | 19.35 | 0.5639 |
| pooled | R6 | repr=bf16, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 129600 | 0.9605 | 1.793e-05 | 0.177 | 0.001735 | 77.89 | 0.5243 |
| pooled | R7 | repr=nvfp4, pool=fp32+fp8_e4m3/native, score=fp8_e4m3/native | 129600 | 0.9539 | 6.126e-05 | 0.177 | 0.002194 | 25.6 | 0.5601 |
| pooled | R8 | repr=bf16, pool=fp32+nvfp4/native, score=fp32/native | 129600 | 0.8893 | 0.0003042 | 0.177 | 0.006587 | 13.68 | 0.6022 |
| pooled | R9 | repr=nvfp4, pool=fp32+nvfp4/native, score=fp32/native | 129600 | 0.8868 | 0.0003551 | 0.177 | 0.006986 | 11.92 | 0.6114 |
