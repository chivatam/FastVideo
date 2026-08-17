| sparsity | quantity | followup_R1_value | study1_frozen_value | note |
| --- | --- | --- | --- | --- |
| 0.9 | mask_jaccard_median[router=nvfp4] | 0.9738 | 0.9738 | R1 reproduces study 1's condition exactly (nvfp4 representation, fp64 scorer); any gap is prompt/layer coverage, not method |
| 0.9 | isolation_ratio | 30.81 | 21.66 | ratio of medians in both studies |
| 0.9 | wrong_mask_excess_median | 2.653e-05 | 3.326e-05 | signed paired difference |
