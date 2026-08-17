| sparsity | quantity | followup_R1_value | study1_frozen_value | note |
| --- | --- | --- | --- | --- |
| 0.8 | mask_jaccard_median[router=nvfp4] | 0.9812 | 0.9807 | R1 reproduces study 1's condition exactly (nvfp4 representation, fp64 scorer); any gap is prompt/layer coverage, not method |
| 0.8 | isolation_ratio | 29.52 | 27 | ratio of medians in both studies |
| 0.8 | wrong_mask_excess_median | 3.437e-05 | 3.096e-05 | signed paired difference |
| 0.9 | mask_jaccard_median[router=nvfp4] | 0.9751 | 0.9738 | R1 reproduces study 1's condition exactly (nvfp4 representation, fp64 scorer); any gap is prompt/layer coverage, not method |
| 0.9 | isolation_ratio | 21.05 | 21.66 | ratio of medians in both studies |
| 0.9 | wrong_mask_excess_median | 4.057e-05 | 3.326e-05 | signed paired difference |
| 0.95 | mask_jaccard_median[router=nvfp4] | 0.9646 | 0.9611 | R1 reproduces study 1's condition exactly (nvfp4 representation, fp64 scorer); any gap is prompt/layer coverage, not method |
| 0.95 | isolation_ratio | 14.35 | 10.08 | ratio of medians in both studies |
| 0.95 | wrong_mask_excess_median | 8.003e-05 | 9.52e-05 | signed paired difference |
