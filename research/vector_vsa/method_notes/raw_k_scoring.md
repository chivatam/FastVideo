# Raw-K scoring semantics

Raw-K changes only the K-side routing representation.

The native query representation is retained exactly: each native 64-token
query region is pooled with FastVideo's existing valid-token-aware
`fused_block_mean`. The routing scale remains `1 / sqrt(head_dim)`.

For each pooled query `Qc`, Raw-K computes:

```text
score[j] = dot(Qc, K[j]) / sqrt(head_dim)
```

against every original padded-sequence K position. Padding positions are
masked before selection. Scores are ranked as raw logits, matching the
rank-preserving convention used by native and Fine-VSA routing.

No K normalization, centroid, local mean, temperature change, softmax
before ranking, learned projection, or query-side change is introduced.

Fine8's pooled score is the arithmetic mean of these same raw token logits
for a full eight-token segment:

```text
dot(Qc, mean(K_8)) = mean_j dot(Qc, K_j)
```

The Raw-score Vec8/Vec16 variants therefore isolate the aggregation rule:
they compute token logits first and then apply max, top-two mean, or
log-sum-exp within a contiguous execution segment.

All candidates retain the same native coarse residual branch. Exact sparse
output always uses the selected original K/V values.
