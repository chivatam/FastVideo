# Cluster-VSA exact-pair accounting

The native geometry has 624 padded KV slots of width 64. Their valid-token
capacities sum to 32,760 and include ragged spatial-boundary slots.

Cluster-VSA changes token membership and capacity order, not the capacity
multiset:

```text
cluster slot width = 64
cluster slot valid capacities = sorted native parent-size multiset
selected slots per query block = 125
nominal support = 125 * 64 = 8,000
```

For every query row, the clustered selector exactly reproduces the valid-size
histogram of the 125 slots selected by native VSA80. It ranks clustered slots
only within each valid-size category and takes the same category counts as the
native row. Reordering the capacity multiset therefore cannot change valid
pair count.

Therefore:

```text
cluster nominal Q-K pairs = native nominal Q-K pairs
cluster valid Q-K pairs = native valid Q-K pairs
```

The equality is checked per query row and again after weighting by each
query block's valid-token count. Padding is never treated as a semantic token.

This accounting prevents Cluster-VSA from receiving an artificial quality
advantage by replacing ragged native selections with 125 fully valid
64-token clusters.
