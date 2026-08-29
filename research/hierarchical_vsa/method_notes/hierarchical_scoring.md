# Hierarchical scoring derivation

The frozen Fine-VSA scorer returns a logit for each pooled child key:

```text
child_logit = Q_coarse · mean(K_child) / sqrt(d)
```

It does not return an attention probability or calibrated mass. Therefore
adding child logits is not mathematically meaningful.

For the primary hierarchical-mass score, each pooled child logit is treated
as applying to its valid tokens. Parent ranking uses:

```text
logsumexp(child_logit + log(valid_child_tokens))
```

This is equivalent, up to a query-row normalization constant, to summing
the token-multiplicity-weighted exponentiated child evidence. Empty padded
children are masked.

The allowed diagnostics are unweighted log-sum-exp, top-2 mean logit, and
maximum child logit. A soft native prior, when evaluated, blends z-scored
hierarchical and native parent scores at the predeclared weights 0, 0.25,
and 0.50. No execution block is guaranteed by the prior.

KV64 candidates select 125 parent descriptors while exactly reproducing the
native selected valid-size histogram for every query row. KV128 candidates
select 62 disjoint adjacent parent pairs and use a size-penalized fixed-count
selection whose valid-token total is asserted not to exceed native VSA80.
