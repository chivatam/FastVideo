# Exact Online-Softmax Merge

This implementation stage is locked behind the held-out offline quality gate.
No production systems work or 72-prompt generation is permitted unless a
Core25 or Core50 candidate passes every frozen offline criterion.

If unlocked, static KV64 and dynamic KV8 kernels will return per-row partial
triples `(m, l, O)`. For disjoint subsets A and B:

`m = max(m_A, m_B)`

`l = exp(m_A-m)*l_A + exp(m_B-m)*l_B`

`O = (exp(m_A-m)*l_A*O_A + exp(m_B-m)*l_B*O_B) / l`

This is mathematically equivalent to one softmax over the union. Independently
normalized outputs must never be added. The implementation must first match a
single-reference union on random ragged tensors, including adversarial logit
ranges, before latency benchmarking. Static IDs, permutations, and valid-token
masks should be precomputed at model load.
