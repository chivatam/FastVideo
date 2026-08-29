# Exact Online-Softmax Merge

The held-out gate selected Core25 and unlocked this implementation stage.
The static KV64 path uses the B200 SM100a kernel, the dynamic tail uses the
frozen Fine8 support with explicit per-row descriptor counts, and a fused
Triton kernel merges their log-base-two LSE outputs.

If unlocked, static KV64 and dynamic KV8 kernels will return per-row partial
triples `(m, l, O)`. For disjoint subsets A and B:

`m = max(m_A, m_B)`

`l = exp(m_A-m)*l_A + exp(m_B-m)*l_B`

`O = (exp(m_A-m)*l_A*O_A + exp(m_B-m)*l_B*O_B) / l`

This is mathematically equivalent to one softmax over the union. Independently
normalized outputs are never added. The implementation validates the merged
output against one Fine8 kernel over the exact static-union-dynamic support
before a measured generation is accepted.

Core25 always fits the minimum valid-token capacity of any 125 native KV64
blocks in the frozen 624-block Wan geometry. Its IDs are prompt invariant and
the full Core25 table is preloaded on the GPU at model setup. Runtime static
selection performs no top-k; only the dynamic Fine8 tail is scored online.
