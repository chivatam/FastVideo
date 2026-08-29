# Exact Pair Accounting

The reference capacity for each head and query block is the actual number of
valid key tokens selected by native VSA80's 125 KV64 descriptors. This handles
ragged boundary blocks without treating padding as work.

For CoreTail:

1. Select the frozen 31- or 62-parent static core.
2. In stable-rank order, admit each whole parent if it fits the query's native
   valid-token capacity. A parent that does not fit is skipped; its fixed
   tensor slot uses a zero-valid descriptor. This deterministic projection is
   required only where ragged native support is smaller than the nominal core.
3. Convert each admitted parent to its eight KV8 children.
4. Remove all admitted core children from the unchanged Fine8 candidate set.
5. Set the dynamic target to `native_valid_tokens - core_valid_tokens`.
6. Continue down the Fine8 ranking until that exact valid-token target is met,
   using zero-valid filler descriptors only for fixed tensor shape.
7. Form the disjoint union of static and dynamic support.

Runtime checks require zero duplicate valid support and exact equality between
the union's valid-token count and native VSA80 for every query. Reported pair
ratio is `selected valid Q-K pairs / native VSA80 valid Q-K pairs`; therefore
its maximum must be at most 1.000000. Nominal support remains 8,000 key tokens
per query, corresponding to 79.9679% sparsity in the established geometry.
