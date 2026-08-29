# Exact pair accounting

Native VSA64 first selects 125 descriptors. Ragged boundary descriptors
mean the valid-token support can be below the nominal `125 × 64 = 8000`.

For every batch/head/query-block row, Vector-VSA first measures:

```text
target_valid_tokens =
sum(native_parent_size[selected_native_parent])
```

Every pooled baseline and every new candidate then matches that target
exactly:

- Raw-K selects the top `target_valid_tokens` valid individual K positions.
- Vec8 and Vec16 use the frozen Fine-VSA exact-token selector, which selects
  fixed-width descriptors by valid-size category until their total equals
  the native target.
- Fixed descriptor capacities remain 8000 token slots nominally.
- Padding descriptors never contribute K/V values or Q-K pairs.

The measured budget is query-size weighted:

```text
sum_query_blocks(query_valid_tokens × selected_valid_K_tokens)
----------------------------------------------------------------
sum_query_blocks(query_valid_tokens × native_selected_valid_K_tokens)
```

The required ratio is exactly `1.0` (within floating-point reporting
tolerance), so nominal sparsity remains approximately 79.97%.
