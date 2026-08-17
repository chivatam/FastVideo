# VSA block-mean pooling assumes zero-filled padding

**Scope:** `fastvideo_kernel.triton_kernels.fused_compress_topk.fused_block_mean`,
and any test or probe that feeds it synthetic tensors.

## The trap

`_fused_block_mean_kernel` loads **all** `BLOCK_ELEMENTS` (64) slots of a tile with
no validity mask, then divides the sum by the *valid* token count from
`variable_block_sizes`:

```python
block_data = tl.load(x_base + offsets).to(tl.float32)   # no mask argument
acc = tl.sum(block_data, axis=0) / vbs                  # vbs = valid count, not 64
```

This is correct in production **only because** VSA's tile buffer zero-fills padding
slots (`scatter_into_tile_buf` writes into a zeroed buffer). Padding contributes
exactly 0 to the sum, so dividing by the valid count yields the true mean of the
valid tokens.

Feed it `torch.randn` over the full padded length — the obvious way to write a
synthetic test at Wan's shape — and every ragged tile gets noise in its pad slots.
At Wan's `(21, 30, 52)` grid, 169 of 624 tiles are ragged and the smallest holds 8
of 64 slots, so a tile can be 8 parts signal to 56 parts noise. A comparison
against a correctly-masked reference then fails by **~270% relative error**, which
looks exactly like a real precision bug in the kernel.

## What it cost

An F2 self-test check (`kernel_pooling_is_bf16_of_fp32_accumulated_mean`) failed
and the arithmetic-ladder ordering check failed with it, reporting the deployed
bf16 selector as *further* from fp64 than a deliberately-degraded
bf16-accumulation arm — a nonsensical ordering that would have been easy to
misread as "VSA's selector is catastrophically imprecise."

## The fix

Build routing inputs the way VSA does, then transpose to the layout the kernel
expects:

```python
non_pad_index = get_non_pad_index(block_sizes, TILE_ELEMENTS)
tile_partition_indices = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
tiled = scatter_into_tile_buf(tokens, (1, seq_len, heads, dim), non_pad_index, None,
                              tile_partition_indices)
query = tiled.transpose(1, 2).contiguous()
```

With that, `fused_block_mean` is **bit-identical** to
`bf16((valid_masked_sum).float() / valid_count)`, and the ladder orders correctly
(`fp32` < `kernel bf16` < `bf16 low-acc`).

## Generalization

Any kernel that takes a `variable_block_sizes`-style ragged descriptor may rely on
padding being zero rather than masking loads. Before concluding such a kernel is
numerically wrong, assert the padding invariant on the input you are handing it.
A cheap standing check is worth keeping:

```python
pad_slots = torch.ones(seq_len, dtype=torch.bool, device=device)
pad_slots[non_pad_index] = False
assert tiled[0][pad_slots].eq(0).all()
```
