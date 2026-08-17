# VSA's `fused_topk_mask` can select `topk + 1` blocks

## What happened

An experiment that compared two block-sparse routing masks asserted the obvious
invariant — both masks retain the same number of key blocks per query block, since
both are top-k with the same `k`. It crashed: one row out of 7,488 had 64 blocks
where the budget was 63. The failure was rare (one row per ~7.5k), appeared only at
some layers/timesteps, and took ~8 minutes of generation to reach, which made it
look like nondeterminism or a probe bug.

It is neither. `fastvideo_kernel.triton_kernels.fused_compress_topk.fused_topk_mask`
really does return `topk + 1` on some rows.

## Why

`_fused_topk_mask_kernel` finds its threshold by 32 iterations of fp32 bisection,
then selects `scores > threshold` plus a cumsum-limited fill from `scores ==
threshold`:

```python
for _i in range(32):
    mid = (lo + hi) * 0.5
    count_ge = tl.sum(((scores_f32 >= mid) & valid_mask).to(tl.int32), axis=0)
    lo = tl.where(count_ge >= topk, mid, lo)
    hi = tl.where(count_ge >= topk, hi, mid)
threshold = lo
above_threshold = scores_f32 > threshold
at_threshold = scores_f32 == threshold
n_needed_at_thresh = topk - tl.sum(above_threshold.to(tl.int32), axis=0)
```

The tie-fill is correct only if `threshold` equals the k-th largest score exactly.
Bisection approaches it from below and, in general, never lands on it: `lo` ends up
strictly less than the k-th value by up to `(hi - lo) / 2^32`. When the k-th and
(k+1)-th scores are *equal* — common in bf16, which has an 8-bit mantissa — both tied
scores then satisfy `scores > threshold`, so `n_above` is `topk + 1`,
`n_needed_at_thresh` goes negative, the tie-fill contributes nothing, and the row
returns one block too many.

The kernel's comment argues 32 iterations resolve ~2e-9, "well below the bf16 ULP",
so the `>`/`==` comparisons are exact. That reasoning is what fails: resolving *finer
than* the ULP is not the same as landing *on* the value, and the tie-fill branch
requires exact equality.

Both conditions are needed, which is why it is rare: a tie at the k-th value, and a
`lo`/`hi` trajectory that doesn't happen to hit the tied value exactly. The
over-selection is scale-invariant — multiplying an offending row by 2 or 0.5 keeps it
failing — so it is not a magnitude/denormal issue.

## What to do

- Do not assume `fused_topk_mask` honours its budget exactly. If your code needs a
  hard guarantee, verify it or use a sort-based selection.
- A stable descending sort reproduces the kernel's intended rule (k largest, ties
  toward the lower index) and always honours the budget:
  `torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :topk]`.
- If you are *measuring* VSA, this behaviour is part of the object under study.
  Record the per-row deviation rate rather than aborting; aborting throws away a
  real finding about the kernel.
- `+inf` in the scores is a separate, total failure of the same bisection: `hi`
  stays `+inf`, the threshold never converges, and rows return far more than `topk`
  (observed: 622 of 624 for `topk=63`). Guard routing inputs for finiteness.

## Reproduction

`artifacts/sparsefp4_followup/configs/f2_kernel_topk_bug.py` replays a saved
offending score row through the installed kernel and confirms 64-for-63,
demonstrates that removing the tie fixes it, and shows scale-invariance.
`f2_explain_violation.py` replays the bisection in PyTorch and prints the
iteration trace, the converged threshold, and `n_above` vs the budget.

Note that constructing a tie synthetically often does *not* reproduce it — a
constructed row can land on a `lo` that happens to equal the tied value. Use a
real dumped row.
