# Stable Core Definition

For calibration prompt `p`, attention unit `u`, and valid KV64 key block `b`,
the stored quantity is:

`M[p,u,b] = mean_q sum_{k in b} softmax(QK^T / sqrt(d))[q,k]`

The mean is over valid full-resolution query tokens in the query KV64 block.
The sum is over valid key tokens only. Q and K are reordered into the native
VSA tiled geometry solely to define matching regions; logits and probabilities
come from exact dense Q and K.

The primary score is:

`C[u,b] = linear_p10_p M[p,u,b]`

Blocks are ranked independently for every step, layer, head, and query region.
The first 31 nonempty blocks form Core25; the first 62 form Core50.

Diagnostics include mean, median, minimum, coefficient of variation, stable
rank versus per-prompt rank correlation, prompt top-set overlap and Jaccard,
dense-mass coverage, mass per valid token, and the frequency with which Fine8
would independently select children of each stable block.
