# Native FastVideo VSA equations

Date verified: 2026-08-29 UTC

This note records the actual Wan/FastVideo VSA implementation before any
compressed-support correction is applied.

## Shapes and block geometry

The Wan checkpoint uses the default VSA tile `(4, 4, 4)`, so every full
coarse block represents `B = 64` fine tokens. At the established
480×832, 81-frame protocol:

- coarse query/KV blocks: `N = 624`;
- native VSA80 exact blocks per coarse query row: `K = 125`;
- omitted coarse blocks: `N - K = 499`;
- coarse decision rows per attention call: `1 × 12 × 624 = 7,488`.

Boundary blocks use their recorded `variable_block_sizes`; multiplicity is
therefore the actual valid-token count, not blindly 64.

## Inputs and learned gates

`WanTransformerBlock_VSA` projects normalized hidden states to fine
`Q`, `K`, `V`, and a learned elementwise `gate_compress`:

```text
Q = to_q(x)
K = to_k(x)
V = to_v(x)
G_compress = to_gate_compress(x)
```

The tensors are reshaped to `[batch, sequence, heads, head_dim]` before the
VSA backend. The checkpoint also has the separate transformer residual gate
`gate_msa`; it is applied only after the attention output projection and is
not part of the VSA coarse/fine merge.

Code: `fastvideo/models/dits/wanvideo.py`, lines 537–568.

## Coarse branch

After conversion to `[B, H, S, D]`, VSA pools each 64-token block:

```text
Q_c[i] = mean(Q tokens in coarse query block i)
K_c[j] = mean(K tokens in coarse KV block j)
V_c[j] = mean(V tokens in coarse KV block j)
```

For head dimension `d`:

```text
C[i,j] = Q_c[i] K_c[j]^T / sqrt(d)
A_c[i,:] = softmax(C[i,:])              # over all 624 blocks
O_c[i] = sum_j A_c[i,j] V_c[j]
```

`O_c[i]` is repeated across the fine queries in coarse query block `i`.

Code: `fastvideo-kernel/python/fastvideo_kernel/ops.py`, lines 111–120.

## Native Top-K and fine branch

The exact block set is:

```text
S_i = TopK(C[i,:], K=125)
```

The sparse kernel then computes fine-token attention only over tokens whose
coarse KV block is in `S_i`:

```text
L_fine[q,t] = Q[q] K[t]^T / sqrt(d)
P_fine[q,:] = softmax(L_fine[q,:] restricted to S_i)
O_fine[q] = sum_{t in S_i} P_fine[q,t] V[t]
```

The fine softmax is independently normalized over selected exact tokens. It
does not use the coarse probabilities or their retained mass.

The Blackwell sparse kernel also produces:

```text
M_exact[q] = log2 sum_{t in S_i} exp(L_fine[q,t])
```

which permits a mathematically correct online merge with another support.

Code:

- `fastvideo-kernel/python/fastvideo_kernel/ops.py`, lines 122–129;
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py`;
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_sm100a.py`.

## Actual native output

The two branches are not normalized together. The kernel returns:

```text
O_native[q] =
    O_fine[q]
    +
    G_compress[q] elementwise-multiply O_c[coarse_block(q)]
```

`G_compress` is a learned vector with one value per token, head, and channel;
it is not a scalar probability gate. In the established trace its mean
absolute value is approximately `0.00287` and RMS is approximately
`0.00404`.

Code: `fastvideo-kernel/python/fastvideo_kernel/ops.py`, lines 131–133.

## Rectification used in this experiment

For the native set `S_i`, define:

```text
m_i = sum_{j in S_i} A_c[i,j]
r_i = 1 - m_i
O_skip_uncond[i] = sum_{j not in S_i} A_c[i,j] V_c[j]
```

`O_skip_uncond` already contains omitted mass `r_i`; it is not separately
renormalized or multiplied by `r_i` again.

To avoid adding a second copy of the native coarse branch, the parameter-free
rectified output is:

```text
O_rectified[q] =
    m_i O_fine[q]
    +
    G_compress[q] elementwise-multiply O_skip_uncond[i]
```

This formulation:

- preserves the exact native Top-K IDs and K=125 sparse kernel;
- accounts for selected probability mass on the independently normalized
  fine output;
- keeps the learned coarse gate on the omitted coarse contribution;
- removes the selected-support component from the coarse branch, avoiding
  obvious selected-support double counting;
- introduces no coefficient search or learned parameter.

## Compressed-halo merge used in this experiment

The compressed halo contains only blocks outside `S_i`. For each fine query:

```text
L_halo[q,j] =
    Q[q] K_c[j]^T / sqrt(d)
    + log(variable_block_sizes[j])
    for j not in S_i
```

The multiplicity term uses the actual valid-token count represented by each
centroid.

The exact sparse kernel output/LSE and compressed-halo output/LSE are merged
under one normalization:

```text
M = max(M_exact, M_halo)
w_exact = 2^(M_exact - M)
w_halo = 2^(M_halo - M)

O_core_halo =
    (w_exact O_fine + w_halo O_halo)
    / (w_exact + w_halo)
```

The checkpoint's learned coarse residual remains unchanged:

```text
O_CH =
    O_core_halo
    +
    G_compress elementwise-multiply O_c
```

Equivalently, CH-VSA replaces only the native selected-support fine branch
with its common-normalization exact-core/omitted-halo counterpart. The native
coarse branch and its learned gate remain untouched. Halo centroids are
strictly disjoint from the exact Top-K, so no exact block is duplicated
inside the core/halo normalization.
