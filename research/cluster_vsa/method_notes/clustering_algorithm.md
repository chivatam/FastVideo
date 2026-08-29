# Cluster-VSA balanced fixed-slot grouping

Cluster-VSA uses a deterministic, training-free recursive ordering of valid
K tokens independently for each attention state.

1. Extract only valid K tokens using the frozen VSA non-padding index.
2. L2-normalize every K vector.
3. Pad the valid-token count to the next power of two with sentinel entries.
4. At each recursive level and independently for every batch/head/group:
   - compute feature variance over valid entries;
   - initialize a direction from the maximum-variance feature;
   - apply exactly two covariance power iterations to obtain a cheap
     approximate principal direction;
   - stably sort by projection on that direction;
   - split at the exact median.
5. Repeat until the balanced working leaves contain at most 64 tokens.
6. Remove sentinels and traverse the leaves in recursive order.
7. Sort the native valid-capacity multiset from largest to smallest, then
   pack the similarity-preserving order into those 624 fixed-width KV64
   slots. Sorting keeps full 64-token recursive leaves intact instead of
   repeatedly cutting them at spatially interleaved ragged boundaries.
   Full slots contain 64 tokens and the same geometric boundary-capacity
   multiset remains explicitly ragged.

The implementation batches all groups at a recursion depth on GPU. For the
32,760-token calibration geometry, it pads to 32,768 and performs nine
balanced split levels.

The primary candidate derives a separate ordering for every K head. The
systems-reuse diagnostic derives one ordering from the mean normalized K
representation across heads and reuses it for every head.

Cluster centroids are means of normalized K vectors and are used only for
routing. Exact sparse attention consumes the original K/V vectors after
permutation. No centroid value enters the final exact output, and no
omitted-support compensation is added.
