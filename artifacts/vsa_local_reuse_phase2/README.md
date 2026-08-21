# VSA Phase 2 — Exact Representation & Execution Strategy for Two-Query KV Reuse

**Question:** inside PR [#1719](https://github.com/hao-ai-lab/FastVideo/pull/1719)'s
existing 2-Q-block CTA, what is the best *exact* representation for
exploiting the ~71.5% selected-set overlap that Phase 1 measured —
shared/private decomposition (A), union + conditional MMA (B1), or union +
dense grouped MMA (B2)?

**Answer: GO-A (shared/private phased execution), with B2 as an explicitly
cheaper-to-land stepping stone that shares the same ring-commit change.**
See `DESIGN.md` for the full kernel mapping and decision matrix.

## Data

Real PR#1719 CTA pairs (q_block 2p, 2p+1) from the Phase-1 720p captures:
3 prompts x 30 layers x 3 steps x 12 heads x 720 pairs = **2,332,800 real
pairs** at K=144 (`build_pair_dataset.py`), plus 512 raw index-row samples
for the exactness references and microbenches.

## Part H — ordering (load-bearing fact)

`q2k_idx` rows are **sorted ascending by KV block id**: `map_to_index`
scans the bool mask in increasing KV order, and the sm_100a kernel consumes
rows in that order. Verified on all 270 captured shards + the production
Triton op. Intersection/union therefore reduce to a linear two-pointer
merge (vectorized: batched `searchsorted`) — no sort, no hashing.

## Part B — real decomposition at K=144 (2.33M pairs)

| quantity | median | mean | P10 | P25 | P50 | P75 | P90 | P99 |
|---|---|---|---|---|---|---|---|---|
| shared | **103** | 98.4 | 62 | 81 | 103 | 120 | 129 | 138 |
| q0_private | **41** | 45.6 | 15 | 24 | 41 | 63 | 82 | 122 |
| q1_private | **41** | 45.6 | 15 | 24 | 41 | 63 | 82 | 122 |
| union | **185** | 189.6 | 159 | 168 | 185 | 207 | 226 | 266 |

(Phase-1's back-of-envelope 103/41/41/185 is exactly the real median.)
Per-layer/step/head breakdowns: `results/{layer,timestep,head}_summary.csv`.

## Part D — exact KV traffic (bf16, 64 tok x 128 dim, K+V = 32 KiB/block)

- baseline: (|S0|+|S1|) x 32 KiB = 9.44 MB/pair-head median (288 blocks)
- union: |U| x 32 KiB (185 blocks median)
- **34.2% total bytes saved** (median per-pair 35.8%, P10 21.5%, P90 44.8%);
  traffic reduction ratio median **1.557x**.
- At the kernel's ring granularity (4 blocks per 256-token K-tile):
  baseline 72 K-tile fetches/pair -> A: 48.5 (-32.6%), B2: 47.8 (-33.7%).

## Part E — MMA work (multiply-add = 2 FLOPs; 4·64·64·128 = 2.10 MFLOP per Q-block x KV-block interaction)

- **A**: `2·|shared| + |p0| + |p1| = |S0|+|S1|` — verified identically equal
  to baseline on all 2.33M pairs. **0% extra FLOPs.**
- **B1**: 0% extra (conditional), but irregular tcgen05 issue (see DESIGN.md).
- **B2**: `2·|U|` interactions → **+31.7% total FLOPs** (median +28.5%,
  P90 +57%).
- Pipeline-iteration view (what softmax/correction warps actually execute):
  baseline median 36 iters/Q-tile; **A: 37 (+2.1%)**; **B2: 47 (+32.7%)**.

## Part F — relative arithmetic intensity (FLOPs / KV byte, baseline = 1.0)

| baseline | A | B1 | B2 |
|---|---|---|---|
| 1.00x | **1.52x** | 1.52x | 2.00x |

Baseline absolute intensity is ~64 FLOP/B — far below B200's ~275 FLOP/B
bf16 ridge, i.e. the KV stream is the structurally scarce resource and the
saved bytes are the point. (Structural model only; no hardware-throughput
claim.)

## Part C — numerical equivalence (real pairs, 64 pairs x 2 Q x fp32/bf16)

Baseline = current per-Q sorted walk with online softmax; streaming
references mirror kernel numerics (bf16 QK/PV inputs, fp32 S and O
accumulation, bf16 P). `results/numerics_summary.csv`:

| mode | strategy | max abs err | rel L2 max | cosine min |
|---|---|---|---|---|
| fp32 | A | 2.0e-8 | 4.2e-7 | 1.000000 |
| fp32 | B1 | **0 (bit-exact)** | 0 | 1.0 |
| fp32 | B2 | **0 (bit-exact)** | 0 | 1.0 |
| bf16 | A | 5.0e-5 | 1.9e-3 | 0.999998 |
| bf16 | B1 | **0 (bit-exact)** | 0 | 1.0 |
| bf16 | B2 | **0 (bit-exact)** | 0 | 1.0 |

B1/B2 are bit-exact because each Q still consumes its member blocks in the
same ascending order and masked blocks contribute exactly zero. A only
reorders blocks (shared first) → fp reassociation at bf16 noise level.

## Part G — metadata construction (B200, 8640 pairs = 12 heads x 720, K=144)

| approach | latency | ns/pair | output bytes | temp peak |
|---|---|---|---|---|
| shared/private (sorted searchsorted) | **1.17 ms** | 135 | 15.0 MB | 88 MB |
| union+membership (sorted searchsorted) | **0.95 ms** | 110 | 12.5 MB | 138 MB |
| shared/private (bitmap) | 2.58 ms | 299 | 12.5 MB | 363 MB |
| shared/private (Triton one-kernel, naive KxK) | 2.51 ms | 291 | 15.0 MB | 25 MB |

Budgets measured at the same shape: **sm_100a kernel 5.52 ms** per 720p
layer call (12 heads), Triton 10.78 ms, and the *existing* per-call
`map_to_index` pass 1.10 ms. Standalone prototype overhead ≈ 17–21% of
kernel time; production fuses pair-metadata emission into the map_to_index
scan (OR/AND of two mask rows) → expected near-zero marginal cost.

Optional MMA probe: batched GEMM with KV resident, M=64 → M=128 costs
**1.39x** (not 2x) — the second query block on already-loaded KV is cheap.

## Files

- `build_pair_dataset.py` — Part A dataset from Phase-1 captures
- `analyze_decomposition.py` — Parts B/D/E/F → `results/`, `plots/`, `summary.json`
- `reference_attention.py` — exact streaming references (baseline/A/B1/B2) + metadata decomposition
- `run_reference_eval.py` — Part C numerics on real pairs
- `metadata_gpu.py`, `metadata_triton.py`, `bench_metadata.py` — Part G
- `test_reference_attention.py` — 24 tests (reconstruction, duplicates, edge cases,
  real K=144 rows, ordering invariance, sorted outputs, GPU==CPU metadata, equivalence)
- `DESIGN.md` — Part J kernel mapping + Part K decision matrix

Reproduce:

```bash
python artifacts/vsa_local_reuse_phase2/build_pair_dataset.py \
    --capture-root /mnt/nvme/outputs/vsa_capture --out /mnt/nvme/outputs/phase2_pairs
python artifacts/vsa_local_reuse_phase2/analyze_decomposition.py \
    --dataset /mnt/nvme/outputs/phase2_pairs --out artifacts/vsa_local_reuse_phase2
python artifacts/vsa_local_reuse_phase2/run_reference_eval.py \
    --sample-pairs /mnt/nvme/outputs/phase2_pairs/sample_pairs.pt --out artifacts/vsa_local_reuse_phase2
python artifacts/vsa_local_reuse_phase2/bench_metadata.py \
    --sample-pairs /mnt/nvme/outputs/phase2_pairs/sample_pairs.pt --out artifacts/vsa_local_reuse_phase2
pytest artifacts/vsa_local_reuse_phase2/ -v
```

## Decision

**GO-A.** For PR #1719's two-query CTA, the best exact local-KV-reuse
representation is shared/private phased execution: it saves ~34% of KV
traffic while adding ~0% compute (+2.1% pipeline iterations) and ~0.1 µs/pair
metadata overhead that fuses into the existing index build. B2 (union +
dense, bit-exact, near-metadata-only kernel change) is the recommended
first milestone since it exercises the identical dual-consumer ring commit,
but its +32.7% softmax/correction-warp inflation lands on the kernel's most
contended resource, so it is a stepping stone, not the destination.

No kernel speedup is claimed in this phase.
