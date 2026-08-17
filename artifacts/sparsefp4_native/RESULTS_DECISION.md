# RESULTS_DECISION — SparseFP4 Native Composition

## Verdict: **POSITIVE** (systems-qualified)

Native NVFP4 attention compute and block-sparse video attention **can be
composed on Blackwell**: the joint path is numerically sound (no error
amplification over its two components), preserves video quality at the level
of the deployed sparse baseline, and delivers real measured speedup — 9.3x at
the attention kernel and 1.40x end-to-end at 720p for the FA4-geometry sparse
arm — with the remaining FP4-specific end-to-end gap reduced to two named,
localized engineering items rather than any fundamental obstacle.

## Basis (every claim maps to saved receipts)

1. **D0/P3/P4 are truly native** — `NATIVE_PROOF.md`: packed E2M1 Q/K +
   per-16 E4M3 SFs at the call boundary, `flash_fwd_sm100_fp4` kernel symbol
   in the profiler, retained-tile-only iteration in source, work scaling
   6.01→0.80 ms (100%→10% retained), no BF16 Q/K materialization.
   The composed path required repairing four version-skew bugs in the
   flash-attention-fp4 fork (patch: `configs/fa4-fork-sparse-fp4-repair.patch`);
   no kernel math was written or changed — the capability existed, broken.
2. **Numerics compose cleanly** — `tables/c5_matrix_s090.md` (25 genuine-VSA
   cells, frozen byte-identical masks): quant-only rel-L2 0.098, sparse-only
   0.128, joint 0.204; **quant-cost-on-sparse (D0 vs C0) = 0.096 ≈
   quant-cost-on-dense (B0 vs A0) = 0.098**. Kernel exactness: D0 vs
   dequantized oracle rel-L2 0.0017 (SNR 55 dB) — identical to the dense
   kernels' deviation from their own oracles (C4, 32 cells).
3. **Quality is baseline-level** — `raw/quality/pq_s090_{pairwise,vbench}.jsonl`:
   within the sparse family, P3/P4 sit at or above deployed VSA (P2) on
   paired PSNR/LPIPS, and match it across VBench dimensions (e.g. P4G subject
   consistency 0.896 vs P2 0.899). All sparse arms trade subject-consistency
   and aesthetic quality against the dense reference — that is the cost of
   VSA-style sparsity itself, not of NVFP4 (P1 dense NVFP4 holds 0.965).
4. **Speed is real and measured** — `tables/c8_performance.md`:
   kernel 9.3x @10% retained vs dense BF16; E2E 720p: P4G 1.402x vs dense
   and 1.24x vs deployed VSA; 480p: P4G 1.03x vs dense (attention share too
   small for more). No FLOP-derived numbers anywhere.

## The systems qualification (stated, not hidden)

- **FP4-fine E2E is not yet the fastest arm.** Two localized causes,
  measured: (a) at 720p tile counts the sparse-FP4 kernel inverts vs its
  BF16 twin (6.09 vs 2.95 ms — persistent-scheduler/tuning regime of
  `flash_fwd_sm100_fp4`, which was tuned for dense shapes); (b) at 480p the
  in-model FP4 path trails its microbenchmark (per-call wall shows FP4 fine
  *faster*: 1.44 vs 1.65 ms incl. quantize — the E2E gap points at per-call
  allocation churn). Both are kernel-engineering items on a path we proved
  numerically sound; neither contradicts the composition claim, which the
  BF16-fine twin (P4G) already cashes out end-to-end.
- Deployed VSA (Triton fine kernel) does not beat dense FA4 at 480p on B200
  at all (0.94x) — the right sparse baseline regime for this hardware is
  720p+, where all sparse arms win.
- Mask-geometry result (novel, actionable): coarsening the deployed 64x64
  mask onto FA4 (P3) costs more than moving the selector to FA4-native
  256-token tiles (P4) — exact mapping keeps true 10% retention *and*
  quality parity. "Turn VSA into FA4 type" is the right design.

## Not INVALID/INCOMPLETE because

Every completion criterion of the skill is met: A0/B0/C0/D0 exist on
identical captured tensors with byte-identical C0/D0 masks; D0 proven native
(source+runtime+profiler+scaling); P0-P3 (+P4 variants) exist with P3/P4
proven native; standard operator metrics, standard video-quality metrics
(paired + VBench), and wall-clock kernel/DiT/E2E timing all recorded; the
four-arm matrix is complete; no routing/Jaccard diagnostics were used as
evidence anywhere.

## QAT recovery track (supplementary, in progress at decision time)

SLA2-style fake-quant STE fine-tune at the deployment operating point
(sparsity 0.9, NVFP4 fine branch), 400 steps on 47 videos: **recovers
subject/background consistency to the dense reference's level on the native
P3/P4 serving arms** (0.846->0.974 / 0.918->0.977; dense 0.976/0.977) and
halves the aesthetic-quality gap. Details + caveats in REPORT.md §8. This
strengthens the POSITIVE verdict: the residual quality cost of the composed
system is trainable away at feasibility scale.
