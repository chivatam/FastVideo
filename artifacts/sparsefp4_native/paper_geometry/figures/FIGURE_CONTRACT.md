# FIGURE_CONTRACT — geometry-alignment spin-out paper

Written before plotting, per the nature-figure skill's contract discipline.
Data files under `data/` are transcribed verbatim from canonical receipts
(`tables/c8_performance_v2.md`, `tables/p4g_vs_p2_quality_bootstrap.md`);
plotting scripts read them from disk and hardcode nothing (ARIS rule).

## Results-level question the figures answer

Why does deployed block-sparse video attention fail to convert 90% sparsity
into wall-clock speedup, and how much does aligning the selector tile to the
kernel's sparse granularity recover?

## Figure 1 — hero / mechanism schematic

- **Core conclusion:** the deployed 64-token selector tile straddles the FA4
  kernel's 256x128 sparse blocks; pooling the mask onto the kernel grid
  inflates retention ~2.4x, while re-aligning the selector tile keeps
  retention exact.
- **Evidence roles:** (a) the mismatch, (b) the failure path, (c) the fix.
- **Archetype:** fork/mechanism diagram; illustrative masks, measured
  inflation factor annotated. Panel letters a/b/c.
- **Integrity note:** masks are synthetic illustrations; only the ~2.4x
  factor is a measurement. Caption must say so.

## Figure 2 — mechanistic evidence (kernel)

- **Core conclusion:** with a 1:1 mask mapping, sparse kernel time tracks
  ideal dense-x-retention scaling (8.9x at 10% kept).
- **Archetype:** log-log line vs ideal reference; single annotation at the
  deployment operating point. Data: `data/kernel_latency.csv`.
- **n / uncertainty:** medians of 50 CUDA-event reps; stated in caption.

## Figure 3 — main result (end-to-end)

- **Core conclusion:** geometry alignment is the dominant E2E lever:
  1.40x vs dense at 720p (1.24x vs deployed VSA); 480p shows the deployed
  slowdown being repaired.
- **Archetype:** paired horizontal bar panels (a: 480p, b: 720p), direct
  labels, no axes. Data: `data/e2e_performance.csv`.
- **n / uncertainty:** medians of 3 steady-state reps (<0.5% dispersion);
  stated in caption. Bars start at zero (honest length encoding).

## Figure 4 — cost evidence (quality)

- **Core conclusion:** the geometry change is comparable on measured VBench
  dimensions with three small significant losses and one large significant
  gain.
- **Archetype:** forest plot of paired deltas with 95% bootstrap CIs; color
  = Holm verdict; per-dimension n shown with each row (unit of replication:
  prompt). Data: `data/quality_p4g_vs_p2.csv`.
- **Integrity note:** zero line explicit; no non-inferiority claim implied
  by the graphic; exact values live in the adjacent table (§5.3), not
  repeated in the figure.

## Shared style contract

- Serif face (DejaVu Serif) to match paper body; mathtext stix.
- Okabe-Ito accents: green #009E73 = ours, blue #0072B2 = deployed baseline,
  vermillion #D55E00 = failure path, grey = neutral/dense.
- No titles inside figures (captions carry them); panel letters bold
  lowercase at top-left.
- Physical sizes: single column ~89 mm (3.5 in) for Fig 2; double column
  ~183 mm (7.2 in) for Figs 1, 3, 4.
- Export: vector PDF (Type 42 fonts) + 300 dpi PNG; provenance manifest
  written next to outputs.
