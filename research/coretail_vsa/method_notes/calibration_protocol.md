# Calibration Protocol

The calibration and held-out prompts were frozen before running dense capture
or examining candidate results.

- Source: public VBench-2.0 full-text prompt benchmark.
- Exclusion: exact normalized text of all 72 subject-consistency development
  prompts.
- Deterministic order: ascending
  `SHA256("coretail-vsa-vbench2-external-v1:" + prompt)`.
- Calibration: first 32 eligible unique prompts.
- Held-out offline evaluation: next 8 eligible unique prompts.
- Overlap among calibration, held-out, and the 72 development prompts: zero.

For each calibration prompt, dense BF16 runs on the frozen 3-step FastWan
checkpoint capture all 90 DiT self-attention calls. The captured unit is
`step × layer × head × query-KV64-block`, with true dense probability mass
recorded for every valid KV64 key block. Padding contributes no mass.

The core statistic is the 10th percentile across 32 prompts using deterministic
linear interpolation. For sorted values `x[0] ... x[31]`, the percentile index
is 3.1 and the score is `0.9*x[3] + 0.1*x[4]`. No alternate percentile or core
ratio will be searched after held-out results are observed.
