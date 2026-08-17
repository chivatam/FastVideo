# Canonical exact-10% VSA256/FA4-aligned operator matrix

25 cells per resolution (5 layers x 5 timesteps) captured from a genuine
sparse-BF16 (P4G-config) trajectory at VSA sparsity 0.90 with 256-token
tiles; C0_256/D0_256 masks byte-identical, mapped 1:1 onto FA4 sparse
geometry (retention exact, no coarsening). Median keep fraction: 0.1006.

| Resolution | Arm | n | MSE med | rel-L2 med | cosine med | SNR dB med | rel-L2 p90 | finite |
|---|---|---|---|---|---|---|---|---|
| 480x832x81 | B0_vs_A0 | 25 | 0.0033651 | 0.0951 | 0.9955 | 20.4 | 0.4216 | True |
| 480x832x81 | C0_256_vs_A0 | 25 | 0.013236 | 0.1918 | 0.9850 | 14.3 | 0.4146 | True |
| 480x832x81 | D0_256_vs_A0 | 25 | 0.01913 | 0.2885 | 0.9702 | 10.8 | 0.4536 | True |
| 480x832x81 | D0_256_vs_C0_256 | 25 | 0.0038916 | 0.0957 | 0.9954 | 20.4 | 0.4084 | True |
| 720x1280x81 | B0_vs_A0 | 25 | 0.0031334 | 0.1018 | 0.9949 | 19.8 | 0.3689 | True |
| 720x1280x81 | C0_256_vs_A0 | 25 | 0.0094636 | 0.2175 | 0.9793 | 13.3 | 0.4780 | True |
| 720x1280x81 | D0_256_vs_A0 | 25 | 0.018079 | 0.3005 | 0.9691 | 10.4 | 0.4695 | True |
| 720x1280x81 | D0_256_vs_C0_256 | 25 | 0.0029782 | 0.0918 | 0.9958 | 20.7 | 0.3564 | True |

Interpretation (no additivity assumed):
- quant-only (B0 vs A0): rel-L2 0.095 (480p) / 0.102 (720p)
- sparse-only at exact 10% (C0_256 vs A0): 0.192 / 0.218
- joint (D0_256 vs A0): 0.289 / 0.301
- conditional quant effect (D0_256 vs C0_256): **0.096 / 0.092** —
  applying NVFP4 to the retained QK tiles perturbs the sparse output by
  approximately the same magnitude as dense NVFP4 perturbs the dense
  output, at both resolutions, under the exact deployment geometry.

Raw: raw/operator/c5b_exact10.jsonl. Capture provenance:
/mnt/nvme/scratch/sparsefp4_native/cap256-{480,720}/provenance.json.
