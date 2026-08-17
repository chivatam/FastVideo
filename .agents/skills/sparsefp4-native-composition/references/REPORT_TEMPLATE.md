# Native Sparse NVFP4 Report

## 1. Verdict
STRONG POSITIVE / POSITIVE / NEGATIVE BUT USEFUL / SYSTEMS NO-GO / INVALID-INCOMPLETE

## 2. Native proof
### D0
- source:
- kernel:
- NVFP4 packing/scales:
- sparse indices:
- profiler/runtime receipt:
- proof unselected tiles skipped:
- proof no BF16/FP16 QK materialization before MMA:

### P3
Same fields.

## 3. Controlled 2x2 operator matrix

| Arm | Sparse | QK | Native | MSE | rel-L2 | cosine | SNR |
|---|---:|---|---:|---:|---:|---:|---:|
| A0 | | | | | | | |
| B0 | | | | | | | |
| C0 | | | | | | | |
| D0 | | | | | | | |

## 4. Video quality

| System | VBench | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| P0 | | | | |
| P1 | | | | |
| P2 | | | | |
| P3 | | | | |

## 5. Performance

| System | Attn ms | Attn speedup | DiT ms | E2E s | E2E speedup | Peak mem |
|---|---:|---:|---:|---:|---:|---:|
| P0 | | | | | | |
| P1 | | | | | | |
| P2 | | | | | | |
| P3 | | | | | | |

## 6. Established-style ablations
- sparsity
- resolution
- timestep
- QK/PV precision
- tile geometry if relevant

## 7. Failure analysis
Standard MSE/SNR first. Custom routing metrics only if required.

## 8. Limitations

## 9. Exact defensible paper claim

## 10. Paper update

## 11. Artifact index
Every headline number -> path + run ID + n.
