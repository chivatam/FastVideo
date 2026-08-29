# Adaptive VSA + NVFP4

This directory contains the isolated, resumable experiment harness for the
8xB200 training-free adaptive VSA + NVFP4 study.

The production FastVideo defaults are unchanged. Runtime instrumentation is
installed only by the research worker and is controlled by explicit command
line options.

Artifacts are written through:

```text
artifacts/adaptive_vsa_fp4/
```

The artifact root is expected to reside on local NVMe.

Core commands:

```bash
python -m research.adaptive_vsa_fp4.scripts.build_manifest
python -m research.adaptive_vsa_fp4.scripts.run_grid --help
python -m research.adaptive_vsa_fp4.scripts.collect --help
python -m research.adaptive_vsa_fp4.scripts.evaluate --help
python -m research.adaptive_vsa_fp4.scripts.summarize --help
```
