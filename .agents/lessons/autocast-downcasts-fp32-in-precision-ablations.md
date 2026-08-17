# Autocast silently downcasts fp32 matmul in precision studies

**Scope:** any side-channel measurement, probe backend, or numerical ablation that
runs inside FastVideo's denoising loop and declares a precision per arm.

## The trap

`DenoisingStage` wraps the transformer call in

```python
with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
```

`torch.autocast` casts the inputs of autocast-eligible ops — `matmul`, `mm`, `bmm`,
`linear` — **down** to the autocast dtype. That includes **fp32** inputs:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    torch.matmul(a_fp32, b_fp32).dtype   # -> torch.bfloat16
    torch.matmul(a_fp64, b_fp64).dtype   # -> torch.float64  (fp64 is exempt)
```

An attention backend that computes side-channel arms inside `forward()` therefore
runs *inside* autocast. Any arm labelled fp32 actually computes in bf16. Reductions
(`sum`), elementwise ops (`mul`) and fp64 matmuls are unaffected, so the corruption
is confined to exactly the arms that matter and leaves no error message.

## What it cost

Phase F1 crossed representation × arithmetic across 12 arms. The fp32 arms (R2, R3)
and the bf16/fp32-accumulate arms (R4, R5) produced **bit-identical masks in
4320/4320 cells**. That was interpreted as a substantive finding — "bf16 with fp32
accumulation is numerically indistinguishable from fp32 for block scoring" — and it
was checked for aliasing by confirming the two arms took different code paths and
that other arm pairs *did* differ. Those checks passed, because the arms really were
distinct code paths that autocast then collapsed onto the same arithmetic.

A full 10-prompt, 3-sparsity run (~150 MB × 8 shards) had to be discarded. The
confound only surfaced when the F2 VSA probe reported `VB_FP32` reproducing the
deployed bf16 mask *exactly*, which was too strong to be plausible.

## The fix

Wrap the entire measurement in an explicit opt-out, capturing the ambient state
first so records describe the trajectory rather than the guard:

```python
ambient = {**ambient_fp32_state(), **autocast_state()}
with declared_precision_arithmetic():      # torch.autocast(..., enabled=False)
    self._measure(..., ambient)
```

Wrap the block coarsely (all arms, all stages) rather than per-op: a per-operation
opt-out is what gets forgotten when a new arm is added later.

For fp32 specifically, autocast is only half the problem — TF32 is the other half.
`exact_fp32_matmul()` disables both.

## Standing gates

`f1_selftest.py` now asserts all four properties, so a regression fails loudly:

- `autocast_downcasts_unguarded_fp32_matmul` — the hazard still exists (guards are needed)
- `declared_precision_arithmetic_restores_fp32` — the guard works
- `fp32_scorer_is_autocast_invariant` — the scorer path is guarded
- `fp32_and_bf16_arms_differ_under_autocast` — the arms are distinguishable where it counts

## Generalization

Two arms producing bit-identical results is evidence of a *shared computation*, not
of numerical equivalence. Before reporting such a match, verify the ambient
execution context does not force the two paths to the same dtype. Cheap standing
check for any precision ablation:

```python
assert not torch.equal(arm_fp32, arm_bf16), "arms collapsed — check autocast/TF32"
```
