# H14: Calibrated Safe Core + Dynamic Fine8 Tail

Pure Fine8 substantially improves output fidelity at the exact VSA80 pair
budget, but its prompt-dependent routing can omit rare pathways and produced
five new generation failures. H14 tests whether some attention support is
consistently important across unrelated prompts.

CoreTail-VSA reserves part of every query's matched VSA80 valid-token capacity
for a prompt-invariant KV64 core learned from true dense attention. The
remaining capacity is filled by the frozen Fine8 ranking, excluding children
already covered by the core. The method introduces no training and no increase
in exact attention density.

Only two predeclared variants are tested:

- Core25: 31 static KV64 parents plus a Fine8 tail.
- Core50: 62 static KV64 parents plus a Fine8 tail.

The experiment asks whether the combination retains Fine8-level fidelity,
reduces its catastrophic tail, and shifts useful support back to a
GPU-efficient static execution structure.
