# G4a Attempt 59a Gate MatMulV2 transpose_x2 Probe

Date frozen: 2026-07-24

## Question

Does preserving the original `[18944, 3584]` gate weight and expressing the
transpose through GE MatMulV2's `transpose_x2=true` reproduce eager ACLNN?

## Motivation and Single Variable

Attempt 58a changed the GE op type but retained an explicit weight Transpose
and `transpose_x2=false`. It produced a new TBE kernel but exactly the same
native NPZ as Attempt 57a. Eager ACLNN instead launches the built-in
`MatMulV2_ND_ND_FP16_FP16_false_true_all_98499` with original weight layout,
160 bytes of tiling data, argument size 200 and prefetch count 3.

Attempt 59a retains all Attempt 58a controls and changes only the gate
projection contract: the private op receives the untransposed gate weight and
lowers to `MatMulV2(..., transpose_x2=true)`.

## Export Gate

- All 82 eager arrays are bitwise identical to Attempt 58a.
- AIR ABI remains eight inputs and 19 outputs.
- Exactly one `GateMatMulV2TransposeX2` consumes `[1,3584]` and the original
  `[18944,3584]` weight directly; its second input is not a Transpose node and
  `transpose_x2=true`.
- ExactQk, Bf16Barrier, Bf16Materialize and total MatMul/MatMulV2 counts remain
  unchanged from Attempt 58a.
- Physical NPU 7 is empty before and after export.

## Native Decision

- Mechanism supported: pre-gate native arrays remain bitwise identical to
  Attempt 58a; step-1 gate pre-activation and gate become bitwise exact; no
  output gains a tolerance failure.
- Mechanism rejected: step-1 gate pre-activation retains the three mismatches.
- Confounded: eager, pre-gate, ABI or graph controls change.

This is a layer-0 mechanism probe and cannot pass complete G4a.
