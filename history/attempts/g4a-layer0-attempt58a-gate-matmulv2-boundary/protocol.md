# G4a Attempt 58a Gate MatMulV2 Single-Variable Probe

Date frozen: 2026-07-24

## Question

Does replacing only the layer-0 gate projection lowering from GE `MatMul` to
GE `MatMulV2` remove the step-1 gate pre-activation mismatch?

## Controlled Scope

- Fixed B=1, four frozen prompt tokens and real physical-slot Paged-KV state.
- Frozen Qwen2.5-7B layer-0 weights and Attempt 53k inputs.
- Attempt 57a graph structure, output ABI, ExactQk, score barrier and gate
  materialization are retained.
- A private `g4a_matmulv2.mm` op executes `torch.mm` in eager mode and lowers
  only the gate projection to `GateMatMulV2` in AIR.
- O, up and down projections retain TorchAir's default GE `MatMul` lowering.

## Export Gate

- Every eager array is bitwise identical to Attempt 57a, including the exposed
  gate pre-activation.
- Eager updated layer-0 K/V is bitwise identical to complete Attempt 53k for
  all four steps, and every diagnostic tensor is finite.
- AIR ABI contains the same eight inputs and 19 outputs as Attempt 57a.
- AIR contains exactly one `GateMatMulV2` node with operands `[1, 3584]` and
  `[3584, 18944]`, plus one ExactQk, one Bf16Barrier and one Bf16Materialize.
- Total projection counts change from four GE MatMul and three MatMulV2 nodes
  to three GE MatMul and four MatMulV2 nodes.
- Physical NPU 7 is empty before and after export.

## Native Decision

- Pass mechanism probe: pre-gate native arrays remain bitwise identical to
  Attempt 57a; step-1 gate pre-activation and gate both become bitwise exact;
  no output gains a tolerance failure.
- Reject mechanism: step-1 gate pre-activation still differs.
- Confounded: eager arrays, pre-gate native arrays, or frozen graph controls
  change.

This probe tests one layer-0 lowering mechanism. It cannot pass complete G4a.
