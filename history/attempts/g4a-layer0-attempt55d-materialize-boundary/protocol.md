# G4a Attempt 55d Layer-0 BF16 Materialization Probe

Date frozen: 2026-07-24

## Question

Does forcing the layer-0 gate and up tensors through two opaque BF16
materialization kernels eliminate the step-1 gate mismatch observed in
Attempt 55a?

## Controlled Scope

- Fixed B=1, four frozen prompt tokens and real physical-slot Paged-KV state.
- Frozen Qwen2.5-7B layer-0 weights and Attempt 53k inputs.
- ExactQk plus exactly one score barrier, matching Attempt 53k.
- Exactly two additional `Bf16Materialize` nodes, one after gate and one after
  up, before their elementwise product.
- The full layer-0 attention and MLP are preserved. Additional graph outputs
  only expose numerical boundaries; they do not feed the computation.

## Export Gate

- Eager updated layer-0 K/V is bitwise identical to the complete Attempt 53k
  eager reference for all four steps.
- Every eager diagnostic tensor is finite.
- AIR ABI contains eight inputs and all declared outputs.
- AIR contains exactly one ExactQk, one Bf16Barrier, and two Bf16Materialize
  nodes.
- Physical NPU 7 is empty before and after export.

## Native Decision

- The 55d eager reference must be array-wise identical to the 55a eager
  reference.
- Native pre-MLP outputs must remain array-wise identical to 55a, excluding a
  graph-scale compilation confounder.
- The two materialize kernels must each launch once per graph run: eight total
  launches across four steps.
- The mechanism is supported only if step-1 gate becomes bitwise exact and its
  first mismatch moves to a later output, with no new tolerance failure.
- Otherwise, reject this materialization boundary as the explanation and
  localize standalone SiLU or MatMul numerical semantics next.

This probe localizes G4a only and cannot pass the complete-decoder gate.
