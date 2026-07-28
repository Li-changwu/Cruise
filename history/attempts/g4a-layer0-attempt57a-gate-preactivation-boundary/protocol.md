# G4a Attempt 57a Gate MatMul-to-SiLU Boundary Probe

Date frozen: 2026-07-24

## Question

Does the step-1 gate mismatch first appear in the gate projection MatMul or in
the following SiLU?

## Controlled Scope

- Fixed B=1, four frozen prompt tokens and real physical-slot Paged-KV state.
- Frozen Qwen2.5-7B layer-0 weights and Attempt 53k inputs.
- ExactQk plus exactly one score materialization barrier, matching Attempt 53k.
- One Bf16Materialize node between gate projection and SiLU.
- One additional `gate_preactivation` output; all other outputs retain their
  Attempt 55a semantics and order relative to one another.
- The full layer-0 attention and MLP are preserved. Additional graph outputs
  only expose numerical boundaries; they do not feed the computation.

## Export Gate

- Eager updated layer-0 K/V is bitwise identical to the complete Attempt 53k
  eager reference for all four steps.
- Every eager diagnostic tensor is finite.
- AIR ABI contains eight inputs and all declared outputs.
- AIR contains exactly one ExactQk, one Bf16Barrier, and one Bf16Materialize
  node.
- Physical NPU 7 is empty before and after export.

## Native Decision

- If step-1 gate pre-activation differs first, localize to gate MatMul.
- If pre-activation is bitwise exact and gate differs first, localize to SiLU.
- If outputs before pre-activation change from 55a, classify the result as
  confounded and do not claim either operation.

This probe localizes G4a only and cannot pass the complete-decoder gate.
