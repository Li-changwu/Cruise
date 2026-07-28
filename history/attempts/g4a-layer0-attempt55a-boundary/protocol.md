# G4a Attempt 55a Layer-0 Boundary Probe

Date frozen: 2026-07-23

## Question

Does native GE first diverge in layer-0 Attention or in layer-0 MLP when the
probe uses the exact one-barrier computation from the complete Attempt 53k
decoder?

## Controlled Scope

- Fixed B=1, four frozen prompt tokens and real physical-slot Paged-KV state.
- Frozen Qwen2.5-7B layer-0 weights and Attempt 53k inputs.
- ExactQk plus exactly one score materialization barrier, matching Attempt 53k.
- The full layer-0 attention and MLP are preserved. Additional graph outputs
  only expose numerical boundaries; they do not feed the computation.

## Export Gate

- Eager updated layer-0 K/V is bitwise identical to the complete Attempt 53k
  eager reference for all four steps.
- Every eager diagnostic tensor is finite.
- AIR ABI contains eight inputs and all declared outputs.
- AIR contains exactly one ExactQk and one Bf16Barrier node.
- Physical NPU 7 is empty before and after export.

## Native Decision

- Attention boundary fails first: test a second post-softmax BF16 barrier.
- Attention passes and MLP fails: split gate/up/product/down and control GE
  fusion or accumulation precision there.
- Probe passes but complete graph fails: instrument the complete graph because
  graph-scale fusion or memory planning is the remaining variable.

This probe localizes G4a only and cannot pass the complete-decoder gate.
