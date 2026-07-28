# G4a Attempt 52d: Opaque-BF16-Barrier Layer-0 Attention/KV AIR

Date: 2026-07-23

## Question

Does the independently verified Attempt-54b opaque BF16 barrier prevent GE
from eliminating the scale-to-BF16 rounding boundary and thereby restore the
complete frozen Qwen layer-0 Attention/KV recurrence?

## Controlled Change

The model, revision, four hidden rows, initial caches, positions, weights,
precision mode, ExactQk kernel and all 16 frozen diagnostic outputs remain
those of Attempt 52c. The only graph change is:

1. the exact custom BF16 QK kernel produces raw QK;
2. scaling executes in FP32 and casts to BF16;
3. `Bf16Barrier` copies the BF16 bits through an opaque AIV kernel before the
   score is consumed by masking and FP32 softmax.

The 72-byte tiling tensor is an explicit graph input. A seventeenth BF16 QK
output makes the repaired rounding boundary independently observable.

## Pass Rule

Attempt 52 passes only if:

- its eager route reproduces all 16 frozen Attempt-7 outputs elementwise;
- native GE with `must_keep_origin_dtype` reproduces all 16 candidate eager
  outputs within `rtol=5e-3, atol=5e-3`;
- QK, attention, recurrent K, recurrent V and position are elementwise exact
  at all four positions;
- the observed AIR input/output ABI matches the generated manifest;
- both ExactQk and Bf16Barrier are present in the AIR and actually launched;
- the run uses idle physical NPU 7 before and after execution.

This is still a layer-slice prerequisite. It does not pass G4a without every
decoder layer, logits and Paged KV.
