# G4a Attempt 52: Repaired Full Layer-0 Attention/KV AIR

Date: 2026-07-23

## Question

Does the QK lowering that passed Attempt 51 restore the complete frozen Qwen
layer-0 Attention/KV slice, including the downstream softmax, V aggregation,
output projection and recurrent KV state?

## Controlled Change

The model, revision, four hidden rows, initial caches, positions, weights,
precision mode and all 16 frozen diagnostic outputs remain those of G2e
Attempt 7/8. Only the QK subgraph changes:

1. the exact custom BF16 QK kernel from Attempts 47-51 produces raw QK;
2. scaling executes in FP32;
3. the scaled score is explicitly materialized as BF16 before masking and
   softmax.

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
- the run uses idle physical NPU 7 and the Attempt-47 custom-op package.

This is still a layer-slice prerequisite. It does not pass G4a without every
decoder layer, logits and Paged KV.

