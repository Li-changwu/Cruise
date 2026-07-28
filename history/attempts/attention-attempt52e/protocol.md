# G4a Attempt 52e: QK-and-Softmax BF16 Barriers

Date: 2026-07-23

## Question

After Attempt 52d made QK exact, do the remaining 1/5 FP16 Attention mismatches
come from GE crossing the second FP32-to-BF16 boundary between softmax and
the probability-value matrix multiplication?

## Controlled Change

The model, revision, four hidden rows, initial caches, positions, weights,
precision mode, ExactQk kernel and all 16 frozen diagnostic outputs remain
those of Attempt 52d. The only graph change is a second barrier:

1. the first barrier preserves the scaled BF16 QK score, as in Attempt 52d;
2. softmax still executes in FP32 and casts to BF16;
3. the second `Bf16Barrier` preserves those probability bits before P x V.

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
- ExactQk and exactly two Bf16Barrier nodes are present in the AIR;
- the native runtime records at least two Bf16Barrier launches;
- the run uses idle physical NPU 7 before and after execution.

This is still a layer-slice prerequisite. It does not pass G4a without every
decoder layer, logits and Paged KV.
