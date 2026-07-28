# G4a Attempt 58a Native Gate MatMulV2 Probe

Date frozen: 2026-07-24

## Claim Boundary

This controlled diagnostic tests whether GE `MatMulV2`, used only for the
layer-0 gate projection, reproduces the eager ACLNN MatMulV2 result. It cannot
pass the complete G4a decoder gate.

## Controls

- New Attempt 58a AIR, eager reference and unchanged 19-output ABI.
- Immutable Attempt 57a eager/native artifacts as the direct baseline.
- Four recurrent layer-0 Paged-KV steps on idle physical NPU 7.
- `must_keep_origin_dtype`, no `RESOURCE_CONFIG_PATH`, no compiler cache.
- Actual ExactQk, Bf16Barrier and one Bf16Materialize launch record required.

## Pass Criteria

- All output files have the ABI-declared shape and dtype, all BF16 outputs are
  finite, next position is exact, and unaddressed Paged-KV remains unchanged.
- Attempt 58a eager arrays are bitwise identical to Attempt 57a.
- Native arrays before gate pre-activation are bitwise identical to Attempt
  57a.
- Attempt 57a step-1 first mismatch is gate pre-activation, while Attempt 58a
  step-1 gate pre-activation and gate are both bitwise exact.
- Every output remains within `rtol=5e-3, atol=5e-3`.
- NPU 7 is empty before and after execution.

If gate pre-activation remains different, GE MatMulV2 does not explain the
eager/native discrepancy and the next probe must compare kernel tiling rather
than expanding the converter.
