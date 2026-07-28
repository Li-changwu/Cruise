# G4a Attempt 59a Native Gate MatMulV2 transpose_x2 Probe

Date frozen: 2026-07-24

## Claim Boundary

This diagnostic tests whether preserving the original gate weight layout and
using `transpose_x2=true` closes the eager/native gate projection mismatch. It
cannot pass the complete G4a decoder gate.

## Controls

- New Attempt 59a AIR, eager reference and unchanged 19-output ABI.
- Immutable Attempt 58a eager/native artifacts as the direct baseline.
- Four recurrent layer-0 Paged-KV steps on idle physical NPU 7.
- `must_keep_origin_dtype`, no `RESOURCE_CONFIG_PATH`, no compiler cache.
- ExactQk, Bf16Barrier, one Bf16Materialize and the target gate MatMulV2 launch
  must all be present.

## Pass Criteria

- Shape, dtype, finiteness, next-position and unaddressed-KV invariants pass.
- Attempt 59a eager arrays and all pre-gate native arrays are bitwise identical
  to Attempt 58a.
- Attempt 58a step-1 first mismatch is gate pre-activation; Attempt 59a step-1
  gate pre-activation and gate are both bitwise exact.
- Every output remains within `rtol=5e-3, atol=5e-3`.
- Target gate launch uses an AIC MatMulV2 kernel with coreDim 19.
- NPU 7 is empty before and after execution.
