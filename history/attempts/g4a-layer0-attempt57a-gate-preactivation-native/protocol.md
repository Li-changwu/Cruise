# G4a Attempt 57a Native Gate MatMul-to-SiLU Localization

Date frozen: 2026-07-24

## Claim Boundary

This diagnostic distinguishes gate projection MatMul error from SiLU error by
observing a materialized BF16 pre-activation. It cannot pass the complete G4a
decoder gate.

## Controls

- New Attempt 57a AIR, eager reference, and ABI with one extra output.
- Immutable Attempt 55a eager/native artifacts as common-output baselines.
- Four recurrent layer-0 Paged-KV steps on idle physical NPU 7.
- `must_keep_origin_dtype`, no `RESOURCE_CONFIG_PATH`, no compiler cache.
- Actual ExactQk, Bf16Barrier, and one Bf16Materialize launch record are
  required.

## Valid Diagnostic

- All 19 output files have the ABI-declared shape and dtype.
- All BF16 outputs are finite and next position is exact.
- Unaddressed Paged-KV elements remain bitwise unchanged.
- Every output is compared with `rtol=5e-3, atol=5e-3` and the first exact and
  tolerance mismatch are recorded for every step.
- NPU 7 is empty before and after execution.
- Common eager arrays and all native outputs before gate pre-activation remain
  bitwise identical to Attempt 55a.

## Decision

- Step-1 pre-activation differs: localize to gate projection MatMul.
- Step-1 pre-activation is exact but gate differs: localize to SiLU.
- Any pre-gate baseline change: classify as confounded and rerun a narrower
  standalone probe.
