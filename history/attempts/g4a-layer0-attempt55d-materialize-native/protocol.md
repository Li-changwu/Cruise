# G4a Attempt 55d Native Layer-0 BF16 Materialization Probe

Date frozen: 2026-07-24

## Claim Boundary

This diagnostic tests whether explicit BF16 materialization after the gate and
up projections removes Attempt 55a's step-1 gate mismatch. It cannot pass the
complete G4a decoder gate.

## Controls

- New Attempt 55d AIR with exactly two Bf16Materialize nodes.
- Immutable Attempt 55a eager/native outputs as the differential baseline.
- Four recurrent layer-0 Paged-KV steps on idle physical NPU 7.
- `must_keep_origin_dtype`, no `RESOURCE_CONFIG_PATH`, no compiler cache.
- Actual ExactQk, Bf16Barrier, and eight Bf16Materialize runtime launch records
  are required.

## Valid Diagnostic

- All 18 output files have the ABI-declared shape and dtype.
- All BF16 outputs are finite and next position is exact.
- Unaddressed Paged-KV elements remain bitwise unchanged.
- Every output is compared with `rtol=5e-3, atol=5e-3` and the first exact and
  tolerance mismatch are recorded for every step.
- NPU 7 is empty before and after execution.
- The 55a and 55d eager references are array-wise identical.
- All native outputs before gate are array-wise identical between 55a and 55d.

## Decision

- Support the fusion-boundary hypothesis only if step-1 gate becomes bitwise
  exact, the first mismatch moves later, and no tolerance regression appears.
- If the pre-gate native outputs change, classify the experiment as confounded.
- If step-1 gate remains different, reject this boundary and isolate standalone
  SiLU and MatMul semantics next.
