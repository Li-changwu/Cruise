# G2g Attempt 44: Same-Card Direct-Launch Control

## Question

Does the previously exact direct `batch_matmul_transpose` path remain
elementwise exact on physical NPU 7 for the same four frozen operands and eager
reference used by the static-tiling GE experiment?

## Frozen Variables

- Physical NPU: 7, confirmed idle immediately before registration.
- Input NPZ SHA-256:
  `9db5b65ee01523d58c5f4b48e7dcf1ba1997ea02ed1dfbcb9e284750484f3b66`.
- Eager reference SHA-256:
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Attempt-44 Python harness SHA-256:
  `5e18309d27ec62fc2df3659c3bfe494f148217b4907df0d8d980a8fbc77e3768`.
- Imported operand/helper module SHA-256:
  `95d4120926ea4e69e20806a2a20b225676bfaefaf1b02956c62afe5c3f137149`.
- Direct kernel library SHA-256:
  `a8ae3257147cfed087bf3f289577068592aaae8eeeb390f2b921f73b178fb185`.
- PyTorch extension SHA-256:
  `eee2f71f57e59b3c30009a583035dd2be026767076abeeaecd9a6cf802f96edd`.
- Four positions, three custom launches per position, BF16 divide by
  `sqrt(128)`, exact equality and `rtol=5e-3`, `atol=5e-3` are unchanged.

## Decision Rules

Attempt 44 passes only if registration succeeds, all three custom runs at each
position are deterministic, raw custom BMM exactly equals native NPU BMM,
native and custom scaled values both exactly equal eager, and exactly twelve
`LaunchKernelWithHandle` records independently show `kernelType=0`,
`coreDim=24`, and `schemMode=2` on device 7.

A pass excludes cross-card numerical variation on physical NPU 7 and leaves
the GE launch/execution policy as the cause of Attempt 43's sparse mismatch. A
failure would prevent that attribution and require a hardware/card control.
