# G4a Attempt 55a Native Layer-0 Boundary Localization

Date frozen: 2026-07-23

## Claim Boundary

This diagnostic identifies the first layer-0 tensor at which native GE differs
from the eager probe. It cannot pass the complete G4a decoder gate.

## Controls

- Immutable Attempt 55a AIR, eager reference and ABI.
- Four recurrent layer-0 Paged-KV steps on idle physical NPU 7.
- `must_keep_origin_dtype`, no `RESOURCE_CONFIG_PATH`, no compiler cache.
- Actual ExactQk and Bf16Barrier runtime launch records are required.

## Valid Diagnostic

- All 18 output files have the ABI-declared shape and dtype.
- All BF16 outputs are finite and next position is exact.
- Unaddressed Paged-KV elements remain bitwise unchanged.
- Every output is compared with `rtol=5e-3, atol=5e-3` and the first exact and
  tolerance mismatch are recorded for every step.
- NPU 7 is empty before and after execution.

## Decision

- First tolerance failure at masked scores/probabilities/attention value:
  isolate softmax-to-PxV and test a post-softmax materialization barrier.
- First failure at attention projection/residual: isolate O projection/add.
- Attention boundary passes and first failure is in post norm/gate/up/product/
  down projection: isolate the named MLP operation.
- All layer-0 outputs pass: graph-scale behavior, not the layer slice, remains
  the leading explanation.
