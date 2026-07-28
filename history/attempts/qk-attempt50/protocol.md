# G4a QK Attempt 50: Raw Boundary and Scaling Semantics

Date: 2026-07-22

## Question

Does the sparse GE/AIR mismatch originate in the `ExactQk` AIC kernel output,
or in the BF16 scaling operation that follows it? If the raw output is exact,
can an explicit FP32 scale followed by BF16 rounding reproduce the frozen eager
semantics exactly?

## Frozen Inputs

- Physical device: idle Ascend 910B2 NPU 7, exposed as process device 0.
- Custom-op package and explicit 72-byte tiling input: Attempt 47/48.
- Q and K operands: the four frozen positions from G2e Attempt 8.
- Eager scaled reference: G2e Attempt 7.
- Raw BF16 reference: the direct-launch output from Attempt 44.
- GE precision mode: `must_keep_origin_dtype`.

The AIR returns four outputs from the same `ExactQk` invocation:

1. `raw_qk`: BF16 kernel output before scaling.
2. `legacy_scaled`: current BF16 `raw / sqrt(128)` followed by FP32 cast.
3. `fp32_div_bf16`: FP32 division, explicit BF16 rounding, then FP32 cast.
4. `fp32_mul_bf16`: FP32 reciprocal multiplication, explicit BF16 rounding,
   then FP32 cast.

## Decision Rules

- Kernel/GE embedding is cleared only if `raw_qk` is elementwise exact at all
  four positions versus the direct-launch reference.
- The scaling semantic difference is resolved only if at least one explicit
  candidate is elementwise exact at all four positions versus eager.
- A repaired scaling candidate must also preserve all positions that already
  passed in the legacy route.
- This attempt resolves only the frozen QK boundary. It does not establish
  full-decoder, logits, Paged-KV, or Device-UDF epoch correctness.

