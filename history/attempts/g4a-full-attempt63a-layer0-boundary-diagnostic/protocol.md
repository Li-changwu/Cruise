# G4a Attempt 63a: full-graph layer-0 boundary localization

## Frozen claim

Keep the Attempt 62a layer stack and expose 16 ordered layer-0 boundaries from
input normalization through the MLP residual. The first exact mismatch chooses
the next single-mechanism correction.

## Frozen controls

- Physical NPU 7 only.
- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- B=1, four recurrent decoder steps, fixed initial Paged-KV and block table.
- All Attempt 62a computation and 197 MatMulV2 contract nodes remain unchanged.
- ExactQk and Bf16Barrier remain unchanged.
- Eager reference baseline: Attempt 53k.

## Export acceptance

- The original 21 eager arrays remain arraywise identical to Attempt 53k.
- ABI has the same eight inputs and adds only diagnostic outputs.
- Exactly 113 `LinearTransposeX2*` and 84 `QkvLinearTransposeX2*` nodes,
  all `MatMulV2` with `transpose_x2=true` and original-weight x2 sources.
- Counts: MatMul=0, MatMulV2=197, ExactQk=28, Bf16Barrier=28,
  BatchMatMul=29.

## Claim boundary

This experiment cannot pass G4a. Its only decision is the first layer-0
boundary whose native value differs from eager.
