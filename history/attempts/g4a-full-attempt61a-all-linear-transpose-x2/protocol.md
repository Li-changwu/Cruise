# G4a Attempt 61a: all linear projections use the transpose-x2 contract

## Frozen claim

Replacing the remaining 84 Q/K/V bias projections with GE `MatMulV2` over the
original checkpoint weight layout, fused bias and `transpose_x2=true` removes
the earliest full-decoder native semantic drift while preserving eager exactly.

## Frozen controls

- Physical NPU 7 only.
- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- B=1, four recurrent decoder steps, fixed initial Paged-KV and block table.
- The 113 Attempt 60a-r2 no-bias projections remain unchanged.
- ExactQk and Bf16Barrier remain unchanged.
- Eager reference baseline: Attempt 53k.

## Export acceptance

- Attempt 60a eager NPZ is arraywise and SHA-256 identical to Attempt 53k.
- ABI remains eight inputs and four outputs.
- Exactly 113 `LinearTransposeX2*` and 84 `QkvLinearTransposeX2*` nodes,
  all `MatMulV2` with `transpose_x2=true` and original-weight x2 sources.
- Counts: MatMul=0, MatMulV2=197, ExactQk=28, Bf16Barrier=28,
  BatchMatMul=29.

## Claim boundary

This attempt can pass only G4a. Device-side greedy sampling, EOS, epoch control,
fixed B=2/4, active masks, and vLLM-Ascend integration remain out of scope.
