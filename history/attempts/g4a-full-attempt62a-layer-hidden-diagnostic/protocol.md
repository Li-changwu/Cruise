# G4a Attempt 62a: full-graph per-layer hidden localization

## Frozen claim

Expose a stacked BF16 hidden state after every complete decoder layer while
leaving the Attempt 61a computation unchanged. Compare these 28 boundaries
between eager and native to identify the first divergent layer per step.

## Frozen controls

- Physical NPU 7 only.
- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- B=1, four recurrent decoder steps, fixed initial Paged-KV and block table.
- All 197 Attempt 61a MatMulV2 contract nodes remain unchanged.
- ExactQk and Bf16Barrier remain unchanged.
- Eager reference baseline: Attempt 53k.

## Export acceptance

- The original 21 eager arrays remain arraywise identical to Attempt 53k.
- ABI has the same eight inputs and adds only one diagnostic output.
- Exactly 113 `LinearTransposeX2*` and 84 `QkvLinearTransposeX2*` nodes,
  all `MatMulV2` with `transpose_x2=true` and original-weight x2 sources.
- Counts: MatMul=0, MatMulV2=197, ExactQk=28, Bf16Barrier=28,
  BatchMatMul=29.

## Claim boundary

This experiment cannot pass G4a. Its only decision is the first layer whose
native hidden state differs from the eager hidden state.
