# G4a Attempt 64a: projection-residual defusion

## Frozen claim

Attempt 63a changed all layer-0 boundary values to eager-exact but failed the
Attempt 62a common-output invariant. A launch-plan differential showed why:
the Attempt 62a layer-0 `o_proj` and `down_proj` MatMulV2 kernels each consume a
third residual tensor (`arg_size=32`), while Attempt 63a uses two-input
MatMulV2 kernels (`arg_size=24`) followed by explicit Add kernels.

Attempt 64a tests one mechanism: prevent projection-residual epilogue fusion.
It starts from the uninstrumented Attempt 62a computation and inserts an opaque
`Bf16Materialize` after every `o_proj` and `down_proj`, before the two residual
adds. No graph outputs are added or removed.

## Frozen controls

- Physical NPU 7 only.
- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Fixed B=1 and four recurrent Paged-KV decoder steps.
- Eight inputs and the five Attempt 62a outputs remain unchanged.
- Original weights, all 197 MatMulV2 transpose-x2 contracts, ExactQk and the
  score Bf16Barrier remain unchanged.
- The only graph change is 56 Bf16Materialize nodes: two per decoder layer.

## Acceptance

- All 25 eager arrays are bitwise identical to Attempt 62a/Attempt 53k common
  content.
- AIR counts are MatMul=0, MatMulV2=197, ExactQk=28, Bf16Barrier=28,
  Bf16Materialize=56 and BatchMatMul=29.
- All 56 target `o_proj`/`down_proj` native kernels have `arg_size=24` and
  `prefetch_count_1=0`; 56 materialize kernels actually launch.
- The four-step logits, Paged-KV, greedy token and position checks pass frozen
  tolerance, and no per-layer hidden tensor fails tolerance.

## Claim boundary

Passing supports residual defusion in the five-output diagnostic graph. G4a
still requires a clean four-output decoder graph without `layer_hiddens`.
