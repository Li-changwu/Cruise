# G2g Attempt 49: Native GE Explicit-Tiling ExactQk

## Question

Does the three-input `ExactQk` AIR execute the original dynamic AIC QK through
native GE and recover exact eager semantics when the same 72 tiling bytes arrive
through a validated ordinary graph-input address?

## Frozen Variables

- Hardware: idle Ascend 910B2 physical NPU 7, process device 0.
- Package: Attempt 47, SHA-256
  `6e4435ce0c85b4c18ca672a957a023c77b0f83810c86aeca6ac60e8be6b7e18e`.
- AIR: Attempt 48, SHA-256
  `00f846fa80fc09f43a760f28e64927027b7be9dc8c54be7d7b0442532a4269b9`.
- Nine-file native input tree SHA-256:
  `11371e992d0634958b5583d425e77013dbbd701e4aeb8e3b7dfcae6f37246768`.
- Explicit tiling file SHA-256:
  `1fc40ec0d67e231128773a5448cdd3333bdd1b97ea8a67bb7ba881d43b0da51f`.
- Native three-input Host SHA-256:
  `8ce3a52c16cfb80301f81565cee7b622c9a48e44b21f1679cd171615a5736e0c`.
- Eager reference SHA-256:
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Four positions, shapes, BF16 divide, FP32 output, exact plus tolerance gates,
  24 blocks, and schedule mode 0 remain unchanged.
- Online compiler cache: new empty `cache-attempt49/`.

## Decision Rules

Attempt 49 passes only if artifact checks, Host compilation, fresh online
compilation, one `ExactQk` launch independently containing `arg_size=48`,
`kernelType=0`, `coreDim=24`, and `schemMode=0`, all four native `RunGraph`
calls, and all four elementwise-exact eager comparisons pass.

A result 145 would reject an ordinary explicit input as a sufficient address
repair. The same sparse 1-ULP mismatch would reject static constantization as
the sole numerical cause. A full pass establishes minimal fixed-shape GE/AIR
QK embeddability; full-Qwen replacement and Device-UDF recurrence remain
separate gates.
