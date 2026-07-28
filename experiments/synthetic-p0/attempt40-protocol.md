# G2g Attempt 40: Native GE ExactQk With Schedule Mode 2

## Question

Can the numerically validated `ExactQk` AIC kernel execute through the native
GE/AIR path when its Host tiling policy uses `SetScheduleMode(2)`, the schedule
mode used by the successful direct PyTorch custom-operator launch?

## Frozen Variables

- Hardware: one idle Ascend 910B2, physical NPU 7, process device 0.
- Package: Attempt 39 `ExactQk`, SHA-256
  `f184a2d495e4ae52e96a9b4d35be7534d840322e6162b7dea8de41a693158f40`.
- Compute source: original AIC QK kernel, SHA-256
  `d136c5f31814ad677edcb209b2920f4ec6ee3537297aa13b572120a400b192e7`.
- AIR: Attempt 28 minimal `ExactQk` AIR, SHA-256
  `f02a3753a7a0a118bf27982d561d6eb7efc650e45ef3eb00e070072ca7e3478a`.
- Inputs: the eight frozen Attempt 28 operand files, tree SHA-256
  `dc6fd7b690a497f14b9ca48c34a12361ad1fb1d432bc1cc66326cde37d485587`.
- Eager reference: Attempt 7 QK reference, SHA-256
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Positions: 0, 1, 2, and 3; output shape `[1,28,1,8]` FP32.
- Correctness: elementwise exact equality and `rtol=5e-3`, `atol=5e-3`.
- Online compiler cache: new empty `cache-attempt40/`.

Physical NPU 7 replaces the historically used NPU 6 because NPU 6 is occupied
by an unrelated live process. Both are 910B2 devices in the same server. This
prevents interference but limits the result to same-model, same-server hardware
equivalence rather than bitwise cross-card characterization.

## Decision Rules

Attempt 40 passes only if artifact checks, Host compilation, fresh online
compilation, a launch explicitly reporting `schemMode=2`, all four native
`RunGraph` calls, and all four exact eager comparisons pass. A device failure
with result 145 would falsify schedule mode 2 as a sufficient repair. A pass
would establish minimal GE/AIR custom-QK embeddability; it would not yet prove
full Qwen replacement, Device-UDF recurrence, or latency benefit.
