# G2g Attempt 46: Native GE Static-Tiling Schedule Mode 0

## Question

Does changing only GE's supported schedule mode from 1 to 0 remove the sparse
numerical mismatch while preserving successful native execution of the
static-tiling AIC QK?

## Frozen Variables

- Hardware: idle Ascend 910B2 physical NPU 7, process device 0.
- Package: Attempt 45, SHA-256
  `47be1f718dffd57fd8f0328032e7165d860e6f51d20e955ba1f13503cea3762c`.
- Static compute source remains SHA-256
  `3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc`.
- Only Host scheduling changed from Attempt 42: `SetScheduleMode(0)`.
- AIR SHA-256:
  `f02a3753a7a0a118bf27982d561d6eb7efc650e45ef3eb00e070072ca7e3478a`.
- Eight-input tree SHA-256:
  `dc6fd7b690a497f14b9ca48c34a12361ad1fb1d432bc1cc66326cde37d485587`.
- Eager reference SHA-256:
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Positions, shapes, dtypes, native Host, comparator, exact plus tolerance
  thresholds, `coreDim=24`, and all static tiling constants are unchanged.
- Online compiler cache: new empty `cache-attempt46/`.

## Decision Rules

Attempt 46 passes only if artifact checks, Host compilation, fresh online
compilation, a single AIC `ExactQk` launch independently containing
`kernelType=0`, `coreDim=24`, and `schemMode=0`, four successful `RunGraph`
calls, and elementwise exact equality at all four positions pass.

Successful execution with the same sparse mismatch would falsify schedule mode
1 as the sole cause. A 145 failure would show mode 0 is not a usable repair.
An exact pass would establish a fixed-shape minimal GE/AIR QK route, but still
not full Qwen, Device-UDF recurrence, or vLLM integration.
