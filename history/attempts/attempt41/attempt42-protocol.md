# G2g Attempt 42: Native GE Static-Tiling ExactQk

## Question

Does the Attempt 41 static-tiling package execute the original AIC QK
computation through native GE/AIR and reproduce the frozen eager QK values,
now that the failing fifth-argument reads are removed?

## Frozen Variables

- Hardware: one idle Ascend 910B2, physical NPU 7, process device 0.
- Package: Attempt 41, SHA-256
  `7b709fe0c5f700d6c3a82712fc8b538fed255e7e2d158e38e1d8eb679ec78124`.
- Static compute source: SHA-256
  `3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc`.
- AIR: Attempt 28 minimal `ExactQk` AIR, SHA-256
  `f02a3753a7a0a118bf27982d561d6eb7efc650e45ef3eb00e070072ca7e3478a`.
- Inputs: the eight frozen Attempt 28 operand files, tree SHA-256
  `dc6fd7b690a497f14b9ca48c34a12361ad1fb1d432bc1cc66326cde37d485587`.
- Eager reference: Attempt 7 QK reference, SHA-256
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Positions: 0, 1, 2, and 3; output `[1,28,1,8]` FP32.
- Correctness: elementwise exact equality and `rtol=5e-3`, `atol=5e-3`.
- Online compiler cache: new empty `cache-attempt42/`.
- Expected launch: AIC, `coreDim=24`, `schemMode=1`.

Physical NPU 7 is used because it is idle immediately before the experiment.
No occupied card is touched.

## Decision Rules

Attempt 42 passes only if artifact checks, Host compilation, fresh online
compilation, an AIC launch reporting `schemMode=1`, all four native `RunGraph`
calls, and all four exact eager comparisons pass. A device result 145 would
falsify static tiling as a sufficient repair. Numerical mismatch after four
successful runs would prove embeddability but reject semantic fidelity.

A pass establishes minimal GE/AIR embeddability of fixed-shape QK. It does not
yet prove full Qwen graph replacement, Device-UDF recurrence, vLLM integration,
or an end-to-end latency gain.
