# G2g Attempt 41: Static-Tiling ExactQk Package

## Question

Can the numerically validated original AIC `ExactQk` kernel be made embeddable
in the native GE/AIR path by replacing only its reads from GE's failing fifth
`gm_tiling_data` argument with the exact compile-time constants for the frozen
shape, while using GE-supported `SetScheduleMode(1)`?

## Frozen Variables

- Current input package: Attempt 39, SHA-256
  `f184a2d495e4ae52e96a9b4d35be7534d840322e6162b7dea8de41a693158f40`.
- Original compute source before this change: SHA-256
  `d136c5f31814ad677edcb209b2920f4ec6ee3537297aa13b572120a400b192e7`.
- Static-tiling compute source: SHA-256
  `3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc`.
- Mode-1 Host tiling source: SHA-256
  `1b313413a3044a5d39ccd072fe4517c19856cd55cb8aa8a26348d07b54ca32f5`.
- Tiling vector is frozen from the byte-level Host prediction:
  `[28,1,128,8,16,512,16,1,1,1,28,5,2336,24,0,0,0,0]`.
- The selected branch is the original `einsum_0_n_bf16_nd` branch implied by
  `tilingKey >> 2 == 584`.
- The five-argument entry ABI, A/B/C addresses, workspace argument, original
  matmul computation, `blockDim=24`, BF16 types, fixed shapes, generated tiling
  aliases, and Ascend 910B target remain unchanged.

## Decision Rules

Attempt 41 is build/package-only. It passes only if input and staged-file
integrity pass, the current artifacts are archived without overwriting prior
attempts, offline OPC compilation succeeds, a changed package is produced and
installed, the installed source is byte-identical to the frozen static source,
and the source contains no dereference of `gm_tiling_data` while retaining both
AIC declarations and `SetScheduleMode(1)`.

A pass authorizes a separate native GE runtime attempt with a new empty compiler
cache and the four frozen QK operand pairs. It proves neither runtime execution
nor numerical fidelity by itself.
