# G2g Attempt 45: Static-Tiling Schedule-Mode-0 Package

## Question

Can the byte-identical static-tiling AIC QK source be packaged with GE
`SetScheduleMode(0)` so that a subsequent experiment can isolate schedule mode
as the cause of the sparse mode-1 numerical mismatch?

## Frozen Variables

- Input package: Attempt 41, SHA-256
  `7b709fe0c5f700d6c3a82712fc8b538fed255e7e2d158e38e1d8eb679ec78124`.
- Static compute source: SHA-256
  `3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc`.
- Input mode-1 Host source: SHA-256
  `1b313413a3044a5d39ccd072fe4517c19856cd55cb8aa8a26348d07b54ca32f5`.
- New mode-0 Host source: SHA-256
  `6e04aff60c4320def8969ad10a808466db2b5fb1c9a99ab5a3c71ad410f5d89f`.
- Kernel body, static tiling constants, fixed shapes, BF16 types, entry ABI,
  `blockDim=24`, tiling key, and generated aliases remain unchanged.

## Decision Rules

Attempt 45 is build-only and does not use an NPU. It passes only if all input
hashes match, the mode-1 package and Host source are archived without
overwriting prior attempts, exactly the Host schedule source changes, offline
OPC build and installation succeed, the package changes, the installed kernel
remains byte-identical, and the Host source contains mode 0 but neither mode 1
nor mode 2.

A pass authorizes a fresh-cache native GE execution with the same four frozen
QK inputs. It does not itself establish runtime or correctness.
