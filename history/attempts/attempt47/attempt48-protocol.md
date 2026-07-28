# G2g Attempt 48: Export Three-Input ExactQk AIR

## Question

Can TorchAir export a static minimal AIR whose `ExactQk` node has A, B, and a
72-byte explicit tiling tensor as ordinary graph inputs under the Attempt 47
custom-op contract?

## Frozen Variables

- Hardware: idle physical NPU 7 for the one eager-wrapper sanity execution.
- Package: Attempt 47, SHA-256
  `6e4435ce0c85b4c18ca672a957a023c77b0f83810c86aeca6ac60e8be6b7e18e`.
- Exporter SHA-256:
  `ad566a77f2f0f3c1b82880936d088906858655595a54b83ceab290ab4ccac4d8`.
- Attempt-8 input NPZ SHA-256:
  `9db5b65ee01523d58c5f4b48e7dcf1ba1997ea02ed1dfbcb9e284750484f3b66`.
- Eager reference SHA-256:
  `d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f`.
- Explicit little-endian `uint32` vector:
  `[28,1,128,8,16,512,16,1,1,1,28,5,2336,24,0,0,0,0]`.
- A/B mapping, downstream BF16 divide, unsqueeze, FP32 output, fixed shapes,
  TorchAir mode, and step-1 exact eager sanity remain unchanged.

## Decision Rules

Attempt 48 passes only if artifact checks and environment setup pass, the
step-1 eager wrapper exactly equals the frozen reference, TorchAir emits an AIR
and pbtxt, the graph contains an `ExactQk` node with `explicit_tiling`, the
saved `tiling.bin` is exactly 72 bytes and decodes to the frozen vector, and
the four A/B pairs plus tiling file are frozen by a tree hash.

This attempt exports artifacts only; it does not run the AIR through native GE
and gives no embeddability or full-Qwen verdict.
