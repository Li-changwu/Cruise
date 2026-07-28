# Real-Qwen P0 Host-vs-Device Recurrence Protocol

## Question

For the same real Qwen2.5-7B layer-0 attention/KV AIR, does moving N repeated
`RunFlowModel` calls into a Device UDF reduce Host Feed/Fetch interactions,
Host CPU time, and wall time while preserving the Host AIR-loop result?

## Frozen Artifacts

- Hardware: one idle Ascend 910B2, physical NPU 7, process device 0.
- AIR SHA-256:
  `eb0effc65a3bba7977430d38a316f374a8f7ba0716065162198153d9191240e3`.
- Reference/input NPZ SHA-256:
  `02ac4d7507ae0dca353fe447e8be4e1d8ddbe8be4b8e8d7f98475d4bc04470cc`.
- Graph config SHA-256:
  `5c940de05183a6ab4d611a2534e0a9e47e5a054a2d0c400c68dc1239fb587095`.
- Func config SHA-256:
  `b1297f8f18e3a4d047adfc8c19974e4ba1f168e9db5a6b60d86722d598f1c6ba`.
- Device controller source SHA-256:
  `598775cf2811c7fa8e5ff84cb00335fddc4a596f12f7ac4f642728ae5a3f6e48`.
- Device controller binary SHA-256:
  `eb5d0e638f70323192408276d0203dbd84d8e747311107d89f7d82c310db92b3`.
- Benchmark harness SHA-256:
  `6f6d9c612b2f393083791f46f33bd78b8d96d777908e2f8053fd0ad9cbda6fbd`.

## Frozen Design

- N sweep: 1, 2, 4. The existing hidden table has four rows, so larger N is
  outside this real-layer artifact's valid state space.
- Per N: build Host and Device graphs once, warm up each route 3 times, then
  collect 10 repetitions.
- Measurement order alternates Host-first and Device-first by repetition.
- Host route: N Feed/Fetch pairs, with K/V/position returned to Host and fed to
  the next AIR invocation.
- Device route: one Feed and one final Fetch; the Device UDF invokes the same
  AIR N times and feeds K/V/position back on device.
- Timers: process wall time and Host-process CPU time around each complete
  route; graph construction and warmup are excluded.
- Correctness: all four final tensors must be elementwise identical between
  Host and Device for all repetitions, and final position must equal N.

## Decision Rules

The mechanism gate passes only if every run exits 0, all Host/Device outputs
are exact, and Device Feed/Fetch counts remain one. The performance gate passes
if the Device median wall time is lower for both N=2 and N=4. N=1 is retained
as the expected fixed-overhead/negative regime and is not required to win.

Host CPU reduction is reported but not made a hard gate because process CPU
accounting includes DataFlow client-side serialization and runtime helper work.

## Claim Boundary

This AIR uses real Qwen layer-0 weights and recurrent KV state, but prior G2d
evidence shows it is not eager-equivalent under `rtol=5e-3`, `atol=5e-3`.
Therefore a pass proves controller-path equivalence and real-layer recurrence
overhead reduction only. It does not prove a semantically accepted full
decoder, paged KV, sampling, dynamic batching, or vLLM integration.
