# P0 Iteration-Control Microbenchmark Protocol

## Claim Under Test

For a fixed recurrent state-update graph on Ascend 910B2, a Device UDF can
perform `N` graph invocations after one Host feed while preserving the final
state and reducing Host interaction and total steady-state latency relative to
a Host graph-replay loop.

## Hardware And Software

- Server: an Ascend 910B2 research server (local SSH alias redacted).
- Device: idle physical NPU 7, Ascend 910B2; exposed as process device 0.
- CANN: 9.0.0; `npu-smi`: 26.0.rc1.
- PyTorch/torch_npu: 2.9.0/2.9.0.
- Workload: recurrent INT32 Add over four state elements.
- Sweep: `N = 1, 2, 4, 8, 16, 32`.
- Warmup: one full transaction for GE/DataFlow; ten transactions for PyTorch.
- Measurement repetitions: 20 per N and route.

## Routes

1. `TorchEager`: Host submits one in-place NPU Add per iteration.
2. `NPUGraph`: Host calls `NPUGraph.replay()` once per iteration.
3. `HostGEGraph`: Host calls `Session::RunGraph()` once per iteration.
4. `DeviceUDF`: Host calls one `FeedDataFlowGraph()` and one
   `FetchDataFlowGraph()`; the UDF calls `RunFlowModel()` N times on device.

`TorchEager` and `NPUGraph` are paired PyTorch baselines. `HostGEGraph` and
`DeviceUDF` are the strongest identical-artifact causal pair because both use
the same GE Add graph and state contract.

## Metrics And Gates

- Correctness: every final state must exactly match the expected N-step state.
- Host submissions: `N` for Eager/NPUGraph/HostGEGraph; one feed for DeviceUDF.
- Host completions: one final synchronize/fetch per measured transaction.
- Latency: median and distribution of synchronized wall time.
- Host CPU: process CPU through submission and through final synchronization.
- Primary feasibility gate: DeviceUDF exact correctness at every N and constant
  Host feed count.
- Performance gate: DeviceUDF median wall time below HostGEGraph at N >= 2.

This synthetic result does not establish full Qwen/vLLM correctness or benefit.
It establishes the device-resident recurrent-control mechanism and crossover.
