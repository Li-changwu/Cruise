# P5 Persistent Decode Qualification protocol

P5 exposes the P4-qualified Persistent Device Model Owner through a narrow
vLLM/OpenAI-compatible `/v1/completions` service. The HTTP process batches only
request-boundary admissions, streams the Device Committed Prefix, and sends
asynchronous cancellation or shutdown events. The model-load-scoped
AICPU/DataFlow Owner autonomously advances every internal Scheduling Quantum
and invokes the AICore Decoder. There is no Host token-step API, Host-visible
Decode epoch, `ResidentEpochScheduler`, legacy sidecar backend, or runtime
fallback to that frozen scaffold.

## Formal workload

The pinned Qwen2.5-7B-Instruct revision is
`a09a35458c702b33eeacc393d103063234e8bc28`, with TP=1, PP=1, greedy sampling,
streaming, a 128-token prompt, 256 output tokens, closed-loop concurrency four,
and 32 measured requests per independent model start. One C4 cohort warms each
freshly loaded route before the measurement window. Model loading and warmup
are excluded from Host CPU and service timing; every measured request is
included.

The stock Graph route reads the single canonical model asset at
`/workspace/cruise-assets/models/Qwen2.5-7B-Instruct-a09a35458c702b33eeacc393d103063234e8bc28`.
The matrix preflight requires an exact revision marker and verifies the
per-file `.cruise-model-sha256` manifest before any service is started. Model
weights, download caches, and copies are forbidden in run or tmpfs storage.
The frozen manifest SHA-256 is
`651d64436c415afd6faf3d82f14086c2df54d99c02f66f81a15b83a2d17de5f9`.

The balanced order is frozen as:

```text
graph-1, owner-1,
owner-2, graph-2,
graph-3, owner-3
```

Each start launches and unloads its own service process tree. The Graph route
is stock vLLM-Ascend PIECEWISE ACLGraph. Its plugin allowlist contains only the
installed Ascend platform and official Ascend model, loader, connector, and
service-profiling registrations; the frozen Cruise epoch plugin cannot load.
The official Ascend `fuse_norm_quant` pass is disabled because the pinned CANN
9.0.0 `libopapi.so` does not export its required `aclnnAddRmsNormBias` symbol.
For the same missing symbol, the Graph launcher selects the existing
vLLM-Ascend `torch_npu.npu_add_rms_norm` layernorm fallback. This module-local
compatibility selection does not change other custom ops, enable eager
execution, or introduce a Cruise execution path.
Both routes receive byte-identical HTTP bodies from the same client and must
return exact token IDs, text, finish reason, streaming boundary, and usage
semantics.

Before a formal matrix, `CRUISE_P5_STOP_AFTER_GRAPH1=1` may be used for a
single-start hardware smoke. It runs the same `graph-1` workload and lifecycle
guards, exits only after that result passes, and does not produce a P5 gate or
qualification claim.
Likewise, `CRUISE_P5_STOP_AFTER_FIRST_PAIR=1` stops only after `graph-1` and
`owner-1` pass the independent exact-pair verifier; it also makes no aggregate
P5 qualification claim.

Owner startup emits only bounded lifecycle markers for GE initialization,
graph construction, `CompileGraph`, output-drain creation, readiness, protocol
failure, and exit. Per-token Device output is never copied into the service
log. The native startup timeout is 1,140 seconds, 60 seconds shorter than the
outer HTTP readiness timeout, so a failure can retain at most eight 8 KiB CANN
log tails before scratch cleanup instead of persisting raw compiler logs.

## CPU and performance gate

Whole-process-tree CPU is sampled from `/proc` every 20 ms from immediately
before the measured requests until immediately after them. The meter sums each
process's own `utime+stime`, keys identities by PID plus start time, and retains
the last sample for descendants that exit during the window. The benchmark
client and bounded-log process are outside the measured service tree.

Across the three starts, Owner must improve over same-round Graph by at least:

- 50% in whole-process-tree Host CPU per output token;
- 15% in per-request TPOT p50;
- 15% in per-request TPOT p95;
- 15% in output tokens per second.

The gate additionally requires 32 admissions, exactly 3,064 AICore calls and
8,192 Owner commit events per measured start, zero Host Decode steps, exact
Graph/Owner semantics, six unique cold starts, and an empty legacy-route module
audit. TTFT remains observable; ADR 0017 defers its final 5% guard to P6.

## Storage boundary

Every matrix uses one PID-scoped `/dev/shm/cruise-p5-*` directory, a 2 GiB
scratch limit, lifecycle mark/finalize, pre/post shared-memory audits, bounded
service logs, and a mutable symlink weight view removed on every exit. Builds,
caches, driver logs, copied weights, and raw profiler data are never durable.
Only the six compact start JSON files, the independent gate, hashes, storage
snapshots, and bounded diagnostic excerpts may remain in `cruise-runs`.

## Current state

The P5 service, CPU meter, six-start orchestrator, and independent verifier are
implemented. This is implementation evidence only. P5 remains open until the
full target-NPU matrix produces a passing `p5-gate.json`. One profiler-guided
redesign and complete rerun is allowed after an initial performance miss; a
second miss stops P6.

The canonical model is present and manifest-verified at the path frozen above;
no model download is required. The latest completed first pair is
`persistent-decode-p5-first-pair-20260814-r5`. Its Graph route achieved 7.544
ms Host CPU per output token, 18.533/18.746 ms TPOT p50/p95, and 198.97 output
tokens/s. Owner achieved 2.291 ms Host CPU per output token, but
43.444/43.614 ms TPOT p50/p95 and 61.31 output tokens/s. Owner therefore proved
a 69.6% Host saving while missing both latency and throughput targets; it also
reported 3,071 measured AICore calls instead of the frozen 3,064. This is a
failed P5 pair, not a qualification result.

The active redesign replaces the slow decomposed-attention compute closure
with FusedInferAttentionScore while preserving the same Persistent Device Model
Owner. The weight-free diagnostic in `persistent_decoder_p3/fia_graphpp_probe`
proved that CANN's public offline builder can create a self-contained 18.9 MiB
OM and that `aclmdlExecute` produces the exact BF16 output. The same FIA AIR
still fails in GraphPp with `BinaryGetFunctionByEntry failed, funcEntry=0`.
`Graph::LoadFromSerializedModelArray` cannot bridge the OM into GraphPp because
it expects a serialized GE model definition and rejects the OM with status
`1343225857`. CANN 9.0's installed public DataFlow headers expose GraphPp,
FlowGraphPp, and FunctionPp but no precompiled-model process point. Private
`ModelPp` symbols are not an allowed ABI.

Consequently, Host-side `aclmdlExecute` is diagnostic evidence only and may not
become a Decode step path. P5 stays on the Device Owner architecture and is
blocked on one of two supported compute-plane outcomes: an officially exposed
precompiled-model process point for this CANN stack, or a GraphPp-compatible
FIA/custom AICore implementation. The legacy Owner remains the correctness
reference while this single permitted P5 redesign is open.
