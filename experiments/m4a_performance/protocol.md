# M4a Performance Preflight Protocol

M4a is an early research-value gate. It does not close M2, M3, or M4. A
positive result sends the project back through M2 and M3 before formal M4
qualification; a negative result records a performance blocker without
weakening the M4 thresholds.

## Frozen claim and controls

The falsifiable claim is that, inside the current single-card support envelope,
Cruise reduces cross-token Host control enough to improve streaming TPOT p50 by
at least 10%, avoid a p95 regression larger than 5%, improve output tokens/s by
at least 10%, and reduce Host CPU per output token by at least 30% versus the
strongest stock route in the same blocked run.

The three routes are:

1. unmodified vLLM-Ascend eager execution (`--enforce-eager`);
2. unmodified vLLM-Ascend PIECEWISE ACLGraph execution;
3. Cruise with stock ACLGraph prefill and the DataFlow resident decode path.

Every route uses the same Qwen2.5-7B-Instruct revision, tokenizer, NPU, 512 MiB
KV-cache budget, synchronous scheduling, maximum batch size four, warmup
manifest, and measured request manifest. Every route uses the versioned
single-primary-EOS generation config in this directory instead of inheriting
the model's additional EOS token. Every request explicitly fixes all supported
greedy-sampling fields, including a repetition penalty of one. This preserves
the declared single-EOS support boundary without admitting arbitrary stop
tokens. Initialization is excluded and reported separately. Each route receives
three independent service starts in the blocked order:

```text
eager-1, graph-1, cruise-1,
cruise-2, graph-2, eager-2,
graph-3, cruise-3, eager-3
```

The strongest baseline for a scenario is selected between eager and ACLGraph
using the lower primary latency metric. All other claims for that scenario use
the same selected baseline; metrics are not allowed to choose separate
baselines opportunistically.

## Workloads and measurements

The versioned workload covers short and decode-heavy requests, concurrency
1/4, closed-loop and bursty arrival, and an overload concurrency of eight while
the server admits at most four sequences. The K=6 decode path returns each
committed token as a separate DELTA event. Streaming cases retain token and
chunk arrival timestamps, so burst cadence and inter-token jitter remain visible
instead of being folded into a single multi-token API event.

For every request the runner retains bounded token IDs, finish semantics,
latency, TTFT where observable, per-request TPOT where observable, and
inter-token gaps. For every scenario it records request/output throughput and
process-tree Host CPU per output token. Cruise additionally writes one
benchmark-only counter file at clean process exit containing Host schedules,
Device epochs, epoch-length distribution, Feed/Fetch calls, KV-import mode,
and per-component EngineCore, Python scheduler, sidecar/native, socket, and
Device-KV transfer wall/CPU totals. These counters are disabled outside M4a.

The current streaming contract sets vLLM `RequestOutputKind.DELTA` and Cruise
keeps every token as a separate consumable event while the Device performs the
bounded epoch. The benchmark requires K=6 epochs, direct Device KV imports,
and 1,280/1,280 eligible decode tokens on the Device route for all three
Cruise starts. It separately reports near-zero intra-burst gaps and jitter;
these are not relabeled as smooth token cadence.

## Decision and storage

The comparison has two independent outcomes:

- `execution_pass`: all nine starts completed, exact output semantics matched,
  mode identities were proven, and the independent verifier reconstructed the
  reported metrics;
- `qualification_pass`: the four predeclared performance thresholds all
  passed on `decode-stream-c4`.

A threshold failure is a valid M4a result and must not make the evidence runner
discard the data. It leaves Cruise opt-in, keeps M4 open, and triggers an
attribution step using K=6 control-plane timing, eager versus graph, Host CPU,
Device graph time, IPC, streaming cadence, and profiler evidence.

For profiler attribution only, the Cruise route uses
`profile_sidecar.py`. It loads and warms the same AIR, controller, native
sidecar, Unix socket, and B=4 K=6 plan without starting a colocated vLLM
EngineCore. This avoids the profiler-time HBM collision between two model
copies. The sidecar first loads the graph and completes warmup with profiling
disabled. It then starts CANN `TASK_TIME_L0` collection in-process, executes
the representative epochs, and stops collection before releasing the runner.
This ordering also avoids retaining profiler-time GE compilation memory while
weights are loaded. The output is labeled `cruise-sidecar-only`, performs no
initial Device KV import, and is valid only for Device graph timeline
attribution. It does not replace API semantics, Host process-tree CPU,
streaming cadence, or service-level TPOT and throughput evidence from the
formal runs.

All builds, caches, logs, sockets, generated GraphPp weights, and profiler data
remain in marker-owned `/dev/shm` scratch. The existing content-addressed
runtime-weight bundle is reused in place. Only bounded JSON, counter files,
hashes, and diagnostic excerpts persist under `/workspace/cruise-runs`.
