# M4a Performance Preflight Protocol

M4a is an early research-value gate. It does not close M2, M3, or M4. A
positive result sends the project back through M2 and M3 before formal M4
qualification; a negative result records a performance blocker without
weakening the M4 thresholds.

## Frozen claim and controls

The falsifiable claim is that, inside the current single-card support envelope,
Cruise reduces cross-token Host control enough to improve both median and p95
streaming TPOT by at least 15% and reduce Host CPU per output token by at least
30% versus the strongest stock route.

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
the server admits at most four sequences. Streaming cases provide token arrival
timestamps. Non-streaming cases expose the K=2 upper-bound path without
mislabeling normalized request latency as TPOT.

For every request the runner retains bounded token IDs, finish semantics,
latency, TTFT where observable, per-request TPOT where observable, and
inter-token gaps. For every scenario it records request/output throughput and
process-tree Host CPU per output token. Cruise additionally writes one
benchmark-only counter file at clean process exit containing Host schedules,
Device epochs, epoch-length distribution, Feed/Fetch calls, KV imports, and
native wall/CPU totals. These counters are disabled outside M4a.

The current streaming contract sets vLLM `RequestOutputKind.DELTA`, which forces
Cruise to K=1. Streaming is therefore the primary end-to-end test and an
intentional negative regime. Non-streaming K=2 results may explain a gap but
cannot substitute for the TPOT gate.

## Decision and storage

The comparison has two independent outcomes:

- `execution_pass`: all nine starts completed, exact output semantics matched,
  mode identities were proven, and the independent verifier reconstructed the
  reported metrics;
- `qualification_pass`: the three predeclared performance thresholds all
  passed on `decode-stream-c4`.

A threshold failure is a valid M4a result and must not make the evidence runner
discard the data. It leaves Cruise opt-in, keeps M4 open, and triggers an
attribution step using K=1 versus K=2, eager versus graph, Host CPU, and device
idle-gap evidence.

All builds, caches, logs, sockets, generated GraphPp weights, and profiler data
remain in marker-owned `/dev/shm` scratch. The existing content-addressed
runtime-weight bundle is reused in place. Only bounded JSON, counter files,
hashes, and diagnostic excerpts persist under `/workspace/cruise-runs`.
