# M4a Three-Route Performance Preflight on NPU0-1

Status: negative preflight recorded on 2026-07-31. All nine service runs were
semantically correct and cleaned up, but Cruise failed both route coverage and
all three frozen performance thresholds. This result does not close M2, M3,
or formal M4.

## Environment and Controls

- SSH target: `NPU0-1`
- Physical NPU: `0`, Ascend 910B2
- Conda environment: `vllm-hust-dev`
- Cruise source: `0f1d24077a79640d753e04cd6a5bf643d2460fe8`
- vLLM source: `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7`
- vLLM-Ascend source: `e967f235ba66edb48a28a6d943aee9455fee70cf`
- Runtime-weight manifest:
  `2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761`
- Fixed model, tokenizer, single primary EOS, greedy request parameters,
  512 MiB stock KV budget, maximum batch size four, and synchronous scheduling
- Blocked order: `eager-1`, `graph-1`, `cruise-1`, `cruise-2`, `graph-2`,
  `eager-2`, `graph-3`, `cruise-3`, `eager-3`

No model, AIR, generated weight, cache, or raw profiler tree is retained in
Git. All mutable runtime material was confined to marker-owned `/dev/shm`.

## Execution Result

Every individual service result has `pass=true`: warmups and scenarios passed,
server shutdown was clean, and eager, PIECEWISE ACLGraph, and Cruise identities
were proven. Token IDs, finish reasons, stop reasons, and completion boundaries
matched exactly across all nine starts.

The comparison nevertheless has `execution_pass=false` because the frozen
route-coverage check failed. Each Cruise start executed 1,251 of 1,280 expected
decode request-tokens on the Device route, a 97.734% hit rate. The missing 29
tokens were executed by stock Host scheduling when a prefill and an otherwise
eligible decode appeared in the same non-single-token schedule.

Each Cruise start produced identical counters:

| Counter | Value |
|---|---:|
| Device epochs | 484 |
| Feed calls | 484 |
| Fetch calls | 484 |
| Device request-tokens | 1,251 / 1,280 |
| K=1 epochs | 336 |
| K=2 epochs | 148 |
| Paged-KV imports | 139 |
| Host schedule calls | 277 |

The post-run independent verifier reconstructed all reported metrics, hashes,
ordering, semantics, and the same route-coverage failure. It therefore records
`execution_pass=false` and `qualification_pass=false`; it does not convert a
negative result into a pass.

## Frozen Primary Result

The primary scenario is `decode-stream-c4`. Values aggregate three independent
starts per route.

| Route | TPOT p50 (ms) | TPOT p95 (ms) | TPOT p99 (ms) | TTFT p50 (ms) | Host CPU/token (ms) | Output token/s |
|---|---:|---:|---:|---:|---:|---:|
| Eager | 57.354 | 59.054 | 59.386 | 113.095 | 21.057 | 60.72 |
| ACLGraph | **28.801** | **35.109** | **38.218** | **86.348** | **13.140** | **106.01** |
| Cruise | 193.757 | 200.718 | 203.146 | 88.995 | 915.878 | 22.36 |

ACLGraph is the strongest applicable baseline. Relative to it, Cruise changed
median TPOT by -572.76%, p95 TPOT by -471.70%, and Host CPU/token by
-6,870.22%. All three frozen gates failed. The current-reference values needed
for a 15%/15%/30% pass would be at most 24.48 ms median TPOT, 29.84 ms p95
TPOT, and 9.20 ms Host CPU/token; a future formal run must recompute these
limits from its own strongest baseline.

## Attribution

1. **Streaming remains K=1.** The OpenAI streaming contract forces one model
   step per epoch, so the primary path still performs one Host schedule, one
   Feed, and one Fetch per output token. It does not yet amortize cross-token
   Host coordination. K=2 was observed only in non-streaming explanatory
   cases and cannot substitute for the streaming gate.
2. **The Paged-KV handoff is a Host round trip.** `capture_kv_snapshot` copies
   each selected stock-vLLM KV block from NPU to CPU, constructs a full static
   B=4 payload, writes it to `/dev/shm`, and then feeds it into DataFlow. The
   current payload is 58,720,256 bytes (56 MiB) even when one row is selected.
   With 139 imports, each full Cruise start constructs 7.60 GiB of transient
   payload. Files are unlinked and scratch is deleted, so this is a latency and
   Host-CPU problem rather than persistent-root growth.
3. **The latency shape matches the handoff.** In the representative primary
   run, Cruise inter-token p50 was 29.247 ms while p95 was 1,002.058 ms. The
   approximately one-second outlier recurred at request admission, while later
   resident tokens were near ACLGraph's steady latency. Host process-tree CPU
   reached 205.01 seconds during 9.94 seconds of primary wall time.
4. **Mixed prefill/decode loses Device coverage.** The deterministic 1,251 of
   1,280 result repeated in all three starts. Scheduler isolation currently
   begins only after Device ownership, so an initial eligible decode can still
   be advanced by a mixed Host schedule.

## Profiler Boundary

A separate focused graph/Cruise run completed successfully at source
`932f2ff`. Its primary metrics reproduced the formal direction: graph TPOT p50
was 29.526 ms and Cruise TPOT p50 was 192.005 ms. Cruise again proved one
Feed/Fetch per 68 Device epochs.

Dynamic `msprof --pid` returned 255 for both EngineCore and the Cruise sidecar
because neither target exposed the required CANN dynamic-profiling socket. No
`task_time_*.csv` was exported. Consequently AI Core task timestamps and idle
gaps are recorded as `not_observed_by_current_msprof_path`; Host logical timing
is not presented as an accelerator idle-gap measurement.

## M4b Entry Conditions

M4b is corrective performance engineering, not a threshold waiver:

1. replace the full NPU-to-Host-to-DataFlow KV snapshot with unified Device KV
   ownership or a proven direct Device transfer; a compact sparse Host payload
   is acceptable only as an intermediate attribution experiment;
2. isolate mixed prefill/decode scheduling before the first eligible decode and
   reach 1,280/1,280 Device request-tokens on the frozen workload;
3. support a bounded K>1 Device epoch with incremental streaming token egress,
   so streaming does not require one Feed/Fetch per token;
4. rerun the unchanged three-start gate, including exact semantics, Host CPU,
   throughput, TTFT, route coverage, cleanup, and independent verification;
5. keep formal M4 open until AI Core gap evidence is observable and every
   declared threshold passes.

## Rejected-Run Chronology

Four earlier run IDs remain rejected and are not used for performance claims:

- r1 inherited the model's `repetition_penalty=1.05`; all 625 eligible decode
  decisions were rejected as unsupported sampling penalties.
- r2 completed API semantics, but shutdown raced EngineCore counter flush and
  left no trustworthy resident-route counters.
- r3 fixed greedy fields but inherited an additional model EOS token; all 625
  eligible decisions were rejected as extra stop conditions.
- r4 executed the real sidecar path and passed request semantics, but EngineCore
  did not run its `atexit` counter flush. The bounded append-only event journal
  introduced for r5 removed this dependency.

Only r5 contains three complete, counter-proven starts per route. These
rejections are retained to prevent accidental reuse of semantically different
or incompletely identified measurements.

## Compact Evidence

Formal raw JSON is retained under
[`m4a-performance-npu01-20260731-r5/`](m4a-performance-npu01-20260731-r5/).
Focused attribution JSON is retained under
[`m4a-attribution-npu01-20260731-r4/`](m4a-attribution-npu01-20260731-r4/).

| File | SHA-256 |
|---|---|
| `comparison.json` | `fe95b92ffd16d218caab86018a075cad0ec92432bd7e88988a8a8d62e1c5acc0` |
| `verifier.json` | `596ff1259943c42a1c4e92d5c15c7079605670f918083b5dfabdaf26010ba8f9` |
| `profile-summary.json` | `fe6cb4b5617d5bcd4b81bd23c522e1bd9617415fb1c08dc9c0e0211af1560449` |
| `runtime-weights-manifest.json` | `2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761` |
