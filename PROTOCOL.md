# Attempt 74: Minimal Host-UDF ABI and Cross-Epoch Control Cost

Date frozen: 2026-07-28

## Gate objective

Remove the logically dead Paged-KV and diagnostic output payloads from every
resident epoch, preserve the accepted Attempt 73 multi-epoch semantics, and
measure the remaining Host control path. Logical tensor payload is reported
separately from profiler-observed Host-Device memcpy traffic.

## Single variable

The decoder AIR, external weights, tiling, Qwen2.5-7B model, NPU 7, CANN
9.0.0, B=4 graph, K=2 epoch, greedy sampling, warmup, vLLM classes, request
inputs, and repetition budget remain fixed. Only the FlowGraph Host-UDF ABI
changes.

Old ABI has 10 inputs and 10 outputs. New ABI has 8 inputs and 2 outputs:

```text
new inputs:
  token, position, sequence_length, slot_mapping,
  active_mask, block_table, tiling, control

new outputs:
  token_history, control_output
```

The decoder `RunFlowModel` closure remains 9-input/4-output. The Device UDF
allocates and zeros resident key/value tensors on its first `Proc`, retains
them across later epochs, and clears only the two physical blocks owned by a
row when its generation changes.

## Logical byte ledger

| ABI | Input bytes | Output bytes | Total bytes |
|---|---:|---:|---:|
| old 10/10 | 58,720,516 | 78,184,928 | 136,905,444 |
| new 8/2 | 260 | 368 | 628 |

The logical reduction is 136,904,816 bytes per epoch. Removing key/value in
both directions accounts for 117,440,512 bytes. These values describe tensor
payloads declared at the FlowGraph boundary; they are not PCIe, HCCS, DMA, or
physical memcpy measurements.

## Correctness experiment

The new ABI must repeat the Attempt 73 trace with one EngineCore, one sidecar,
and one persistent Device UDF:

```text
[A] -> [A,B] -> [A,C]
```

A remains row 0/generation 1 for six tokens. B uses row 1/generation 2 and C
reuses row 1/generation 3. Every epoch uses K=2, one socket request/response,
one DataFlow Feed/Fetch, and exactly two `RunFlowModel` calls. All tokens,
request accounting, generation acknowledgements, cleanup behavior, and the
sub-10-second first service epoch rule remain unchanged.

## Controlled overhead experiment

Run four independent EngineCore processes in blocked ABBA order:

```text
old-1 -> new-1 -> new-2 -> old-2
```

Each process performs the same full warmup followed by 15 measured B=4/K=2
service epochs. Every measured epoch admits four one-token requests, generates
two tokens per request, drains the scheduler with a control-only cleanup step,
and then admits the next generation. This yields 30 old and 30 new service
samples.

For each service epoch record:

- four `EngineCore.add_request` calls;
- one `EngineCore.step` and one `post_step`;
- one Unix-socket send and one response receive;
- one `FeedDataFlowGraph` and one `FetchDataFlowGraph`;
- two device `RunFlowModel` calls;
- Python/EngineCore process CPU and wall time around step plus post-step;
- native sidecar `CLOCK_PROCESS_CPUTIME_ID` and Feed-to-Fetch wall time;
- declared input/output ABI bytes;
- absolute wall-clock timestamps for profiler correlation.

Initialization, warmup, admission, and cleanup are reported separately from
the measured service epoch.

## Profiler rule

Run an old and a new EngineCore benchmark through application `msprof` with
ACL, GE, runtime API, and task-time export enabled. Search every exported CSV
for rows that expose both a directional H2D/D2H memcpy operation and one
unambiguous byte field.

- If both routes expose such rows, report their observed directional bytes and
  the exact report/column provenance.
- If either route does not, report `not_observed`, preserve the report headers
  and candidate rows, and explain the missing field or direction.
- Never substitute the logical ABI byte ledger for observed physical traffic.

Application `msprof` covers initialization, warmup, and the measured epochs,
so its byte totals, when available, are process-wide corroborating evidence;
they are not divided by the epoch count.

## Per-epoch runtime transfer rule

Every sidecar is launched with a small `LD_PRELOAD` interposer for the
`rtMemcpy` and `rtMemcpyAsync` symbols used by `libge_executor`. The
interposer records the API, `CLOCK_REALTIME` start/end timestamps, copy kind,
requested byte count, and return status. These rows are joined to the absolute
`time.time_ns` window of each measured `EngineCore.step` plus `post_step`.

An epoch is observed only when it contains at least one successful H2D and one
successful D2H runtime memcpy call, with no copy crossing the epoch boundary. The four
ABBA blocks must provide 15 such epochs each. Filtered per-epoch rows and their
SHA256 values are retained in evidence. This metric is an observed CANN
runtime-copy request count; it is not the declared ABI ledger and is not
described as PCIe/HCCS wire traffic.

## Pass rules

1. Structural verification proves new 8/2, old 10/10, unchanged decoder ABI,
   device allocation of K/V, and absence of Host K/V/logits/final-state edges.
2. The new-ABI multi-epoch correctness trace passes independently.
3. All four ABBA blocks pass with exactly 15 samples and identical semantic,
   model, hardware, and runtime controls.
4. Every sample reports the frozen Host API counts, positive native CPU time,
   Python CPU time, wall time, and the correct logical byte ledger.
5. The analyzer reports 30 old and 30 new observations for every metric. No
   speedup threshold is required; a slowdown is retained as a valid result.
6. Profiler transfer status is either directly `observed` with byte/direction
   provenance or explicitly `not_observed` with preserved inspection evidence.
7. Runtime memcpy bytes are observed in both directions for every one of
   the 60 measured epochs and independently verified from filtered rows.
8. Independent verifiers, unit tests, SHA256 manifests, NPU idle checks,
   storage finalization, and scratch cleanup all pass.

## Storage and support boundary

Formal execution is allowed only through:

```bash
/root/ascend-control-g4-20260723/storage-control/run_guarded_attempt.sh \
  /root/ascend-control-g4-20260723/attempt74-src
```

Builds, relocated AIR, runtime and external weights, cache, CANN logs, profiler
raw data, sockets, and temporary files live below marker-protected
`/dev/shm/a74r1`. Root retains only source and compact evidence.

Passing applies to Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling,
one-token prompts, static B=4, K=2, greedy sampling, and one 910B2/CANN 9.0.0
server. It does not establish general sampling, prefill, preemption,
continuous batching, API-server performance, or physical transfer bytes that
the installed profiler does not expose.
