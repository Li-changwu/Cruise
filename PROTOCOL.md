# Attempt 74: Minimal Host-UDF ABI and Cross-Epoch Control Cost

Date frozen: 2026-07-28

## Gate objective

Remove the logically dead Paged-KV and diagnostic output payloads from every
resident epoch, preserve the accepted Attempt 73 multi-epoch semantics, and
measure the remaining Host control path. Logical tensor payload is reported
separately from profiler-observed Host-Device memcpy traffic.

## Single variable

The decoder AIR, external weights, tiling, Qwen2.5-7B model, physical NPU,
CANN release, B=4 graph, K=2 epoch, greedy sampling, warmup, vLLM classes,
request inputs, and repetition budget remain fixed within a run. Only the
FlowGraph Host-UDF ABI changes. The current deployment-validation target is
physical NPU 7 on Ascend 910B2 with CANN 8.5.1.

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

The frozen CANN 9.0.0 run uses `--storage-limit=2048`. Releases such as the
current CANN 8.5.1 deployment that require an explicit unit suffix set
`CRUISE_MSPROF_STORAGE_LIMIT=2048MB`; this changes only profiler retention
syntax, not the measured workload.

Custom OPP archives are relocatable. `CRUISE_CUSTOM_SET_ENV`,
`CRUISE_BARRIER_SET_ENV`, and `CRUISE_MATERIALIZE_SET_ENV` identify the three
installed `vendor/bin/set_env.bash` files, but the driver derives each current
vendor root from the script location instead of sourcing its install-time
absolute paths. The corresponding `CRUISE_*_OPP_VENDOR` variables can override
that derivation explicitly. All three vendor trees must contain `op_impl`,
`op_proto`, and `op_api/lib` before any model is loaded.

If an installed CANN application profiler cannot initialize GE in the
resident sidecar, set `CRUISE_MSPROF_MODE=off` together with a non-empty
`CRUISE_MSPROF_UNAVAILABLE_REASON`. The driver then runs the same four ABBA
blocks without application `msprof`, preserves the explicit reason, and
reports the process-wide profiler metric as `not_observed`. This does not
relax the per-epoch DataFlow payload gate below.

## Per-epoch transfer-trace rule

Every sidecar is launched with an `LD_PRELOAD` interposer that records three
separate classes of events:

- `dataflow_tensor`: `Tensor::GetSize()` for every actual tensor passed to
  `DFlowSessionImpl::FeedDataFlowGraph` and returned by
  `DFlowSessionImpl::FetchDataFlowGraph`;
- `runtime_memcpy`: the CANN `rtMemcpy`/`rtsMemcpy` API families, including
  synchronous, asynchronous, batch, descriptor, offset, and 2-D variants;
- `mbuf_diagnostic`: selected Mbuf/Buff allocation and size APIs used only to
  establish whether steady-state buffers are allocated again.

Every record contains `CLOCK_REALTIME` start/end timestamps, requested tensor
or copy bytes, kind when available, and return status. The records are joined
to the absolute `time.time_ns` window of each measured `EngineCore.step` plus
`post_step`.

An epoch is observed only when it contains exactly one successful DataFlow
Feed and one successful Fetch whose ordered tensor sizes match the route ABI.
Thus every old epoch must expose 10 input and 10 output tensors totaling
58,720,516 B and 78,184,928 B; every new epoch must expose 8 input and 2 output
tensors totaling 260 B and 368 B. All four ABBA blocks must provide 15 such
epochs, with no event crossing an epoch boundary.

Runtime memcpy is reported independently as `observed` or `observed_zero`.
Zero is a valid measurement: on CANN 8.5.1, startup uses `rtMemcpy`, while the
steady-state DataFlow epochs reuse Mbufs and do not pass through any interposed
runtime memcpy API. Filtered per-epoch rows, raw traces, and SHA256 values are
retained, and the independent verifier reconstructs all 60 windows from the
raw trace.

The four formal ABBA raw traces are copied from scratch into compact evidence
only after a 16 MiB per-file bound is enforced. Their hashes, the filtered
rows, the Git source origin and commit, and every nested evidence file are
covered by the final recursive integrity manifest. This keeps the result
independently verifiable after scratch cleanup.

The DataFlow tensor payload is an observed runtime API-boundary payload, not a
declared constant substituted by the analyzer. It is also not claimed to be
PCIe, HCCS, DMA, or other physical-link traffic. A physical-byte claim remains
`not_observed` unless a profiler exposes both direction and byte fields.

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
7. All 60 epochs expose complete DataFlow Feed/Fetch tensor payloads matching
   their ABI. Runtime memcpy activity is independently reported as observed or
   observed-zero; neither result is replaced with declared ABI bytes.
8. Independent verifiers, unit tests, SHA256 manifests, NPU idle checks,
   storage finalization, and scratch cleanup all pass.

## Storage and support boundary

Formal execution uses a clean GitHub checkout and `run_attempt74.sh`, with all
machine-specific locations supplied through the documented `CRUISE_*`
environment variables. Builds, relocated AIR, runtime and external weights,
cache, CANN logs, profiler raw data, sockets, transfer traces, and temporary
files live below a marker-protected `/dev/shm` scratch directory. Root retains
only the source checkout, archived OPP inputs, and compact evidence.
The model-running phase also uses a scratch working directory because some
CANN releases emit `fusion_result.json` relative to the process cwd. When the
source is a Git checkout, clean-worktree gates before and after execution
reject any runtime artifact that escapes this boundary.

Passing applies to Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling,
one-token prompts, static B=4, K=2, greedy sampling, and the recorded
910B2/CANN server configuration. It does not establish general sampling, prefill, preemption,
continuous batching, API-server performance, or physical transfer bytes that
the installed profiler does not expose.
