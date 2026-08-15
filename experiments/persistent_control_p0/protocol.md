# P0 Persistent Control Spike protocol

## Claim

One AICPU/DataFlow Owner Instance can remain alive across active and quiescent
periods, autonomously advance synthetic Device Scheduling Quanta, accept Host
Control Events while running, and publish incremental output without a Host
control operation per quantum.

This experiment contains no Decoder or AICore compute. An observed AICore task
count other than zero is therefore unexpected; deterministic AICore recurrence
belongs to P1.

Placement is fail-closed. The target process point must select the `Ascend`
resource, load as `device_type=0` through the Device UDF proxy, and complete a
one-quantum admission/shutdown probe. `node_type=local` only identifies the
local topology node and is not sufficient to classify execution. An Aarch
selection, `device_type=1`, local-UDF untar, or Host UDF executor is a Host
placement failure.

## Event ABI

Every Host event is an `INT64[8]` tensor:

```text
[owner_instance_id, event_type, event_seq, generation,
 target_quanta, cancel_seq, seed, reserved]
```

`event_type` is `1=ADMIT`, `2=CANCEL`, or `3=SHUTDOWN`. The Host may submit an
event at any time. It never sends a continue, step, row-selection, or quantum
command.

## Output ABI

Every Device output is an `INT64[8]` tensor:

```text
[owner_instance_id, output_type, event_seq, generation,
 commit_seq, state, remaining_quanta, status]
```

`output_type` is `1=ADMIT_ACK`, `2=COMMIT`, `3=RETIRE_COMPLETE`,
`4=RETIRE_CANCELLED`, `5=QUIESCENT`, `6=SHUTDOWN`, or `7=REJECTED`.
`COMMIT` is emitted once per synthetic quantum while the same FlowFunc Proc is
still alive. `SHUTDOWN` carries the DataFlow EOS flag.

## State machine

The Owner blocks on its input `FlowMsgQueue` while quiescent. Admission starts a
generation. While active, the Owner polls for control events before each
synthetic quantum, advances the state, and publishes a commit. Completion or
cancellation emits retirement and quiescence without returning from Proc. Only
an explicit shutdown event ends Proc and the Owner Instance.

## Hard gate

Each independent start runs four generations: three complete at least 1024
quanta and one is cancelled after at least 128 observed commits. This creates at
least three quiescence/readmission transitions. Three process starts must load
and unload cleanly, preserve exact output sequences, and leave no material HBM,
queue, or process growth. Host event-feed count is fixed by external events and
does not scale with the number of commits; per-output fetches are classified as
Asynchronous Host Output Drain rather than control.

The full profile is a fourth independent start and must pass the same exact
sequence checks. Its Host PID is bound to one Device `udf_executor` PID whose
target FlowFunc initialization, processor initialization, single positive-time
execution, exact output count, and clean exit appear in one ordered Device log.
The profiler command must successfully enable task time, AICPU, and AICore
collection, record its enable/disable lifecycle, and report zero AICore tasks.
CANN 9 does not necessarily export a long-running DataFlow FlowFunc as an
ordinary `AI_CPU` task, so a generic `AI_CPU` row is neither required nor
sufficient; attributed Device UDF lifecycle and metrics are the controller
activity gate. Raw driver logs are retained on failure, but a passing run keeps
only the bounded attribution extract.
