# Persistent cancellation P2 protocol

P2 preserves the P1 Persistent Device Model Owner and deterministic AICore
recurrence while adding generation-aware admission, cancellation, retirement,
and exact-event deduplication. The Host sends only external control events and
drains outputs asynchronously. It never chooses the row used for admission or
starts a Device Scheduling Quantum.

## Device recurrence

The Owner holds four row states. An admission declares request identity,
generation, output budget, Output Credit, and seed; the Device assigns the
first compatible free row and returns that row in the Admit Ack. Each Device
Scheduling Quantum invokes one four-element AICore Add and advances every
active row with Output Credit by one committed token. The synthetic recurrence
is paced at one millisecond per emitted commit so the test outfeed remains
drainable; this pace is not a Host scheduling command or a performance claim.

Before every quantum the Device drains all currently available Host Control
Events. When no row is eligible it blocks on the event queue. A cancellation
therefore becomes visible at a scheduling boundary, never rolls back the
Committed Prefix, emits exactly one Retire Cancelled acknowledgement, and
prevents every later commit for that request generation.

## Event tensor

Every event is one `DT_INT64[12]` tensor:

```text
[owner, type, event_seq, request, generation, row,
 target_steps, credit, cancel_seq, seed, flags, reserved]
```

Types are `1=ADMIT`, `2=CREDIT`, `3=CANCEL`, and `4=SHUTDOWN`. Admission uses
`row=-1`; the Device chooses the row. Credit and cancellation identify the row
returned by the Admit Ack but do not select Device work. An exact retry of the
latest event identity returns a Cumulative Control Acknowledgement without
reapplying the effect. A lower sequence, same sequence with different content,
wrong Owner Instance, stale generation, invalid row, or invalid capacity is
rejected without mutating request state.

`event_seq` is the declared Control Event Identity sequence, not the DataFlow
transport transaction id. The Host assigns every Feed a separate strictly
increasing transport sequence even when it deliberately retries or reorders a
protocol event. This lets the Owner observe and reject stale protocol input
without the transport rejecting the test before it reaches the Device.

## Output tensor

Every output is one `DT_INT64[16]` tensor:

```text
[owner, type, event_seq, request, generation, row,
 commit_seq, state, remaining, credit, cancel_seq, status,
 aicore_calls, total_commits, total_retired, quiescent_count]
```

Types are `1=ADMIT_ACK`, `2=COMMIT`, `3=RETIRE_COMPLETE`,
`4=RETIRE_CANCELLED`, `5=CREDIT_ACK`, `6=QUIESCENT`,
`7=CUMULATIVE_ACK`, `8=REJECTED`, and `9=SHUTDOWN`. Status codes distinguish
wrong owner, stale or conflicting event sequence, no capacity, stale
generation, invalid row or budget, stale cancel, inactive generation, unknown
event, malformed message, and exact duplicate.

## Required evidence

Each independent full start executes a deterministic matrix proving:

- cancellation with zero initial credit before the first commit;
- cancellation during AICore recurrence and rejection after completion;
- exact duplicate admission, credit, and cancellation without a repeated
  effect;
- reordered event, wrong-owner event, and stale-generation cancellation
  rejection;
- four-row capacity rejection followed by reuse of the cancelled row only
  after its Retire Acknowledgement;
- no commit after either retirement type and no state change in a reused
  generation caused by a stale cancellation.

The same Owner then executes 250 cohorts of four active requests. Every row
must commit at least once before the Host sends its external cancellation, so
the run contains exactly 1,000 in-flight cancellations. Host monotonic time
from successful Cancel Feed initiation to observed Retire Cancelled output is
recorded for every stress request. The gate requires exactly 1,000 samples,
p95 at most 50 ms, p99 at most 100 ms, contiguous recurrence for every
Committed Prefix, exactly one retirement per admitted generation, and zero
post-retirement commits.

Three mechanism starts and one profiled start must select the Ascend Device
controller, use unique Owner Instances, return HBM to within two percentage
points of baseline, release every visible process, and pass evidence hashes.
The profiled count and identity of the target Add tasks must exactly match the
profiled Owner's reported AICore call count. P2 is correctness and cancellation
latency evidence, not Decoder or service performance qualification.
