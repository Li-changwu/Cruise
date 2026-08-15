# P1 Persistent Recurrence protocol

## Claim

One P0-qualified Persistent Device Model Owner can invoke an AICore recurrence
once per Device Scheduling Quantum, publish a Committed Prefix before shutdown,
and isolate an output-blocked row while other eligible rows continue. The Host
sends only admission, Output Credit, and shutdown events; it never sends a
step, continue, row-selection, or quantum command.

P1 uses a weight-free `INT32[4]` Add GraphPp. The Add result is the next
Device-owned recurrent state and the committed synthetic token. This proves the
AICPU/DataFlow control-plane to AICore compute-plane contract without claiming
Decoder semantics or performance.

The synthetic recurrence is Device-paced at one millisecond per emitted commit.
This keeps the weight-free Add from outrunning the asynchronous output transport;
it is not a Host scheduling signal or a performance claim. A real Decoder is
expected to provide a longer natural compute interval.

## Event ABI

Every Host event is an `INT64[16]` tensor:

```text
[owner, type, event_seq, cohort, batch_size,
 credit0, credit1, credit2, credit3,
 seed0, seed1, seed2, seed3, reserved0, reserved1, reserved2]
```

`type` is `1=ADMIT`, `2=CREDIT`, or `3=SHUTDOWN`. Credits are additive and
bounded. Admission is accepted only while quiescent; P2 owns the wider stale,
duplicate, cancellation, and generation matrix.

## Output ABI

Every Device output is an `INT64[12]` tensor:

```text
[owner, type, event_seq, cohort, row, commit_seq,
 state, token, remaining, credit_remaining, status, reserved]
```

`type` is `1=ADMIT_ACK`, `2=COMMIT`, `3=RETIRE_COMPLETE`, `4=QUIESCENT`,
`5=CREDIT_ACK`, `6=SHUTDOWN`, or `7=REJECTED`. A commit consumes one row-local
Output Credit and becomes externally visible only after the invoked AICore Add
returns successfully.

## Exact scenario

Each Owner Instance runs two cohorts. Cohort 1 admits B=1 with 256 credits and
must produce exactly 256 contiguous commits. Cohort 2 admits B=4 with credits
`[16, 256, 256, 256]`. Row 0 must stop exactly at commit 16 while rows 1-3
reach commit 256 and retire. One asynchronous credit event adds 240 credits to
row 0, which then reaches commit 256 and retires.
All 1,280 commits must be visible before shutdown.

The Device invokes the recurrence GraphPp exactly 752 times: 256 for B=1, 256
while at least one B=4 row has initial credit, and 240 after row 0 receives more
credit. Host feed count is exactly four and does not scale with commits or
AICore calls.

## Hard gate

Three independent starts must preserve exact row state, token, remaining,
credit, commit, retirement, quiescence, and pre-shutdown output sequences with
unique Owner Instances. Placement must select `Ascend` and `device_type=0`.
Profiling must bind the target Device UDF lifecycle to the profiled Host run,
show the recurrence GraphPp or its Add kernel on AICore/AIVector, and contain no
unattributed Host-controlled recurrence. Every start must release processes and
return HBM within two percentage points of the preflight value.
