# Cruise Serving

Cruise is a narrow vLLM-Ascend serving capability whose value is to move
Decode control responsibility to the NPU and convert reduced Host/Device
coordination into better token-generation performance without semantic or
stability regressions.

## Language

**Decode Control Offload**:
The transfer of steady-state Decode iteration control from the Host CPU to a
Persistent Device Model Owner. The Host does not initiate token steps or
Device Scheduling Quanta.
_Avoid_: K=6 support, bounded delegation, batching optimization

**Host-Visible Decode Epoch**:
A bounded delegation in which the Host starts up to K Device Decode steps and
regains scheduling control when they finish. It is a prototype mechanism, not
the target Decode Control Offload architecture.
_Avoid_: Device-resident control, persistent Decode

**Device Scheduling Quantum**:
A bounded, Device-internal execution interval used by the Persistent Device
Model Owner to balance progress, event handling, and resource policy. It is
chained by the NPU and is not a Host-visible Decode command.
_Avoid_: Epoch, Host iteration, Feed/Fetch cycle

**Host Control Event**:
An asynchronous admission, cancellation, lease, or service-policy change sent
to the Persistent Device Model Owner without initiating a Decode iteration.
It may declare constraints but cannot select a row, lane, or quantum length.
_Avoid_: Decode command, epoch plan, step request, run-row command

**Control Event Identity**:
The Owner Instance, request or row generation, and declared monotonic event
sequence that make an at-least-once Host Control Event safe to retry and
deduplicate. It is independent of transport order so retries and stale events
can be observed without reusing a queue identity.
_Avoid_: Queue position, timestamp, exactly-once delivery

**Cumulative Control Acknowledgement**:
The Device response that confirms a Control Event Identity has already been
applied, allowing an exact retry to complete without applying its effect again.
_Avoid_: Per-token acknowledgement, exactly-once transport, Decode continuation

**Device Control Plane**:
The AICPU/DataFlow portion of the Persistent Device Model Owner that receives
Host Control Events and chooses Device work without Host scheduling.
_Avoid_: Host sidecar scheduler, pure AICore requirement

**Device Compute Plane**:
The AICore execution owned by the Device Control Plane for Prefill, Decode,
sampling, and state advancement.
_Avoid_: Device scheduler, Host execution

**Controller-Aware Graph Family**:
A model-load-scoped set of phase-specific whole-model graphs selected by the
Device Control Plane. Each graph preserves the strongest supported compute
structure for its phase while the Persistent Device Model Owner retains request
and scheduling authority.
_Avoid_: Universal graph, Host graph dispatcher, per-operator execution

**Shared Device State Contract**:
The identity, layout, ownership, and aliasing rules that allow graphs in one
Controller-Aware Graph Family to advance the same Device KV Lease without a
Host copy or an implicit full-cache reconstruction.
_Avoid_: KV tensor handoff, Host cache, matching tensor shapes

**Epoch Reference Scaffold**:
The frozen fixed-K Host-Visible Decode Epoch path retained only as a correctness
oracle, measurement reference, and bring-up aid for the Persistent architecture.
_Avoid_: Fallback product path, M4b candidate architecture

**Owner Instance**:
One uniquely identified lifetime of a Persistent Device Model Owner. State and
acknowledgements from different Owner Instances must never be combined.
_Avoid_: Process ID, model name, epoch generation

**Quiescent Owner**:
A Persistent Device Model Owner with no active requests that retains its model
resources and event channels while blocking for the next Host Control Event.
_Avoid_: Busy polling, unloaded model, stopped owner

**Persistent Control Spike**:
The weight-free P0 proof that the target CANN substrate can sustain one Owner
Instance, autonomous quanta, live event ingress, incremental output, and
quiescence without steady-state Host scheduling.
_Avoid_: Decoder prototype, performance candidate, epoch benchmark

**Controller Activity Attribution**:
Evidence that the target Persistent Control Spike, rather than unrelated Device
work, remained active and produced the observed commits during a measured run.
_Avoid_: Any AI_CPU task, successful profiler exit, route inference

**Mechanism Implemented**:
A capability exists in a candidate design and passes implementation-level
checks; it makes no claim about target-hardware behavior or performance.
_Avoid_: Complete, done, validated

**Candidate Hardware Validated**:
The exact candidate identity has passed its declared correctness and lifecycle
checks on supported target hardware.
_Avoid_: Tested, works on NPU

**Host Saving**:
A reduction in whole-service-process-tree Host CPU per output token that is
attributable to Decode control work rather than work shifted between processes.
_Avoid_: Scheduler saving, fewer calls

**End-to-End Benefit**:
A user-visible improvement in token-generation latency or throughput with
bounded TTFT and no semantic or stability regression.
_Avoid_: Host saving, route coverage, control-plane amortization

**Committed Prefix**:
The monotonically growing token and Device-state prefix that is externally
visible and can never be replayed or withdrawn.
_Avoid_: Partial output, epoch result

**Asynchronous Host Output Drain**:
The one-way Host consumption of Committed Prefix updates for client streaming.
It may apply bounded backpressure but never acknowledges or starts Device work.
_Avoid_: Fetch-to-continue, Decode response, scheduling acknowledgement

**Output Credit**:
A bounded per-row output slot that must be available before the Device advances
and publishes the row's next Committed Prefix update.
_Avoid_: Scheduling token, unbounded output buffer, per-token control ACK

**Output-Blocked Row**:
An admitted row that temporarily lacks Output Credit and is therefore
ineligible for Device execution while other eligible rows may continue.
_Avoid_: Cancelled request, stalled owner, Host-paused row

**Device KV Lease**:
A bounded set of Paged-KV capacity admitted by the Host and owned by the Device
for position and KV progression until completion or explicit retirement. The
first Persistent path grants the full declared budget before admission.
_Avoid_: Resident row, fixed KV slot, KV copy

**Retire Acknowledgement**:
The Device confirmation that a row generation has a final Committed Prefix and
can no longer produce tokens, permitting its KV lease and identity to be reused.
_Avoid_: Cancel response, Host cleanup

**Quantum-Boundary Serial Admission**:
The staged admission model in which the Device owner serializes Prefill,
retirement, and cohort changes with Decode at an internal scheduling boundary.
It makes no Prefill/Decode concurrency claim.
_Avoid_: Epoch-Boundary Admission, continuous batching, concurrent prefill

**Concurrent Prefill/Decode**:
Measured overlap of Prefill and resident Decode execution on the Device that
reduces combined completion time without violating either workload's service
or resource gates.
_Avoid_: Co-resident processes, overlapping Host calls

**Latency-Protected Decode Lane**:
The latency-critical execution lane whose Device opportunity is preserved while
bounded Prefill work uses remaining resources under the Owner's policy.
_Avoid_: Equal-share scheduling, Host preemption, Decode-only execution

**Persistent Device Model Owner**:
The single long-lived Device-side authority that owns model weights, Paged-KV
state, and the execution lanes used for both Prefill and Decode from model load
until model unload, including periods in the Quiescent Owner state.
_Avoid_: Device Model Owner, sidecar model, duplicate runtime, colocated models

**Performance Qualified**:
The exact candidate has met all accepted performance, semantic, and stability
gates in a repeated same-round comparison with the strongest applicable
baseline.
_Avoid_: Faster, promising, performance passed

**Persistent Decode Qualified**:
The P5 state in which a stable resident Decode cohort meets the accepted Host,
TPOT, throughput, semantic, and lifecycle gates before concurrent Prefill.
_Avoid_: Performance Qualified, Stable v1.0, serial serving complete

**Stability Soak**:
The separate 10,000-request P6 run that validates long-lived Owner, request,
output, and KV behavior rather than supplying latency qualification samples.
_Avoid_: Three-start benchmark, smoke test, success-rate estimate from 96 requests

**Strongest Applicable Baseline**:
The supported stock serving route with the best measured performance for the
declared workload under the same environment and run protocol.
_Avoid_: Stock, baseline

**Decode-Heavy Target**:
The closed-loop concurrency-four primary workload with 32 requests, exactly 128
prompt tokens, and 256 output tokens generated with EOS ignored so steady-state
Decode dominates one-time request setup and tail effects.
_Avoid_: Short decode, seven-token decode

**Stable v1.0**:
A released Cruise version for which M0-M5 and every final acceptance rule are
complete with reproducible evidence.
_Avoid_: Developer Preview, feature complete, production-ready
