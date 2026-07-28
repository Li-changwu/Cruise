# G4c Attempt 70a: B=4 Device Fallback Recovery

Date frozen: 2026-07-26

Attempt 70a preserves the accepted 69e-r5 decoder, Device UDF, tensor ABI,
inputs, and DataFlow graph. It changes only the Host runner and comparator to
exercise four failures that must be rejected before `RunFlowModel`.

## Prerequisites

- Attempt 69e-r5 correctness result SHA256:
  `9a64b4a5488f0c38f5542825971b3ae2f75ed5084238ee16d43240cb52a3656c`.
- Attempt 69c-r2 B=4 AIR SHA256:
  `263b2acf291e13f6a84042ded53c8dccabb1fa847dcdcbbbe0ece418610ad1e3`.
- The live r5 `/dev/shm` scratch contains the accepted inputs and the single
  379-file external-weight set. Attempt 70a reuses that weight set and does not
  materialize another 15 GB copy.

## Question

Does the B=4 Device UDF reject invalid epoch controls and capacity overflow
without invoking the decoder graph or mutating token, position, sequence
length, slot mapping, active mask, or Paged KV state?

## Frozen Cases

| Case | Control/input change | Expected status |
|---|---|---:|
| `invalid-max-steps` | `max_steps=9` while maximum is 8 | 201 |
| `capacity-exceeded` | active request 0 starts at position 7 with `K=2` | 202 |
| `unsupported-sampling` | `sampling_mode=1` | 205 |
| `unsupported-graph` | `graph_variant=1` | 206 |

All cases use the accepted `k8-all-active` input. The capacity case updates
request 0's position, length, and slot consistently before submission, so the
only invalid condition is insufficient logical capacity.

## Pass Rules

- Each case completes one Feed and one Fetch and returns its exact expected
  status.
- Control output reports `model_calls=0`, fallback flag 1, executed steps 0,
  and finish reason 3 for every request.
- Logits history remains all zero and token history remains all `-1`.
- Final token, position, sequence length, slot mapping, active mask, key cache,
  and value cache are elementwise exact copies of the submitted state.
- NPU 7 is idle immediately before the suite and after GE finalization.
- Runtime model failure, malformed output, non-finite logits, and allocation
  failure are not claimed by this attempt because they require fault injection
  after `RunFlowModel` begins.

## Storage And Log Invariants

- New inputs, build products, recovery outputs, GE cache, and CANN process logs
  live in a fresh `/dev/shm` scratch root.
- The accepted r5 external weights are reused in place as the only live weight
  set; their count must remain 379.
- Persistent evidence stays below 512 MiB and command logs retain at most a
  32 MiB head and 32 MiB tail. The watchdog enforces a 100 GiB root reserve,
  128 GiB `/dev/shm` reserve, 24 GiB persistent G4 cap, and 64 GiB scratch cap.
- More than 64 MiB of growth outside the current evidence directory terminates
  the command. A successful run removes its scratch only after evidence hashes
  and the final NPU-idle check pass.
- Existing evidence and scratch targets are never overwritten.

## Claim Boundary

Passing closes the explicitly requested invalid-control, capacity, sampling,
and graph-variant fallback gate. Stable K>=2 performance and runtime fault
injection remain open; vLLM-Ascend integration must not start yet.
