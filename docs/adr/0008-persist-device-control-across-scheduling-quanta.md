---
status: accepted
---

# Persist Device control across scheduling quanta

Cruise uses a Persistent Device Model Owner that autonomously chains internal
Device Scheduling Quanta for steady-state Decode. The Host sends only
asynchronous admission, cancellation, Device KV Lease, and service-policy
events, and receives Committed Prefix, retirement, and fault events; it never
starts the next token step or scheduling quantum, selects a row or execution
lane, or supplies a quantum length. The owner is created with model load,
survives empty-request periods, and is destroyed only at model unload or a
declared fault boundary. This accepts a more demanding persistent-control and
event-transport design in exchange for actually moving Decode control authority
to the NPU. With no active requests it becomes a Quiescent Owner: weights,
required KV pools, and event channels remain resident while AICore execution
blocks instead of spinning or relying on Host polling for liveness.
