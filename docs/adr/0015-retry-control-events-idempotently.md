---
status: accepted
---

# Retry control events idempotently

Host Control Events use at-least-once delivery rather than assuming that CANN
queues provide exactly-once semantics. Every event carries a Control Event
Identity consisting of the Owner Instance, request or row generation, and a
monotonic event sequence; the Device applies it idempotently and returns an
asynchronous cumulative acknowledgement so the Host can retry without creating
duplicate admission, cancellation, lease, or policy effects. Events for an old
owner, reused generation, or non-advancing sequence are rejected and never
affect Device scheduling.
