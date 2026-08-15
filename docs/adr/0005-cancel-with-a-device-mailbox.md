# Cancel with a Device mailbox

Cancellation is a one-way, event-driven mailbox message identified by row,
generation, and a monotonic cancel sequence; it is not a per-token scheduling
round trip. Delivery is at least once, so duplicate and stale sequences are
idempotently ignored. The NPU checks cancellation between token steps,
preserves every token already in the Committed Prefix, commits no later token
after observing the message, and emits a Retire Acknowledgement before the Host
releases the KV lease or reuses the row. The target cancellation latency is at
most 50 ms at p95 and 100 ms at p99. A partial-feed and graph-visibility
prototype must prove this behavior; stream- or device-wide abort is not an
acceptable substitute.
