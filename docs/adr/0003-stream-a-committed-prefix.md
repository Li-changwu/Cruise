# Stream a committed prefix

Each token becomes externally visible only after the NPU has advanced and
committed the corresponding Decode and KV state, producing a monotonically
growing Committed Prefix while the Persistent Device Model Owner continues to
run. The Host drains and forwards that Device output stream without scheduling
the next token or Device Scheduling Quantum. This Asynchronous Host Output
Drain may dequeue each token or chunk and apply bounded backpressure, but the
Device does not use a Host acknowledgement to schedule more work. Each row has
bounded Output Credits; the Device schedules a token only when it can atomically
publish the resulting state and commit sequence. A row without credit becomes
output-blocked while other rows remain eligible, and an all-blocked Owner waits
for capacity instead of dropping output, buffering without bound, or busy
polling. Splitting a completed Host-Visible Decode Epoch into near-simultaneous
HTTP chunks is not incremental streaming. An in-graph outfeed prototype must
prove pre-completion visibility and acceptable Host, AICPU, and TPOT cost before
this path can become the product architecture; failure of that gate triggers an
architectural pivot rather than a burst-output waiver.

The Device does not infer client failure from exhausted credit. A disconnected
client or a service-configured output-stall timeout causes the Host to send an
ordinary asynchronous cancellation event; until then only the affected row is
output-blocked. The timeout is service policy, not a command selecting Device
work.
