---
status: accepted
---

# Separate stability from performance sampling

The three-start, 32-request Decode-Heavy Target qualifies latency, throughput,
and Host cost, while a separate P6 Stability Soak runs at least 10,000 requests
with zero Cruise-caused wrong tokens, errors, hangs, Owner restarts, or KV
corruption. Client cancellation and disconnect are classified separately and
must satisfy their own retirement contract. If P6 fails concurrency, QoS, HBM,
or stability, P5 may remain a Persistent Decode Qualified research candidate,
but Cruise is neither Performance Qualified nor Stable v1.0 and serial admission
cannot be relabeled as final concurrency.
