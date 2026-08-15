---
status: accepted
---

# Split Device control from AICore compute

The Persistent Device Model Owner may use AICPU/DataFlow as its Device Control
Plane while using AICore as its Device Compute Plane. Requiring all scheduling
logic to execute inside a pure AICore graph would make asynchronous admission,
cancellation, output, and policy handling substantially harder without proving
better service performance. This split qualifies only if no Host interaction is
required per token or Device Scheduling Quantum and separate attribution shows
that AICPU cost does not prevent the accepted Host CPU, TPOT, throughput, TTFT,
or stability gates.
