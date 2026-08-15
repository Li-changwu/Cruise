---
status: accepted
---

# Prove persistent control before porting the Decoder

Cruise first builds a weight-free Persistent Control Spike before integrating
the real Decoder. One model-load-scoped Owner must autonomously chain Device
Scheduling Quanta, accept admission and cancellation while running, expose
output before shutdown, become quiescent and wake again, and show no per-token
or per-quantum Host control Run, Feed, or fetch-to-continue operation. An
Asynchronous Host Output Drain may consume each Committed Prefix update, but
the Device cannot wait for its acknowledgement or use it to start more work. If
the CANN control and transport substrate cannot express those invariants,
Decoder migration stops and the substrate or architecture is reconsidered
instead of hiding Host control inside the full model path.

The spike passes only with one Owner Instance per model load, at least 1024
autonomously chained synthetic quanta, three run/quiesce/readmit cycles, live
admission and cancellation ingress, pre-shutdown output visibility, and no
Host control-call count that grows with quanta or output tokens. Three clean
load/unload runs must show no HBM, queue, or process leak, and the evidence must
separate Host, AICPU, and AICore traces and counters. This is structural
feasibility evidence, not LLM performance qualification.
