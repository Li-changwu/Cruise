---
status: accepted
---

# Protect Decode while Prefill uses slack

P6 uses a Latency-Protected Decode Lane and runs bounded Prefill chunks only
with remaining Device opportunity. The Host supplies declarative deadline,
priority, and resource constraints, while the Persistent Device Model Owner
chooses lanes and timing; no Host command switches execution back to Decode.
Exact AICore allocation is selected by a measured sweep rather than hard-coded
before profiling, but Prefill may never monopolize the Device long enough to
violate the isolated Decode TPOT p95 or Prefill TTFT 5% guards.
