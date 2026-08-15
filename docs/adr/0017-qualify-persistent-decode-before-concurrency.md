---
status: accepted
---

# Qualify Persistent Decode before concurrency

P5 reaches Persistent Decode Qualified only when a prefilled stable cohort beats
the same-round Graph route across three independent starts by at least 50% in
whole-process-tree Host CPU per output token, 15% in TPOT p50 and p95, and 15%
in output throughput, with exact semantics, incremental output, and lifecycle
checks passing. P4 reports serial-serving TTFT but may defer the final 5% TTFT
limit to P6. One profiler-attributed redesign and full P5 rerun is allowed after
an initial miss; a second Host, TPOT, or throughput failure stops P6 work so
concurrency cannot hide a Decode path that lacks independent value.
