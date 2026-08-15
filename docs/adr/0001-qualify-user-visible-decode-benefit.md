# Qualify user-visible Decode benefit

Cruise performance qualifies only when the same-round Decode-Heavy Target beats
the strongest applicable Graph baseline by at least 15% in TPOT p50 and p95,
15% in output throughput, and 50% in whole-process-tree Host CPU per output
token, while TTFT p50 and p95 regress by no more than 5%. The primary target is
streaming with a 128-token prompt, 256-token output budget, concurrency four,
and 32 requests per independent start; short and mixed workloads remain
regression guards. These thresholds keep Decode Control Offload subordinate to
user-visible performance instead of treating reduced Host work as success by
itself.
