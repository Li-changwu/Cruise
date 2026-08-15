# Unify Prefill and Decode under one Device owner

The final concurrent architecture uses one Persistent Device Model Owner for
shared model weights, Paged-KV state, and separate Prefill and Decode execution
lanes. Across three independent starts it must demonstrate overlapping AICore
task intervals, at least 10% shorter combined makespan than isolated Prefill
plus Decode, no more than 5% regression in Decode TPOT p95 or Prefill TTFT
against their isolated references, and peak HBM at or below 90% of physical
capacity. Running two full model contexts in stock vLLM and a sidecar is an
experiment, not the target architecture, because co-residence does not prove
concurrency and leaves insufficient HBM and isolation margin.
