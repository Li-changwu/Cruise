# Experiments

`synthetic-p0/` contains the original three-route feasibility microbenchmark
and its protocol. It established the device-resident recurrence mechanism
before full decoder work began.

Later source snapshots live under `history/attempts/`. Raw measurements are
kept outside Git; compact accepted findings are summarized in the protocol and
status documents referenced by the root README.

`m4a_performance/` is the early three-route performance preflight. It compares
stock eager, stock ACLGraph, and Cruise with a versioned API workload, blocked
three-start ordering, exact semantic comparison, process-tree Host CPU
measurement, benchmark-only resident-route counters, and an independent
verifier. Its explicit profiling mode pauses after warmup, attaches only to a
representative scenario, limits transient `msprof` output to marker-owned
`/dev/shm`, and persists only compact summaries. The NPU0-1 preflight is a
recorded negative result; M4a does not close the formal M2, M3, or M4
milestones.
