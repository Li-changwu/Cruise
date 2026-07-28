# G4c Attempt 70b: B=4 Stable Resident-Epoch Performance

Date frozen: 2026-07-26

Attempt 70b preserves the accepted 69e-r5 AIR, Device UDF, inputs, greedy/EOS
semantics, and Paged-KV layout. It adds a performance-only runner that keeps
both the Host graph and Device DataFlow graph resident in one process.

## Question

After route initialization and warmup, does Device-resident iteration reduce
wall time and Host process CPU time for every K>=2 while preserving K Host
submissions versus one Device Feed/Fetch transaction?

## Controlled Design

- Fixed true B=4 inputs for K=2, 4, and 8; K=1 is excluded because the
  correctness run showed cold initialization dominates it.
- One GE lifetime holds the Host graph session and Device DataFlow session, so
  model loading and GraphPp materialization are outside measured intervals.
- Each K has three warmup pairs followed by 15 measured pairs.
- Pair order alternates `Host->Device`, `Device->Host` by iteration.
- Host and Device receive fresh copies of the identical frozen input for every
  epoch. Output file writes are disabled, but Device Fetch still returns and
  validates all ten output tensors.
- The single accepted r5 379-file external-weight set is reused in `/dev/shm`.

## Pass Rules

For each K=2,4,8:

- Exactly 3 warmups and 15 measured samples exist for each route.
- Route order follows the frozen alternating sequence.
- Host reports K model submissions; Device reports one Host submission, one
  Feed, and one Fetch.
- Device median wall time and median Host process CPU time are lower.
- Median paired speedup is at least 1.10x.
- Device is faster in at least 13 of 15 measured pairs.
- Device wall-time Q3 is below Host wall-time Q1.

Wall and process CPU time are measured inside each epoch call. Session/model
initialization, warmups, input-file loading, result serialization, and
post-processing are excluded from those intervals for both routes.

## Storage And Log Invariants

- Inputs, prepared runner, GE cache, CANN logs, and the small `perf.tsv` live
  under `/dev/shm`; the r5 external weights remain the only live weight set.
- Persistent evidence remains below 512 MiB and every captured log remains below
  65 MiB after head/tail truncation.
- The runtime watchdog enforces 100 GiB root reserve, 128 GiB `/dev/shm` reserve,
  24 GiB G4 persistent budget, 512 MiB evidence budget, and 64 GiB scratch
  budget. Existing targets are never overwritten.
- More than 64 MiB of growth outside the current evidence directory terminates
  the command. Successful runs remove their scratch only after evidence hashes
  and the final NPU-idle check pass.
- Device launch requires both an empty process list and HBM usage at or below
  5% for three consecutive five-second samples. This prevents a just-finished
  shared job from being mistaken for a memory-ready device.

## Claim Boundary

Passing proves stable benefit for fixed B=4 K=2/4/8 on this 910B2/CANN 9.0.0
server. It does not establish dynamic batching, request insertion, preemption,
cross-platform generality, or vLLM-Ascend end-to-end benefit.
