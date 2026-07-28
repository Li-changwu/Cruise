# G4c Attempt 70b-r3-r1: B=4 Blocked-ABBA Performance Evidence Replay

Date frozen: 2026-07-26

Attempt 70b-r3-r1 preserves the accepted 69e-r5 AIR, Device UDF, inputs,
greedy/EOS semantics, and Paged-KV layout. Attempt 70b tried to keep Host and
Device graphs resident in one GE process, but constructing the second graph
required another 15,251,506,481-byte weight copy and failed before any
performance sample. That is an infrastructure failure, not a performance
result.

Attempt 70b-r1 completed all H1 samples but exited 139 after successful Session
destruction and `GEFinalize`, before the stack-owned `ge::Graph` was destroyed.
Attempt 70b-r2 changed only Graph ownership and teardown ordering: Session was
released first, the explicitly owned Graph second, and GE was finalized last.

The first r2 launch passed preparation but an external shared-NPU job occupied
physical NPU 7 throughout the complete 3600-second `pre-h1` window, so it
produced no performance sample. The resource-only r2-r1 retry kept the r2 host
runner byte-identical and extended each NPU-ready wait to 21600 seconds. It
passed the performance gate, but scratch cleanup removed the four raw
performance TSVs after finalization. Attempt 70b-r3 keeps that runner and
protocol byte-equivalent while preserving the four compact TSVs for independent
analysis replay. Waiting duration remains outside all measured intervals.

Attempt 70b-r3 passed H1, but an external shared-NPU job entered after the D1
idle gate and consumed enough HBM for Device FlowModel loading to fail with
runtime error 207001. Driver evidence distinguishes the attempt logical-device
hostpid from the external hostpids. This resource-only retry keeps the runner,
measurements, and analysis unchanged while extending every required idle window
from 15 seconds to 300 seconds.

## Question

After route-local initialization and warmup, does Device-resident iteration
reduce epoch wall time and Host process CPU time for every K>=2 while
preserving K Host submissions versus one Device Feed/Fetch transaction?

## Controlled Design

- Fixed true B=4 inputs for K=2, 4, and 8. K=1 correctness remains covered by
  69e-r5; the performance gate is explicitly K>=2.
- A single route is resident in each process, avoiding the unmeasured
  two-graph weight duplication. Four sessions run in blocked-ABBA order:
  H1(8 samples), D1(8), D2(7), H2(7).
- H1/D1 form block 1 (`Host->Device`); D2/H2 form block 2
  (`Device->Host`). Pairing equal iteration indices yields 15 Host/Device
  pairs per K while balancing route order across the two blocks.
- Every route/block/K receives three unmeasured warmups before its measured
  samples. Each of the four sessions has an independent GE cache and CANN log
  path.
- Host and Device load identical frozen inputs. `RunHostCase` performs K graph
  submissions. `RunDeviceCase` performs one Feed and one Fetch and validates
  all ten returned tensors' count, byte sizes, and dtypes. Output-file writes
  are disabled inside measured intervals.
- Wall and process CPU timers cover only the epoch call. GE initialization,
  graph construction, input-file loading, warmup, serialization, and analysis
  are excluded for both routes.

This blocked design controls the combined-session memory failure and first/last
route drift. It does not claim cycle-level interleaving: each block is a
route-resident session because both graphs cannot coexist on this server.

## Pass Rules

For each K=2,4,8:

- Exactly three warmups per route/block and 15 measured pairs exist.
- Block order and route positions match H1-D1-D2-H2.
- Host reports K model submissions; Device reports one Host submission, one
  Feed, and one Fetch.
- Device median wall time and median Host process CPU time are lower.
- Median paired speedup is at least 1.10x.
- Device is faster in at least 13 of 15 measured pairs.
- Device wall-time Q3 is below Host wall-time Q1.

The four block files must have nondecreasing modification timestamps and the
complete result must contain exactly 126 rows.

## Storage And Device Invariants

- Inputs, runner, Device UDF controller workspace/build, four GE caches, CANN
  logs, and performance TSVs live under marker-protected `/dev/shm` scratch.
  No new durable weight or build copy is created.
- Root reserve is at least 100 GiB, `/dev/shm` reserve at least 128 GiB, G4
  persistent use at most 24 GiB, evidence at most 512 MiB, and scratch at most
  64 GiB. Unexpected persistent growth above 64 MiB terminates a heavy command.
- Every block launch and the final state require an empty NPU process list plus
  HBM usage at or below 5% for 60 consecutive five-second samples.
- Successful scratch removal requires finalized evidence hashes and the final
  NPU-ready check. Failure scratch remains marker-protected for diagnosis.
- Only the four compact `perf-block.tsv` files, result JSON, bounded logs, and
  integrity metadata persist on the root filesystem. Inputs, weights, caches,
  builds, and CANN logs remain scratch-only and are deleted after finalization.

## Claim Boundary

Passing proves stable benefit for fixed B=4 K=2/4/8 on this 910B2/CANN 9.0.0
server under the blocked-ABBA protocol. It does not establish dynamic batching,
request insertion, preemption, cross-platform generality, or vLLM-Ascend
end-to-end benefit.
