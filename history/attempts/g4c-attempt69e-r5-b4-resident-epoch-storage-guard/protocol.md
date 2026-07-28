# G4c Attempt 69e-r5: B=4 Device-Resident Generation Epoch

Date frozen: 2026-07-26

Attempt 69e-r5 preserves the 69e model, controller, cases, comparison
semantics, and r4 storage/NPU waiting guard. It copies r4's successful Host
suite outputs together with the identical inputs and Host binary, checks file
counts, byte counts, and binary SHA256, and runs only the missing Device suite
before applying the unchanged comparator. If the Host outputs are unavailable,
r5 executes the original Host route. Attempt 69e-r4 completed all seven Host
cases, after which another container immediately claimed NPU 7 and prevented
the Device route from starting.

## Prerequisites

- Attempt 69b true B=4 eager reference SHA256:
  `a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8`.
- Attempt 69c-r2 relocatable B=4 AIR SHA256:
  `263b2acf291e13f6a84042ded53c8dccabb1fa847dcdcbbbe0ece418610ad1e3`.
- Attempt 69d-r1 B=4 native GE acceptance is valid.

## Question

Can one Device UDF transaction repeatedly invoke the accepted true B=4 AIR,
perform per-request greedy argmax and EOS control, and advance four independent
Paged-KV states without a Host round trip between decoder steps?

## Frozen Mechanism

- Every iteration invokes the single B=4 AIR exactly once. The UDF does not
  invoke four B=1 graphs or two B=2 graphs.
- Decoder ABI: token `[4,1]`, position `[4]`, sequence length `[4,1]`, key/value
  cache `[28,8,128,4,128]`, slot mapping `[4]`, active mask `[4]`, block table
  `[4,2]`, and explicit tiling `[72]`.
- Requests own disjoint two-block regions under block table
  `[[1,0],[3,2],[5,4],[7,6]]`.
- Control input is
  `[max_steps,eos_0,eos_1,eos_2,eos_3,sampling_mode,graph_variant]`;
  only greedy mode 0 and graph variant 0 are accepted.
- The 24-element control output records fixed status, four EOS values, four
  initial active flags, four executed-step counts, four finish reasons, and
  initial/final active counts.

## Cases

- `K=1,2,4` with four active requests at heterogeneous starting positions
  `[0,1,2,3]`.
- `K=8` with all requests starting at position 0 and staying inside capacity.
- Alternating active/empty slots.
- Finished/active/empty/active slots.
- Controlled independent EOS: requests 0 and 2 use their real first generated
  tokens as EOS and stop after one step; requests 1 and 3 retain configured
  Qwen EOS `151645` and continue to `K=4`.

## Pass Rules

- Host B=4 graph loop and Device UDF match at `rtol=5e-3, atol=5e-3` for every
  active per-step logits tensor and complete final Paged KV.
- Every generated token is the greedy argmax of its real request logits.
- Token histories and all final token/position/length/slot/active/control fields
  match exactly.
- Empty, already-finished, and newly EOS-finished slots remain frozen on later
  iterations; inactive request cache blocks and all unaddressed cache elements
  remain elementwise exact.
- K=1 Host and Device results preserve Attempt 69b B=4 eager continuity.
- Host submits the B=4 graph once per model call. Device uses one Feed and one
  Fetch for the complete batch epoch.
- NPU 7 is idle before Host, between routes, and after Device.

## Storage And Log Invariants

- Inputs, Host/Device tensor outputs, build products, GE cache, GraphPp external
  weights, and CANN process logs live under one fresh `/dev/shm` scratch root.
- The persistent G4 root receives only source, protocol, result/status files,
  integrity manifests, NPU state, storage snapshots, and bounded diagnostic
  logs. The evidence directory must stay below 1 GiB.
- Every captured command stream is fully consumed and hashed, but the
  persistent log retains at most 32 MiB from its head and 32 MiB from its tail.
- Preflight rejects root free space below 50 GiB, `/dev/shm` free space below
  64 GiB, a persistent G4 tree above 32 GiB, an unapproved persistent child
  above 2 GiB, a persistent log above 128 MiB, non-scratch heavy paths, an
  existing target, or a non-idle physical NPU 7.
- While every captured command is running, a watchdog rechecks root and
  `/dev/shm` reserves, the 1 GiB evidence limit, the 32 GiB persistent-tree
  limit, and a 192 GiB scratch limit. A breach terminates the command before
  either filesystem can be exhausted.
- Failed runs keep bounded evidence on persistent storage and their full
  transient state in `/dev/shm`; neither location is overwritten on retry.

## Claim Boundary

Passing closes G4c fixed B=2/4 correctness and Host-call-count semantics. Stable
performance benefit, recovery, and vLLM-Ascend scheduler integration remain
open under the final G4 gate.
