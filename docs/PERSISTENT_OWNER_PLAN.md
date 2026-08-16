# Persistent Device Model Owner execution plan

Cruise replaces Host-visible fixed-K Decode epochs with one model-load-scoped
Persistent Device Model Owner. Development is evidence-gated: a later phase may
not begin until the previous phase passes on the target NPU, and the frozen
Epoch Reference Scaffold is never promoted as a fallback product path.

## Ordered phases

| Phase | Scope | Exit evidence |
|---|---|---|
| P0 | Weight-free persistent AICPU/DataFlow control, live event ingress, incremental outfeed, quiescence and wakeup | One Owner per load, at least 1024 autonomous quanta, three readmission cycles, three clean load/unload starts, no Host control call proportional to quanta |
| P1 | Deterministic AICore recurrence and Committed Prefix outfeed | Exact B=1/B=4 256-step recurrence, pre-shutdown output, Output Credit isolation, three NPU starts |
| P2 | Admission, cancellation, generation isolation and retirement | Full stale/duplicate/reordered matrix, retirement-before-reuse, 1000 cancels with p95 at most 50 ms and p99 at most 100 ms |
| P3 | Pinned Qwen2.5-7B-Instruct greedy Decoder and multi-page Device KV Lease | Exact B=1 short/full and B=4 mixed cases, at least 384-token lease, 100% Device Decode coverage, no Host KV progression |
| P4 | Quantum-Boundary Serial Admission service | Exact 128/256 streaming closed-loop C4 workload, 32 requests, lifecycle and regression cases |
| P5 | Persistent Decode Qualification | Versus same-round Graph: Host CPU/token 50% lower, TPOT p50/p95 15% better, throughput 15% better across three starts |
| P6 | Shared-owner Concurrent Prefill/Decode | Real AICore overlap, 10% makespan benefit, isolated QoS within 5%, peak HBM at most 90%, final service gates and 10,000-request soak |

## Stop conditions

P0 tests CANN DataFlow/FlowFunc first and at most one other officially supported
Device-resident control substrate. Failure of both stops the architecture on
the current CANN 9 and Ascend 910B2 stack. P5 permits one profiler-driven
redesign and full rerun; a second Host, TPOT, or throughput miss stops P6. A P6
failure leaves at most a Persistent Decode Qualified research candidate and
does not complete Cruise or Stable v1.0.

If the single permitted P5 redesign cannot pass its exact attention component
through a public supported Device path, it cannot enter the second pair or full
rerun. That leaves P5 Stopped / Unqualified on the current stack and keeps P6
closed; it is not reclassified as a second performance measurement. ADR 0020
defines the evidence and reopening requirements for this condition.

## Current gate state

P0 passed the full target-NPU gate in
`persistent-control-p0-profile-20260808-r14`. The candidate selected the
`Ascend` resource with `device_type=0`, passed three mechanism starts and one
profiled start, bound the profiled Host process to one target Device UDF
executor, reported zero AICore tasks as required by P0, returned HBM to its 58%
baseline, released all visible processes, and passed evidence-integrity checks.
The profiled Host/controller source SHA-256 values were
`89399661186bbee16f726e914cc068a9025f16df26e5198a4976d6ec7a6c97c6` and
`da5dc7f21ef5a92719c1d26a551bffd58ef8eae301ce1f260091c91b7c27517a`.

P1 passed the full target-NPU gate in
`persistent-recurrence-p1-profile-20260809-r4`. Owners `1001`, `1002`, and
`1003` each produced exactly 1,291 outputs and 1,280 commits from four Host
feeds, stopped the output-blocked B=4 row at commit 16 while its peers reached
256, and completed 752 autonomous GraphPp calls. Profile Owner `2001` repeated
the same sequence; the profiler recorded exactly 752 AIVector tasks, all 752
were attributed to
`persistent_recurrence_p1_graph_pp/persistent_recurrence_p1_add`, and the
profiled Host process was bound to one target Device UDF executor. Placement
selected `Ascend` with `device_type=0`, all four starts returned HBM to the 5%
baseline, and no visible Device process remained.

The initial r4 analysis rejected the profile because it read compiled kernel
names from `task_time` but not the full Device op identity from `op_summary`.
The analyzer was tightened to accept only the exact target op or its GraphPp
path suffix, covered by positive hashed-kernel and negative unrelated-Add
tests, and rerun offline against the unchanged r4 NPU evidence. The authoritative
gate result is `evidence/result-reanalyzed.json` with
`requested_gate_pass=true`; the original failed result remains as an audit
record. The profiled Host/controller source SHA-256 values were
`e676f35b932a3309fd9f25e91d7bfab3d621ec0f560980ed8f80d6f2778c24f5` and
`9f33ad760c4d9ba44d8983be507bd63457993bb96327e494d41433043cbbd5e5`.

P2 passed the full target-NPU gate in
`persistent-cancel-p2-profile-20260809-r5`, with the three mechanism starts
independently reproduced in `persistent-cancel-p2-mechanism-20260809-r4`.
Owners `1001`, `1002`, and `1003` and profiled Owner `2001` each completed the
full stale/duplicate/reordered matrix and 1,000 stress cancellations. Every
run reported 255 quiescent transitions, 1,009 retirements, five expected
rejections, four cumulative duplicate acknowledgements, and 1,000 valid
latency samples. Mechanism cancellation p95/p99 were respectively
3.242/5.536 ms, 3.301/5.715 ms, and 3.357/5.783 ms; the profiled run was
3.287/4.527 ms. Placement selected `Ascend` with `device_type=0`, no Host UDF
executor evidence, target Add task count exactly matched the profiled Owner's
567 AICore calls, HBM stayed at the 5% baseline after every start, all visible
processes were released, and evidence-integrity checks passed. The profiled
Host/controller source SHA-256 values were
`a0d48efbf6abb54be356a2c147a5b74b1570f05591d2db4511ad5ee364b6614c` and
`19ea59600ef190e07b9569cdff6ff212ad535768b223527c3d0a50a21fcf55f2`.

P3 passed its target-NPU correctness and lifecycle gate. The exact B=1 short
and 128-prompt/256-output cases passed independent cold Graph comparison in
`persistent-decoder-p3-gate-b1-short-20260814T091600Z` and
`persistent-decoder-p3-gate-b1-full-20260814T093753Z`. The natural EOS case
uses the pinned model's primary `151645` token and passed in
`persistent-decoder-p3-gate-b1-eos-20260814T104120Z`. The earlier
`persistent-decoder-p3-oracle-b1-eos-20260814T095946Z` result is superseded and
non-gating because it reached the output budget without EOS; that observation
led to a mandatory EOS assertion in the oracle.

The final B=4 candidate completed three independent cold starts in
`persistent-decoder-p3-b4-mixed-final-r1-20260814T111441Z`, `r2`, and `r3`.
Each used five Host feeds for four admissions plus shutdown, made 383 AICore
calls, committed 344 tokens, retired all four requests, and exactly matched the
same cold Graph oracle. All three Owner summaries and structured comparisons
are byte-identical. The authoritative aggregate is
`persistent-decoder-p3-three-start-gate-20260814T114811Z`, with 100% Device
Decode coverage and explicit AIR, external-weight, and source identities.

`persistent-decoder-p3-b1-backpressure-20260814T104222Z` completed the full
256-token output after a bounded delayed Host drain. The real-Decoder lifecycle
run `persistent-decoder-p3-lifecycle-20260814T105230Z` covered cancellation
before a commit, cancellation after two commits, Output Credit replenishment,
duplicate acknowledgements, stale and bad-generation rejection, retirement,
and row reuse. Every completed run released its mutable external-weight view,
tmpfs scratch, Host process, and visible NPU process. The canonical AIR SHA-256
is `53e7c8407ac638af4c8b50865e8cc1c5ea428aedfa399b56c9bf014b792cc60b`;
the 721-file external-weight bundle identity is
`f93a2828d8b3ded10acc3831d114abe8b57738ad3adfc357a0f2bb047939866b`.

P3 is Candidate Hardware Validated.

P4 passed its target-NPU service and regression gate. The formal run
`persistent-admission-p4-primary-c4-20260814T123734Z` completed 32 exact
128-prompt/256-output requests as eight Device-staged C4 cohorts: 8,192
single-token outputs, 33 Host feeds, 3,064 AICore calls, four rows reused eight
times each, and exact cold Graph oracle equivalence. The regression run
`persistent-admission-p4-regression-20260814T122710Z` covered natural EOS,
short output, burst admission, four expected C8 overload rejections followed by
successful retry, four Output Credit acknowledgements, cancellation, retirement,
and row reuse. The authoritative aggregate is
`persistent-admission-p4-gate-20260814T125100Z`.

P4 is Candidate Hardware Validated. P5 Persistent Decode Qualification is the
next active phase. P4's approximately 48.13 output tokens/s, 44.38 ms median
per-request TPOT, and 5.70 s median serial-serving TTFT are observability-only;
they have no same-round Graph baseline or whole-process-tree Host CPU
attribution. No P4 result is a vLLM performance qualification or concurrent
Prefill/Decode claim; those remain blocked on P5 and P6.

The 2026-08-15 committed-source replay binds P3 and P4 to commit `950df7c`.
Three independent P3 cold starts were byte-identical and matched the cold
Graph oracle exactly. P4 primary again completed 32 requests, 8,192 commits,
33 Feed calls, and exactly 3,064 AICore calls; its lifecycle regression also
passed with the same source identity. Compact evidence is in
`evidence/PERSISTENT-OWNER-SOURCE-BOUND-REPLAY-20260815.md`.

P5 now has an implementation-ready vLLM-compatible Owner entry, a sampled
whole-process-tree CPU meter, and an independent six-start matrix verifier in
`experiments/persistent_decode_p5`. Its Owner runtime imports neither the
frozen Host-visible scheduler nor the legacy sidecar backend. This does not
advance the gate state: P5 remains open until the interleaved Graph/Owner
matrix passes exact semantics and all four ADR 0017 performance thresholds.

The first complete pair in `persistent-decode-p5-first-pair-20260814-r5`
confirmed the intended Host saving but failed the performance gate. Relative to
Graph, Owner reduced Host CPU per output token from 7.544 ms to 2.291 ms, while
TPOT p50 increased from 18.533 ms to 43.444 ms and throughput fell from 198.97
to 61.31 output tokens/s. Its measured AICore count was 3,071 rather than the
required 3,064. The canonical 15 GiB model asset is present and verified; this
failure is not caused by a missing model or a restarted implementation.

The active P5 redesign is now bounded to a supported FusedInferAttentionScore
compute-plane path. A weight-free public-API probe proved exact execution of a
self-contained 18.9 MiB FIA OM, but GraphPp still fails to restore the FIA
kernel entry. The public `Graph::LoadFromSerializedModelArray` API rejects the
OM format, and the installed CANN 9.0 DataFlow headers declare no precompiled
model process point. Cruise will not call Host ACL once per Decode step and
will not bind to the undeclared private `ModelPp` ABI. P5 remains open at this
compute-plane integration blocker; P6 remains closed.

A current-source FIA replay on 2026-08-15 reproduced that boundary before the
first admission Feed: `BinaryGetFunctionByEntry` failed with `funcEntry=0` and
runtime status `107000` at
`persistent_decoder_p3_graph_pp/FusedInferAttentionScore`. The same committed
P3/P4 source passed on the accepted legacy compute plane, so this is a P5
compute-plane blocker rather than a legacy correctness regression.

A bounded public custom-op probe then narrowed the blocker. In
`persistent-decoder-p3-custom-graphpp-20260815-r3`, ordinary Graph and public
GraphPp each loaded one generated `Bf16Materialize` custom AICore kernel,
launched it once, and returned a bitwise-exact BF16 output. This proves that
`funcEntry=0` is not a general GraphPp boundary for every custom AICore kernel
on this CANN 9 stack. It does not establish FIA compatibility: Cruise must
first show that the FIA host tiling, tiling-data ABI, and OpDef are available
through public installed resources, or build a fixed-shape attention closure.
Only an exact B=4, K=384 Graph/GraphPp attention component may advance to the
second full Graph/Owner pair. Compact evidence is in
`evidence/PERSISTENT-OWNER-CUSTOM-GRAPHPP-20260815.md`.

The bounded public FIA package attempt then generated a valid 31-input AIR from
the public TorchAir `custom_op` extension point and installed the official
ops-transformer v9.0.0 ordinary and relocatable objects in an isolated custom
OPP. Ordinary Graph and public GraphPp still did not select the static
`FusedInferAttentionScore_3b093497...` object. Both fell back to online
`te_fusedinferattentionscore_*` compilation and failed before a launch because
the package's relative IncreFlashAttention header was unavailable. The mini
component therefore did not pass, and no full P3 FIA graph, second Graph/Owner
pair, or six-start matrix was run. P5 is Stopped / Unqualified on this CANN 9 /
Ascend 910B2 stack; P6 remains closed under ADR 0020. Compact evidence is in
`evidence/PERSISTENT-OWNER-FIA-GRAPHPP-20260815.md`.

Controller-Aware Graph V2 first exported a stock `ScatterPaKvCache + RefData`
structure, but public GraphPp could not assign a DataFlow data index to the
external `RefData` transfer node. V2 therefore does not use external `RefData`
as its shared-state boundary.

The replacement Device-state handle lifetime probe passed on 2026-08-15 at
clean commit `7055eb4`. FunctionPp allocated one 8 KiB synthetic Device buffer,
retained the same `FlowMsg` and address, and invoked one custom AICore GraphPp
twice. The dependent updates zero -> BF16 1.0 -> 2.0, complete-buffer scans,
and cross-call checksums were exact. Host cache input/output stayed at zero;
ordinary Graph also executed the same custom kernel exactly. Compact evidence
is in `evidence/PERSISTENT-OWNER-V2-DEVICE-KV-UPDATE-20260815.md`.

This advances only the public Device State Handle mechanism. The next ordered
gate must prove the target B4/K384 Paged-KV layout and an explicit in-graph
update-to-reader dependency before any Prefill/Decode graph uses the mechanism.
P5 remains Stopped / Unqualified; no P5 pair, six-start matrix, or P6 run is
authorized by this component result.

The target-layout ordering gate passed on 2026-08-15 at clean commit
`babcf33`. One FunctionPp-owned 3 MiB state tensor used the formal
`[2, 12, 32, 128, 16]` B4/K384 PA-NZ layout. `DevicePagedKvUpdate` updated
positions 0, 127, 128, and 383, and its compact ticket directly fed a
`DevicePagedKvRead` in the same graph. Ordinary Graph and two public GraphPp
invocations were exact; the Device controller's complete state scans rejected
missing, misplaced, or extra writes. GraphPp Host KV I/O remained zero.

This closes only `V2-KV-ORDER`. It does not establish attention execution,
internal Device-to-Device copy absence, whole-model graph export, or service
performance. The next graph-family integration must retain this dependency and
state boundary while adding a real attention consumer. P5 remains Stopped /
Unqualified and P6 remains closed.

The public FIA JIT path was subsequently repaired without a private ABI or
system OPP modification. A minimal FIA component and a real five-dimensional
PA-NZ cache `[12, 4, 8, 128, 16]` both executed exactly through ordinary Graph
and public GraphPp. This removed the earlier `funcEntry=0` and cache-layout
component blockers, but did not reopen P5.

The one permitted combined KV-update-to-FIA gate then failed on 2026-08-15 at
clean commit `864fb94`. AIR export and structure passed, including shared
key/value inputs, an explicit update-ticket-to-FIA dependency, compact outputs,
and zero `RefData` or `TensorMove`. TorchAir had reordered the six exported
`Data` indices to key, value, metadata, query, mask, and block table, while the
Graph config and C++ callers retained key, value, query, metadata, mask, and
block table. GE therefore received BF16 at the update operator's INT32 metadata
input, rejected engine assignment, and launched none of the three target
kernels. Ordinary Graph failed before execution; GraphPp failed compilation
with `1343225857` before FunctionPp allocated the long-lived KV buffers.

Because the frozen combined gate required every input/lifetime, exactness,
no-full-KV-I/O, and dual-route condition in one run, this input-index failure
stops the path. No corrected rerun, full Decoder graph-family integration,
second P5 pair, or six-start matrix is authorized. Compact evidence is in
`evidence/PERSISTENT-OWNER-V2-KV-ATTENTION-20260815.md`. P5 remains Stopped /
Unqualified and P6 remains closed.

One separately authorized ABI-r2 diagnostic then derived the six input
positions, dtypes, and shapes from AIR and generated both Graph descriptors and
C++ constants from the checked manifest. Export/static ABI passed at clean
commit `e414092`. Ordinary Graph loaded the artifact and launched
`DevicePagedKvUpdate`, `DeviceQueryAfterKvUpdate`, and
`FusedInferAttentionScore` once each, but `RunGraph` returned runtime status
`107000` without compact outputs. The input-order defect was therefore fixed,
but ordinary Graph exactness was not established.

The failure bundle retained only the first 24 sorted driver logs and omitted
the detailed execution-process tails, so this result cannot assign the invalid
parameter to a particular kernel or output-recovery boundary. The staged stop
rule prevented GraphPp, full Decoder integration, another P5 pair, and the
six-start matrix. Compact evidence is in
`evidence/PERSISTENT-OWNER-V2-KV-ATTENTION-ABI-R2-20260816.md`. P5 remains
Stopped / Unqualified and P6 remains closed.

The follow-up owner-only revision removed ordinary Graph from the ownership
gate while retaining it as a frozen negative diagnostic. FunctionPp allocated
one 3 MiB CANN-owned key/value state, retained the same FlowMsg objects and
Device addresses, and passed them directly to GraphPp. Two Feed/Fetch rounds,
two complete cache scans, attention, tickets, mask, table, and query all passed
exactly. Host KV I/O was 0/0, and bounded transfer logs contained no full-cache
size match. Load-time task registration and two-call semantic evidence are now
reported separately.

This passes the bounded E0-E4 ownership component at source commit `6a628e8`
plus the captured worktree patch. It does not reopen P5: the next independently
tracked step is 28-layer Decoder integration and correctness before a new
Graph/Owner pair. Compact evidence is in
`evidence/PERSISTENT-OWNER-V2-KV-ATTENTION-OWNER-20260816.md`. P5 remains
Stopped / Unqualified and P6 remains closed.
