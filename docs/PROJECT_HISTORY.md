# Project History

This document maps the source snapshots retained in `history/attempts/` to the
research questions they addressed. Attempt numbers are experiment identifiers,
not software releases.

## Stage 0: Device-Control Feasibility

The synthetic P0 established that one Host Feed/Fetch can enclose N recurrent
`RunFlowModel` calls in a Device UDF. The identical-artifact Host GE versus
Device UDF comparison crossed over at N=2 and reached 3.65x at N=32. The
real-Qwen layer-slice P0 then showed elementwise-identical recurrent attention
and KV state for N=1,2,4. The bounded controller added runtime graph selection,
EOS/max-step termination, and capacity rejection.

Sources: `experiments/synthetic-p0/`, `history/attempts/real-qwen-p0/`, and
`history/attempts/bounded-decode-controller/`.

## Stage 1: Full-Decoder Semantic Closure

Attempts 41-51 isolated sparse QK differences in the GE online-compiled path.
Attempts 52-65 progressively audited attention, BF16 materialization,
boundaries, transpose/layout choices, fusion, residual paths, and all linear
layers. Attempt 65 is the clean full-decoder packaging milestone used by later
DataFlow work.

Sources: `history/attempts/attempt41/` through
`history/attempts/g4a-full-attempt65-clean*/`.

## Stage 2: DataFlow and Device UDF Bring-Up

Attempts 66a and 66b moved from DataFlow smoke tests to BF16 Device UDF I/O,
all required operators, repeated decoder invocation, and persistent controller
state. These experiments defined the compiler/runtime constraints later used
by the resident epoch.

Sources: `history/attempts/g4b-attempt66*/`.

## Stage 3: Batched Full-Decoder Epochs

Attempts 67-70 established B=2 and then B=4 eager, AIR, native, and resident
epoch routes. They added active-row masking, full decoder logits, greedy
sampling, persistent Paged-KV, runtime graph metadata, recovery contracts,
storage guards, and the blocked-ABBA performance protocol.

The authoritative G4 B=4 performance run used one Feed/Fetch for the Device
route and K Host submissions for the Host route. Median paired speedups were
1.55x at K=2, 3.56x at K=4, and 5.36x at K=8. Full details and claim boundaries
are in `history/attempts/g4/G4-STATUS-20260724.md`.

Sources: `history/attempts/g4c-attempt67*/` through
`history/attempts/g4c-attempt70*/`.

## Stage 4: vLLM Integration

Attempt 71 introduced the scheduler contract and eligibility envelope. Attempt
72 connected one vLLM EngineCore step to the native sidecar. Attempt 73 proved
multi-epoch cohort evolution and safe row generation reuse with the trace
`[A] -> [A,B] -> [A,C]`.

Attempt 74, now at the repository root, removes dead Host-UDF Paged-KV and
diagnostic payloads. The boundary changes from 10 inputs/10 outputs and
136,905,444 declared bytes per epoch to 8 inputs/2 outputs and 628 declared
bytes. The decoder ABI and accepted Attempt 73 semantics remain fixed.

The formal CANN 8.5.1 blocked-ABBA run then observed the same old/new byte
counts on the actual DataFlow Feed/Fetch tensors for all 60 measured epochs.
Median Host-control wall time changed from 212.208 ms to 59.951 ms (3.54x).
No covered runtime memcpy or Mbuf event occurred inside any measured epoch;
all such records were startup-only. Physical-link bytes remain unobserved
because application `msprof` cannot initialize the resident sidecar on this
CANN release. The accepted result boundary is recorded in
`evidence/ATTEMPT74-CANN851-R5.md`.

Sources: `history/attempts/vllm-integration-attempt71-*` through
`history/attempts/vllm-integration-attempt73-*`, followed by the active root.

## Stage 5: Productization Roadmap

Cruise has passed its research-feasibility gate, but the current `0.1.0` line is
still a research baseline rather than a stable product. In particular, its
`0.1.0` package version is development metadata; it is not a statement of
production readiness. The next work is governed by the productization tracker
referenced below and by this version-controlled roadmap.

### Governance and Source of Truth

- This document is the canonical roadmap. The GitHub issue mirrors its status
  and provides discussion, ownership, and links to pull requests and evidence.
- A checkbox may be marked complete only by a merged commit plus reproducible
  test or experiment evidence. A smoke run, synthetic result, or unmerged
  branch is not completion evidence.
- Changes to scope, thresholds, or support claims must update this document in
  the same pull request. Regressions reopen the relevant gate.
- Releases are evidence-gated rather than date-gated. Unsupported requests
  must preserve stock vLLM behavior or fail before device state is mutated.
- Performance evidence must use fixed versions, a same-machine baseline, and
  at least three independent service starts. Negative results are retained.

Productization tracker:
[`Li-changwu/Cruise#1`](https://github.com/Li-changwu/Cruise/issues/1).

### First Stable Support Contract

The first stable release is intentionally narrow. It targets one Ascend 910B2
with the declared CANN, torch-npu, vLLM, and vLLM-Ascend compatibility matrix;
Qwen2.5-7B-Instruct; TP=PP=1; synchronous scheduling; text-only requests; and
greedy decoding. It must accept real prompts through the vLLM API server,
support continuous request arrival at epoch boundaries for up to four resident
rows, stream tokens in order, and fall back safely for requests outside the
device route. Broader sampling, other model families, LoRA, speculative
decoding, multimodal input, TP/PP, and multi-node operation are later scope,
not hidden v1.0 requirements.

### Release Gates

| Release level | Required gates | Meaning |
|---|---|---|
| Research baseline | Accepted through Stage 4 | Reproducible feasibility evidence; not a serving product |
| Developer Preview | M0-M1 | Installable end-to-end serving path within the narrow support contract |
| Beta | M0-M3 | Fault-contained and observable serving path suitable for controlled users |
| Release Candidate | M0-M4 | Product-level correctness, stability, and performance evidence complete |
| Stable v1.0 | M0-M5 and all final acceptance rules | Supported, documented, versioned, and rollback-capable release |

### M0: Product Contract and Reproducible Deployment

- [x] Publish an exact compatibility matrix covering hardware, driver, CANN,
  torch-npu, Python, vLLM, vLLM-Ascend, model revision, graph artifacts, and
  external-weight hashes.
- [x] Replace the experiment-only environment-variable bundle with a validated
  user configuration and a `doctor` command that reports every missing or
  incompatible dependency without loading the model.
- [x] Version the Python contract, sidecar wire protocol, Host-UDF ABI, graph
  configuration, and external assets; reject incompatible combinations before
  model execution.
- [x] Provide documented clean install, start, stop, upgrade, rollback, and
  uninstall procedures without editing vLLM or vLLM-Ascend source files.
- [x] Add a bounded no-NPU smoke path and a one-command NPU installation check.

M0 checkpoint (2026-07-29): four of five implementation items are verified.
The isolated non-editable wheel install, smoke, package-data/entry-point check,
uninstall, 60-test frozen-environment suite, exact NPU/driver diagnosis, ABI
verifier, repository audit, and marker-checked cleanup all passed. Evidence is
in [`M0-PRODUCT-READINESS-20260729.md`](../evidence/M0-PRODUCT-READINESS-20260729.md).

M0 portability increment (2026-07-30): compatibility schema v2 separates
host-independent capability requirements from exact evidence-backed profiles.
The pre-model doctor now reports structured rejection codes with expected,
observed, and remediation fields for unsupported architecture/NPU, device
count/health/occupancy, DataFlow and compiler availability, software versions,
and `/dev/shm` capacity. On NPU0-1, the check correctly distinguished an
installed but inactive CANN 9.0.0 DataFlow package from a missing component;
after exposing its Python 3.11 site-packages path, the candidate profile passed
on physical NPU 0. The target environment suite passed 100 tests. This is an
installation-capability checkpoint only: the CANN 9.0.0 profile remains
`candidate-m0-validation`, and no model or runtime asset was used by this
increment.

M0 NPU0-1 checkpoint (2026-07-31): a clean `main` checkout on the large
`/workspace` volume produced and installed a non-editable wheel in
`vllm-hust-dev`; package smoke and the CANN 9.0.0 candidate-profile doctor
passed on idle physical NPU 0. A later asset audit corrected the initial
missing-model diagnosis: the machine already contained the exact frozen
Qwen2.5-7B-Instruct, AIR, tiling, baseline, and custom OPP trees under an
existing `/dev/shm/cruise-m0-assets-npu01-r1` owner path. No model was
downloaded or moved. Cruise materialized the 342-file, 15,231,237,408-byte
runtime bundle once into a manifest-addressed `/workspace/cruise-assets`
store; a second deep invocation returned `reused=true` with no duplicate or
staging residue. Runtime doctor failures preserve stable `code`, `expected`,
`observed`, and `remediation` fields. The active Triton build exposes the new
`extra.cann` namespace, so the capability contract accepts either qualified
symbol and the plugin installs a process-local legacy alias when needed. The
complete remote suite passes 109 tests with four CANN ownership warnings.
Evidence is in
[`M0-CANN900-NPU01-20260731.md`](../evidence/M0-CANN900-NPU01-20260731.md).
This checkpoint does not close M0 or authorize M1 execution.

M0 closed on NPU0-1 physical NPU 0 (2026-07-31). Commit `8408a68` passed two
independent real-model
`build -> install -> smoke -> NPU doctor -> runtime doctor --deep -> start ->
stop -> verify -> cleanup -> uninstall` cycles in `vllm-hust-dev`. Each cycle
verified 13 EngineCore cases, 13 Feed/Fetch pairs, and 49 device model calls;
left no package, process, socket, scratch, or source-tree residue; and retained
only 88 KiB of structured evidence. The 342-file, 15,231,237,408-byte runtime
bundle remained a single content-addressed read-only asset on `/workspace`.
The qualification also found and fixed pre-start CANN cache-directory creation
and post-cleanup sidecar artifact verification defects. Evidence is in
[`M0-LIFECYCLE-NPU01-20260731.md`](../evidence/M0-LIFECYCLE-NPU01-20260731.md).
The NPU0-1 profile is now `m0-qualified-developer-preview`; this closes M0 but
does not close M1 or claim Stable v1.0.

Exit evidence: a clean checkout can be installed and diagnosed on a supported
machine using only documented commands, and a repeated install/start/stop cycle
leaves no untracked source files or unbounded persistent artifacts.

### M1: End-to-End Serving Semantics

- [x] Implement a real prefill-to-resident-decode ownership transition and
  prove token and Paged-KV equivalence against unmodified vLLM for nontrivial
  prompts.
- [x] Support continuous admission, completion, and row reuse at epoch
  boundaries for mixed arrival times and output lengths while retaining the
  generation-checked row lease.
- [ ] Preserve OpenAI-compatible streaming and non-streaming response order,
  EOS, `max_tokens`, disconnect, cancellation, and request-finalization
  semantics.
- [x] Define explicit `prepared`, `executing`, and `committed` states. Only a
  proven pre-execution failure may replay on the Host; an ambiguous
  post-mutation failure must never duplicate token or KV advancement.
- [ ] Route unsupported sampling or features to an unmodified Host path before
  device ownership begins, without requiring a server restart.

Exit evidence: a differential suite of at least 1,000 deterministic requests
spanning prompt/output-length bins, batch sizes 1-4, EOS and cancellation
boundaries has exact token IDs, finish reasons, streaming order, and scheduler
accounting. All ineligible cases preserve baseline behavior.

M1 first increment (2026-07-29): commit `857c007` adds the state-owner field,
the `prepared -> executing -> committed` Python contract, sidecar protocol v4,
native bridge transitions around Feed/Fetch, and a fail-safe fallback adapter.
The dependency-light suite has 50 passing tests; the frozen server environment
has 70 passing tests; and CANN 8.5.1 compiled both new/old bridge and server
targets. The evidence is in
[`M1-COMMIT-STATE-20260729.md`](../evidence/M1-COMMIT-STATE-20260729.md).
M1 second increment (2026-07-29): stock vLLM executed a real three-token
prefill on `NPUWorker`, after which Cruise copied the scheduler-owned Paged-KV
block once, imported it into the resident cache, executed K=2, transferred
state ownership to the device, and executed the remaining K=1 epoch through
the unchanged 260-byte/368-byte steady ABI. The Host snapshot and Device cache
checksums both equalled `3477654769`; stock and Cruise produced the exact token
sequence `[2776, 4460, 311, 1855]`. Python contract v3, sidecar protocol v5,
generation-checked transfer files, checksum fail-stop behavior, child-only OPP
environment isolation, and sidecar shutdown from stock `NPUWorker` are now in
the active source. The dependency-light and frozen-environment suites report
51 and 73 passing tests, respectively. Evidence is in
[`M1-PREFILL-TRANSFER-20260729.md`](../evidence/M1-PREFILL-TRANSFER-20260729.md).

M1 third increment (2026-07-29): separate stock and Cruise processes executed
four simultaneous-prefill cohorts covering B=1,2,3,4, prompt lengths 2-5, and
mixed output budgets 2-5. All 10 request outputs, terminal reasons, and final
scheduler accounting matched exactly. Each cohort imported all active Paged-KV
rows once; the four Host/Device checksum pairs matched; all later epochs stayed
Device-owned with the 260-byte/368-byte steady ABI. B=3 and B=4 also proved
completion-driven active-count shrink while surviving row generations remained
stable. The frozen server suite has 78 passing tests. Evidence is in
[`M1-BATCHED-PREFILL-20260729.md`](../evidence/M1-BATCHED-PREFILL-20260729.md).

M1 fourth increment (2026-07-29): A remained Device-owned while B and then C
each executed an isolated stock prefill. The two mixed resident epochs imported
only the new row; A retained row 0/generation 1, while B used row 1/generation
2 and C reused row 1/generation 3 after B completed. All three output token
sequences, terminal reasons, and final scheduler accounting matched stock
vLLM. All three Host/Device Paged-KV checksum pairs matched, and every Device
epoch retained one Feed and one Fetch. The scheduler now fail-stops before a
Host step can execute stale Device-owned state. The frozen server suite has 83
passing tests. Evidence is in
[`M1-CONTINUOUS-ADMISSION-20260729.md`](../evidence/M1-CONTINUOUS-ADMISSION-20260729.md).

M1 remains open. The next ordered gates are: stock-equivalent unsupported
request routing before ownership transfer; API streaming and non-streaming
ordering; then EOS, disconnect/cancellation, simultaneous-arrival expansion,
and the 1,000-request differential exit suite.

### M2: Lifecycle, Recovery, and Resource Safety

- [ ] Add sidecar supervision, bounded startup and request timeouts, readiness
  and liveness checks, graceful shutdown, stale-socket cleanup, and one
  well-defined restart policy.
- [ ] Inject failures before Feed, during device execution, after Fetch, on
  sidecar exit, on malformed response, and under NPU/storage pressure; verify
  the commit-state rule for every fault point.
- [ ] Add capacity admission and backpressure for resident rows, KV blocks,
  scratch space, logs, and external assets. Overload must be bounded and must
  not corrupt admitted requests.
- [ ] Ensure model startup failure, client disconnect, SIGTERM, and normal exit
  release processes, sockets, device resources, and marker-protected scratch.

Exit evidence: a 24-hour or 10,000-request soak, whichever is longer, completes
with no incorrect response, hang, orphan sidecar, or monotonic Host/HBM growth.
After warmup, measured RSS and HBM drift stay within 2%; default persistent
runtime output, excluding packages, models, and retained evidence, stays below
100 MiB.

### M3: Observability and Operator Workflow

- [ ] Export route eligibility and rejection reasons, selected epoch length,
  resident-row/KV occupancy, Feed/Fetch counts, sidecar latency, Host CPU,
  fallback, restart, timeout, and device-error metrics.
- [ ] Add structured, bounded logs with request correlation and lifecycle
  events, while excluding prompt text, generated text, credentials, and raw
  model tensors by default.
- [ ] Expose startup capability, health, and support-matrix status through the
  documented server/operator interface.
- [ ] Publish a quickstart, configuration reference, troubleshooting guide,
  failure/rollback runbook, and known-limitations page validated from a clean
  installation.

Exit evidence: an operator can distinguish Host fallback, device execution,
overload, unsupported input, sidecar failure, and device failure without a
debug build or unbounded profiling capture.

### M4: End-to-End Performance Qualification

- [ ] Compare unmodified vLLM-Ascend eager execution, its supported graph path,
  and Cruise under the same API-server workload, versions, weights, NPU,
  warmup, and request trace.
- [ ] Cover short and decode-heavy requests, concurrency 1-4 and overload,
  steady and bursty arrivals, EOS variation, and mixed output budgets.
- [ ] Report TTFT, inter-token latency/TPOT, end-to-end latency, throughput,
  Host CPU/token, accelerator idle gaps, route hit rate, error rate, HBM, and
  all p50/p95/p99 distributions. Initialization is reported separately.
- [ ] Retain per-run configuration, raw bounded measurements, independent
  verifier output, checksums, and negative results using the storage policy.

Exit evidence: on the declared decode-heavy target, both median and p95 TPOT
improve by at least 15% and Host CPU/token falls by at least 30% versus the
strongest applicable baseline. Across the full supported matrix, throughput
regression must not exceed 3%, TTFT regression must not exceed 5%, and request
success must remain at or above 99.9%. Otherwise the device route remains
opt-in and the failed threshold is documented rather than waived.

### M5: Release Engineering and Stable v1.0

- [ ] Add source-unit, integration, protocol-compatibility, packaging, and
  documentation checks to pull-request CI; run NPU correctness, fault, soak,
  and performance gates on a controlled hardware workflow.
- [ ] Build versioned wheel/source releases and a separate checksummed asset
  manifest. No model, AIR, weight, profiler, cache, or runtime artifact enters
  the source distribution.
- [ ] Adopt semantic versioning, changelog and deprecation rules, license and
  security reporting documents, support policy, and a tested rollback path.
- [ ] Reproduce the Release Candidate from two independent clean deployments
  and publish the exact commands, compatibility manifest, results, and known
  limitations.
- [ ] Close all correctness, data-corruption, hang, unbounded-resource, and
  installation blockers; lower-severity open issues must be listed in release
  notes with workarounds or explicit support exclusions.

Exit evidence: a user can install the tagged release, run `doctor`, start a
supported OpenAI-compatible server, execute the published correctness and
performance smoke workloads, observe its health, stop it cleanly, and roll
back using only released documentation and artifacts.

### Final Acceptance Rules

Stable v1.0 is accepted only when all M0-M5 checkboxes and their exit evidence
are complete. The 44-test research suite and Attempt 74 performance result
remain necessary regression evidence, but they are not sufficient for a
product release. Synthetic loops, a single successful request, or an
unversioned server workspace may never substitute for API-level correctness,
fault injection, soak, clean deployment, and same-spec performance evidence.
