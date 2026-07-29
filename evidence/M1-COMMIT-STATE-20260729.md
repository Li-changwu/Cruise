# M1 Commit-State Checkpoint

Date: 2026-07-29 (Asia/Shanghai)

## Scope

This checkpoint qualifies the first M1 increment at commit `857c007`. It
establishes the execution ownership and replay boundary; it is not evidence of
complete prefill-to-decode serving semantics.

## Contract

Each resident request declares its pre-epoch state owner as `host` or `device`.
Each native epoch response carries one of:

| State | Meaning | Host replay |
|---|---|---|
| `prepared` | validation and input construction completed; Feed was not submitted | permitted only when every request is Host-owned |
| `executing` | Feed may have submitted or device mutation may have started | forbidden |
| `committed` | Fetch and output validation completed | forbidden; result is already authoritative |

The sidecar wire contract is v4 with a 128 B request and 352 B response. The
native bridge sets `executing` immediately before Feed and `committed` only
after Fetch and output-shape/token-control validation. A socket error after
submission is classified as `executing` because partial delivery cannot prove
that the device did not mutate.

## Controlled Evidence

| Check | Result |
|---|---|
| Dependency-light suite | 50 passed |
| Full `vllm-hust-dev` suite | 70 passed, 4 upstream deprecation warnings |
| Prepared failure with Host-owned plan | Host fallback attached with one-step accounting |
| Prepared failure with device-owned row | rejected, no replay |
| Executing/committed failure | rejected, no replay |
| Native source transition order | prepared < executing < Feed < Fetch < committed |
| CANN 8.5.1 native build | new/old bridge, new/old server, trace library and AIR relocation target built |
| ABI verifier and repository audit | passed |

The build emitted only the existing GE `LoadFromFile`/`SaveToFile` deprecation
warnings. No model execution was claimed in this checkpoint because the M0
external-asset provisioning gate remains open.

## Boundary

This increment prevents the most dangerous M1 regression, namely replaying a
request after an ambiguous device mutation. It does not yet provide real
multi-token prefill, host-to-device KV ownership transfer, streaming/non-streaming
API responses, continuous admission at epoch boundaries, disconnect/cancellation
handling, or NPU fault-injection evidence. Those claims remain unchecked in
`PROJECT_HISTORY.md`.
