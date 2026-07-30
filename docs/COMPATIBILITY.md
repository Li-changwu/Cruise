# Compatibility and Contract Matrix

Cruise compatibility has two separate layers. `capability_requirements`
describes host-independent prerequisites such as architecture, supported NPU
products, DataFlow, the device cross-compiler, device health and shared-memory
capacity. A profile records one exact software and asset combination together
with its validation status and evidence. Matching the capability layer alone
does not make an untested stack supported.

The machine-readable source is
`src/vllm_ascend_resident_epoch/compatibility.json`. No hostname appears in the
contract: a different server passes when it provides the required capabilities
and exactly matches a selected, qualified profile. `cruise doctor` rejects all
other combinations before model loading.

## Required Capabilities

The current product line requires aarch64, at least one healthy and idle Ascend
910B2, importable CANN DataFlow Python support, `meta_flow_func.h`,
`aarch64-target-linux-gnu-g++`, an Ascend Triton runtime exposing either the
legacy `triton.language.extra.ascend.libdevice.pow` name or the 3.2.1
`triton.language.extra.cann.libdevice.pow` name, and at least 32 GiB free on
`/dev/shm`. Cruise installs an in-process `ascend -> cann` namespace alias
before loading vLLM-Ascend when only the latter exists; no environment file is
modified. Adding a new accelerator model is a manifest and qualification
change, not a hostname special case.

## Formally Validated Profile

Profile: `attempt74-910b2-cann851-r5`

| Component | Exact validated value |
|---|---|
| Architecture | aarch64 |
| Accelerator | one Ascend 910B2 |
| npu-smi | 25.2.1 |
| Ascend driver | 25.2.1 |
| Python | 3.11.15 |
| CANN | 8.5.1 |
| torch / torch-npu | 2.9.0 / 2.9.0 |
| vLLM | `0.20.1.post1.dev363+gec4847981`, commit `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7` |
| vLLM-Ascend | `0.19.1.post1.dev254+g9cf69cacf`, commit `e967f235ba66edb48a28a6d943aee9455fee70cf` |
| Model | content-addressed Qwen2.5-7B-Instruct snapshot |
| Workload | TP=PP=1, B<=4, K in 1/2/4/8, greedy, one-token prompt |

The model snapshot is identified by its configuration SHA256
`7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`
and safetensors index SHA256
`624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028`.
The 342-file, 15,231,237,408-byte runtime-weight set is identified by manifest
SHA256
`2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761`.
AIR, tiling, controller, graph, and baseline hashes are recorded in the
machine-readable profile and the Attempt 74 evidence.

## NPU0-1 Qualified Profile

Profile `attempt74-910b2-cann900-npu01-r1` records the NPU0-1 qualification
target: Ascend 910B2, CANN/driver/npu-smi 9.0.0/26.0.rc1/26.0.rc1, Python
3.11.15, torch-npu 2.9.0.post2, and the declared vLLM/vLLM-Ascend commits. Two
independent real-model install, deep-doctor, start, stop, cleanup, and uninstall
cycles passed on physical NPU 0. Its status is therefore
`m0-qualified-developer-preview`. This qualifies the bounded M0 deployment
contract; it does not establish Stable v1.0 serving semantics.

## Versioned Contracts

| Contract | Version or shape |
|---|---:|
| Python scheduler/result contract | 3 |
| Runtime configuration schema | 1 |
| Sidecar wire protocol | 5 |
| Sidecar request / response | 136 B / 352 B |
| Host-UDF ABI | 2, 8 inputs / 2 outputs |
| Internal decoder ABI | 1, 9 inputs / 4 outputs |

Python constants live in `version.py`; the native wire constants live in
`native/resident_epoch_protocol.h`. Dependency-light tests require both copies
and the packaged compatibility manifest to agree. The sidecar rejects a wire
protocol mismatch, while runtime configuration validation rejects incompatible
Host-UDF tensor counts and asset identities before model execution.

## Validation Levels

The checks are deliberately split so installation can be diagnosed without
copying model data:

```bash
cruise smoke
cruise doctor --mode source

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH:-}"
cruise doctor --mode npu \
  --profile attempt74-910b2-cann851-r5 \
  --device 7

cruise doctor --mode runtime --config /etc/cruise/cruise.json
cruise doctor --mode runtime --config /etc/cruise/cruise.json --deep
```

`source` checks only packaged contracts. `npu` first checks the portable
capabilities, then the exact selected profile. It also rejects an unavailable,
unhealthy, unsupported, or occupied device. `runtime` additionally checks
paths, executable permissions, UDF shape, hashes, weight count/bytes, OPP
layout and scratch capacity. `--deep` hashes every external-weight file and is
intended for deployment qualification rather than every process start.

Every failed JSON check includes a stable `code`, `expected`, `observed`, and
`remediation`. The major rejection classes are unsupported architecture or
accelerator, insufficient or unavailable NPU, device health or occupancy,
missing/inactive DataFlow, missing CANN headers/compiler, incompatible package
or CANN/driver versions, a missing Ascend Triton runtime, insufficient shared
memory, and missing or non-executable runtime assets. A missing runtime bundle is reported as
`missing-runtime-asset` with the exact expected path and an instruction to
provision the profile's content-addressed bundle; substituting another model
or artifact is never treated as remediation. A candidate profile adds an
`unvalidated-compatibility-profile` warning even when all checks pass.

## M1 Validated Extension

The 2026-07-31 M1 exit evidence closes the declared narrow serving contract on
NPU0-1 physical NPU 0: Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling,
greedy decoding, and the content-addressed runtime bundle recorded by the
profile. The differential covers 1,000 deterministic requests in 400 cohorts,
B=1,2,3,4, prompt lengths 2-5, output budgets 2-7, EOS, cancellation,
unsupported `min_tokens`, and generation-checked Paged-KV row reuse. It also
passes the eight-case OpenAI API semantic matrix for streaming/non-streaming,
single/batch, EOS, `max_tokens`, unsupported fallback, disconnect, and a
post-disconnect probe.

This is a correctness boundary, not a general compatibility claim. It does
not establish arbitrary prompt lengths, non-greedy sampling, LoRA, TP/PP,
multimodal serving, long-running soak, or performance qualification.

## Claim Boundary

The CANN 8.5.1 profile is a formally validated research baseline, not Stable
v1.0. The CANN 9.0.0 NPU0-1 profile is qualified for the bounded M0/M1
Developer Preview correctness contract, while M2-M5 remain open. Adding
another version or accelerator requires its own profile and same-spec
correctness evidence; broad version ranges must not be inferred.
