# Compatibility and Contract Matrix

Cruise uses an explicit compatibility profile. A profile is an exact record of
one validated stack, not a promise that nearby versions will work. The
machine-readable source is
`src/vllm_ascend_resident_epoch/compatibility.json`; `cruise doctor` rejects a
mismatch before the runtime path is started.

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
cruise doctor --mode npu \
  --profile attempt74-910b2-cann851-r5 \
  --device 7

cruise doctor --mode runtime --config /etc/cruise/cruise.json
cruise doctor --mode runtime --config /etc/cruise/cruise.json --deep
```

`source` checks only packaged contracts. `npu` checks the exact software stack,
architecture, NPU product and health. `runtime` additionally checks paths,
executable permissions, UDF shape, hashes, weight count/bytes, OPP layout and
scratch capacity. `--deep` hashes every external-weight file and is intended
for deployment qualification rather than every process start.

## M1 Validated Extension

The 2026-07-29 M1 checkpoints extend the validated correctness envelope to
simultaneous stock-vLLM prefills at B=1,2,3,4, prompt lengths 2-5, and greedy
output budgets 2-5. Each admitted request imports one 128-token Paged-KV block
for all 28 layers exactly once. Mixed-budget B=3 and B=4 cohorts shrink as
requests complete while surviving rows keep their generations and continue
through the steady 8-input/2-output ABI. This bounded matrix does not establish
arbitrary prompt lengths, admission of new requests into a Device-owned cohort,
or row reuse after a nontrivial prefill.

## Claim Boundary

This profile is a formally validated research baseline, not Stable v1.0. CANN
9.0.0 supported earlier feasibility experiments, but it is not part of this
exact product profile. Adding another version requires its own profile and
same-spec correctness evidence; broad version ranges must not be inferred.
