# M0 Product-Readiness Checkpoint

Date: 2026-07-29 (Asia/Shanghai)

## Scope

This checkpoint qualifies the product-contract implementation through Cruise
commit `36e21155a879e8fb173a5608457fc676242f269a`. It does not close M0 and does
not claim a Developer Preview serving product. The remaining gate is a repeated
real-model `install -> runtime doctor -> start -> stop -> cleanup` cycle using
the new configuration path.

## Contract and Test Evidence

| Check | Result |
|---|---|
| Dependency-light suite | 44 passed on Windows with Python 3.11 |
| Full frozen-environment suite | 60 passed, 4 upstream deprecation warnings |
| Minimal Host-UDF ABI verifier | Passed; 8 inputs, 2 outputs, 628 declared bytes |
| Repository audit | Passed; 1,018 files, 5.64 MiB |
| GitHub Actions for asset-profile binding | Passed ([run 30424328813](https://github.com/Li-changwu/Cruise/actions/runs/30424328813)) |

The seven profile-identity negative cases independently changed graph config,
tiling, external-weight manifest, external-weight file count, external-weight
bytes, model config, and model index. Every mismatch was rejected while the
runtime configuration was loaded. AIR remains deployment-specific because
weight-path relocation changes its bytes, but its configured SHA256 remains
mandatory and is checked against the deployed file.

## Supported-Machine Diagnosis

`cruise doctor --mode npu` passed on one Ascend 910B2 using the declared
`attempt74-910b2-cann851-r5` profile:

- aarch64 and healthy physical device 7;
- Python 3.11.15, CANN 8.5.1, torch/torch-npu 2.9.0;
- exact vLLM and vLLM-Ascend development versions from the profile;
- npu-smi 25.2.1 and an independently reported Ascend driver 25.2.1.

The only diagnostic warning was the intentional
`formally-validated-research-baseline` maturity status.

## Wheel Qualification

Commit `36e2115` produced
`vllm_ascend_resident_epoch-0.1.0-py3-none-any.whl`:

- size: 38,769 bytes;
- SHA256: `168943b069bcb7ecc5d1a0ca1248b354905d3ad19916a30ff872e62a832a536b`;
- installed with `pip install --no-deps` into an isolated venv below `/dev/shm`;
- `cruise version --json` and `cruise smoke --json` passed;
- the console entry point and packaged `compatibility.json` were present;
- the packaged profile contained the explicit driver identity;
- installation metadata reported `editable=False`;
- `pip uninstall` removed both the command and importable package.

The marker-checked wheel/venv directory and the earlier native-build directory
were removed after validation. No `cruise-m0-*` path remained under `/dev/shm`,
and the server checkout remained clean at the tested commit.

## Open M0 Exit Condition

The large experimental trees were intentionally removed from the root disk,
and the source repository correctly excludes AIR, tiling, generated controller
packages, and 15.2 GB of external weights. Consequently, this checkpoint did
not recreate those assets merely to claim a lifecycle pass.

Before M0 closes, Cruise must provide a reproducible, content-addressed asset
provisioning step outside the root disk and execute two consecutive real-model
cycles through the documented configuration. Each cycle must pass deep runtime
diagnosis, start and stop the current EngineCore/sidecar path, remove all
marked scratch, leave no process or socket, preserve a clean checkout, and
retain less than 100 MiB of persistent runtime output.
