# Developer Preview Installation

This procedure installs the current Cruise research baseline without modifying
vLLM or vLLM-Ascend source files. It produces a Developer Preview control path,
not the OpenAI-compatible server promised by M1.

## Prerequisites

- a machine matching a published compatibility profile;
- the matching vLLM and vLLM-Ascend environment;
- CANN DataFlow headers, libraries and compiler;
- the content-addressed model, AIR, tiling, custom OPP and runtime-weight
  artifacts recorded by the profile;
- at least the configured free capacity on `/dev/shm`.

Large model, compiler and runtime artifacts are external assets. They are not
downloaded by `pip` and must not be copied into this repository or a persistent
`ascend-control-*` experiment clone.

## Install

Build and install a wheel from a clean checkout in the target environment:

```bash
git clone https://github.com/Li-changwu/Cruise.git
cd Cruise
conda activate vllm-hust-dev
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install --no-deps dist/vllm_ascend_resident_epoch-*.whl
cruise smoke
```

Check the NPU stack before preparing model assets:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cruise doctor --mode npu \
  --profile attempt74-910b2-cann851-r5 \
  --device 7
```

Build the native sidecar outside the source tree:

```bash
cmake -S native -B /dev/shm/cruise-native-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /dev/shm/cruise-native-build --parallel
```

Controller packaging, AIR relocation and runtime-weight materialization remain
the same content-checked steps used by `run_attempt74.sh`. Put their generated
outputs under a marker-protected `/dev/shm` asset root, retain the manifest on
persistent storage, and fill a machine-specific copy of
`config/cruise.example.json`. Never edit the tracked example with host paths.

## Validate and Start

```bash
cruise config validate /etc/cruise/cruise.json --check-paths
cruise doctor --mode runtime --config /etc/cruise/cruise.json

cruise run --config /etc/cruise/cruise.json -- \
  python /path/to/Cruise/run_engine_core_native.py \
    --model-config /path/to/model/config-only \
    --baseline-result /path/to/attempt69e-r5-result.json \
    --output /path/to/bounded-evidence/result.json
```

The current command is the accepted EngineCore qualification harness. A
general API-server command is deliberately not documented as supported until
M1 closes real prefill, continuous admission, cancellation and fallback.

`cruise run` validates the runtime configuration again, sources CANN, builds
the internal environment, starts the requested command in a new process group,
forwards SIGINT/SIGTERM, and removes its marked per-run scratch directory on
both success and failure.

## Upgrade and Rollback

Build the new wheel from a clean commit, run `cruise smoke` and `doctor`, stop
the old process, then install the wheel. Asset and protocol incompatibility is
rejected before startup. Until tagged releases exist, rollback is commit based:

```bash
git switch --detach <accepted-commit>
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install --no-deps --force-reinstall dist/*.whl
cruise doctor --mode runtime --config /etc/cruise/cruise.json
```

Do not reuse an AIR, UDF, sidecar or weight manifest across a failed contract
check.

## Stop and Uninstall

Send SIGTERM or Ctrl-C to the `cruise run` process. After the child and sidecar
exit, remove the empty marker-managed root and uninstall the package:

```bash
cruise cleanup --config /etc/cruise/cruise.json
python -m pip uninstall vllm-ascend-resident-epoch
```

`cruise cleanup` refuses recursive deletion. It removes only an empty scratch
root with the exact Cruise marker; active run directories or unknown files are
left untouched and reported as an error.

