# Developer Preview Installation

This procedure installs the current Cruise research baseline without modifying
vLLM or vLLM-Ascend source files. It produces the qualified Developer Preview
control path. The narrow M1 OpenAI-compatible semantic contract is accepted on
the published NPU0-1 profile, but Stable v1.0 operations remain future work.

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
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH:-}"
cruise doctor --mode npu \
  --profile attempt74-910b2-cann851-r5 \
  --device 7
```

Some CANN packages install DataFlow below the toolkit but do not add it to the
active Conda environment. In that case `doctor` returns
`inactive-device-control-runtime` with the exact detected `PYTHONPATH` fix.
Do not switch to a system Python with a different ABI; activate the profile's
Python first, then expose the matching CANN site-packages directory.

Build the native sidecar outside the source tree:

```bash
cmake -S native -B /dev/shm/cruise-native-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /dev/shm/cruise-native-build --parallel
```

Provision the immutable 342-file AIR runtime-weight bundle once on a dedicated
data volume. The destination is fixed by the qualified manifest digest; a
second invocation deep-checks and reuses the same bundle instead of creating
another 15.2 GB copy:

```bash
asset_root=/data/cruise-assets
weight_digest=2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761
python materialize_runtime_weights.py \
  --model-dir /data/models/Qwen2.5-7B-Instruct-a09a354 \
  --persistent-asset-root "${asset_root}" \
  --output-dir "${asset_root}/runtime-weights/${weight_digest}" \
  --manifest "${asset_root}/manifests/${weight_digest}.json"
```

This command never downloads a model. It accepts only the frozen model config
and index hashes, refuses a non-empty unmarked asset root, writes through a
process-specific staging directory, removes that staging directory on normal
failure, and publishes only the expected manifest SHA256. Use the exact
resulting weight path when relocating the AIR; changing the path requires a new
relocated AIR and `air_sha256`.

Controller packaging, native builds, AIR relocation output, GraphPp-generated
weights, caches, logs, and sockets remain per-cycle artifacts below marked
`/dev/shm` scratch. Fill a machine-specific copy of
`config/cruise.example.json` with the persistent weight bundle and temporary
deployment paths. Never edit the tracked example with host paths.

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

The current command is the accepted EngineCore qualification harness. The
OpenAI API semantic gate is exercised by the versioned M1 differential runner;
an unattended general-purpose API-server quickstart remains deferred to the
M2/M3 lifecycle, observability, and operational milestones.

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
