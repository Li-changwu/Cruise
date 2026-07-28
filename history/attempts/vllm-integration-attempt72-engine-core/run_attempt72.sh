#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt72-src
evidence=${root}/evidence-attempt72r1-engine-core
scratch=/dev/shm/a72r1
build=${scratch}/native-build
controller_workspace=${scratch}/controller
runtime_config=${scratch}/config/resident_epoch_func.json
weights=${scratch}/external-weights
runtime_export=${scratch}/runtime-export
runtime_air=${scratch}/qwen_b4_decoder_step_attempt72r1.air
cache=${scratch}/cache
cann_logs=${scratch}/cann-logs
tmp=${scratch}/t
frozen_air=${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
tiling=${root}/raw-attempt69d-r1-b4-native/native-inputs/case0/explicit_tiling.bin
baseline=${root}/evidence-attempt69e-r5-b4-resident-epoch/attempt69e-r5-result.json
model_config=${src}/tests/fixtures/qwen2-7b-config
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
old_weight_prefix=${root}/export-attempt69c-b4

guard=${src}/storage_guard/storage_guard.sh
if [[ ! -f "${guard}" || ! -f "${frozen_air}" || ! -f "${tiling}" ||
      ! -f "${baseline}" || ! -f "${model_config}/config.json" ||
      ! -f "${model}/model.safetensors.index.json" ||
      ! -f "${src}/materialize_runtime_weights.py" ||
      ! -f "${src}/run_engine_core_native.py" ||
      ! -f "${src}/verify_engine_core_result.py" ||
      ! -f "${src}/controller/g4c_b4_resident_epoch.cpp" ||
      ! -f "${src}/controller/CMakeLists.txt" ||
      ! -f "${src}/native/relocate_air_paths.cpp" ]] ||
   ! grep -q '"pass": true' "${baseline}"; then
  exit 96
fi

source "${guard}"
export STORAGE_GUARD_LARGE_ALLOWLIST=export-attempt67b-b2:export-attempt69c-b4
export STORAGE_GUARD_MAX_SCRATCH_GIB=64
export STORAGE_GUARD_NPU_WAIT_SECONDS=21600
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5
storage_guard_preflight "${root}" "${evidence}" "${scratch}" 7 100 24 128
preflight_status=$?
[[ ${preflight_status} -eq 0 ]] || exit ${preflight_status}

for heavy_path in "${build}" "${controller_workspace}" "${weights}" \
                  "${runtime_export}" "${runtime_air}" "${cache}" \
                  "${cann_logs}" "${tmp}" "${runtime_config}"; do
  storage_guard_assert_scratch_path "${heavy_path}" || exit $?
done
mkdir -p "${build}" "${weights}" "${cache}" "${cann_logs}" "${tmp}" \
  "$(dirname -- "${runtime_config}")"

status=${evidence}/status.tsv
printf 'case\texit_status\n' >"${status}"
finalized=0

capture_after() {
  npu-smi info -t proc-mem -i 7 >"${evidence}/npu7-processes-after.txt" \
    2>&1 || true
  npu-smi info -t usages -i 7 >"${evidence}/npu7-usages-after.txt" \
    2>&1 || true
  du -sh "${weights}" >"${evidence}/external-weights-size.txt" 2>&1 || true
  find "${weights}" -type f -printf '%P\t%s bytes\n' | sort \
    >"${evidence}/external-weights-files.txt" 2>&1 || true
  storage_guard_snapshot exit "${evidence}/storage-exit.tsv" || true
}

on_exit() {
  local exit_status=$?
  trap - EXIT
  set +e
  capture_after
  if [[ ${finalized} -eq 0 ]]; then
    storage_guard_finalize
  fi
  exit "${exit_status}"
}
trap on_exit EXIT

run_step() {
  local name=$1 timeout_value=$2
  shift 2
  storage_guard_run_log "${evidence}/${name}.stdout.log" \
    "${evidence}/${name}.stdout.meta.json" "${timeout_value}" -- "$@"
  local step_status=$?
  printf '%s\t%s\n' "${name}" "${step_status}" >>"${status}"
  return "${step_status}"
}

wait_npu_ready() {
  local label=$1
  storage_guard_wait_for_npu_idle 7 "${STORAGE_GUARD_NPU_WAIT_SECONDS}" \
    "${root}" 24 "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
    "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"
  local ready_status=$?
  if [[ ${ready_status} -eq 0 ]]; then
    printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_STATE}" \
      >"${evidence}/npu7-processes-${label}.txt"
    printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_USAGE}" \
      >"${evidence}/npu7-usages-${label}.txt"
    printf 'npu7-ready-%s\t0\n' "${label}" >>"${status}"
    return 0
  fi
  printf 'npu7-ready-%s\t%s\n' "${label}" "${ready_status}" >>"${status}"
  return "${ready_status}"
}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_PROCESS_LOG_PATH=${cann_logs}
export ASCEND_CACHE_PATH=${cache}
export TMPDIR=${tmp}
export TORCHINDUCTOR_CACHE_DIR=${cache}/torchinductor
export TRITON_CACHE_DIR=${cache}/triton
export XDG_CACHE_HOME=${cache}/xdg
export PYTHONDONTWRITEBYTECODE=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json

custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f \
  -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f \
  -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" &&
   -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}

export PYTHONPATH=${src}/src:${src}:/root/vllm-hust:/root/vllm-ascend-hust:${PYTHONPATH:-}
export VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY=vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine
export VLLM_ASCEND_RESIDENT_EPOCH_AIR=${runtime_air}
export VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG=${src}/config/graph_config.json
export VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG=${runtime_config}
export VLLM_ASCEND_RESIDENT_EPOCH_TILING=${tiling}
export VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS=${weights}
export VLLM_ASCEND_RESIDENT_EPOCH_LIBRARY=${build}/libresident_epoch_bridge.so
export VLLM_ASCEND_RESIDENT_EPOCH_SERVER=${build}/resident_epoch_server
export VLLM_ASCEND_RESIDENT_EPOCH_SOCKET=${scratch}/resident-epoch.sock
export VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT=3600
export VLLM_ASCEND_RESIDENT_EPOCH_STEPS=8
export VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY=8

find "${src}" -type f ! -path '*/__pycache__/*' \
  ! -path '*/.pytest_cache/*' -print0 | sort -z | xargs -0 sha256sum \
  >"${evidence}/source-integrity.log"
sha256sum "${frozen_air}" "${tiling}" "${baseline}" \
  >"${evidence}/frozen-artifact-integrity.log"
readlink -f "${model}"/model-*.safetensors | sort \
  >"${evidence}/model-shards.txt"
sha256sum "${model}/config.json" "${model}/model.safetensors.index.json" \
  >"${evidence}/model-metadata-integrity.log"
git -C /root/vllm-hust rev-parse HEAD >"${evidence}/vllm-commit.txt"
git -C /root/vllm-ascend-hust rev-parse HEAD \
  >"${evidence}/vllm-ascend-commit.txt"

run_step cann-python-smoke 120s python3 -c 'import tbe'
cann_python_status=$?
[[ ${cann_python_status} -eq 0 ]] || exit ${cann_python_status}

run_step unit-tests 300s python3 -m pytest -q "${src}/tests"
unit_status=$?
[[ ${unit_status} -eq 0 ]] || exit ${unit_status}

run_step prepare-controller 120s cp -a "${src}/controller" \
  "${controller_workspace}"
controller_status=$?
[[ ${controller_status} -eq 0 ]] || exit ${controller_status}

run_step prepare-runtime-config 120s python3 \
  "${src}/prepare_runtime_config.py" \
  --template "${src}/config/resident_epoch_func.json" \
  --controller-workspace "${controller_workspace}" \
  --output "${runtime_config}"
config_status=$?
[[ ${config_status} -eq 0 ]] || exit ${config_status}

run_step cmake 600s cmake -S "${src}/native" -B "${build}"
cmake_status=$?
[[ ${cmake_status} -eq 0 ]] || exit ${cmake_status}
run_step build 1800s cmake --build "${build}" --parallel 2
build_status=$?
[[ ${build_status} -eq 0 ]] || exit ${build_status}
sha256sum "${build}/libresident_epoch_bridge.so" \
  "${build}/resident_epoch_server" "${build}/relocate_air_paths" \
  >"${evidence}/native-bridge-integrity.log"

run_step materialize-runtime-weights 3600s python3 \
  "${src}/materialize_runtime_weights.py" \
  --model-dir "${model}" \
  --output-dir "${runtime_export}" \
  --manifest "${evidence}/runtime-weights-manifest.json"
materialize_status=$?
[[ ${materialize_status} -eq 0 ]] || exit ${materialize_status}

runtime_weight_file_count=$(find "${runtime_export}" -maxdepth 1 -type f \
  ! -name '*.*' | wc -l)
runtime_weight_bytes=$(find "${runtime_export}" -maxdepth 1 -type f \
  ! -name '*.*' -printf '%s\n' | awk '{total += $1} END {print total + 0}')
printf 'files\t%s\nbytes\t%s\n' "${runtime_weight_file_count}" \
  "${runtime_weight_bytes}" >"${evidence}/runtime-weights-summary.tsv"
if [[ ${runtime_weight_file_count} -ne 342 || \
      ${runtime_weight_bytes} -ne 15231237408 ]]; then
  printf 'runtime-weights-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'runtime-weights-integrity\t0\n' >>"${status}"
find "${runtime_export}" -maxdepth 1 -type f ! -name '*.*' -print0 | \
  sort -z | xargs -0 sha256sum \
  >"${evidence}/runtime-weights-integrity.log"

run_step relocate-runtime-air 600s "${build}/relocate_air_paths" \
  "${frozen_air}" "${runtime_air}" "${old_weight_prefix}" \
  "${runtime_export}" "${evidence}/attempt72r1-air-relocation.json"
relocate_status=$?
[[ ${relocate_status} -eq 0 ]] || exit ${relocate_status}
sha256sum "${runtime_air}" \
  >"${evidence}/runtime-air-integrity.log"

wait_npu_ready pre-native || exit $?
run_step engine-core-native 10800s python3 "${src}/run_engine_core_native.py" \
  --model-config "${model_config}" \
  --baseline-result "${baseline}" \
  --output "${evidence}/attempt72r1-engine-core-result.json"
native_status=$?
[[ ${native_status} -eq 0 ]] || exit ${native_status}

run_step engine-core-result-verify 120s python3 \
  "${src}/verify_engine_core_result.py" \
  "${evidence}/attempt72r1-engine-core-result.json"
verify_status=$?
[[ ${verify_status} -eq 0 ]] || exit ${verify_status}

weight_file_count=$(find "${weights}" -maxdepth 1 -type f | wc -l)
weight_bytes=$(storage_guard_used_bytes "${weights}")
printf 'files\t%s\nbytes\t%s\n' "${weight_file_count}" "${weight_bytes}" \
  >"${evidence}/external-weights-summary.tsv"
if [[ ${weight_file_count} -ne 379 || ${weight_bytes} -lt 15000000000 ||
      ${weight_bytes} -gt 16000000000 ]]; then
  printf 'external-weights-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'external-weights-integrity\t0\n' >>"${status}"

wait_npu_ready final
idle_status=$?
capture_after
sha256sum "${evidence}/attempt72r1-engine-core-result.json" "${status}" \
  "${evidence}/native-bridge-integrity.log" \
  >"${evidence}/result-integrity.log" 2>/dev/null || true
storage_guard_finalize
finalize_status=$?
[[ ${finalize_status} -eq 0 ]] || exit ${finalize_status}
finalized=1
cat "${evidence}/attempt72r1-engine-core-result.json"
trap - EXIT
[[ ${idle_status} -eq 0 ]] || exit ${idle_status}
storage_guard_cleanup_scratch
