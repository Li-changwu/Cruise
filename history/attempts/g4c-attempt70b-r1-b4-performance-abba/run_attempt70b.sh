#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt70b-r1-src
evidence=${root}/evidence-attempt70b-r1-b4-performance
scratch=/dev/shm/ascend-control-g4-20260726/attempt70b-r1-b4-performance
prepared=${G4_PREPARED_DIR:-/dev/shm/ascend-control-g4-20260726/attempt69e-r5-b4-resident-epoch}
prepared_evidence=${root}/evidence-attempt69e-r5-b4-resident-epoch
prepared_runner=${G4_PREPARED_RUNNER_DIR:-/dev/shm/ascend-control-g4-20260726/attempt70b-r1-prepared-runner}
raw=${scratch}/raw
inputs=${raw}/inputs
perf_outputs=${raw}/perf
caches=${scratch}/cache
build=${scratch}/build
weights=${prepared}/external-weights
cann_logs=${scratch}/cann-logs
controller_workspace=${scratch}/controller
air=${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${root}/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json
prepared_result=${prepared_evidence}/attempt69e-r5-result.json

guard=${src}/storage_guard/storage_guard.sh
if [[ ! -f "${guard}" || ! -f "${air}" || ! -f "${reference}" ||
      ! -f "${acceptance}" || ! -f "${prepared_result}" ||
      ! -f "${prepared_evidence}/status.tsv" ||
      ! -d "${prepared}/raw/inputs" || ! -d "${weights}" ]] ||
   ! grep -q '"pass": true' "${acceptance}" ||
   ! grep -q '"pass": true' "${prepared_result}" ||
   ! grep -Fqx $'device-epoch-suite\t0' "${prepared_evidence}/status.tsv" ||
   [[ $(find "${weights}" -maxdepth 1 -type f | wc -l) -ne 379 ]]; then
  exit 96
fi

source "${guard}"
export STORAGE_GUARD_LARGE_ALLOWLIST=export-attempt67b-b2:export-attempt69c-b4
export STORAGE_GUARD_MAX_SCRATCH_GIB=64
export STORAGE_GUARD_NPU_WAIT_SECONDS=3600
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5
storage_guard_preflight "${root}" "${evidence}" "${scratch}" 7 100 24 128
preflight_status=$?
[[ ${preflight_status} -eq 0 ]] || exit ${preflight_status}

for heavy_path in "${raw}" "${inputs}" "${perf_outputs}" "${caches}" \
                  "${build}" "${cann_logs}" "${controller_workspace}"; do
  storage_guard_assert_scratch_path "${heavy_path}" || exit $?
done
storage_guard_require_child "${weights}" /dev/shm || exit $?
mkdir -p "${raw}" "${perf_outputs}" "${caches}" "${cann_logs}"

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
  storage_guard_wait_for_npu_idle 7 3600 "${root}" 24 \
    "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
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
export G4_EXTERNAL_WEIGHT_DIR=${weights}
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH

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
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH

find "${src}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${evidence}/source-integrity.log"
sha256sum "${air}" "${reference}" "${acceptance}" "${prepared_result}" \
  >"${evidence}/frozen-artifact-integrity.log"

run_step analyzer-selftest 120s python3 "${src}/test_analyze_abba_performance.py"
analyzer_test_status=$?
[[ ${analyzer_test_status} -eq 0 ]] || exit ${analyzer_test_status}

run_step prepare-controller 120s cp -a "${src}/controller" \
  "${controller_workspace}"
controller_status=$?
[[ ${controller_status} -eq 0 ]] || exit ${controller_status}
if [[ ! -f "${controller_workspace}/CMakeLists.txt" ||
      ! -f "${controller_workspace}/g4c_b4_resident_epoch.cpp" ]]; then
  printf 'prepare-controller-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'prepare-controller-integrity\t0\n' >>"${status}"

run_step reuse-prepared 900s bash -c '
  set -euo pipefail
  cp -a --reflink=auto "$1/raw/inputs" "$2/raw/inputs"
' _ "${prepared}" "${scratch}"
reuse_status=$?
[[ ${reuse_status} -eq 0 ]] || exit ${reuse_status}
source_input_bytes=$(storage_guard_used_bytes "${prepared}/raw/inputs")
copied_input_bytes=$(storage_guard_used_bytes "${inputs}")
source_input_files=$(find "${prepared}/raw/inputs" -type f | wc -l)
copied_input_files=$(find "${inputs}" -type f | wc -l)
if [[ ${source_input_bytes} -ne ${copied_input_bytes} ||
      ${source_input_files} -ne ${copied_input_files} ]]; then
  printf 'reuse-prepared-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'reuse-prepared-integrity\t0\n' >>"${status}"

if storage_guard_require_child "${prepared_runner}" /dev/shm &&
   [[ -x "${prepared_runner}/g4c_b4_epoch_runner" &&
      -f "${prepared_runner}/host-source.sha256" ]] &&
   (cd "${src}/host" && sha256sum -c \
      "${prepared_runner}/host-source.sha256" >/dev/null 2>&1); then
  mkdir -p "${build}"
  cp -a "${prepared_runner}/g4c_b4_epoch_runner" "${build}/"
  printf 'reuse-runner\t0\n' >>"${status}"
else
  run_step cmake 600s cmake -S "${src}/host" -B "${build}"
  cmake_status=$?
  [[ ${cmake_status} -eq 0 ]] || exit ${cmake_status}
  run_step build 1800s cmake --build "${build}" --parallel 2
  build_status=$?
  [[ ${build_status} -eq 0 ]] || exit ${build_status}
fi
sha256sum "${build}/g4c_b4_epoch_runner" \
  >"${evidence}/performance-runner-integrity.log"

run_perf_block() {
  local name=$1 route=$2 block=$3 repeats=$4
  wait_npu_ready "pre-${name}" || return $?
  mkdir -p "${perf_outputs}/${name}" "${caches}/${name}" \
    "${cann_logs}/${name}"
  export ASCEND_CACHE_PATH=${caches}/${name}
  export ASCEND_PROCESS_LOG_PATH=${cann_logs}/${name}
  if [[ "${route}" == perf-device-block ]]; then
    export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
  else
    unset RESOURCE_CONFIG_PATH
  fi
  run_step "performance-${name}" 10800s \
    "${build}/g4c_b4_epoch_runner" "${route}" "${air}" \
    "${src}/config/graph_config.json" \
    "${src}/config/g4c_b4_resident_epoch_func.json" \
    "${inputs}" "${perf_outputs}/${name}" "${block}" "${repeats}" 0 0
}

run_perf_block h1 perf-host-block 1 8 || exit $?
run_perf_block d1 perf-device-block 1 8 || exit $?
run_perf_block d2 perf-device-block 2 7 || exit $?
run_perf_block h2 perf-host-block 2 7 || exit $?

run_step analyze-abba-performance 1800s python3 \
  "${src}/analyze_abba_performance.py" \
  --h1 "${perf_outputs}/h1/perf-block.tsv" \
  --d1 "${perf_outputs}/d1/perf-block.tsv" \
  --d2 "${perf_outputs}/d2/perf-block.tsv" \
  --h2 "${perf_outputs}/h2/perf-block.tsv" \
  --output "${evidence}/attempt70b-r1-result.json"
analysis_status=$?

find "${inputs}" "${perf_outputs}" -type f -print0 | sort -z |
  xargs -0 sha256sum >"${evidence}/scratch-output-integrity.log"
wait_npu_ready final
idle_status=$?
capture_after
sha256sum "${evidence}/attempt70b-r1-result.json" "${status}" \
  >"${evidence}/result-integrity.log" 2>/dev/null || true
storage_guard_finalize
finalize_status=$?
[[ ${finalize_status} -eq 0 ]] || exit ${finalize_status}
finalized=1
cat "${evidence}/attempt70b-r1-result.json" 2>/dev/null ||
  tail -240 "${evidence}/analyze-abba-performance.stdout.log"
trap - EXIT
[[ ${analysis_status} -eq 0 ]] || exit ${analysis_status}
[[ ${idle_status} -eq 0 ]] || exit ${idle_status}
storage_guard_cleanup_scratch
