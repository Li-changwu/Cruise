#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt70a-src
evidence=${root}/evidence-attempt70a-b4-recovery
scratch=/dev/shm/ascend-control-g4-20260726/attempt70a-b4-recovery
prepared=${G4_PREPARED_DIR:-/dev/shm/ascend-control-g4-20260726/attempt69e-r5-b4-resident-epoch}
prepared_evidence=${root}/evidence-attempt69e-r5-b4-resident-epoch
prepared_runner=${G4_PREPARED_RUNNER_DIR:-/dev/shm/ascend-control-g4-20260726/attempt70a-prepared-runner}
raw=${scratch}/raw
cache=${scratch}/cache
build=${scratch}/build
weights=${prepared}/external-weights
cann_logs=${scratch}/cann-logs
air=${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${root}/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json
prepared_result=${prepared_evidence}/attempt69e-r5-result.json
inputs=${raw}/inputs
device_outputs=${raw}/recovery
cases=(invalid-max-steps capacity-exceeded unsupported-sampling \
       unsupported-graph)

guard=${src}/storage_guard/storage_guard.sh
if [[ ! -f "${guard}" || ! -f "${air}" || ! -f "${reference}" ||
      ! -f "${acceptance}" || ! -f "${prepared_result}" ||
      ! -f "${prepared_evidence}/status.tsv" ||
      ! -d "${prepared}/raw/inputs" ||
      ! -d "${weights}" ]] ||
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
storage_guard_preflight "${root}" "${evidence}" "${scratch}" 7 100 24 128
preflight_status=$?
[[ ${preflight_status} -eq 0 ]] || exit ${preflight_status}

for heavy_path in "${raw}" "${cache}" "${build}" "${cann_logs}" \
                  "${inputs}" "${device_outputs}"; do
  storage_guard_assert_scratch_path "${heavy_path}" || exit $?
done
storage_guard_require_child "${weights}" /dev/shm || exit $?
mkdir -p "${raw}" "${cache}" "${cann_logs}" "${device_outputs}"
for case_name in "${cases[@]}"; do
  mkdir -p "${device_outputs}/${case_name}"
done

status=${evidence}/status.tsv
printf 'case\texit_status\n' >"${status}"
finalized=0

capture_after() {
  npu-smi info -t proc-mem -i 7 >"${evidence}/npu7-processes-after.txt" 2>&1 || true
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

record_npu_idle() {
  local label=$1
  local output=${evidence}/npu7-processes-${label}.txt
  if ! npu-smi info -t proc-mem -i 7 >"${output}" 2>&1 ||
     ! grep -Fq 'No process in device.' "${output}"; then
    printf 'npu7-idle-%s\t95\n' "${label}" >>"${status}"
    return 95
  fi
  printf 'npu7-idle-%s\t0\n' "${label}" >>"${status}"
}

run_step() {
  local name=$1 timeout_value=$2
  shift 2
  storage_guard_run_log "${evidence}/${name}.stdout.log" \
    "${evidence}/${name}.stdout.meta.json" "${timeout_value}" -- "$@"
  local step_status=$?
  printf '%s\t%s\n' "${name}" "${step_status}" >>"${status}"
  return "${step_status}"
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
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
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
sha256sum "${air}" "${reference}" "${acceptance}" \
  >"${evidence}/frozen-artifact-integrity.log"

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
   (cd "${src}/host" && sha256sum -c "${prepared_runner}/host-source.sha256" \
      >/dev/null 2>&1); then
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
sha256sum "${build}/g4c_b4_epoch_runner" >"${evidence}/recovery-runner-integrity.log"

if storage_guard_wait_for_npu_idle 7 3600 "${root}" 24 \
   "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"; then
  printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_STATE}" \
    >"${evidence}/npu7-processes-pre-device.txt"
  printf 'npu7-idle-pre-device\t0\n' >>"${status}"
else
  wait_device_status=$?
  printf 'npu7-idle-pre-device\t%s\n' "${wait_device_status}" >>"${status}"
  exit "${wait_device_status}"
fi
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
run_step recovery-suite 10800s "${build}/g4c_b4_epoch_runner" recovery "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4c_b4_resident_epoch_func.json" \
  "${inputs}" "${device_outputs}" 0 0 0 0
recovery_status=$?
if [[ ${recovery_status} -ne 0 ]]; then
  tail -320 "${evidence}/recovery-suite.stdout.log"
  exit ${recovery_status}
fi

run_step compare-recovery 1800s python3 "${src}/compare_recovery.py" \
  --input-dir "${inputs}" --output-dir "${device_outputs}" \
  --output "${evidence}/attempt70a-result.json"
compare_status=$?

find "${inputs}" "${device_outputs}" -type f -print0 |
  sort -z | xargs -0 sha256sum >"${evidence}/scratch-output-integrity.log"
capture_after
if grep -Fq 'No process in device.' "${evidence}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${evidence}/attempt70a-result.json" "${status}" \
  >"${evidence}/result-integrity.log" 2>/dev/null || true
storage_guard_finalize
finalize_status=$?
[[ ${finalize_status} -eq 0 ]] || exit ${finalize_status}
finalized=1
cat "${evidence}/attempt70a-result.json" 2>/dev/null || \
  tail -240 "${evidence}/compare-recovery.stdout.log"
trap - EXIT
[[ ${compare_status} -eq 0 ]] || exit ${compare_status}
[[ ${idle_status} -eq 0 ]] || exit ${idle_status}
storage_guard_cleanup_scratch
