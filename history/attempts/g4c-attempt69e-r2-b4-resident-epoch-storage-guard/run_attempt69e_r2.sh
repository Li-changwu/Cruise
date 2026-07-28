#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt69e-r2-src
evidence=${root}/evidence-attempt69e-r2-b4-resident-epoch
scratch=/dev/shm/ascend-control-g4-20260726/attempt69e-r2-b4-resident-epoch
raw=${scratch}/raw
cache=${scratch}/cache
build=${scratch}/build
weights=${scratch}/external-weights
cann_logs=${scratch}/cann-logs
air=${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${root}/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json
inputs=${raw}/inputs
host_outputs=${raw}/host
device_outputs=${raw}/device
cases=(k1-heterogeneous k2-heterogeneous k4-heterogeneous k8-all-active \
       active-empty-alternating finished-active-empty-active \
       independent-early-eos)

guard=${src}/storage_guard/storage_guard.sh
if [[ ! -f "${guard}" || ! -f "${air}" || ! -f "${reference}" ||
      ! -f "${acceptance}" ]] || ! grep -q '"pass": true' "${acceptance}"; then
  exit 96
fi
source "${guard}"
export STORAGE_GUARD_LARGE_ALLOWLIST=export-attempt67b-b2:export-attempt69c-b4
export STORAGE_GUARD_MAX_SCRATCH_GIB=192
storage_guard_preflight "${root}" "${evidence}" "${scratch}" 7 50 32 64
preflight_status=$?
[[ ${preflight_status} -eq 0 ]] || exit ${preflight_status}

for heavy_path in "${raw}" "${cache}" "${build}" "${weights}" "${cann_logs}" \
                  "${inputs}" "${host_outputs}" "${device_outputs}"; do
  storage_guard_assert_scratch_path "${heavy_path}" || exit $?
done
mkdir -p "${raw}" "${cache}" "${build}" "${weights}" "${cann_logs}" \
  "${host_outputs}" "${device_outputs}"
for case_name in "${cases[@]}"; do
  mkdir -p "${host_outputs}/${case_name}" "${device_outputs}/${case_name}"
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

run_step prepare-inputs 1800s python3 "${src}/prepare_inputs.py" \
  --reference "${reference}" --output-dir "${inputs}"
prepare_status=$?
[[ ${prepare_status} -eq 0 ]] || exit ${prepare_status}

run_step cmake 600s cmake -S "${src}/host" -B "${build}"
cmake_status=$?
[[ ${cmake_status} -eq 0 ]] || exit ${cmake_status}
run_step build 1800s cmake --build "${build}" --parallel 2
build_status=$?
[[ ${build_status} -eq 0 ]] || exit ${build_status}

record_npu_idle pre-host || exit $?
run_step host-epoch-suite 10800s "${build}/g4c_b4_epoch_runner" host "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4c_b4_resident_epoch_func.json" \
  "${inputs}" "${host_outputs}" -1 -1 -1 -1
host_status=$?
if [[ ${host_status} -ne 0 ]]; then
  tail -240 "${evidence}/host-epoch-suite.stdout.log"
  exit ${host_status}
fi
record_npu_idle after-host || exit $?

mapfile -t early_eos <"${host_outputs}/early_eos_tokens.txt"
if [[ ${#early_eos[@]} -ne 4 ]]; then
  printf 'early-eos-tokens\t93\n' >>"${status}"
  exit 93
fi
for eos_value in "${early_eos[@]}"; do
  if ! [[ ${eos_value} =~ ^[0-9]+$ ]]; then
    printf 'early-eos-tokens\t93\n' >>"${status}"
    exit 93
  fi
done
printf 'early-eos-tokens\t0\n' >>"${status}"

record_npu_idle pre-device || exit $?
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
run_step device-epoch-suite 10800s "${build}/g4c_b4_epoch_runner" device "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4c_b4_resident_epoch_func.json" \
  "${inputs}" "${device_outputs}" "${early_eos[@]}"
device_status=$?
if [[ ${device_status} -ne 0 ]]; then
  tail -320 "${evidence}/device-epoch-suite.stdout.log"
  exit ${device_status}
fi

run_step compare 1800s python3 "${src}/compare_epochs.py" \
  --host-dir "${host_outputs}" --device-dir "${device_outputs}" \
  --input-dir "${inputs}" --eager-reference "${reference}" \
  --output "${evidence}/attempt69e-r2-result.json"
compare_status=$?

find "${inputs}" "${host_outputs}" "${device_outputs}" -type f -print0 |
  sort -z | xargs -0 sha256sum >"${evidence}/scratch-output-integrity.log"
capture_after
if grep -Fq 'No process in device.' "${evidence}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${evidence}/attempt69e-r2-result.json" "${status}" \
  >"${evidence}/result-integrity.log" 2>/dev/null || true
storage_guard_finalize
finalize_status=$?
[[ ${finalize_status} -eq 0 ]] || exit ${finalize_status}
finalized=1
cat "${evidence}/attempt69e-r2-result.json" 2>/dev/null || \
  tail -240 "${evidence}/compare.stdout.log"
trap - EXIT
[[ ${compare_status} -eq 0 ]] || exit ${compare_status}
exit ${idle_status}
