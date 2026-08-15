#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../.." && pwd)
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_id=${CRUISE_RUN_ID:-persistent-control-p0-$(date -u +%Y%m%dT%H%M%SZ)}
run_root=${persistent_root}/${run_id}
evidence=${CRUISE_EVIDENCE_DIR:-${run_root}/evidence}
scratch=${CRUISE_SCRATCH_DIR:-/dev/shm/cruise-${run_id}}
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
quantum_delay_us=${CRUISE_P0_QUANTUM_DELAY_US:-1000}
profile_enabled=${CRUISE_P0_PROFILE:-1}
run_mode=${CRUISE_P0_MODE:-full}
ascend_log_level=${CRUISE_ASCEND_LOG_LEVEL:-3}
placement_log_level=${CRUISE_P0_PLACEMENT_LOG_LEVEL:-0}
ascend_slog_stdout=${CRUISE_ASCEND_SLOG_STDOUT:-0}
conda_sh=${CRUISE_CONDA_SH:-/home/changwu/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-vllm-hust-dev}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
guard=${source_dir}/storage_guard/storage_guard.sh
template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
controller_source=${script_dir}/controller
host_source=${script_dir}/persistent_control_p0_host.cpp
analyzer=${script_dir}/analyze_p0.py
placement_analyzer=${script_dir}/placement_evidence.py
config_writer=${script_dir}/prepare_p0_config.py
resource_writer=${source_dir}/prepare_resource_config.py

[[ "${physical_npu}" =~ ^[0-9]+$ ]]
[[ "${quantum_delay_us}" =~ ^[0-9]+$ && "${quantum_delay_us}" -le 1000000 ]]
[[ "${profile_enabled}" == 0 || "${profile_enabled}" == 1 ]]
[[ "${run_mode}" == full || "${run_mode}" == placement-diagnostic ]]
[[ "${ascend_log_level}" =~ ^[0-4]$ ]]
[[ "${placement_log_level}" =~ ^[0-4]$ ]]
[[ "${ascend_slog_stdout}" == 0 || "${ascend_slog_stdout}" == 1 ]]
for required in "${conda_sh}" "${cann_set_env}" "${guard}" "${template}" \
  "${host_source}" "${analyzer}" "${placement_analyzer}" "${config_writer}" \
  "${resource_writer}" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_control_p0.cpp"; do
  [[ -f "${required}" ]] || {
    printf 'missing required input: %s\n' "${required}" >&2
    exit 96
  }
done

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=8
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((512 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=60
export STORAGE_GUARD_NPU_STABLE_SAMPLES=1
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 2

finalize() {
  local command_status=$? finalize_status=0 cleanup_status=0
  local driver_evidence=driver-logs
  trap - EXIT
  set +e
  if [[ ${command_status} -ne 0 ]]; then
    driver_evidence=failure-driver-logs
  fi
  if [[ -n "${driver_logs:-}" && -d "${driver_logs}" ]]; then
    mkdir -p "${evidence}/${driver_evidence}"
    cp -a "${driver_logs}/." "${evidence}/${driver_evidence}/"
  fi
  storage_guard_finalize
  finalize_status=$?
  if [[ ${finalize_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${command_status} -ne 0 ]]; then
    exit "${command_status}"
  fi
  if [[ ${finalize_status} -ne 0 ]]; then
    exit "${finalize_status}"
  fi
  exit "${cleanup_status}"
}
trap finalize EXIT

build=${scratch}/build
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
driver_cache=${scratch}/driver-cache
driver_logs=${scratch}/driver-logs
driver_tmp=${scratch}/driver-tmp
profile_root=${scratch}/profile
profile_evidence=${evidence}/profile
controller_workspace=${scratch}/controller
mkdir -p "${build}" "${config_dir}" "${deploy_root}" "${driver_cache}" \
  "${driver_logs}" "${driver_tmp}" "${profile_root}" "${profile_evidence}" \
  "${controller_workspace}"
for path in "${build}" "${config_dir}" "${deploy_root}" "${driver_cache}" \
  "${driver_logs}" "${driver_tmp}" "${profile_root}" \
  "${controller_workspace}"; do
  storage_guard_assert_scratch_path "${path}"
done
cp "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_control_p0.cpp" \
  "${controller_workspace}/"

source "${conda_sh}"
conda activate "${conda_env}"
source "${cann_set_env}"
ascend_toolchain=${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
[[ -x "${ascend_toolchain}" ]] || {
  printf 'missing Ascend FunctionPp toolchain: %s\n' "${ascend_toolchain}" >&2
  exit 96
}
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export RESOURCE_CONFIG_PATH=${config_dir}/numa_config.json
export ASCEND_GLOBAL_LOG_LEVEL=${ascend_log_level}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ascend_slog_stdout}
export ASCEND_PROCESS_LOG_PATH=${driver_logs}
export ASCEND_CACHE_PATH=${driver_cache}
export TMPDIR=${driver_tmp}
export XDG_CACHE_HOME=${driver_cache}/xdg
export PYTHONDONTWRITEBYTECODE=1

python "${resource_writer}" --template "${template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
function_config=${config_dir}/persistent_control_p0_func.json
toolchain_config=${config_dir}/persistent_control_p0_toolchain.json
deploy_config=${config_dir}/persistent_control_p0_deploy.json
python "${config_writer}" --workspace "${controller_workspace}" \
  --ascend-toolchain "${ascend_toolchain}" --output "${function_config}" \
  --toolchain-output "${toolchain_config}" \
  --deploy-output "${deploy_config}"

host_binary=${build}/persistent_control_p0_host
g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -pthread \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/external" \
  "${host_source}" \
  -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${host_binary}" \
  >"${evidence}/host-compile.log" 2>&1
sha256sum "${host_binary}" "${host_source}" \
  "${controller_workspace}/persistent_control_p0.cpp" \
  "${function_config}" "${toolchain_config}" "${deploy_config}" \
  "${RESOURCE_CONFIG_PATH}" \
  >"${evidence}/artifact-integrity.log"

npu-smi info -t usages -i "${physical_npu}" >"${evidence}/npu-usage-before.txt"
wait_for_npu_release() {
  local label=$1
  for _ in $(seq 1 60); do
    if npu-smi info -t proc-mem -i "${physical_npu}" 2>&1 | \
      grep -Fq 'No process in device.'; then
      npu-smi info -t proc-mem -i "${physical_npu}" \
        >"${evidence}/npu-processes-${label}.txt"
      npu-smi info -t usages -i "${physical_npu}" \
        >"${evidence}/npu-usage-${label}.txt"
      return 0
    fi
    sleep 1
  done
  return 95
}

placement_result=${evidence}/placement-result.json
placement_extract=${evidence}/placement-driver-extract.log
export ASCEND_GLOBAL_LOG_LEVEL=${placement_log_level}
storage_guard_run_log "${evidence}/placement-probe.log" \
  "${evidence}/placement-probe.meta.json" 300 -- \
  "${host_binary}" "${function_config}" "${deploy_config}" \
  9001 "${quantum_delay_us}" placement_probe
wait_for_npu_release after-placement-probe
set +e
python "${placement_analyzer}" --log-root "${driver_logs}" \
  --log-root "${evidence}/placement-probe.log" \
  --output "${placement_result}" --extract-output "${placement_extract}" \
  >"${evidence}/placement-analyze.log"
placement_status=$?
set -e
storage_guard_assert_scratch_path "${driver_logs}"
find "${driver_logs}" -type f -delete
export ASCEND_GLOBAL_LOG_LEVEL=${ascend_log_level}
storage_guard_runtime_budget_ok 0
if [[ ${placement_status} -ne 0 ]]; then
  printf 'P0 placement gate failed; see %s\n' "${placement_result}" >&2
  exit "${placement_status}"
fi
if [[ "${run_mode}" == placement-diagnostic ]]; then
  printf 'PERSISTENT_CONTROL_P0_PLACEMENT_COMPLETE evidence=%s\n' "${evidence}"
  exit 0
fi

for start in 1 2 3; do
  owner=$((1000 + start))
  storage_guard_run_log "${evidence}/start-${start}.log" \
    "${evidence}/start-${start}.meta.json" 900 -- \
    "${host_binary}" "${function_config}" "${deploy_config}" \
    "${owner}" "${quantum_delay_us}"
  wait_for_npu_release "after-start-${start}"
done

profile_status=${evidence}/profile-status.tsv
write_profile_status() {
  local exit_status=$1
  {
    printf 'profile_exit_status\t%s\n' "${exit_status}"
    printf 'aicpu\ton\n'
    printf 'ai_core\ton\n'
    printf 'task_time\tl1\n'
  } >"${profile_status}"
}
write_profile_status 96
if [[ "${profile_enabled}" == 1 ]]; then
  set +e
  storage_guard_run_log "${evidence}/profile.log" \
    "${evidence}/profile.meta.json" 1200 -- \
    msprof --output="${profile_root}" --runtime-api=on --ge-api=l0 \
      --task-time=l1 --ai-core=on --aicpu=on \
      --storage-limit=200MB \
      "${host_binary}" "${function_config}" "${deploy_config}" \
      2001 "${quantum_delay_us}"
  profile_exit=$?
  set -e
  write_profile_status "${profile_exit}"
  wait_for_npu_release after-profile
  cp -a "${profile_root}/." "${profile_evidence}/"
  storage_guard_runtime_budget_ok 0
fi

analysis_args=(
  --run-log "${evidence}/start-1.log"
  --run-log "${evidence}/start-2.log"
  --run-log "${evidence}/start-3.log"
  --npu-before "${evidence}/npu-usage-before.txt"
  --npu-after "${evidence}/npu-usage-after-start-1.txt"
  --npu-after "${evidence}/npu-usage-after-start-2.txt"
  --npu-after "${evidence}/npu-usage-after-start-3.txt"
  --profile-status "${profile_status}"
  --profile-root "${profile_evidence}"
  --placement-result "${placement_result}"
  --output "${evidence}/result.json"
)
if [[ "${profile_enabled}" == 0 ]]; then
  analysis_args+=(--mechanism-only)
else
  analysis_args+=(
    --profile-log "${evidence}/profile.log"
    --controller-log-root "${driver_logs}"
    --controller-extract-output "${evidence}/profile-controller-extract.log"
    --npu-after-profile "${evidence}/npu-usage-after-profile.txt"
  )
fi
set +e
python "${analyzer}" "${analysis_args[@]}" >"${evidence}/analyze.log"
analysis_status=$?
set -e
if [[ ${analysis_status} -eq 0 ]]; then
  storage_guard_assert_scratch_path "${driver_logs}"
  find "${driver_logs}" -type f -delete
else
  exit "${analysis_status}"
fi
if [[ "${profile_enabled}" == 0 ]]; then
  printf 'PERSISTENT_CONTROL_P0_MECHANISM_COMPLETE profile=skipped evidence=%s\n' \
    "${evidence}"
else
  printf 'PERSISTENT_CONTROL_P0_COMPLETE evidence=%s\n' "${evidence}"
fi
