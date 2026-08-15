#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../.." && pwd)
p3_dir=${source_dir}/experiments/persistent_decoder_p3
scenario=${1:-${CRUISE_P4_SCENARIO:-primary-c4}}
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-admission-p4-${scenario}-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${CRUISE_EVIDENCE_DIR:-${run_root}/evidence}
scratch=${CRUISE_P4_SCRATCH:-/dev/shm/cruise-p4-service-${physical_npu}-$$}
air=${CRUISE_P4_AIR:-/workspace/cruise-assets/p3-air384-r4-xfs/qwen_b4_p3_decoder_step.runtime.air}
external_weights=${CRUISE_P4_EXTERNAL_WEIGHT_DIR:-/workspace/cruise-assets/runtime-weights/p3-ge-external-f93a2828d8b3ded1}
oracle=${CRUISE_P4_ORACLE_SUMMARY:-/workspace/cruise-runs/persistent-decoder-p3-oracle-b1-full-20260814T091826Z/evidence/summary.json}
conda_sh=${CRUISE_CONDA_SH:-/home/changwu/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
resource_writer=${source_dir}/prepare_resource_config.py
config_writer=${p3_dir}/prepare_p3_config.py
controller_source=${p3_dir}/controller
host_source=${script_dir}/persistent_admission_p4_host.cpp
workload=${script_dir}/workload.json
verifier=${script_dir}/verify_p4_gate.py
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
external_view_tool=${p3_dir}/manage_ge_external_view.py
external_view=${run_root}/.ge-external-view

[[ "${scenario}" == primary-c4 || "${scenario}" == regression ]] || {
  printf 'unsupported P4 scenario: %s\n' "${scenario}" >&2
  exit 96
}
for required in "${cann_set_env}" "${template}" "${guard}" \
  "${cann_python_env}/bin/python" "${lifecycle_tool}" \
  "${external_view_tool}" "${resource_writer}" "${config_writer}" \
  "${air}" "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" "${host_source}" \
  "${workload}" "${verifier}"; do
  [[ -f "${required}" ]] || {
    printf 'missing required input: %s\n' "${required}" >&2
    exit 96
  }
done
if [[ "${scenario}" == primary-c4 && ! -f "${oracle}" ]]; then
  printf 'missing P4 cold Graph oracle: %s\n' "${oracle}" >&2
  exit 96
fi
[[ -d "${external_weights}" ]] || {
  printf 'missing external weights: %s\n' "${external_weights}" >&2
  exit 96
}
[[ "${scratch}" == /dev/shm/* ]] || {
  printf 'P4 scratch must be under /dev/shm: %s\n' "${scratch}" >&2
  exit 96
}

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=${CRUISE_P4_MAX_SCRATCH_GIB:-2}
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=${CRUISE_P4_MAX_EVIDENCE_BYTES:-$((64 * 1024 * 1024))}
export STORAGE_GUARD_NPU_WAIT_SECONDS=${CRUISE_P4_NPU_WAIT_SECONDS:-60}
export STORAGE_GUARD_NPU_STABLE_SAMPLES=${CRUISE_P4_NPU_STABLE_SAMPLES:-1}
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
export STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=2
python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
  --assets-root "${CRUISE_RUNTIME_ASSET_ROOT:-/workspace/cruise-assets}" \
  --max-runs-gib 20 --max-run-gib 2 --summary-only
python3 "${lifecycle_tool}" shm-audit --summary-only
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 2
python3 "${lifecycle_tool}" mark --runs-root "${persistent_root}" \
  --run-dir "${run_root}" --retention-class evidence --retention-days 30 \
  --max-run-gib 2
sha256sum "${host_source}" "${workload}" "${verifier}" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" "${config_writer}" \
  "${external_view_tool}" "${air}" "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json" \
  >"${evidence}/source-identity.sha256"
if [[ "${scenario}" == primary-c4 ]]; then
  sha256sum "${oracle}" >"${evidence}/oracle-identity.sha256"
fi

build=${scratch}/build
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
controller_workspace=${scratch}/controller
cache=${scratch}/cache
driver_logs=${scratch}/driver-logs
tmp=${scratch}/tmp
status=${evidence}/status.tsv
mkdir -p "${build}" "${config_dir}" "${deploy_root}" \
  "${controller_workspace}" "${cache}" "${driver_logs}" "${tmp}"
cp "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" \
  "${controller_workspace}/"

if [[ -n "${conda_env}" ]]; then
  [[ -f "${conda_sh}" ]] || {
    printf 'missing Conda activation script: %s\n' "${conda_sh}" >&2
    exit 96
  }
  source "${conda_sh}"
  conda activate "${conda_env}"
fi
source "${cann_set_env}"
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
ascend_toolchain=${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
[[ -x "${ascend_toolchain}" ]] || {
  printf 'missing Ascend FunctionPp toolchain: %s\n' "${ascend_toolchain}" >&2
  exit 96
}

export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export RESOURCE_CONFIG_PATH=${config_dir}/numa.json
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export PYTHONDONTWRITEBYTECODE=1

python "${resource_writer}" --template "${template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
function_config=${config_dir}/function.json
graph_config=${config_dir}/graph.json
toolchain_config=${config_dir}/toolchain.json
deploy_config=${config_dir}/deploy.json
python "${config_writer}" \
  --workspace "${controller_workspace}" \
  --ascend-toolchain "${ascend_toolchain}" \
  --function-output "${function_config}" \
  --graph-output "${graph_config}" \
  --toolchain-output "${toolchain_config}" \
  --deploy-output "${deploy_config}"

host_binary=${build}/persistent_admission_p4_host
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
  >"${scratch}/host-compile.log" 2>&1

review_intermediates() {
  local label=$1
  storage_guard_runtime_budget_ok 0
  storage_guard_snapshot "${label}" "${evidence}/storage-${label}.tsv"
  find "${driver_logs}" -type f -size +16M -delete
}
review_intermediates after-compile

extract_failure_logs() {
  local source_file relative target
  [[ -d "${driver_logs}" ]] || return 0
  mkdir -p "${evidence}/failure-driver-logs"
  while IFS= read -r -d '' source_file; do
    relative=${source_file#"${driver_logs}"/}
    target=${evidence}/failure-driver-logs/${relative}
    mkdir -p "$(dirname -- "${target}")"
    tail -c $((512 * 1024)) -- "${source_file}" >"${target}"
  done < <(find "${driver_logs}" -type f -print0 | sort -z | head -z -n 32)
}

wait_for_release() {
  for _ in $(seq 1 120); do
    if npu-smi info -t proc-mem -i "${physical_npu}" 2>&1 | \
      rg -q 'No process in device\.'; then
      return 0
    fi
    sleep 1
  done
  return 95
}

finalize() {
  local command_status=$? view_status=0 finalize_status=0 cleanup_status=0 lifecycle_status=0
  local retention_class=evidence retention_days=30
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >>"${status}"
  if [[ ${command_status} -ne 0 ]]; then extract_failure_logs; fi
  if [[ -n "${summary:-}" && -f "${summary}" &&
        ! -f "${evidence}/summary.json" ]]; then
    cp --reflink=auto "${summary}" "${evidence}/summary.json"
  fi
  if [[ -d "${external_view}" ]]; then
    python3 "${external_view_tool}" cleanup --run-root "${run_root}" \
      --view "${external_view}" \
      --receipt "${evidence}/ge-external-view-cleanup.json"
    view_status=$?
  fi
  storage_guard_finalize
  finalize_status=$?
  if [[ ${finalize_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${finalize_status} -eq 0 && ${cleanup_status} -eq 0 ]]; then
    if [[ ${command_status} -ne 0 ]]; then
      retention_class=diagnostic
      retention_days=7
    fi
    python3 "${lifecycle_tool}" finalize --runs-root "${persistent_root}" \
      --run-dir "${run_root}" --retention-class "${retention_class}" \
      --retention-days "${retention_days}"
    lifecycle_status=$?
    if [[ ${lifecycle_status} -eq 0 ]]; then
      python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
        --assets-root "${CRUISE_RUNTIME_ASSET_ROOT:-/workspace/cruise-assets}" \
        --max-runs-gib 20 --max-run-gib 2 --summary-only
      lifecycle_status=$?
    fi
    if [[ ${lifecycle_status} -eq 0 ]]; then
      python3 "${lifecycle_tool}" shm-audit --summary-only
      lifecycle_status=$?
    fi
  fi
  if [[ ${command_status} -ne 0 ]]; then exit "${command_status}"; fi
  if [[ ${view_status} -ne 0 ]]; then exit "${view_status}"; fi
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

summary=${scratch}/summary.json
wait_for_release
python3 "${external_view_tool}" create --source "${external_weights}" \
  --run-root "${run_root}" --view "${external_view}"
timeout_seconds=${CRUISE_P4_TIMEOUT:-7200}
if [[ "${scenario}" == regression ]]; then
  timeout_seconds=${CRUISE_P4_TIMEOUT:-3600}
fi
(
  cd -- "${scratch}"
  storage_guard_run_log "${evidence}/owner.stdout.log" \
    "${evidence}/owner.stdout.meta.json" \
    "${timeout_seconds}s" -- env \
    ASCEND_PROCESS_LOG_PATH="${driver_logs}" \
    ASCEND_CACHE_PATH="${cache}" \
    TMPDIR="${tmp}" \
    CRUISE_P4_RUN_ROOT="${run_root}" \
    CRUISE_P4_EXTERNAL_WEIGHT_DIR="${external_view}" \
    "${host_binary}" "${function_config}" "${graph_config}" \
    "${deploy_config}" "${air}" "$((7000 + physical_npu))" \
    "${scenario}" "${summary}"
)
[[ -f "${summary}" ]] || {
  printf 'P4 host did not produce a summary\n' >&2
  exit 20
}
cp --reflink=auto "${summary}" "${evidence}/summary.json"
if [[ "${scenario}" == primary-c4 ]]; then
  python3 "${verifier}" --summary "${evidence}/summary.json" \
    --oracle "${oracle}" --output "${evidence}/gate.json"
else
  python3 "${verifier}" --summary "${evidence}/summary.json" \
    --output "${evidence}/gate.json"
fi
review_intermediates after-owner
wait_for_release

find "${driver_logs}" -type f -size +16M -delete
printf 'P4_SERVICE_RUN_COMPLETE scenario=%s scratch=%s summary=%s\n' \
  "${scenario}" "${scratch}" "${summary}"
