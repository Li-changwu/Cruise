#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
v2_dir=$(cd -- "${script_dir}/.." && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-owner-v2-kv-graphpp-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_V2_KV_GRAPHPP_SCRATCH:-/dev/shm/cruise-v2-kv-graphpp-${physical_npu}-$$}
python_bin=${CRUISE_V2_GRAPHPP_PYTHON:-$(command -v python3)}
cann_home=${CRUISE_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}
cann_set_env=${CRUISE_CANN_SET_ENV:-${cann_home}/set_env.sh}
guard=${source_dir}/storage_guard/storage_guard.sh
hardware_policy=${v2_dir}/hardware_policy.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
resource_writer=${source_dir}/prepare_resource_config.py
resource_template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
exporter=${script_dir}/export_kv_alias.py
inspector=${script_dir}/inspect_kv_alias.py
config_writer=${script_dir}/prepare_graphpp_config.py
host_source=${script_dir}/kv_alias_graphpp_probe.cpp
verifier=${script_dir}/verify_graphpp_pair.py

for required in "${python_bin}" "${cann_set_env}" "${guard}" \
  "${hardware_policy}" "${lifecycle_tool}" "${resource_writer}" \
  "${resource_template}" "${exporter}" "${inspector}" "${config_writer}" \
  "${host_source}" "${verifier}"; do
  [[ -f "${required}" ]] || {
    printf 'missing V2 KV GraphPp input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${scratch}" == /dev/shm/cruise-v2-kv-graphpp-* ]] || {
  printf 'V2 KV GraphPp scratch must be PID-scoped under /dev/shm: %s\n' \
    "${scratch}" >&2
  exit 96
}

source "${guard}"
source "${hardware_policy}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=1
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((128 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=60
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
  --assets-root /workspace/cruise-assets --summary-only
python3 "${lifecycle_tool}" shm-audit --summary-only
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 1
python3 "${lifecycle_tool}" mark --runs-root "${persistent_root}" \
  --run-dir "${run_root}" --retention-class diagnostic --retention-days 7 \
  --max-run-gib 1

export_dir=${scratch}/export
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
driver_logs=${scratch}/driver-logs
cache=${scratch}/cache
tmp=${scratch}/tmp
build=${scratch}/build
mkdir -p "${driver_logs}" "${cache}" "${tmp}" "${build}"

finalize() {
  local command_status=$? recovery_status=0 finalize_status=0 cleanup_status=0
  local lifecycle_status=0
  trap - EXIT
  set +e
  cd "${source_dir}"
  printf 'driver-exit\t%s\n' "${command_status}" >"${evidence}/status.tsv"
  if [[ -n ${V2_HBM_BASELINE_MB} ]]; then
    v2_wait_for_hbm_recovery "${evidence}" "${physical_npu}" \
      "${V2_HBM_BASELINE_MB}"
    recovery_status=$?
  fi
  if [[ ( ${command_status} -ne 0 || ${recovery_status} -ne 0 ) && \
        -d "${driver_logs}" ]]; then
    mkdir -p "${evidence}/failure-driver-logs"
    while IFS= read -r log; do
      tail -c $((512 * 1024)) -- "${log}" \
        >"${evidence}/failure-driver-logs/$(basename -- "${log}")"
    done < <(find "${driver_logs}" -type f -print | sort | head -n 24)
  fi
  storage_guard_finalize
  finalize_status=$?
  if [[ ${finalize_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${finalize_status} -eq 0 && ${cleanup_status} -eq 0 ]]; then
    python3 "${lifecycle_tool}" finalize --runs-root "${persistent_root}" \
      --run-dir "${run_root}" --retention-class diagnostic --retention-days 7
    lifecycle_status=$?
    python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
      --assets-root /workspace/cruise-assets --summary-only || lifecycle_status=$?
  fi
  if [[ ${command_status} -ne 0 ]]; then exit "${command_status}"; fi
  if [[ ${recovery_status} -ne 0 ]]; then exit "${recovery_status}"; fi
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

v2_capture_hbm_baseline "${evidence}" "${physical_npu}"
source "${cann_set_env}"
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export RESOURCE_CONFIG_PATH=${config_dir}/numa.json
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_LOG_LEVEL:-0}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_PROCESS_LOG_PATH=${driver_logs}
export ASCEND_CACHE_PATH=${cache}
export TMPDIR=${tmp}
export PYTHONDONTWRITEBYTECODE=1

wait_for_release() {
  for _ in $(seq 1 60); do
    if npu-smi info -t proc-mem -i "${physical_npu}" 2>&1 | \
      rg -q 'No process in device\.'; then
      return 0
    fi
    sleep 1
  done
  return 95
}

"${python_bin}" "${resource_writer}" --template "${resource_template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
"${python_bin}" "${config_writer}" \
  --graph-output "${config_dir}/graph.json" \
  --deploy-output "${config_dir}/deploy.json" \
  --expected-output "${scratch}/expected-output.bin"

set +e
cd "${scratch}"
storage_guard_run_log "${evidence}/export.log" "${evidence}/export.meta.json" \
  600s -- "${python_bin}" "${exporter}" --output-dir "${export_dir}"
export_status=$?
cd "${source_dir}"
set -e
printf 'export-exit\t%s\n' "${export_status}" >"${evidence}/export-status.tsv"
[[ ${export_status} -eq 0 || ${export_status} -eq 139 ]] || \
  exit "${export_status}"
"${python_bin}" - "${export_dir}/export-result.json" \
  "${export_dir}/kv_alias_probe.air" "${export_dir}/dynamo.pbtxt" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("pass") is not True or result.get("structure_pass") is not True:
    raise SystemExit(1)
if not all(Path(path).is_file() for path in sys.argv[2:]):
    raise SystemExit(1)
PY
wait_for_release

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -pthread \
  -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${host_source}" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${build}/kv_alias_graphpp_probe" \
  >"${evidence}/host-compile.log" 2>&1

: >"${evidence}/mode-status.tsv"
for mode in graph dataflow; do
  mode_driver_logs=${driver_logs}/${mode}
  mkdir -p "${mode_driver_logs}"
  export ASCEND_PROCESS_LOG_PATH=${mode_driver_logs}
  set +e
  cd "${scratch}"
  storage_guard_run_log "${evidence}/${mode}.log" \
    "${evidence}/${mode}.meta.json" 600s -- \
    "${build}/kv_alias_graphpp_probe" "${mode}" \
    "${export_dir}/kv_alias_probe.air" "${config_dir}/graph.json" \
    "${config_dir}/deploy.json" "${scratch}/${mode}-output.bin" \
    "${scratch}/${mode}.json"
  mode_status=$?
  cd "${source_dir}"
  set -e
  printf '%s-exit\t%s\n' "${mode}" "${mode_status}" \
    >>"${evidence}/mode-status.tsv"
  [[ -f "${scratch}/${mode}.json" ]] && \
    cp --reflink=auto "${scratch}/${mode}.json" "${evidence}/${mode}.json"
  find "${mode_driver_logs}" -type f -print0 | sort -z | \
    xargs -0 -r rg -i 'LaunchKernel: kernel info.*(scatter.*pa.*kv|scatter_pa_kv_cache)' \
    >"${evidence}/${mode}-launch-metadata.txt" || true
  wait_for_release
done

cp --reflink=auto "${export_dir}/export-result.json" "${evidence}/"
cp --reflink=auto "${export_dir}/graph-structure.json" "${evidence}/"
cp --reflink=auto "${export_dir}/dynamo.pbtxt" "${evidence}/"
cp --reflink=auto "${config_dir}/graph.json" "${evidence}/graph-config.json"
cp --reflink=auto "${config_dir}/deploy.json" "${evidence}/deploy-config.json"
sha256sum "${export_dir}/kv_alias_probe.air" | \
  sed 's#  .*#  kv_alias_probe.air#' >"${evidence}/air-identity.sha256"
sha256sum "${exporter}" "${inspector}" "${config_writer}" "${host_source}" \
  "${verifier}" "${hardware_policy}" "${script_dir}/run_graphpp_pair_on_910b.sh" \
  >"${evidence}/source-identity.sha256"
git -C "${source_dir}" status --porcelain=v1 \
  >"${evidence}/source-worktree-status.txt"
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
"${python_bin}" - <<'PY' >"${evidence}/package-identity.txt"
import importlib.metadata
for name in ("torch", "torch-npu"):
    print(f"{name}=={importlib.metadata.version(name)}")
PY

set +e
"${python_bin}" "${verifier}" \
  --expected-output "${scratch}/expected-output.bin" \
  --graph-result "${evidence}/graph.json" \
  --graph-output "${scratch}/graph-output.bin" \
  --dataflow-result "${evidence}/dataflow.json" \
  --dataflow-output "${scratch}/dataflow-output.bin" \
  --output "${evidence}/verifier.json"
verifier_status=$?
set -e
storage_guard_snapshot kv-graphpp-complete \
  "${evidence}/storage-kv-graphpp-complete.tsv"
exit "${verifier_status}"
