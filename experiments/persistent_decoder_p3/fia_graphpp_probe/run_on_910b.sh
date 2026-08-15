#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-decoder-p3-mini-fia-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_MINI_FIA_SCRATCH:-/dev/shm/cruise-mini-fia-${physical_npu}-$$}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
python_bin=${CRUISE_MINI_FIA_PYTHON:-$(command -v python3)}
probe_modes=${CRUISE_MINI_FIA_MODES:-graph dataflow}
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
resource_writer=${source_dir}/prepare_resource_config.py
template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
ops_source=${CRUISE_FIA_OPS_SOURCE:-/dev/shm/cruise-ops-transformer-9.0.0-audit}
opp_builder=${script_dir}/build_isolated_opp.sh

for required in "${cann_set_env}" "${python_bin}" "${guard}" \
  "${lifecycle_tool}" "${resource_writer}" "${template}" \
  "${cann_python_env}/bin/python" \
  "${script_dir}/export_mini_fia.py" \
  "${script_dir}/prepare_probe_config.py" \
  "${opp_builder}" \
  "${script_dir}/audit_isolated_opp.py" \
  "${script_dir}/preserve_op_kernel_hierarchy.patch" \
  "${script_dir}/mini_fia_om_probe.cpp" \
  "${script_dir}/mini_fia_graphpp_probe.cpp"; do
  [[ -f "${required}" ]] || { printf 'missing mini FIA input: %s\n' "${required}" >&2; exit 96; }
done
read -r -a selected_modes <<<"${probe_modes}"
[[ ${#selected_modes[@]} -gt 0 ]] || { printf 'no mini FIA probe modes selected\n' >&2; exit 96; }
declare -A seen_modes=()
om_precedes_serialized=0
for mode in "${selected_modes[@]}"; do
  [[ "${mode}" == om || "${mode}" == graph || "${mode}" == dataflow || \
     "${mode}" == serialized-dataflow ]] || {
    printf 'invalid mini FIA probe mode: %s\n' "${mode}" >&2
    exit 96
  }
  [[ -z "${seen_modes[${mode}]:-}" ]] || {
    printf 'duplicate mini FIA probe mode: %s\n' "${mode}" >&2
    exit 96
  }
  seen_modes[${mode}]=1
  if [[ "${mode}" == om ]]; then
    om_precedes_serialized=1
  elif [[ "${mode}" == serialized-dataflow && ${om_precedes_serialized} -ne 1 ]]; then
    printf 'serialized-dataflow requires a preceding om mode\n' >&2
    exit 96
  fi
done
[[ "${scratch}" == /dev/shm/cruise-mini-fia-* ]] || {
  printf 'mini FIA scratch must be PID-scoped under /dev/shm: %s\n' "${scratch}" >&2
  exit 96
}

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=1
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((64 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=60
export STORAGE_GUARD_NPU_STABLE_SAMPLES=1
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
  --assets-root /workspace/cruise-assets --summary-only
python3 "${lifecycle_tool}" shm-audit --summary-only
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 1
python3 "${lifecycle_tool}" mark --runs-root "${persistent_root}" \
  --run-dir "${run_root}" --retention-class diagnostic --retention-days 7 \
  --max-run-gib 1

build=${scratch}/build
fia_build_root=${scratch}/fia-opp-source
fia_install_root=${scratch}/fia-opp-install
opp_proxy=${scratch}/opp
export_dir=${scratch}/export
external_weight_dir=${scratch}/external-weights
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
driver_logs=${scratch}/driver-logs
cache=${scratch}/cache
tmp=${scratch}/tmp
mkdir -p "${build}" "${external_weight_dir}" "${config_dir}" "${deploy_root}" \
  "${opp_proxy}" \
  "${driver_logs}" "${cache}" "${tmp}"

finalize() {
  local command_status=$? finalize_status=0 cleanup_status=0 lifecycle_status=0
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >"${evidence}/status.tsv"
  for compile_log in "${scratch}"/*-host-compile.log; do
    [[ -f "${compile_log}" ]] || continue
    tail -c $((512 * 1024)) -- "${compile_log}" \
      >"${evidence}/$(basename -- "${compile_log}")"
  done
  if [[ ${command_status} -ne 0 && -d "${driver_logs}" ]]; then
    mkdir -p "${evidence}/failure-driver-logs"
    while IFS= read -r log; do
      tail -c $((512 * 1024)) -- "${log}" \
        >"${evidence}/failure-driver-logs/$(basename -- "${log}")"
    done < <(find "${driver_logs}" -type f -print | sort | head -n 24)
  fi
  if [[ -d "${fia_install_root}" ]]; then
    find "${fia_install_root}" -type d -exec chmod u+w {} +
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
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

git -C "${source_dir}" status --porcelain=v1 \
  >"${evidence}/source-worktree-status.txt"
[[ ! -s "${evidence}/source-worktree-status.txt" ]] || {
  printf 'mini FIA hardware probe requires a clean source worktree\n' >&2
  exit 94
}
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
source "${cann_set_env}"
system_opp=${ASCEND_OPP_PATH}
for component in built-in include lib64 bin Ascend; do
  [[ -e "${system_opp}/${component}" ]] || {
    printf 'missing system OPP component: %s\n' "${component}" >&2
    exit 96
  }
  ln -s "${system_opp}/${component}" "${opp_proxy}/${component}"
done
export ASCEND_OPP_PATH=${opp_proxy}
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
custom_set_env=$("${opp_builder}" --source "${ops_source}" \
  --work-root "${fia_build_root}" --install-root "${fia_install_root}" \
  --evidence "${evidence}")
[[ -f "${custom_set_env}" ]] || {
  printf 'missing isolated FIA set_env.bash: %s\n' "${custom_set_env}" >&2
  exit 95
}
export ASCEND_OPP_PATH=${system_opp}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "${custom_set_env}"
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export RESOURCE_CONFIG_PATH=${config_dir}/numa.json
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH=${driver_logs}
export ASCEND_CACHE_PATH=${cache}
export TMPDIR=${tmp}

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

"${python_bin}" "${resource_writer}" --template "${template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
"${python_bin}" "${script_dir}/prepare_probe_config.py" \
  --graph-output "${config_dir}/graph.json" \
  --deploy-output "${config_dir}/deploy.json"
set +e
storage_guard_run_log "${evidence}/export.log" "${evidence}/export.meta.json" \
  300s -- "${python_bin}" "${script_dir}/export_mini_fia.py" \
  --output-dir "${export_dir}"
export_status=$?
set -e
printf 'export-exit\t%s\n' "${export_status}" >"${evidence}/export-status.tsv"
[[ ${export_status} -eq 0 || ${export_status} -eq 139 ]] || exit "${export_status}"
"${python_bin}" - "${export_dir}/export-result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("pass") is not True:
    raise SystemExit(1)
PY
cp --reflink=auto "${export_dir}/export-result.json" "${evidence}/export-result.json"
cp --reflink=auto "${export_dir}/dynamo.pbtxt" "${evidence}/dynamo.pbtxt"
wait_for_release

if [[ -n "${seen_modes[om]:-}" ]]; then
  g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
    -fstack-protector-all -pthread \
    -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
    "${script_dir}/mini_fia_om_probe.cpp" \
    -Wl,--no-as-needed "${ASCEND_HOME_PATH}/lib64/libge_compiler.so" \
    "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
    "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
    "${ASCEND_HOME_PATH}/lib64/libascendcl.so" \
    -Wl,--as-needed -o "${build}/mini_fia_om_probe" \
    >"${scratch}/om-host-compile.log" 2>&1
  cp --reflink=auto "${scratch}/om-host-compile.log" \
    "${evidence}/om-host-compile.log"
fi

if [[ -n "${seen_modes[graph]:-}" || -n "${seen_modes[dataflow]:-}" || \
      -n "${seen_modes[serialized-dataflow]:-}" ]]; then
  g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
    -fstack-protector-all -pthread \
    -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
    "${script_dir}/mini_fia_graphpp_probe.cpp" \
    -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
    "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
    "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
    "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
    "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
    "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
    "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
    -Wl,--no-whole-archive -o "${build}/mini_fia_graphpp_probe" \
    >"${scratch}/graphpp-host-compile.log" 2>&1
  cp --reflink=auto "${scratch}/graphpp-host-compile.log" \
    "${evidence}/graphpp-host-compile.log"
fi

: >"${evidence}/mode-status.tsv"
result_paths=()
for mode in "${selected_modes[@]}"; do
  set +e
  cd "${scratch}"
  if [[ "${mode}" == om ]]; then
    storage_guard_run_log "${evidence}/om.log" "${evidence}/om.meta.json" \
      900s -- "${build}/mini_fia_om_probe" \
      "${export_dir}/mini_fia.air" "${scratch}/mini_fia" \
      "${scratch}/om.json"
  else
    model_input=${export_dir}/mini_fia.air
    if [[ "${mode}" == serialized-dataflow ]]; then
      model_input=${scratch}/mini_fia.om
    fi
    storage_guard_run_log "${evidence}/${mode}.log" "${evidence}/${mode}.meta.json" \
      300s -- "${build}/mini_fia_graphpp_probe" "${mode}" \
      "${model_input}" "${config_dir}/graph.json" \
      "${config_dir}/deploy.json" "${scratch}/${mode}.json" \
      "${external_weight_dir}"
  fi
  mode_status=$?
  cd "${source_dir}"
  set -e
  printf '%s-exit\t%s\n' "${mode}" "${mode_status}" \
    >>"${evidence}/mode-status.tsv"
  if [[ -f "${scratch}/${mode}.json" ]]; then
    cp --reflink=auto "${scratch}/${mode}.json" "${evidence}/${mode}.json"
  fi
  if [[ "${mode}" == om && -f "${scratch}/mini_fia.om" ]]; then
    sha256sum "${scratch}/mini_fia.om" | \
      awk '{print $1 "  mini_fia.om"}' >"${evidence}/mini_fia.om.sha256"
  fi
  result_paths+=("${evidence}/${mode}.json")
  wait_for_release
done

"${python_bin}" - "${result_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
if not all(result.get("pass") is True for result in results):
    raise SystemExit(1)
print("P3_MINI_FIA_SELECTED_PROBES_PASS")
PY
storage_guard_snapshot mini-fia-complete "${evidence}/storage-mini-fia-complete.tsv"
