#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
v2_dir=$(cd -- "${script_dir}/.." && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-owner-v2-kv-order-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_V2_KV_ORDER_SCRATCH:-/dev/shm/cruise-v2-kv-order-${physical_npu}-$$}
python_bin=${CRUISE_V2_KV_ORDER_PYTHON:-$(command -v python3)}
cann_home=${CRUISE_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}
cann_set_env=${CRUISE_CANN_SET_ENV:-${cann_home}/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
guard=${source_dir}/storage_guard/storage_guard.sh
hardware_policy=${v2_dir}/hardware_policy.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
resource_writer=${source_dir}/prepare_resource_config.py
resource_template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
custom_template=${cann_home}/tools/new_op_project_template/custom_op
exporter=${script_dir}/export_probe.py
inspector=${script_dir}/inspect_probe_graph.py
config_writer=${script_dir}/prepare_probe_config.py
host_source=${script_dir}/paged_kv_order_probe_host.cpp
verifier=${script_dir}/verify_probe.py
controller_source=${script_dir}/controller
op_kernel_dir=${script_dir}/op_kernel
op_host_dir=${script_dir}/op_host

for required in "${python_bin}" "${cann_set_env}" "${guard}" \
  "${hardware_policy}" "${lifecycle_tool}" "${resource_writer}" \
  "${resource_template}" "${cann_python_env}/bin/python" \
  "${custom_template}/CMakeLists.txt" "${exporter}" "${inspector}" \
  "${config_writer}" "${host_source}" "${verifier}" \
  "${op_kernel_dir}/device_paged_kv_update.cpp" \
  "${op_kernel_dir}/device_paged_kv_read.cpp" \
  "${op_host_dir}/device_paged_kv_update_def.cpp" \
  "${op_host_dir}/device_paged_kv_update_infershape.cpp" \
  "${op_host_dir}/device_paged_kv_update_tiling.cpp" \
  "${op_host_dir}/device_paged_kv_read_def.cpp" \
  "${op_host_dir}/device_paged_kv_read_infershape.cpp" \
  "${op_host_dir}/device_paged_kv_read_tiling.cpp" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/paged_kv_order_controller.cpp"; do
  [[ -f "${required}" ]] || {
    printf 'missing V2 Paged-KV order probe input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${scratch}" == /dev/shm/cruise-v2-kv-order-* ]] || {
  printf 'V2 Paged-KV order scratch must be PID-scoped under /dev/shm: %s\n' \
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

project=${scratch}/project
opp_proxy=${scratch}/opp
install=${scratch}/install
export_dir=${scratch}/export
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
controller_workspace=${scratch}/controller
driver_logs=${scratch}/driver-logs
cache=${scratch}/cache
tmp=${scratch}/tmp
build=${scratch}/build
mkdir -p "${opp_proxy}" "${config_dir}" "${deploy_root}" \
  "${controller_workspace}" "${driver_logs}" "${cache}" "${tmp}" "${build}"

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
  for mode in graph dataflow; do
    [[ -f "${evidence}/${mode}.log" ]] || continue
    tail -c $((1024 * 1024)) -- "${evidence}/${mode}.log" \
      >"${evidence}/${mode}-error-excerpt.log"
  done
  if [[ ( ${command_status} -ne 0 || ${recovery_status} -ne 0 ) && \
        -d "${driver_logs}" ]]; then
    mkdir -p "${evidence}/failure-driver-logs"
    while IFS= read -r log; do
      tail -c $((512 * 1024)) -- "${log}" \
        >"${evidence}/failure-driver-logs/$(basename -- "${log}")"
    done < <(find "${driver_logs}" -type f -print | sort | head -n 24)
  fi
  if [[ -d "${install}" ]]; then
    find "${install}" -type d -exec chmod u+w {} +
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

git -C "${source_dir}" status --porcelain=v1 \
  >"${evidence}/source-worktree-status.txt"
[[ ! -s "${evidence}/source-worktree-status.txt" ]] || {
  printf 'V2 hardware probe requires a clean source worktree\n' >&2
  exit 94
}
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
v2_capture_hbm_baseline "${evidence}" "${physical_npu}"
source "${cann_set_env}"
system_opp=${ASCEND_OPP_PATH}
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
ascend_toolchain=${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
[[ -x "${ascend_toolchain}" ]] || {
  printf 'missing Ascend FunctionPp toolchain: %s\n' "${ascend_toolchain}" >&2
  exit 96
}
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

cp -a "${custom_template}" "${project}"
cp "${op_kernel_dir}"/* "${project}/op_kernel/"
cp "${op_host_dir}"/* "${project}/op_host/"
cp "${controller_source}/CMakeLists.txt" \
  "${controller_source}/paged_kv_order_controller.cpp" \
  "${controller_workspace}/"
ln -s "${system_opp}/built-in" "${opp_proxy}/built-in"
ln -s "${system_opp}/include" "${opp_proxy}/include"
ln -s "${system_opp}/lib64" "${opp_proxy}/lib64"
ln -s "${system_opp}/bin" "${opp_proxy}/bin"
ln -s "${system_opp}/Ascend" "${opp_proxy}/Ascend"
export ASCEND_OPP_PATH=${opp_proxy}
cmake -S "${project}" -B "${project}/build_out" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SOURCE_PACKAGE=True \
  -DENABLE_BINARY_PACKAGE=True -DASCEND_COMPUTE_UNIT=ascend910b \
  -Dvendor_name=cruise_paged_kv_order_probe \
  -DASCEND_CANN_PACKAGE_PATH="${cann_home}" \
  -DASCEND_PYTHON_EXECUTABLE="${cann_python_env}/bin/python" \
  -DCMAKE_INSTALL_PREFIX="${project}/build_out" -DENABLE_CROSS_COMPILE=False \
  >"${evidence}/custom-configure.log" 2>&1
cmake --build "${project}/build_out" --target binary package -j8 \
  >"${evidence}/custom-build.log" 2>&1
export ASCEND_OPP_PATH=${system_opp}
package=${project}/build_out/custom_opp_openEuler_aarch64.run
"${package}" --quiet --install-path="${install}" \
  >"${evidence}/custom-install.log" 2>&1
custom_set_env=$(find "${install}" -type f -name set_env.bash | head -n 1)
[[ -n "${custom_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"

"${python_bin}" "${resource_writer}" --template "${resource_template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
"${python_bin}" "${config_writer}" \
  --workspace "${controller_workspace}" \
  --ascend-toolchain "${ascend_toolchain}" \
  --graph-output "${config_dir}/graph.json" \
  --function-output "${config_dir}/function.json" \
  --toolchain-output "${config_dir}/toolchain.json" \
  --deploy-output "${config_dir}/deploy.json"

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
  "${export_dir}/paged_kv_order_probe.air" \
  "${export_dir}/dynamo.pbtxt" <<'PY'
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
  -Wl,--no-whole-archive -o "${build}/paged_kv_order_probe_host" \
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
    "${build}/paged_kv_order_probe_host" "${mode}" \
    "${export_dir}/paged_kv_order_probe.air" "${config_dir}/graph.json" \
    "${config_dir}/function.json" "${config_dir}/deploy.json" \
    "${scratch}/${mode}.json"
  mode_status=$?
  cd "${source_dir}"
  set -e
  printf '%s-exit\t%s\n' "${mode}" "${mode_status}" \
    >>"${evidence}/mode-status.tsv"
  [[ -f "${scratch}/${mode}.json" ]] && \
    cp --reflink=auto "${scratch}/${mode}.json" "${evidence}/${mode}.json"
  find "${mode_driver_logs}" -type f -print0 | sort -z | \
    xargs -0 -r rg -i \
      'LaunchKernel: kernel info.*kernel_name=te_devicepagedkv(update|read)_' \
    >"${evidence}/${mode}-launch-metadata.txt" || true
  wait_for_release
done

cp --reflink=auto "${export_dir}/export-result.json" "${evidence}/"
cp --reflink=auto "${export_dir}/graph-structure.json" "${evidence}/"
cp --reflink=auto "${export_dir}/dynamo.pbtxt" "${evidence}/"
cp --reflink=auto "${config_dir}/graph.json" "${evidence}/graph-config.json"
cp --reflink=auto "${config_dir}/function.json" \
  "${evidence}/function-config.json"
cp --reflink=auto "${config_dir}/deploy.json" "${evidence}/deploy-config.json"
sha256sum "${package}" "${export_dir}/paged_kv_order_probe.air" \
  >"${evidence}/artifact-identity.sha256"
sha256sum "${exporter}" "${inspector}" "${config_writer}" "${host_source}" \
  "${verifier}" "${hardware_policy}" "${script_dir}/run_on_910b.sh" \
  "${controller_source}"/* "${op_host_dir}"/* "${op_kernel_dir}"/* \
  >"${evidence}/source-identity.sha256"
sha256sum "${ASCEND_HOME_PATH}/include/flow_func/flow_msg.h" \
  "${ASCEND_HOME_PATH}/include/flow_func/meta_run_context.h" \
  >"${evidence}/public-flow-api.sha256"
"${python_bin}" - <<'PY' >"${evidence}/package-identity.txt"
import importlib.metadata
for name in ("torch", "torch-npu"):
    print(f"{name}=={importlib.metadata.version(name)}")
PY

set +e
"${python_bin}" "${verifier}" \
  --structure "${evidence}/graph-structure.json" \
  --graph-result "${evidence}/graph.json" \
  --graph-log "${evidence}/graph-launch-metadata.txt" \
  --dataflow-result "${evidence}/dataflow.json" \
  --dataflow-log "${evidence}/dataflow-launch-metadata.txt" \
  --output "${evidence}/verifier.json"
verifier_status=$?
set -e
storage_guard_snapshot kv-order-complete \
  "${evidence}/storage-kv-order-complete.tsv"
exit "${verifier_status}"
