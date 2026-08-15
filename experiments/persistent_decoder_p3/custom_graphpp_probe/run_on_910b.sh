#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-decoder-p3-custom-graphpp-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_CUSTOM_GRAPHPP_SCRATCH:-/dev/shm/cruise-custom-graphpp-${physical_npu}-$$}
cann_home=${CRUISE_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}
cann_set_env=${CRUISE_CANN_SET_ENV:-${cann_home}/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
python_bin=${CRUISE_CUSTOM_GRAPHPP_PYTHON:-$(command -v python3)}
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
resource_writer=${source_dir}/prepare_resource_config.py
resource_template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
custom_template=${cann_home}/tools/new_op_project_template/custom_op
frozen_source=${source_dir}/history/attempts/bf16-materialize-attempt56r1

for required in "${cann_set_env}" "${python_bin}" "${guard}" \
  "${lifecycle_tool}" "${resource_writer}" "${resource_template}" \
  "${cann_python_env}/bin/python" "${custom_template}/CMakeLists.txt" \
  "${frozen_source}/bf16_materialize.cpp" \
  "${frozen_source}/bf16_materialize_def.cpp" \
  "${frozen_source}/bf16_materialize_infershape.cpp" \
  "${frozen_source}/bf16_materialize_tiling.cpp" \
  "${frozen_source}/export_probe.py" \
  "${script_dir}/prepare_probe_config.py" \
  "${script_dir}/bf16_graphpp_probe.cpp" \
  "${script_dir}/verify_probe.py"; do
  [[ -f "${required}" ]] || {
    printf 'missing custom GraphPp probe input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${scratch}" == /dev/shm/cruise-custom-graphpp-* ]] || {
  printf 'custom GraphPp scratch must be PID-scoped under /dev/shm: %s\n' \
    "${scratch}" >&2
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

project=${scratch}/project
opp_proxy=${scratch}/opp
install=${scratch}/install
export_dir=${scratch}/export
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
driver_logs=${scratch}/driver-logs
cache=${scratch}/cache
tmp=${scratch}/tmp
build=${scratch}/host-build
mkdir -p "${opp_proxy}" "${config_dir}" "${deploy_root}" "${driver_logs}" \
  "${cache}" "${tmp}" "${build}"

finalize() {
  local command_status=$? finalize_status=0 cleanup_status=0 lifecycle_status=0
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >"${evidence}/status.tsv"
  for retained_log in "${scratch}"/*.log; do
    [[ -f "${retained_log}" ]] || continue
    tail -c $((1024 * 1024)) -- "${retained_log}" \
      >"${evidence}/$(basename -- "${retained_log}")"
  done
  if [[ ${command_status} -ne 0 && -d "${driver_logs}" ]]; then
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
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

source "${cann_set_env}"
system_opp=${ASCEND_OPP_PATH}
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export RESOURCE_CONFIG_PATH=${config_dir}/numa.json
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_EXPORT_LOG_LEVEL:-3}
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

"${python_bin}" "${resource_writer}" --template "${resource_template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
"${python_bin}" "${script_dir}/prepare_probe_config.py" \
  --graph-output "${config_dir}/graph.json" \
  --deploy-output "${config_dir}/deploy.json"

cp -a "${custom_template}" "${project}"
cp "${frozen_source}/bf16_materialize.cpp" "${project}/op_kernel/"
cp "${frozen_source}/bf16_materialize_def.cpp" "${project}/op_host/"
cp "${frozen_source}/bf16_materialize_infershape.cpp" "${project}/op_host/"
cp "${frozen_source}/bf16_materialize_tiling.cpp" "${project}/op_host/"
ln -s "${system_opp}/built-in" "${opp_proxy}/built-in"
ln -s "${system_opp}/include" "${opp_proxy}/include"
ln -s "${system_opp}/lib64" "${opp_proxy}/lib64"
ln -s "${system_opp}/bin" "${opp_proxy}/bin"
ln -s "${system_opp}/Ascend" "${opp_proxy}/Ascend"
export ASCEND_OPP_PATH=${opp_proxy}
cmake -S "${project}" -B "${project}/build_out" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SOURCE_PACKAGE=True \
  -DENABLE_BINARY_PACKAGE=True -DASCEND_COMPUTE_UNIT=ascend910b \
  -Dvendor_name=cruise_bf16_probe \
  -DASCEND_CANN_PACKAGE_PATH="${cann_home}" \
  -DASCEND_PYTHON_EXECUTABLE="${cann_python_env}/bin/python" \
  -DCMAKE_INSTALL_PREFIX="${project}/build_out" -DENABLE_CROSS_COMPILE=False \
  >"${scratch}/custom-configure.log" 2>&1
cmake --build "${project}/build_out" --target binary package -j8 \
  >"${scratch}/custom-build.log" 2>&1
export ASCEND_OPP_PATH=${system_opp}
package=${project}/build_out/custom_opp_openEuler_aarch64.run
"${package}" --quiet --install-path="${install}" \
  >"${scratch}/custom-install.log" 2>&1
custom_set_env=$(find "${install}" -type f -name set_env.bash | head -n 1)
[[ -n "${custom_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"

set +e
cd "${scratch}"
storage_guard_run_log "${evidence}/export.log" "${evidence}/export.meta.json" \
  600s -- "${python_bin}" "${frozen_source}/export_probe.py" \
  --output-dir "${export_dir}"
export_status=$?
cd "${source_dir}"
set -e
printf 'export-exit\t%s\n' "${export_status}" >"${evidence}/export-status.tsv"
[[ ${export_status} -eq 0 || ${export_status} -eq 139 ]] || \
  exit "${export_status}"
"${python_bin}" - "${export_dir}/export-result.json" \
  "${export_dir}/bf16_materialize_probe.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/input.bin" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("execution_success") is not True:
    raise SystemExit(1)
if not all(Path(path).is_file() for path in sys.argv[2:]):
    raise SystemExit(1)
PY
wait_for_release
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_LOG_LEVEL:-0}

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -pthread \
  -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${script_dir}/bf16_graphpp_probe.cpp" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${build}/bf16_graphpp_probe" \
  >"${scratch}/host-compile.log" 2>&1

: >"${evidence}/mode-status.tsv"
for mode in graph dataflow; do
  mode_driver_logs=${driver_logs}/${mode}
  mkdir -p "${mode_driver_logs}"
  export ASCEND_PROCESS_LOG_PATH=${mode_driver_logs}
  set +e
  cd "${scratch}"
  storage_guard_run_log "${evidence}/${mode}.log" \
    "${evidence}/${mode}.meta.json" 600s -- \
    "${build}/bf16_graphpp_probe" "${mode}" \
    "${export_dir}/bf16_materialize_probe.air" "${config_dir}/graph.json" \
    "${config_dir}/deploy.json" "${export_dir}/input.bin" \
    "${scratch}/${mode}-output.bin" "${scratch}/${mode}.json"
  mode_status=$?
  cd "${source_dir}"
  set -e
  printf '%s-exit\t%s\n' "${mode}" "${mode_status}" \
    >>"${evidence}/mode-status.tsv"
  [[ -f "${scratch}/${mode}.json" ]] && \
    cp --reflink=auto "${scratch}/${mode}.json" "${evidence}/${mode}.json"
  rg -i 'LaunchKernel: kernel info.*kernel_name=te_bf16materialize_' \
    "${mode_driver_logs}" >"${evidence}/${mode}-launch-metadata.txt" || true
  wait_for_release
done

cp --reflink=auto "${export_dir}/export-result.json" \
  "${evidence}/export-result.json"
cp --reflink=auto "${export_dir}/dynamo.pbtxt" "${evidence}/dynamo.pbtxt"
cp --reflink=auto "${config_dir}/graph.json" "${evidence}/graph-config.json"
cp --reflink=auto "${config_dir}/deploy.json" "${evidence}/deploy-config.json"
sha256sum "${package}" "${export_dir}/bf16_materialize_probe.air" \
  "${export_dir}/input.bin" >"${evidence}/artifact-integrity.log"
sha256sum "${frozen_source}"/bf16_materialize*.cpp \
  "${frozen_source}/export_probe.py" "${script_dir}"/*.py \
  "${script_dir}/bf16_graphpp_probe.cpp" "${script_dir}/run_on_910b.sh" \
  >"${evidence}/source-integrity.log"

set +e
"${python_bin}" "${script_dir}/verify_probe.py" \
  --input "${export_dir}/input.bin" \
  --graph-result "${evidence}/graph.json" \
  --graph-output "${scratch}/graph-output.bin" \
  --graph-log "${evidence}/graph-launch-metadata.txt" \
  --dataflow-result "${evidence}/dataflow.json" \
  --dataflow-output "${scratch}/dataflow-output.bin" \
  --dataflow-log "${evidence}/dataflow-launch-metadata.txt" \
  --output "${evidence}/verifier.json"
verifier_status=$?
set -e
storage_guard_snapshot custom-graphpp-complete \
  "${evidence}/storage-custom-graphpp-complete.tsv"
exit "${verifier_status}"
