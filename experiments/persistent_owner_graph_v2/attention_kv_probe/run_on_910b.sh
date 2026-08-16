#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
v2_dir=$(cd -- "${script_dir}/.." && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
fia_probe_dir=${source_dir}/experiments/persistent_decoder_p3/fia_graphpp_probe
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
max_stage=${CRUISE_V2_KV_ATTENTION_MAX_STAGE:-dataflow}
allow_diagnostic_dirty=${CRUISE_ALLOW_DIAGNOSTIC_DIRTY:-0}
run_id=${CRUISE_RUN_ID:-persistent-owner-v2-kv-attention-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_V2_KV_ATTENTION_SCRATCH:-/dev/shm/cruise-v2-kv-attention-${physical_npu}-$$}
python_bin=${CRUISE_V2_KV_ATTENTION_PYTHON:-$(command -v python3)}
cann_home=${CRUISE_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}
cann_set_env=${CRUISE_CANN_SET_ENV:-${cann_home}/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
ops_source=${CRUISE_FIA_OPS_SOURCE:-/dev/shm/cruise-ops-transformer-9.0.0-audit}
guard=${source_dir}/storage_guard/storage_guard.sh
hardware_policy=${v2_dir}/hardware_policy.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
resource_writer=${source_dir}/prepare_resource_config.py
resource_template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
custom_template=${cann_home}/tools/new_op_project_template/custom_op
fia_builder=${fia_probe_dir}/build_isolated_opp.sh
exporter=${script_dir}/export_probe.py
inspector=${script_dir}/inspect_probe_graph.py
config_writer=${script_dir}/prepare_probe_config.py
host_source=${script_dir}/attention_kv_probe_host.cpp
verifier=${script_dir}/verify_probe.py
controller_source=${script_dir}/controller
op_kernel_dir=${script_dir}/op_kernel
op_host_dir=${script_dir}/op_host

case "${max_stage}" in
  export|graph|owner|dataflow) ;;
  *)
    printf 'invalid V2 KV attention max stage: %s\n' "${max_stage}" >&2
    exit 96
    ;;
esac
case "${allow_diagnostic_dirty}" in
  0|1) ;;
  *)
    printf 'invalid diagnostic dirty-worktree setting: %s\n' \
      "${allow_diagnostic_dirty}" >&2
    exit 96
    ;;
esac

for required in "${python_bin}" "${cann_set_env}" "${guard}" \
  "${hardware_policy}" "${lifecycle_tool}" "${resource_writer}" \
  "${resource_template}" "${cann_python_env}/bin/python" \
  "${custom_template}/CMakeLists.txt" "${fia_builder}" "${ops_source}/build.sh" \
  "${exporter}" "${inspector}" "${config_writer}" "${host_source}" \
  "${verifier}" "${controller_source}/CMakeLists.txt" \
  "${controller_source}/attention_kv_controller.cpp" \
  "${op_kernel_dir}/device_paged_kv_update.cpp" \
  "${op_kernel_dir}/device_query_after_kv_update.cpp" \
  "${op_host_dir}/device_paged_kv_update_def.cpp" \
  "${op_host_dir}/device_paged_kv_update_infershape.cpp" \
  "${op_host_dir}/device_paged_kv_update_tiling.cpp" \
  "${op_host_dir}/device_query_after_kv_update_def.cpp" \
  "${op_host_dir}/device_query_after_kv_update_infershape.cpp" \
  "${op_host_dir}/device_query_after_kv_update_tiling.cpp"; do
  [[ -f "${required}" ]] || {
    printf 'missing V2 KV attention probe input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${scratch}" == /dev/shm/cruise-v2-kv-attention-* ]] || {
  printf 'V2 KV attention scratch must be PID-scoped under /dev/shm: %s\n' \
    "${scratch}" >&2
  exit 96
}

source "${guard}"
source "${hardware_policy}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=1
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((256 * 1024 * 1024))
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
custom_install=${scratch}/custom-install
fia_work=${scratch}/fia-source
fia_install=${scratch}/fia-install
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

extract_failure_logs() {
  local mode source_file relative target
  [[ -d "${driver_logs}" ]] || return 0
  find "${driver_logs}" -type f -printf '%P\t%s\n' | sort \
    >"${evidence}/driver-log-manifest.tsv"
  for mode in graph dataflow; do
    [[ -d "${driver_logs}/${mode}" ]] || continue
    mkdir -p "${evidence}/failure-driver-logs/${mode}"
    cp -a "${driver_logs}/${mode}/." \
      "${evidence}/failure-driver-logs/${mode}/"
    find "${driver_logs}/${mode}" -type f -print0 | sort -z | \
      xargs -0 -r rg -n -i \
        'error|failed|failure|invalid|exception|fault|107000' | \
      tail -n 8192 >"${evidence}/${mode}-error-index.log" || true
  done
  while IFS= read -r -d '' source_file; do
    relative=${source_file#"${driver_logs}"/}
    target=${evidence}/failure-driver-logs/${relative}
    mkdir -p "$(dirname -- "${target}")"
    tail -c $((512 * 1024)) -- "${source_file}" >"${target}"
  done < <(find "${driver_logs}" -mindepth 1 -maxdepth 1 -type f -print0 | \
    sort -z)
}

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
    extract_failure_logs
  fi
  for root in "${custom_install}" "${fia_install}"; do
    [[ -d "${root}" ]] && find "${root}" -type d -exec chmod u+w {} +
  done
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
  if [[ "${allow_diagnostic_dirty}" != 1 ]]; then
    printf 'V2 KV attention hardware probe requires a clean worktree\n' >&2
    exit 94
  fi
  git -C "${source_dir}" diff --check
  git -C "${source_dir}" diff --binary \
    >"${evidence}/source-worktree.patch"
}
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
v2_capture_hbm_baseline "${evidence}" "${physical_npu}"
source "${cann_set_env}"
system_opp=${ASCEND_OPP_PATH}
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
ascend_toolchain=${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
[[ -x "${ascend_toolchain}" ]] || exit 96
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
  "${controller_source}/attention_kv_controller.cpp" \
  "${controller_workspace}/"
for component in built-in include lib64 bin Ascend; do
  ln -s "${system_opp}/${component}" "${opp_proxy}/${component}"
done
export ASCEND_OPP_PATH=${opp_proxy}
cmake -S "${project}" -B "${project}/build_out" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SOURCE_PACKAGE=True \
  -DENABLE_BINARY_PACKAGE=True -DASCEND_COMPUTE_UNIT=ascend910b \
  -Dvendor_name=cruise_attention_kv_probe \
  -DASCEND_CANN_PACKAGE_PATH="${cann_home}" \
  -DASCEND_PYTHON_EXECUTABLE="${cann_python_env}/bin/python" \
  -DCMAKE_INSTALL_PREFIX="${project}/build_out" -DENABLE_CROSS_COMPILE=False \
  >"${evidence}/custom-configure.log" 2>&1
cmake --build "${project}/build_out" --target binary package -j8 \
  >"${evidence}/custom-build.log" 2>&1
custom_package=${project}/build_out/custom_opp_openEuler_aarch64.run
"${custom_package}" --quiet --install-path="${custom_install}" \
  >"${evidence}/custom-install.log" 2>&1
custom_set_env=$(find "${custom_install}" -type f -name set_env.bash -print -quit)
[[ -f "${custom_set_env}" ]] || exit 95

fia_set_env=$("${fia_builder}" --source "${ops_source}" \
  --work-root "${fia_work}" --install-root "${fia_install}" \
  --evidence "${evidence}")
[[ -f "${fia_set_env}" ]] || exit 95
export ASCEND_OPP_PATH=${system_opp}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "${custom_set_env}"
custom_opp_path=${ASCEND_CUSTOM_OPP_PATH}
source "${fia_set_env}"
export ASCEND_CUSTOM_OPP_PATH=${custom_opp_path}:${ASCEND_CUSTOM_OPP_PATH}

"${python_bin}" "${resource_writer}" --template "${resource_template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"

set +e
cd "${scratch}"
storage_guard_run_log "${evidence}/export.log" "${evidence}/export.meta.json" \
  600s -- "${python_bin}" "${exporter}" --output-dir "${export_dir}"
export_status=$?
cd "${source_dir}"
set -e
printf 'export-exit\t%s\n' "${export_status}" >"${evidence}/export-status.tsv"
[[ ${export_status} -eq 0 || ${export_status} -eq 139 ]] || exit "${export_status}"
"${python_bin}" - "${export_dir}/export-result.json" \
  "${export_dir}/attention_kv_probe.air" "${export_dir}/dynamo.pbtxt" \
  "${export_dir}/graph-abi.json" <<'PY'
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

"${python_bin}" "${config_writer}" \
  --workspace "${controller_workspace}" \
  --abi-input "${export_dir}/graph-abi.json" \
  --header-output "${controller_workspace}/attention_kv_graph_abi.h" \
  --ascend-toolchain "${ascend_toolchain}" \
  --graph-output "${config_dir}/graph.json" \
  --function-output "${config_dir}/function.json" \
  --toolchain-output "${config_dir}/toolchain.json" \
  --deploy-output "${config_dir}/deploy.json"

cp --reflink=auto "${export_dir}/export-result.json" "${evidence}/"
cp --reflink=auto "${export_dir}/graph-structure.json" "${evidence}/"
cp --reflink=auto "${export_dir}/graph-abi.json" "${evidence}/"
cp --reflink=auto "${export_dir}/dynamo.pbtxt" "${evidence}/"
cp --reflink=auto "${config_dir}/graph.json" "${evidence}/graph-config.json"
cp --reflink=auto "${config_dir}/function.json" "${evidence}/function-config.json"
cp --reflink=auto "${config_dir}/deploy.json" "${evidence}/deploy-config.json"
cp --reflink=auto "${controller_workspace}/attention_kv_graph_abi.h" \
  "${evidence}/"
sha256sum "${custom_package}" "${export_dir}/attention_kv_probe.air" \
  "${export_dir}/graph-abi.json" \
  >"${evidence}/artifact-identity.sha256"
sha256sum "${exporter}" "${inspector}" "${config_writer}" "${host_source}" \
  "${verifier}" "${hardware_policy}" "${script_dir}/run_on_910b.sh" \
  "${controller_source}"/* "${op_host_dir}"/* "${op_kernel_dir}"/* \
  >"${evidence}/source-identity.sha256"
sha256sum "${ASCEND_HOME_PATH}/include/flow_func/flow_msg.h" \
  "${ASCEND_HOME_PATH}/include/flow_func/meta_run_context.h" \
  "${ASCEND_HOME_PATH}/include/graph/tensor.h" \
  >"${evidence}/public-api.sha256"
printf '%s\n' "${max_stage}" >"${evidence}/max-stage.txt"

if [[ "${max_stage}" == export ]]; then
  storage_guard_snapshot kv-attention-export-complete \
    "${evidence}/storage-kv-attention-export-complete.tsv"
  exit 0
fi

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -pthread \
  -I"${controller_workspace}" \
  -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${host_source}" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive "${ASCEND_HOME_PATH}/lib64/libascendcl.so" \
  -o "${build}/attention_kv_probe_host" >"${evidence}/host-compile.log" 2>&1

: >"${evidence}/mode-status.tsv"
case "${max_stage}" in
  graph) modes=(graph) ;;
  owner) modes=(dataflow) ;;
  dataflow) modes=(graph dataflow) ;;
esac
for mode in "${modes[@]}"; do
  mode_driver_logs=${driver_logs}/${mode}
  mkdir -p "${mode_driver_logs}"
  export ASCEND_PROCESS_LOG_PATH=${mode_driver_logs}
  set +e
  cd "${scratch}"
  storage_guard_run_log "${evidence}/${mode}.log" \
    "${evidence}/${mode}.meta.json" 900s -- \
    "${build}/attention_kv_probe_host" "${mode}" \
    "${export_dir}/attention_kv_probe.air" "${config_dir}/graph.json" \
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
      'LaunchKernel: kernel info.*kernel_name=te_(devicepagedkvupdate|devicequeryafterkvupdate|fusedinferattentionscore)_' \
    >"${evidence}/${mode}-launch-metadata.txt" || true
  find "${mode_driver_logs}" -type f -print0 | sort -z | \
    xargs -0 -r rg -i \
      'MemcpyAsync|Memcpy|TensorMove|TransData|H2D|D2H|copy task' | \
    sed -n '1,4096p' >"${evidence}/${mode}-transfer-metadata.txt" || true
  wait_for_release
  [[ ${mode_status} -eq 0 ]] || exit "${mode_status}"
  if [[ "${mode}" == graph ]]; then
    "${python_bin}" - "${evidence}/graph.json" \
      "${evidence}/graph-launch-metadata.txt" <<'PY'
import json
import re
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
launches = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
required = (
    r"kernel_name=te_devicepagedkvupdate_",
    r"kernel_name=te_devicequeryafterkvupdate_",
    r"kernel_name=te_fusedinferattentionscore_",
)
if (
    result.get("pass") is not True
    or result.get("attention_exact") is not True
    or result.get("ticket_exact") is not True
    or result.get("metadata_input_index") != 2
    or result.get("query_input_index") != 3
    or not all(re.search(pattern, launches, re.IGNORECASE) for pattern in required)
):
    raise SystemExit(1)
PY
  fi
done

if [[ "${max_stage}" == graph ]]; then
  storage_guard_snapshot kv-attention-graph-complete \
    "${evidence}/storage-kv-attention-graph-complete.tsv"
  exit 0
fi

set +e
verifier_route=combined
[[ "${max_stage}" == owner ]] && verifier_route=owner
verifier_args=(
  --route "${verifier_route}"
  --structure "${evidence}/graph-structure.json"
  --dataflow-result "${evidence}/dataflow.json"
  --dataflow-log "${evidence}/dataflow-launch-metadata.txt"
  --dataflow-transfer-log "${evidence}/dataflow-transfer-metadata.txt"
  --output "${evidence}/verifier.json"
)
if [[ "${verifier_route}" == combined ]]; then
  verifier_args+=(
    --graph-result "${evidence}/graph.json"
    --graph-log "${evidence}/graph-launch-metadata.txt"
    --graph-transfer-log "${evidence}/graph-transfer-metadata.txt"
  )
fi
"${python_bin}" "${verifier}" "${verifier_args[@]}"
verifier_status=$?
set -e
storage_guard_snapshot kv-attention-complete \
  "${evidence}/storage-kv-attention-complete.tsv"
exit "${verifier_status}"
