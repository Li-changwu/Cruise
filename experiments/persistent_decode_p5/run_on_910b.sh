#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../.." && pwd)
p3_dir=${source_dir}/experiments/persistent_decoder_p3
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
stop_after_graph1=${CRUISE_P5_STOP_AFTER_GRAPH1:-0}
stop_after_first_pair=${CRUISE_P5_STOP_AFTER_FIRST_PAIR:-0}
run_id=${CRUISE_RUN_ID:-persistent-decode-p5-matrix-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${CRUISE_EVIDENCE_DIR:-${run_root}/evidence}
scratch=${CRUISE_P5_SCRATCH:-/dev/shm/cruise-p5-matrix-${physical_npu}-$$}
air=${CRUISE_P5_AIR:-/workspace/cruise-assets/p3-air384-fia-37bd7557850a72b4/qwen_b4_p3_decoder_step.runtime.air}
external_weights=${CRUISE_P5_EXTERNAL_WEIGHT_SOURCE:-/workspace/cruise-assets/runtime-weights/p3-ge-external-420a16406d4f8723}
model_revision=a09a35458c702b33eeacc393d103063234e8bc28
model_manifest_sha256=651d64436c415afd6faf3d82f14086c2df54d99c02f66f81a15b83a2d17de5f9
model=${CRUISE_P5_MODEL:-/workspace/cruise-assets/models/Qwen2.5-7B-Instruct-${model_revision}}
tokenizer=${CRUISE_P5_TOKENIZER:-${model}}
model_revision_marker=${model}/.cruise-model-revision
model_sha256_manifest=${model}/.cruise-model-sha256
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
service_python=${CRUISE_P5_SERVICE_PYTHON:-$(command -v python3)}
template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
resource_writer=${source_dir}/prepare_resource_config.py
config_writer=${p3_dir}/prepare_p3_config.py
controller_source=${p3_dir}/controller
native_source=${script_dir}/persistent_owner_transport.cpp
p4_host_source=${source_dir}/experiments/persistent_admission_p4/persistent_admission_p4_host.cpp
workload=${script_dir}/workload.json
runner=${script_dir}/run_p5_benchmark.py
verifier=${script_dir}/verify_p5_gate.py
pair_verifier=${script_dir}/verify_p5_pair.py
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
external_view_tool=${p3_dir}/manage_ge_external_view.py
external_view=${run_root}/.ge-external-view

[[ "${stop_after_graph1}" == 0 || "${stop_after_graph1}" == 1 ]] || {
  printf 'CRUISE_P5_STOP_AFTER_GRAPH1 must be 0 or 1: %s\n' \
    "${stop_after_graph1}" >&2
  exit 96
}
[[ "${stop_after_first_pair}" == 0 || "${stop_after_first_pair}" == 1 ]] || {
  printf 'CRUISE_P5_STOP_AFTER_FIRST_PAIR must be 0 or 1: %s\n' \
    "${stop_after_first_pair}" >&2
  exit 96
}

for required in "${cann_set_env}" "${cann_python_env}/bin/python" \
  "${service_python}" "${template}" "${resource_writer}" "${config_writer}" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" "${native_source}" \
  "${p4_host_source}" "${workload}" "${runner}" "${verifier}" \
  "${pair_verifier}" \
  "${guard}" "${lifecycle_tool}" "${external_view_tool}" "${air}" \
  "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json" \
  "${model}/config.json" "${model}/model.safetensors.index.json" \
  "${model}/tokenizer.json" "${model_revision_marker}" \
  "${model_sha256_manifest}"; do
  [[ -f "${required}" ]] || {
    printf 'missing required P5 input: %s\n' "${required}" >&2
    exit 96
  }
done
for required_dir in "${model}" "${tokenizer}" "${external_weights}"; do
  [[ -d "${required_dir}" ]] || {
    printf 'missing required P5 directory: %s\n' "${required_dir}" >&2
    exit 96
  }
done
[[ $(<"${model_revision_marker}") == "${model_revision}" ]] || {
  printf 'P5 model revision marker mismatch: %s\n' "${model_revision_marker}" >&2
  exit 96
}
(cd "${model}" && sha256sum --check --strict \
  "${model_sha256_manifest}") >/dev/null
read -r actual_model_manifest_sha256 _ < <(sha256sum "${model_sha256_manifest}")
[[ "${actual_model_manifest_sha256}" == "${model_manifest_sha256}" ]] || {
  printf 'P5 model manifest digest mismatch: %s\n' \
    "${model_sha256_manifest}" >&2
  exit 96
}
[[ "${scratch}" == /dev/shm/cruise-p5-* ]] || {
  printf 'P5 scratch must be PID-scoped under /dev/shm/cruise-p5-*: %s\n' \
    "${scratch}" >&2
  exit 96
}

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=${CRUISE_P5_MAX_SCRATCH_GIB:-2}
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=${CRUISE_P5_MAX_EVIDENCE_BYTES:-$((64 * 1024 * 1024))}
export STORAGE_GUARD_NPU_WAIT_SECONDS=${CRUISE_P5_NPU_WAIT_SECONDS:-120}
export STORAGE_GUARD_NPU_STABLE_SAMPLES=${CRUISE_P5_NPU_STABLE_SAMPLES:-2}
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

build=${scratch}/build
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
controller_workspace=${scratch}/controller
driver_logs=${scratch}/driver-logs
status=${evidence}/status.tsv
mkdir -p "${build}" "${config_dir}" "${deploy_root}" \
  "${controller_workspace}" "${driver_logs}"
cp "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" \
  "${controller_workspace}/"
sha256sum "${native_source}" "${p4_host_source}" "${workload}" \
  "${runner}" "${verifier}" "${pair_verifier}" \
  "${source_dir}/src/vllm_ascend_persistent_owner/server.py" \
  "${source_dir}/src/vllm_ascend_persistent_owner/owner_transport.py" \
  "${source_dir}/src/vllm_ascend_persistent_owner/graph_baseline_launcher.py" \
  "${source_dir}/src/vllm_ascend_persistent_owner/graph_rmsnorm_compat.py" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/persistent_decoder_p3.cpp" "${config_writer}" \
  "${external_view_tool}" "${air}" "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json" "${model_revision_marker}" \
  "${model_sha256_manifest}" >"${evidence}/source-identity.sha256"

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
export PYTHONPATH=${source_dir}/src:${source_dir}:${PYTHONPATH:-}
export CRUISE_P5_RUN_ROOT=${run_root}
export CRUISE_P5_EXTERNAL_WEIGHT_DIR=${external_view}

"${cann_python_env}/bin/python" "${resource_writer}" --template "${template}" \
  --physical-npu "${physical_npu}" --deploy-root "${deploy_root}" \
  --output "${RESOURCE_CONFIG_PATH}"
function_config=${config_dir}/function.json
graph_config=${config_dir}/graph.json
toolchain_config=${config_dir}/toolchain.json
deploy_config=${config_dir}/deploy.json
"${cann_python_env}/bin/python" "${config_writer}" \
  --workspace "${controller_workspace}" \
  --ascend-toolchain "${ascend_toolchain}" \
  --function-output "${function_config}" \
  --graph-output "${graph_config}" \
  --toolchain-output "${toolchain_config}" \
  --deploy-output "${deploy_config}"

owner_binary=${build}/persistent_owner_transport
g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -pthread \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/external" \
  "${native_source}" \
  -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${owner_binary}" \
  >"${scratch}/owner-compile.log" 2>&1

review_intermediates() {
  local label=$1
  storage_guard_runtime_budget_ok 0
  storage_guard_snapshot "${label}" "${evidence}/storage-${label}.tsv"
  find "${driver_logs}" -type f -size +16M -delete
}

wait_for_release() {
  for _ in $(seq 1 180); do
    if npu-smi info -t proc-mem -i "${physical_npu}" 2>&1 | \
      rg -q 'No process in device\.'; then
      return 0
    fi
    sleep 1
  done
  return 95
}

finalize() {
  local command_status=$? view_status=0 guard_status=0 cleanup_status=0 lifecycle_status=0
  local retention_class=evidence retention_days=30
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >>"${status}"
  if [[ -d "${external_view}" ]]; then
    python3 "${external_view_tool}" cleanup --run-root "${run_root}" \
      --view "${external_view}" \
      --receipt "${evidence}/ge-external-view-cleanup.json"
    view_status=$?
  fi
  storage_guard_finalize
  guard_status=$?
  if [[ ${guard_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${guard_status} -eq 0 && ${cleanup_status} -eq 0 ]]; then
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
  if [[ ${guard_status} -ne 0 ]]; then exit "${guard_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

review_intermediates after-compile
wait_for_release
python3 "${external_view_tool}" create --source "${external_weights}" \
  --run-root "${run_root}" --view "${external_view}"

labels=(graph-1 owner-1 owner-2 graph-2 graph-3 owner-3)
inputs=()
owner_start=0
for label in "${labels[@]}"; do
  route=${label%%-*}
  index=${label##*-}
  runtime=${scratch}/runtime-${label}
  output=${evidence}/${label}.json
  inputs+=(--input "${output}")
  wait_for_release
  command=(
    "${service_python}" "${runner}"
    --route "${route}"
    --label "${label}"
    --workload "${workload}"
    --output "${output}"
    --runtime-dir "${runtime}"
    --model "${model}"
    --tokenizer "${tokenizer}"
  )
  if [[ "${route}" == owner ]]; then
    owner_start=$((owner_start + 1))
    command+=(
      --owner-executable "${owner_binary}"
      --function-config "${function_config}"
      --graph-config "${graph_config}"
      --deploy-config "${deploy_config}"
      --air "${air}"
      --owner-id "$((5100 + owner_start))"
    )
  fi
  timeout "${CRUISE_P5_START_TIMEOUT:-10800}s" "${command[@]}"
  wait_for_release
  review_intermediates "after-${label}"
  rm -rf -- "${runtime}"
  if [[ "${label}" == graph-1 && "${stop_after_graph1}" == 1 ]]; then
    printf 'P5_GRAPH1_SMOKE_COMPLETE result=%s\n' "${output}"
    exit 0
  fi
  if [[ "${label}" == owner-1 ]]; then
    "${service_python}" "${pair_verifier}" \
      --graph "${evidence}/graph-1.json" \
      --owner "${evidence}/owner-1.json" \
      --output "${evidence}/p5-first-pair.json"
    review_intermediates after-first-pair
    if [[ "${stop_after_first_pair}" == 1 ]]; then
      printf 'P5_FIRST_PAIR_COMPLETE result=%s\n' \
        "${evidence}/p5-first-pair.json"
      exit 0
    fi
  fi
done

"${service_python}" "${verifier}" --workload "${workload}" \
  "${inputs[@]}" --output "${evidence}/p5-gate.json"
review_intermediates after-gate
printf 'P5_MATRIX_COMPLETE gate=%s\n' "${evidence}/p5-gate.json"
