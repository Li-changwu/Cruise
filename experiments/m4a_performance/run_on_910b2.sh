#!/usr/bin/env bash
set -euo pipefail

source_dir=${CRUISE_SOURCE_DIR:?CRUISE_SOURCE_DIR must name the Cruise checkout}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
artifacts=${CRUISE_ARTIFACTS_DIR:-/dev/shm/cruise-m0-assets-npu01-r1}
run_id=${CRUISE_RUN_ID:-m4a-performance-$(date -u +%Y%m%dT%H%M%SZ)}
run_root=${persistent_root}/${run_id}
evidence=${CRUISE_EVIDENCE_DIR:-${run_root}/evidence}
scratch=${CRUISE_SCRATCH_DIR:-/dev/shm/cruise-${run_id}}
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
conda_sh=${CRUISE_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-vllm-hust-dev}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
vllm_root=${CRUISE_VLLM_ROOT:-${HOME}/vllm-hust}
vllm_ascend_root=${CRUISE_VLLM_ASCEND_ROOT:-${HOME}/vllm-ascend-hust}
runtime_asset_root=${CRUISE_RUNTIME_ASSET_ROOT:-/workspace/cruise-assets}
vllm_kv_cache_bytes=${CRUISE_VLLM_KV_CACHE_BYTES:-536870912}
profile_attribution=${CRUISE_M4A_PROFILE_ATTRIBUTION:-0}

model=${artifacts}/model-frozen
export CRUISE_API_TOKENIZER=${CRUISE_API_TOKENIZER:-${model}}
frozen=${artifacts}/frozen
frozen_air=${frozen}/qwen_b4_decoder_step_attempt69c_r2.air
tiling=${frozen}/explicit_tiling.bin
old_weight_prefix=${CRUISE_OLD_WEIGHT_PREFIX:-/root/ascend-control-g4-20260723/export-attempt69c-b4}
workload=${CRUISE_M4A_WORKLOAD:-${source_dir}/experiments/m4a_performance/workload.json}
generation_config=${source_dir}/experiments/m4a_performance/generation_config/generation_config.json
runner=${source_dir}/experiments/m4a_performance/run_benchmark.py
verifier=${source_dir}/experiments/m4a_performance/verify_results.py
profile_analyzer=${source_dir}/experiments/m4a_performance/analyze_profiles.py

build=${scratch}/native-build
controller=${scratch}/controller
config_dir=${scratch}/config
runtime_func_config=${config_dir}/resident_epoch_func.json
runtime_weight_digest=2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761
runtime_weights=${runtime_asset_root}/runtime-weights/${runtime_weight_digest}
runtime_weights_manifest=${runtime_asset_root}/manifests/${runtime_weight_digest}.json
runtime_air=${scratch}/qwen_b4_decoder_step_${run_id}.air
external_weights=${scratch}/external-weights
driver_cache=${scratch}/driver-cache
driver_cann_logs=${scratch}/driver-cann-logs
driver_tmp=${scratch}/driver-tmp
runtime_workdir=${scratch}/runtime-workdir
deploy_root=${scratch}/dataflow-deploy
runs=${scratch}/runs
profile_root=${scratch}/profiles
resource_config_template=${CRUISE_RESOURCE_CONFIG_TEMPLATE:-${source_dir}/experiments/synthetic-p0/numa_config.physical7.json}
resource_config=${scratch}/numa_config.physical${physical_npu}.json
status=${evidence}/status.tsv

guard=${source_dir}/storage_guard/storage_guard.sh
required=(
  "${guard}"
  "${runner}"
  "${verifier}"
  "${profile_analyzer}"
  "${workload}"
  "${generation_config}"
  "${source_dir}/materialize_runtime_weights.py"
  "${source_dir}/prepare_resource_config.py"
  "${source_dir}/prepare_runtime_config.py"
  "${source_dir}/native/CMakeLists.txt"
  "${source_dir}/controller/CMakeLists.txt"
  "${source_dir}/controller/g4c_b4_resident_epoch.cpp"
  "${resource_config_template}"
  "${frozen_air}"
  "${tiling}"
  "${model}/config.json"
  "${model}/model.safetensors.index.json"
  "${runtime_weights_manifest}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || {
    printf 'missing required input: %s\n' "${path}" >&2
    exit 96
  }
done
[[ -f "${conda_sh}" && -f "${cann_set_env}" ]] || exit 96
[[ -d "${runtime_weights}" ]] || exit 96
[[ "${profile_attribution}" == 0 || "${profile_attribution}" == 1 ]] || exit 96
if [[ "${profile_attribution}" == 1 ]]; then
  for command in jq msprof pgrep ps; do
    command -v "${command}" >/dev/null || {
      printf 'missing profiling command: %s\n' "${command}" >&2
      exit 96
    }
  done
fi

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=96
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((100 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=21600
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 100 24 128

finalize() {
  local command_status=$? finalize_status=0 cleanup_status=0
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >>"${status}"
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

mkdir -p "${build}" "${controller}" "${config_dir}" \
  "${external_weights}" "${driver_cache}" "${driver_cann_logs}" \
  "${driver_tmp}" "${runtime_workdir}" "${deploy_root}" "${runs}"
mkdir -p "${profile_root}"
for path in "${build}" "${controller}" "${config_dir}" "${runtime_air}" \
  "${external_weights}" "${driver_cache}" "${driver_cann_logs}" \
  "${driver_tmp}" "${runtime_workdir}" "${deploy_root}" "${runs}" \
  "${profile_root}" \
  "${resource_config}"; do
  storage_guard_assert_scratch_path "${path}"
done

run_step() {
  local name=$1 timeout_value=$2
  shift 2
  if storage_guard_run_log "${evidence}/${name}.stdout.log" \
    "${evidence}/${name}.stdout.meta.json" "${timeout_value}" -- "$@"; then
    printf '%s\t0\n' "${name}" >>"${status}"
  else
    local step_status=$?
    printf '%s\t%s\n' "${name}" "${step_status}" >>"${status}"
    return "${step_status}"
  fi
}

wait_npu_ready() {
  local label=$1
  storage_guard_wait_for_npu_idle "${physical_npu}" 1800 \
    "${persistent_root}" 24 "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
    "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"
  printf 'npu-ready-%s\t0\n' "${label}" >>"${status}"
}

source "${conda_sh}"
conda activate "${conda_env}"
source "${cann_set_env}"
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export CRUISE_PHYSICAL_NPU=${physical_npu}
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_PROCESS_LOG_PATH=${driver_cann_logs}
export ASCEND_CACHE_PATH=${driver_cache}
export TMPDIR=${driver_tmp}
export TORCHINDUCTOR_CACHE_DIR=${driver_cache}/torchinductor
export TRITON_CACHE_DIR=${driver_cache}/triton
export XDG_CACHE_HOME=${driver_cache}/xdg
export PYTHONDONTWRITEBYTECODE=1
export RESOURCE_CONFIG_PATH=${resource_config}

custom_opp=${artifacts}/custom-opp/install-attempt47/vendors/vllm-ascend
barrier_opp=${artifacts}/custom-opp/install-attempt69a-b4-barrier/vendors/vllm-ascend
materialize_opp=${artifacts}/custom-opp/install-attempt56r1/vendors/vllm-ascend
for vendor in "${custom_opp}" "${barrier_opp}" "${materialize_opp}"; do
  [[ -d "${vendor}/op_impl" && -d "${vendor}/op_proto" && \
     -d "${vendor}/op_api/lib" ]] || exit 96
done
unset ASCEND_CUSTOM_OPP_PATH
export VLLM_ASCEND_RESIDENT_EPOCH_CHILD_ASCEND_CUSTOM_OPP_PATH=${materialize_opp}:${barrier_opp}:${custom_opp}
export VLLM_ASCEND_RESIDENT_EPOCH_CHILD_LIBRARY_PATH=${materialize_opp}/op_api/lib:${barrier_opp}/op_api/lib:${custom_opp}/op_api/lib

export PYTHONPATH=${source_dir}/src:${source_dir}:${vllm_root}:${vllm_ascend_root}:${PYTHONPATH:-}
export VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY=vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine
export VLLM_ASCEND_RESIDENT_EPOCH_AIR=${runtime_air}
export VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG=${source_dir}/config/graph_config.json
export VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG=${runtime_func_config}
export VLLM_ASCEND_RESIDENT_EPOCH_TILING=${tiling}
export VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS=${external_weights}
export VLLM_ASCEND_RESIDENT_EPOCH_RUNTIME_WEIGHTS=${runtime_weights}
export VLLM_ASCEND_RESIDENT_EPOCH_SERVER=${build}/resident_epoch_server
export VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT=3600
export VLLM_ASCEND_RESIDENT_EPOCH_STEPS=2
export VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY=8
export CRUISE_VLLM_KV_CACHE_BYTES=${vllm_kv_cache_bytes}

printf 'case\texit_status\n' >"${status}"
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
git -C "${source_dir}" remote get-url origin >"${evidence}/source-origin.txt"
git -C "${source_dir}" status --porcelain --untracked-files=all \
  >"${evidence}/source-worktree-pre.txt"
[[ ! -s "${evidence}/source-worktree-pre.txt" ]] || exit 96

run_step prepare-resource-config 120s python3 \
  "${source_dir}/prepare_resource_config.py" \
  --template "${resource_config_template}" \
  --physical-npu "${physical_npu}" \
  --deploy-root "${deploy_root}" \
  --output "${resource_config}"

{
  printf 'key\tvalue\n'
  printf 'run_id\t%s\n' "${run_id}"
  printf 'source_dir\t%s\n' "${source_dir}"
  printf 'conda_env\t%s\n' "${conda_env}"
  printf 'physical_npu\t%s\n' "${physical_npu}"
  printf 'scratch_dir\t%s\n' "${scratch}"
  printf 'evidence_dir\t%s\n' "${evidence}"
  printf 'model\t%s\n' "${model}"
  printf 'runtime_weights\t%s\n' "${runtime_weights}"
  printf 'runtime_asset_root\t%s\n' "${runtime_asset_root}"
  printf 'resource_config\t%s\n' "${resource_config}"
  printf 'dataflow_deploy_root\t%s\n' "${deploy_root}"
  printf 'vllm_kv_cache_bytes\t%s\n' "${vllm_kv_cache_bytes}"
  printf 'execution_scope\t%s\n' "$([[ "${profile_attribution}" == 1 ]] && printf profiling || printf formal)"
  printf 'formal_m2\topen\n'
  printf 'formal_m3\topen\n'
  printf 'formal_m4\topen\n'
} >"${evidence}/deployment-config.tsv"
sha256sum "${workload}" "${generation_config}" "${frozen_air}" "${tiling}" "${resource_config}" \
  "${runtime_weights_manifest}" >"${evidence}/input-integrity.log"
git -C "${vllm_root}" rev-parse HEAD >"${evidence}/vllm-commit.txt"
git -C "${vllm_ascend_root}" rev-parse HEAD \
  >"${evidence}/vllm-ascend-commit.txt"

run_step unit-tests 900s env \
  PYTHONPATH="${vllm_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m pytest -q "${source_dir}/tests"
run_step prepare-controller 120s cp -a \
  "${source_dir}/controller/." "${controller}/"
run_step prepare-runtime-config 120s python3 \
  "${source_dir}/prepare_runtime_config.py" \
  --template "${source_dir}/config/resident_epoch_func.json" \
  --controller-workspace "${controller}" \
  --output "${runtime_func_config}"
run_step cmake 600s cmake -S "${source_dir}/native" -B "${build}"
run_step build 1800s cmake --build "${build}" --parallel 2
run_step verify-runtime-weights 3600s python3 \
  "${source_dir}/materialize_runtime_weights.py" \
  --model-dir "${model}" \
  --output-dir "${runtime_weights}" \
  --manifest "${runtime_weights_manifest}" \
  --persistent-asset-root "${runtime_asset_root}"
cp "${runtime_weights_manifest}" "${evidence}/runtime-weights-manifest.json"
run_step relocate-runtime-air 600s "${build}/relocate_air_paths" \
  "${frozen_air}" "${runtime_air}" "${old_weight_prefix}" \
  "${runtime_weights}" "${evidence}/air-relocation.json"

cd "${runtime_workdir}"

retain_bounded_log() {
  local source_log=$1 name=$2
  python3 "${source_dir}/storage_guard/bounded_log.py" \
    --output "${evidence}/${name}.log" \
    --metadata "${evidence}/${name}.meta.json" \
    --head-bytes 1048576 --tail-bytes 1048576 <"${source_log}"
}

find_profile_target() {
  local api_pid=$1 pattern=$2 candidates candidate current parent
  for _ in $(seq 1 600); do
    candidates=$(pgrep -f -- "${pattern}" || true)
    for candidate in ${candidates}; do
      current=${candidate}
      while [[ ${current} -gt 1 ]]; do
        if [[ ${current} -eq ${api_pid} ]]; then
          printf '%s\n' "${candidate}"
          return 0
        fi
        parent=$(ps -o ppid= -p "${current}" 2>/dev/null | tr -d ' ')
        [[ "${parent}" =~ ^[0-9]+$ ]] || break
        current=${parent}
      done
    done
    sleep 0.5
  done
  return 1
}

run_profile_route() {
  local mode=$1
  local route_code=${mode:0:1}
  local runtime=${scratch}/p/${route_code}
  local ready=${runtime}/profile-ready.json
  local start=${runtime}/profile-start
  local result=${evidence}/profile-${mode}.json
  local benchmark_stdout=${scratch}/profile-${mode}-benchmark.stdout
  local msprof_stdout=${scratch}/profile-${mode}-msprof.stdout
  local target_pattern target_pid api_pid benchmark_pid profiler_pid
  local benchmark_status=0 profiler_status=0

  wait_npu_ready "pre-profile-${mode}"
  if [[ "${mode}" == cruise ]]; then
    find "${external_weights}" -depth -mindepth 1 -delete
    target_pattern=${build}/resident_epoch_server
  else
    target_pattern='VLLM::EngineCore'
  fi

  timeout --signal=TERM --kill-after=30s 7200s python3 "${runner}" \
    --mode "${mode}" --run-label "${mode}-profile" --model "${model}" \
    --workload "${workload}" --runtime-dir "${runtime}" \
    --only-scenario decode-stream-c4 \
    --profile-ready-file "${ready}" --profile-start-file "${start}" \
    --output "${result}" >"${benchmark_stdout}" 2>&1 &
  benchmark_pid=$!

  for _ in $(seq 1 2400); do
    [[ -f "${ready}" ]] && break
    if ! kill -0 "${benchmark_pid}" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if [[ ! -f "${ready}" ]]; then
    if wait "${benchmark_pid}"; then benchmark_status=0; else benchmark_status=$?; fi
    retain_bounded_log "${benchmark_stdout}" "profile-${mode}-benchmark"
    printf 'profile-%s-ready\t%s\n' "${mode}" "${benchmark_status}" >>"${status}"
    [[ ${benchmark_status} -ne 0 ]] || benchmark_status=97
    return "${benchmark_status}"
  fi

  api_pid=$(jq -er '.api_server_pid' "${ready}")
  if ! target_pid=$(find_profile_target "${api_pid}" "${target_pattern}"); then
    : >"${start}"
    if wait "${benchmark_pid}"; then benchmark_status=0; else benchmark_status=$?; fi
    retain_bounded_log "${benchmark_stdout}" "profile-${mode}-benchmark"
    printf 'profile-%s-target\t124\n' "${mode}" >>"${status}"
    return 124
  fi
  printf '%s\n' "${target_pid}" >"${evidence}/profile-${mode}-target-pid.txt"
  timeout --signal=TERM --kill-after=30s 180s msprof \
    --output="${profile_root}/${mode}" --dynamic=on --pid="${target_pid}" \
    --duration=30 --runtime-api=on --ge-api=l0 --task-time=l1 \
    --ai-core=on --aic-metrics=PipeUtilization --storage-limit=256MB \
    >"${msprof_stdout}" 2>&1 &
  profiler_pid=$!
  sleep 8
  : >"${start}"

  if wait "${benchmark_pid}"; then benchmark_status=0; else benchmark_status=$?; fi
  if wait "${profiler_pid}"; then profiler_status=0; else profiler_status=$?; fi
  retain_bounded_log "${benchmark_stdout}" "profile-${mode}-benchmark"
  retain_bounded_log "${msprof_stdout}" "profile-${mode}-msprof"
  printf 'profile-%s-benchmark\t%s\n' "${mode}" "${benchmark_status}" >>"${status}"
  printf 'profile-%s-msprof\t%s\n' "${mode}" "${profiler_status}" >>"${status}"
  [[ ${benchmark_status} -eq 0 ]]
}

if [[ "${profile_attribution}" == 1 ]]; then
  run_profile_route graph
  run_profile_route cruise
  run_step analyze-profiles 300s python3 "${profile_analyzer}" \
    --profile-root "${profile_root}" --evidence-dir "${evidence}" \
    --status "${status}" \
    --output "${evidence}/profile-summary.json"
  wait_npu_ready final
  git -C "${source_dir}" status --porcelain --untracked-files=all \
    >"${evidence}/source-worktree-final.txt"
  [[ ! -s "${evidence}/source-worktree-final.txt" ]] || exit 93
  sha256sum "${evidence}"/*.json >"${evidence}/result-integrity.log"
  printf 'complete\t0\n' >>"${status}"
  exit 0
fi

order=(eager-1 graph-1 cruise-1 cruise-2 graph-2 eager-2 graph-3 cruise-3 eager-3)
result_args=()
for label in "${order[@]}"; do
  mode=${label%%-*}
  wait_npu_ready "pre-${label}"
  if [[ "${mode}" == cruise ]]; then
    storage_guard_assert_scratch_path "${external_weights}"
    find "${external_weights}" -depth -mindepth 1 -delete
  fi
  result=${evidence}/${label}.json
  run_step "benchmark-${label}" 7200s python3 "${runner}" \
    --mode "${mode}" --run-label "${label}" --model "${model}" \
    --workload "${workload}" --runtime-dir "${runs}/${label}" \
    --output "${result}"
  result_args+=(--result "${result}")
done

comparison_status=0
if run_step compare 300s python3 "${runner}" --mode compare \
  --workload "${workload}" "${result_args[@]}" \
  --output "${evidence}/comparison.json"; then
  :
else
  comparison_status=$?
fi
verification_status=0
if run_step verify 300s python3 "${verifier}" \
  --comparison "${evidence}/comparison.json" --workload "${workload}" \
  "${result_args[@]}" --output "${evidence}/verifier.json"; then
  :
else
  verification_status=$?
fi

wait_npu_ready final
git -C "${source_dir}" status --porcelain --untracked-files=all \
  >"${evidence}/source-worktree-final.txt"
[[ ! -s "${evidence}/source-worktree-final.txt" ]] || exit 93
sha256sum "${evidence}"/*.json >"${evidence}/result-integrity.log"
printf 'complete\t0\n' >>"${status}"
if [[ ${comparison_status} -ne 0 ]]; then
  exit "${comparison_status}"
fi
if [[ ${verification_status} -ne 0 ]]; then
  exit "${verification_status}"
fi
