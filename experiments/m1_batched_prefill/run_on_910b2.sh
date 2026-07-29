#!/usr/bin/env bash
set -euo pipefail

source_dir=${CRUISE_SOURCE_DIR:?CRUISE_SOURCE_DIR must name the Cruise checkout}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/root/ascend-control-cruise-20260729}
artifacts=${CRUISE_ARTIFACTS_DIR:-${persistent_root}/artifacts}
run_id=${CRUISE_RUN_ID:-m1-batched-prefill-$(date -u +%Y%m%dT%H%M%SZ)}
evidence=${CRUISE_EVIDENCE_DIR:-${persistent_root}/evidence-${run_id}}
scratch=${CRUISE_SCRATCH_DIR:-/dev/shm/cruise-${run_id}}
physical_npu=${CRUISE_PHYSICAL_NPU:-7}
conda_sh=${CRUISE_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-vllm-hust-dev}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
vllm_root=${CRUISE_VLLM_ROOT:-/root/vllm-hust}
vllm_ascend_root=${CRUISE_VLLM_ASCEND_ROOT:-/root/vllm-ascend-hust}

model=${artifacts}/model-frozen
frozen=${artifacts}/frozen
frozen_air=${frozen}/qwen_b4_decoder_step_attempt69c_r2.air
tiling=${frozen}/explicit_tiling.bin
old_weight_prefix=${CRUISE_OLD_WEIGHT_PREFIX:-/root/ascend-control-g4-20260723/export-attempt69c-b4}
case_manifest=${source_dir}/experiments/m1_batched_prefill/cases.json
runner=${source_dir}/experiments/m1_batched_prefill/run_differential.py
resource_config=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json

build=${scratch}/native-build
controller=${scratch}/controller
config_dir=${scratch}/config
runtime_func_config=${config_dir}/resident_epoch_func.json
runtime_export=${scratch}/runtime-export
runtime_air=${scratch}/qwen_b4_decoder_step_${run_id}.air
external_weights=${scratch}/external-weights
cache=${scratch}/cache
cann_logs=${scratch}/cann-logs
tmp=${scratch}/tmp
runtime_workdir=${scratch}/runtime-workdir
status=${evidence}/status.tsv

guard=${source_dir}/storage_guard/storage_guard.sh
required=(
  "${guard}"
  "${runner}"
  "${case_manifest}"
  "${resource_config}"
  "${source_dir}/materialize_runtime_weights.py"
  "${source_dir}/prepare_runtime_config.py"
  "${source_dir}/native/CMakeLists.txt"
  "${source_dir}/controller/CMakeLists.txt"
  "${source_dir}/controller/g4c_b4_resident_epoch.cpp"
  "${frozen_air}"
  "${tiling}"
  "${model}/config.json"
  "${model}/model.safetensors.index.json"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || {
    printf 'missing required input: %s\n' "${path}" >&2
    exit 96
  }
done
[[ -f "${conda_sh}" && -f "${cann_set_env}" ]] || exit 96

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=64
export STORAGE_GUARD_NPU_WAIT_SECONDS=21600
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 100 24 128
guard_ready=1

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
  "${external_weights}" "${cache}" "${cann_logs}" "${tmp}" \
  "${runtime_workdir}"
for path in "${build}" "${controller}" "${config_dir}" "${runtime_export}" \
  "${runtime_air}" "${external_weights}" "${cache}" "${cann_logs}" \
  "${tmp}" "${runtime_workdir}"; do
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

source "${conda_sh}"
conda activate "${conda_env}"
source "${cann_set_env}"
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_PROCESS_LOG_PATH=${cann_logs}
export ASCEND_CACHE_PATH=${cache}
export TMPDIR=${tmp}
export TORCHINDUCTOR_CACHE_DIR=${cache}/torchinductor
export TRITON_CACHE_DIR=${cache}/triton
export XDG_CACHE_HOME=${cache}/xdg
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
export VLLM_ASCEND_RESIDENT_EPOCH_SERVER=${build}/resident_epoch_server
export VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT=3600
export VLLM_ASCEND_RESIDENT_EPOCH_STEPS=2
export VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY=8

{
  printf 'key\tvalue\n'
  printf 'run_id\t%s\n' "${run_id}"
  printf 'source_dir\t%s\n' "${source_dir}"
  printf 'conda_env\t%s\n' "${conda_env}"
  printf 'physical_npu\t%s\n' "${physical_npu}"
  printf 'scratch_dir\t%s\n' "${scratch}"
  printf 'evidence_dir\t%s\n' "${evidence}"
  printf 'model\t%s\n' "${model}"
} >"${evidence}/deployment-config.tsv"
sha256sum "${case_manifest}" "${frozen_air}" "${tiling}" \
  >"${evidence}/input-integrity.log"
git -C "${vllm_root}" rev-parse HEAD >"${evidence}/vllm-commit.txt"
git -C "${vllm_ascend_root}" rev-parse HEAD \
  >"${evidence}/vllm-ascend-commit.txt"

run_step unit-tests 600s python3 -m pytest -q "${source_dir}/tests"
run_step prepare-controller 120s cp -a \
  "${source_dir}/controller/." "${controller}/"
run_step prepare-runtime-config 120s python3 \
  "${source_dir}/prepare_runtime_config.py" \
  --template "${source_dir}/config/resident_epoch_func.json" \
  --controller-workspace "${controller}" \
  --output "${runtime_func_config}"
run_step cmake 600s cmake -S "${source_dir}/native" -B "${build}"
run_step build 1800s cmake --build "${build}" --parallel 2
run_step materialize-runtime-weights 3600s python3 \
  "${source_dir}/materialize_runtime_weights.py" \
  --model-dir "${model}" \
  --output-dir "${runtime_export}" \
  --manifest "${evidence}/runtime-weights-manifest.json"
run_step relocate-runtime-air 600s "${build}/relocate_air_paths" \
  "${frozen_air}" "${runtime_air}" "${old_weight_prefix}" \
  "${runtime_export}" "${evidence}/air-relocation.json"

cd "${runtime_workdir}"
export VLLM_ASCEND_RESIDENT_EPOCH_SOCKET=${scratch}/baseline.sock
run_step baseline 10800s python3 "${runner}" \
  --mode baseline --model "${model}" --cases "${case_manifest}" \
  --output "${evidence}/baseline.json"

storage_guard_wait_for_npu_idle "${physical_npu}" 1800 \
  "${persistent_root}" 24 "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
  "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"
export VLLM_ASCEND_RESIDENT_EPOCH_SOCKET=${scratch}/cruise.sock
run_step cruise 10800s python3 "${runner}" \
  --mode cruise --model "${model}" --cases "${case_manifest}" \
  --output "${evidence}/cruise.json"
run_step compare 120s python3 "${runner}" \
  --mode compare --baseline "${evidence}/baseline.json" \
  --cruise "${evidence}/cruise.json" \
  --output "${evidence}/comparison.json"

storage_guard_wait_for_npu_idle "${physical_npu}" 1800 \
  "${persistent_root}" 24 "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
  "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"
sha256sum "${evidence}/baseline.json" "${evidence}/cruise.json" \
  "${evidence}/comparison.json" >"${evidence}/result-integrity.log"
printf 'complete\t0\n' >>"${status}"
