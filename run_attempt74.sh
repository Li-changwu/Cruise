#!/usr/bin/env bash
set -uo pipefail

root=${CRUISE_EXPERIMENT_ROOT:-/root/ascend-control-g4-20260723}
g2g=${CRUISE_G2G_ROOT:-/root/ascend-control-g2g-20260719}
src=${CRUISE_SOURCE_DIR:-${root}/attempt74-src}
baseline_src=${CRUISE_BASELINE_SOURCE_DIR:-${root}/attempt73-src}
run_id=${CRUISE_RUN_ID:-attempt74r1}
evidence=${CRUISE_EVIDENCE_DIR:-${root}/evidence-attempt74r1-minimal-abi}
scratch=${CRUISE_SCRATCH_DIR:-/dev/shm/a74r1}
physical_npu=${CRUISE_PHYSICAL_NPU:-7}
build=${scratch}/native-build
controller_new=${scratch}/controller-new
controller_old=${scratch}/controller-old
config_dir=${scratch}/config
runtime_config_new=${config_dir}/resident_epoch_func_new.json
runtime_config_old=${config_dir}/resident_epoch_func_old.json
weights=${scratch}/external-weights
runtime_export=${scratch}/runtime-export
runtime_air=${scratch}/qwen_b4_decoder_step_${run_id}.air
cache=${scratch}/cache
profiler=${scratch}/profiler
cann_logs=${scratch}/cann-logs
tmp=${scratch}/t
runtime_memcpy=${scratch}/rt-memcpy
frozen_air=${CRUISE_FROZEN_AIR:-${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air}
tiling=${CRUISE_TILING:-${root}/raw-attempt69d-r1-b4-native/native-inputs/case0/explicit_tiling.bin}
baseline=${CRUISE_BASELINE_RESULT:-${root}/evidence-attempt69e-r5-b4-resident-epoch/attempt69e-r5-result.json}
model_config=${CRUISE_MODEL_CONFIG:-${src}/tests/fixtures/qwen2-7b-config}
model=${CRUISE_MODEL_DIR:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}
old_weight_prefix=${CRUISE_OLD_WEIGHT_PREFIX:-${root}/export-attempt69c-b4}
conda_sh=${CRUISE_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-vllm-hust-dev}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
vllm_root=${CRUISE_VLLM_ROOT:-/root/vllm-hust}
vllm_ascend_root=${CRUISE_VLLM_ASCEND_ROOT:-/root/vllm-ascend-hust}
resource_config=${CRUISE_RESOURCE_CONFIG:-${g2g}/numa_config.physical7.json}
msprof_storage_limit=${CRUISE_MSPROF_STORAGE_LIMIT:-2048}

guard=${src}/storage_guard/storage_guard.sh
required=(
  "${guard}"
  "${frozen_air}"
  "${tiling}"
  "${baseline}"
  "${model_config}/config.json"
  "${model}/model.safetensors.index.json"
  "${src}/materialize_runtime_weights.py"
  "${src}/run_multi_epoch_cohort.py"
  "${src}/run_abi_epoch_benchmark.py"
  "${src}/verify_multi_epoch_cohort_result.py"
  "${src}/verify_minimal_abi_source.py"
  "${src}/analyze_abi_comparison.py"
  "${src}/verify_abi_comparison_result.py"
  "${src}/summarize_msprof_transfers.py"
  "${src}/summarize_runtime_memcpy.py"
  "${src}/controller/g4c_b4_resident_epoch.cpp"
  "${src}/controller-old/g4c_b4_resident_epoch.cpp"
  "${src}/native/resident_epoch_bridge.cpp"
  "${src}/native/resident_epoch_bridge_old.cpp"
  "${src}/native/rt_memcpy_trace.cpp"
  "${src}/native/relocate_air_paths.cpp"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || exit 96
done
[[ -d "${baseline_src}" ]] || exit 96
grep -q '"pass": true' "${baseline}" || exit 96

source "${guard}"
export STORAGE_GUARD_LARGE_ALLOWLIST=export-attempt67b-b2:export-attempt69c-b4
export STORAGE_GUARD_MAX_SCRATCH_GIB=64
export STORAGE_GUARD_NPU_WAIT_SECONDS=21600
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5
storage_guard_preflight "${root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 100 24 128
preflight_status=$?
[[ ${preflight_status} -eq 0 ]] || exit ${preflight_status}

for heavy_path in "${build}" "${controller_new}" "${controller_old}" \
                  "${config_dir}" "${weights}" "${runtime_export}" \
                  "${runtime_air}" "${cache}" "${profiler}" \
                  "${cann_logs}" "${tmp}" "${runtime_memcpy}"; do
  storage_guard_assert_scratch_path "${heavy_path}" || exit $?
done
mkdir -p "${build}" "${weights}" "${cache}" "${profiler}" \
  "${cann_logs}" "${tmp}" "${config_dir}"
mkdir -p "${runtime_memcpy}"

status=${evidence}/status.tsv
printf 'case\texit_status\n' >"${status}"
finalized=0

capture_after() {
  npu-smi info -t proc-mem -i "${physical_npu}" \
    >"${evidence}/npu${physical_npu}-processes-after.txt" \
    2>&1 || true
  npu-smi info -t usages -i "${physical_npu}" \
    >"${evidence}/npu${physical_npu}-usages-after.txt" \
    2>&1 || true
  du -sh "${weights}" >"${evidence}/external-weights-size.txt" 2>&1 || true
  find "${weights}" -type f -printf '%P\t%s bytes\n' | sort \
    >"${evidence}/external-weights-files.txt" 2>&1 || true
  storage_guard_snapshot exit "${evidence}/storage-exit.tsv" || true
}

on_exit() {
  local exit_status=$?
  trap - EXIT
  set +e
  capture_after
  if [[ ${finalized} -eq 0 ]]; then
    storage_guard_finalize
  fi
  exit "${exit_status}"
}
trap on_exit EXIT

run_step() {
  local name=$1 timeout_value=$2
  shift 2
  storage_guard_run_log "${evidence}/${name}.stdout.log" \
    "${evidence}/${name}.stdout.meta.json" "${timeout_value}" -- "$@"
  local step_status=$?
  printf '%s\t%s\n' "${name}" "${step_status}" >>"${status}"
  return "${step_status}"
}

wait_npu_ready() {
  local label=$1
  storage_guard_wait_for_npu_idle "${physical_npu}" \
    "${STORAGE_GUARD_NPU_WAIT_SECONDS}" \
    "${root}" 24 "${STORAGE_GUARD_MIN_ROOT_FREE_BYTES}" \
    "${STORAGE_GUARD_MIN_SHM_FREE_BYTES}"
  local ready_status=$?
  if [[ ${ready_status} -eq 0 ]]; then
    printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_STATE}" \
      >"${evidence}/npu${physical_npu}-processes-${label}.txt"
    printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_USAGE}" \
      >"${evidence}/npu${physical_npu}-usages-${label}.txt"
    printf 'npu%s-ready-%s\t0\n' "${physical_npu}" "${label}" >>"${status}"
    return 0
  fi
  printf 'npu%s-ready-%s\t%s\n' "${physical_npu}" "${label}" \
    "${ready_status}" >>"${status}"
  return "${ready_status}"
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

custom_set_env=${CRUISE_CUSTOM_SET_ENV:-$(find "${g2g}/install-attempt47" \
  -type f -name set_env.bash | head -1)}
barrier_set_env=${CRUISE_BARRIER_SET_ENV:-$(find \
  "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)}
materialize_set_env=${CRUISE_MATERIALIZE_SET_ENV:-$(find \
  "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)}
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && \
   -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"
export RESOURCE_CONFIG_PATH=${resource_config}

export PYTHONPATH=${src}/src:${src}:${vllm_root}:${vllm_ascend_root}:${PYTHONPATH:-}
export VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY=vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine
export VLLM_ASCEND_RESIDENT_EPOCH_AIR=${runtime_air}
export VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG=${src}/config/graph_config.json
export VLLM_ASCEND_RESIDENT_EPOCH_TILING=${tiling}
export VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS=${weights}
export VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT=3600
export VLLM_ASCEND_RESIDENT_EPOCH_STEPS=2
export VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY=8

find "${src}" -type f ! -path '*/__pycache__/*' \
  ! -path '*/.pytest_cache/*' -print0 | sort -z | xargs -0 sha256sum \
  >"${evidence}/source-integrity.log"
sha256sum "${frozen_air}" "${tiling}" "${baseline}" \
  >"${evidence}/frozen-artifact-integrity.log"
sha256sum "${baseline_src}/controller/g4c_b4_resident_epoch.cpp" \
  "${baseline_src}/config/graph_config.json" \
  >"${evidence}/attempt73-baseline-source-integrity.log"
readlink -f "${model}"/model-*.safetensors | sort \
  >"${evidence}/model-shards.txt"
sha256sum "${model}/config.json" "${model}/model.safetensors.index.json" \
  >"${evidence}/model-metadata-integrity.log"
git -C "${vllm_root}" rev-parse HEAD >"${evidence}/vllm-commit.txt"
git -C "${vllm_ascend_root}" rev-parse HEAD \
  >"${evidence}/vllm-ascend-commit.txt"

run_step cann-python-smoke 120s python3 -c 'import tbe' || exit $?
run_step unit-tests 600s python3 -m pytest -q "${src}/tests" || exit $?

run_step source-verification 120s python3 "${src}/verify_minimal_abi_source.py" \
  "${src}" --baseline-source "${baseline_src}" \
  --output "${evidence}/${run_id}-source-verification.json" || exit $?

run_step prepare-controller-new 120s cp -a "${src}/controller" \
  "${controller_new}" || exit $?
run_step prepare-controller-old 120s cp -a "${src}/controller-old" \
  "${controller_old}" || exit $?
run_step prepare-runtime-config-new 120s python3 \
  "${src}/prepare_runtime_config.py" \
  --template "${src}/config/resident_epoch_func.json" \
  --controller-workspace "${controller_new}" \
  --output "${runtime_config_new}" || exit $?
run_step prepare-runtime-config-old 120s python3 \
  "${src}/prepare_runtime_config.py" \
  --template "${src}/config/resident_epoch_func_old.json" \
  --controller-workspace "${controller_old}" \
  --output "${runtime_config_old}" || exit $?

run_step cmake 600s cmake -S "${src}/native" -B "${build}" || exit $?
run_step build 1800s cmake --build "${build}" --parallel 2 || exit $?
sha256sum "${build}/libresident_epoch_bridge.so" \
  "${build}/libresident_epoch_bridge_old.so" \
  "${build}/resident_epoch_server" \
  "${build}/resident_epoch_server_old" \
  "${build}/libresident_epoch_memcpy_trace.so" \
  "${build}/relocate_air_paths" \
  >"${evidence}/native-bridge-integrity.log"

run_step materialize-runtime-weights 3600s python3 \
  "${src}/materialize_runtime_weights.py" \
  --model-dir "${model}" \
  --output-dir "${runtime_export}" \
  --manifest "${evidence}/runtime-weights-manifest.json" || exit $?

runtime_weight_file_count=$(find "${runtime_export}" -maxdepth 1 -type f \
  ! -name '*.*' | wc -l)
runtime_weight_bytes=$(find "${runtime_export}" -maxdepth 1 -type f \
  ! -name '*.*' -printf '%s\n' | awk '{total += $1} END {print total + 0}')
printf 'files\t%s\nbytes\t%s\n' "${runtime_weight_file_count}" \
  "${runtime_weight_bytes}" >"${evidence}/runtime-weights-summary.tsv"
if [[ ${runtime_weight_file_count} -ne 342 ||
      ${runtime_weight_bytes} -ne 15231237408 ]]; then
  printf 'runtime-weights-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'runtime-weights-integrity\t0\n' >>"${status}"
find "${runtime_export}" -maxdepth 1 -type f ! -name '*.*' -print0 | \
  sort -z | xargs -0 sha256sum \
  >"${evidence}/runtime-weights-integrity.log"

run_step relocate-runtime-air 600s "${build}/relocate_air_paths" \
  "${frozen_air}" "${runtime_air}" "${old_weight_prefix}" \
  "${runtime_export}" "${evidence}/${run_id}-air-relocation.json" || exit $?
sha256sum "${runtime_air}" >"${evidence}/runtime-air-integrity.log"

configure_route() {
  local label=$1 route=$2
  export ASCEND_CACHE_PATH=${cache}/${label}
  export TORCHINDUCTOR_CACHE_DIR=${cache}/${label}/torchinductor
  export TRITON_CACHE_DIR=${cache}/${label}/triton
  export XDG_CACHE_HOME=${cache}/${label}/xdg
  export VLLM_ASCEND_RESIDENT_EPOCH_SOCKET=${scratch}/${label}.sock
  export VLLM_ASCEND_RESIDENT_EPOCH_MEMCPY_TRACE_LIBRARY=${build}/libresident_epoch_memcpy_trace.so
  export VLLM_ASCEND_RESIDENT_EPOCH_MEMCPY_TRACE_PATH=${runtime_memcpy}/${label}.tsv
  if [[ "${route}" == new ]]; then
    export VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG=${runtime_config_new}
    export VLLM_ASCEND_RESIDENT_EPOCH_LIBRARY=${build}/libresident_epoch_bridge.so
    export VLLM_ASCEND_RESIDENT_EPOCH_SERVER=${build}/resident_epoch_server
  else
    export VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG=${runtime_config_old}
    export VLLM_ASCEND_RESIDENT_EPOCH_LIBRARY=${build}/libresident_epoch_bridge_old.so
    export VLLM_ASCEND_RESIDENT_EPOCH_SERVER=${build}/resident_epoch_server_old
  fi
  mkdir -p "${ASCEND_CACHE_PATH}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"
}

wait_npu_ready pre-semantic || exit $?
configure_route semantic-new new
run_step minimal-abi-multi-epoch 10800s python3 \
  "${src}/run_multi_epoch_cohort.py" \
  --model-config "${model_config}" \
  --baseline-result "${baseline}" \
  --output "${evidence}/${run_id}-multi-epoch-result.json" || exit $?
run_step minimal-abi-multi-epoch-verify 120s python3 \
  "${src}/verify_multi_epoch_cohort_result.py" \
  "${evidence}/${run_id}-multi-epoch-result.json" || exit $?

run_block() {
  local label=$1 route=$2 profile_route=$3
  wait_npu_ready "pre-${label}" || return $?
  configure_route "${label}" "${route}"
  local output=${evidence}/${run_id}-${label}.json
  if [[ "${profile_route}" == yes ]]; then
    local profile_output=${profiler}/${route}
    run_step "benchmark-${label}" 14400s msprof \
      --output="${profile_output}" --ascendcl=on --ge-api=l1 \
      --runtime-api=on --task-time=l1 --ai-core=off --aicpu=off \
      --type=text --storage-limit="${msprof_storage_limit}" \
      python3 "${src}/run_abi_epoch_benchmark.py" \
      --model-config "${model_config}" --baseline-result "${baseline}" \
      --route "${route}" --repetitions 15 --output "${output}"
  else
    run_step "benchmark-${label}" 14400s python3 \
      "${src}/run_abi_epoch_benchmark.py" \
      --model-config "${model_config}" --baseline-result "${baseline}" \
      --route "${route}" --repetitions 15 --output "${output}"
  fi
}

run_block old-1 old yes || exit $?
run_block new-1 new yes || exit $?
run_block new-2 new no || exit $?
run_block old-2 old no || exit $?

find "${profiler}" -type f -printf '%P\t%s\n' | sort \
  >"${evidence}/msprof-files.tsv"
run_step summarize-msprof-transfers 600s python3 \
  "${src}/summarize_msprof_transfers.py" \
  --old-root "${profiler}/old" --new-root "${profiler}/new" \
  --output "${evidence}/${run_id}-msprof-transfer-summary.json" || exit $?

run_step summarize-runtime-memcpy 600s python3 \
  "${src}/summarize_runtime_memcpy.py" \
  --old-1-trace "${runtime_memcpy}/old-1.tsv" \
  --old-1-result "${evidence}/${run_id}-old-1.json" \
  --new-1-trace "${runtime_memcpy}/new-1.tsv" \
  --new-1-result "${evidence}/${run_id}-new-1.json" \
  --new-2-trace "${runtime_memcpy}/new-2.tsv" \
  --new-2-result "${evidence}/${run_id}-new-2.json" \
  --old-2-trace "${runtime_memcpy}/old-2.tsv" \
  --old-2-result "${evidence}/${run_id}-old-2.json" \
  --filtered-dir "${evidence}/rt-memcpy-filtered" \
  --output "${evidence}/${run_id}-runtime-memcpy-summary.json" || exit $?

run_step analyze-abi-comparison 300s python3 \
  "${src}/analyze_abi_comparison.py" \
  --old-1 "${evidence}/${run_id}-old-1.json" \
  --new-1 "${evidence}/${run_id}-new-1.json" \
  --new-2 "${evidence}/${run_id}-new-2.json" \
  --old-2 "${evidence}/${run_id}-old-2.json" \
  --semantic-result "${evidence}/${run_id}-multi-epoch-result.json" \
  --source-verification "${evidence}/${run_id}-source-verification.json" \
  --profiler-summary "${evidence}/${run_id}-msprof-transfer-summary.json" \
  --runtime-memcpy-summary "${evidence}/${run_id}-runtime-memcpy-summary.json" \
  --output "${evidence}/${run_id}-result.json" || exit $?
run_step verify-abi-comparison 300s python3 \
  "${src}/verify_abi_comparison_result.py" \
  "${evidence}/${run_id}-result.json" || exit $?

weight_file_count=$(find "${weights}" -maxdepth 1 -type f | wc -l)
weight_bytes=$(storage_guard_used_bytes "${weights}")
printf 'files\t%s\nbytes\t%s\n' "${weight_file_count}" "${weight_bytes}" \
  >"${evidence}/external-weights-summary.tsv"
if [[ ${weight_file_count} -ne 379 || ${weight_bytes} -lt 15000000000 ||
      ${weight_bytes} -gt 16000000000 ]]; then
  printf 'external-weights-integrity\t93\n' >>"${status}"
  exit 93
fi
printf 'external-weights-integrity\t0\n' >>"${status}"

wait_npu_ready final
idle_status=$?
capture_after
sha256sum "${evidence}/${run_id}-result.json" "${status}" \
  "${evidence}/native-bridge-integrity.log" \
  >"${evidence}/result-integrity.log" 2>/dev/null || true
find "${evidence}" -maxdepth 1 -type f ! -name evidence-integrity.log \
  -print0 | sort -z | xargs -0 sha256sum \
  >"${evidence}/evidence-integrity.log"
storage_guard_finalize
finalize_status=$?
[[ ${finalize_status} -eq 0 ]] || exit ${finalize_status}
finalized=1
cat "${evidence}/${run_id}-result.json"
trap - EXIT
[[ ${idle_status} -eq 0 ]] || exit ${idle_status}
storage_guard_cleanup_scratch
