#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
heavy_load=${CRUISE_P3_ALLOCATOR_HEAVY_LOAD:-0}
scratch=${CRUISE_P3_ALLOCATOR_SCRATCH:-/dev/shm/cruise-p3-allocator-probe-$(date -u +%H%M%S)}
conda_sh=${CRUISE_CONDA_SH:-/home/changwu/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-vllm-hust-dev}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
template=${source_dir}/experiments/synthetic-p0/numa_config.physical7.json
resource_writer=${source_dir}/prepare_resource_config.py
config_writer=${script_dir}/prepare_probe_config.py
controller_source=${script_dir}/controller
host_source=${script_dir}/p3_allocator_probe_host.cpp

for required in "${conda_sh}" "${cann_set_env}" "${template}" \
  "${resource_writer}" "${config_writer}" "${host_source}" \
  "${controller_source}/CMakeLists.txt" \
  "${controller_source}/p3_allocator_probe.cpp"; do
  [[ -f "${required}" ]] || {
    printf 'missing required input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${heavy_load}" == 0 || "${heavy_load}" == 1 ]]

build=${scratch}/build
config_dir=${scratch}/config
deploy_root=${scratch}/dataflow-deploy
controller_workspace=${scratch}/controller
logs=${scratch}/logs
mkdir -p "${build}" "${config_dir}" "${deploy_root}" \
  "${controller_workspace}" "${logs}"
cp "${controller_source}/CMakeLists.txt" \
  "${controller_source}/p3_allocator_probe.cpp" \
  "${controller_workspace}/"

source "${conda_sh}"
conda activate "${conda_env}"
source "${cann_set_env}"
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
toolchain_config=${config_dir}/toolchain.json
deploy_config=${config_dir}/deploy.json
config_args=(
  --workspace "${controller_workspace}"
  --ascend-toolchain "${ascend_toolchain}"
  --function-output "${function_config}"
  --toolchain-output "${toolchain_config}"
  --deploy-output "${deploy_config}"
)
if [[ "${heavy_load}" == 1 ]]; then
  config_args+=(--heavy-load)
fi
python "${config_writer}" "${config_args[@]}"

host_binary=${build}/p3_allocator_probe_host
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

default_cases=(
  tensor-msg-21
  tensor-msg-28
  tensor-msg-42
  factory-wrap-28
  factory-wrap-42
  raw-msg-28
  raw-msg-42
  tensor-list-2x21
  tensor-msg-2x42
  factory-wrap-2x42
  tensor-msg-4x21
  tensor-list-4x21
)
if [[ -n "${CRUISE_P3_ALLOCATOR_CASES:-}" ]]; then
  read -r -a cases <<<"${CRUISE_P3_ALLOCATOR_CASES}"
else
  cases=("${default_cases[@]}")
fi
matrix=${scratch}/matrix.log
: >"${matrix}"
case_index=0
for probe_case in "${cases[@]}"; do
  case_index=$((case_index + 1))
  wait_for_release
  case_root=${scratch}/case-${probe_case}
  case_tmp=${scratch}/t${case_index}
  mkdir -p "${case_root}/driver-logs" "${case_root}/cache" "${case_tmp}"
  ASCEND_PROCESS_LOG_PATH=${case_root}/driver-logs \
  ASCEND_CACHE_PATH=${case_root}/cache \
  TMPDIR=${case_tmp} \
    "${host_binary}" "${function_config}" "${deploy_config}" "${probe_case}" \
    2>&1 | tee "${case_root}/host.log" | tee -a "${matrix}"
  wait_for_release
done

printf 'P3_ALLOCATOR_PROBE_COMPLETE cases=%s scratch=%s\n' \
  "${#cases[@]}" "${scratch}"
