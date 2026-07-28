#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-p0-20260722
raw=${root}/cpu-sweep
sample_dir=/root/ascend-control-g2b-20260718/official-dflow/cpp
host_source=${root}/sample_host_token_loop_cpu.cpp
device_source=${root}/sample_device_token_loop_cpu.cpp
host_binary=${raw}/bin/sample_host_token_loop_cpu
device_binary=${raw}/bin/sample_device_token_loop_cpu
numa=${root}/numa_config.physical7.json
expected_host_source_sha=aa5abfb5a8293fd9352b8ec3e31d9fad37a87eef1605007385177b6898445d1d
expected_device_source_sha=13237a81aa627b271f2a00469061ff815399a10a717933153b09674188ce1e93
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf

if [[ -e "${raw}" ]]; then
  printf 'P0_CPU_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}/bin"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
npu-smi info >"${raw}/npu-before.txt" 2>&1
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT

if [[ $(sha256sum "${host_source}" | awk '{print $1}') != "${expected_host_source_sha}" ||
      $(sha256sum "${device_source}" | awk '{print $1}') != "${expected_device_source_sha}" ||
      $(sha256sum "${numa}" | awk '{print $1}') != "${expected_numa_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"

compile_common=(
  -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv
  -fstack-protector-all -fPIC -I"${ASCEND_HOME_PATH}/include"
  -I"${ASCEND_HOME_PATH}/include/external"
  -I"${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
)
g++ "${compile_common[@]}" "${host_source}" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  -Wl,--no-whole-archive -o "${host_binary}" \
  >"${raw}/host-compile.log" 2>&1
host_compile_status=$?
printf 'host-compile\t%s\n' "${host_compile_status}" >>"${status_file}"
if [[ "${host_compile_status}" -ne 0 ]]; then
  tail -120 "${raw}/host-compile.log"
  exit "${host_compile_status}"
fi

g++ "${compile_common[@]}" "${device_source}" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${device_binary}" \
  >"${raw}/device-compile.log" 2>&1
device_compile_status=$?
printf 'device-compile\t%s\n' "${device_compile_status}" >>"${status_file}"
if [[ "${device_compile_status}" -ne 0 ]]; then
  tail -120 "${raw}/device-compile.log"
  exit "${device_compile_status}"
fi
sha256sum "${host_source}" "${device_source}" "${host_binary}" \
  "${device_binary}" "${numa}" "${root}/run_p0_cpu_sweep.sh" \
  >"${raw}/artifact-integrity.log"

run_case() {
  local case_name=$1
  shift
  timeout 900 "$@" >"${raw}/${case_name}.stdout.log" 2>&1
  local status=$?
  printf '%s\t%s\n' "${case_name}" "${status}" >>"${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    tail -160 "${raw}/${case_name}.stdout.log"
    exit "${status}"
  fi
}

cd "${sample_dir}/output"
for n in 1 2 4 8 16 32; do
  run_case "host-ge-n${n}" "${host_binary}" "${n}" 100000 1 0 20
  run_case "device-udf-n${n}" "${device_binary}" "${n}" 100000 1 20
done
printf 'P0_CPU_SWEEP_COMPLETE\n'
