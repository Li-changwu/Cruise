#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-real-p0-20260722
source_root=/root/ascend-control-g2d-20260718
sample_dir=${source_root}/official-dflow/cpp
air=${source_root}/export/qwen_attention_kv.air
reference=${source_root}/export/reference.npz
graph_config=${sample_dir}/config/qwen_attention_graph.json
func_config=${sample_dir}/config/qwen_attention_controller_func.json
controller_source=${source_root}/device-qwen-attn-udf/qwen_attention_controller.cpp
controller_binary=${source_root}/device-qwen-attn-udf/build/_qwen_attention_controller/Ascend/release/libqwen_attention_controller.so
numa=/root/ascend-control-g2g-20260719/numa_config.physical7.json
harness=${root}/real_qwen_p0_benchmark.py
protocol=${root}/protocol.md

expected_air_sha=eb0effc65a3bba7977430d38a316f374a8f7ba0716065162198153d9191240e3
expected_reference_sha=02ac4d7507ae0dca353fe447e8be4e1d8ddbe8be4b8e8d7f98475d4bc04470cc
expected_graph_config_sha=5c940de05183a6ab4d611a2534e0a9e47e5a054a2d0c400c68dc1239fb587095
expected_func_config_sha=b1297f8f18e3a4d047adfc8c19974e4ba1f168e9db5a6b60d86722d598f1c6ba
expected_controller_source_sha=598775cf2811c7fa8e5ff84cb00335fddc4a596f12f7ac4f642728ae5a3f6e48
expected_controller_binary_sha=eb5d0e638f70323192408276d0203dbd84d8e747311107d89f7d82c310db92b3
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_harness_sha=6f6d9c612b2f393083791f46f33bd78b8d96d777908e2f8053fd0ad9cbda6fbd
expected_protocol_sha=b1f49b227b095addadd9e05b8cd61ab1276383f370e500e9a694229747e1ff9b

raw=${root}/raw
if [[ -e "${raw}" ]]; then
  printf 'REAL_QWEN_P0_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT
npu-smi info >"${raw}/npu-before.txt" 2>&1

if [[ $(sha256sum "${air}" | cut -d' ' -f1) != "${expected_air_sha}" ||
      $(sha256sum "${reference}" | cut -d' ' -f1) != "${expected_reference_sha}" ||
      $(sha256sum "${graph_config}" | cut -d' ' -f1) != "${expected_graph_config_sha}" ||
      $(sha256sum "${func_config}" | cut -d' ' -f1) != "${expected_func_config_sha}" ||
      $(sha256sum "${controller_source}" | cut -d' ' -f1) != "${expected_controller_source_sha}" ||
      $(sha256sum "${controller_binary}" | cut -d' ' -f1) != "${expected_controller_binary_sha}" ||
      $(sha256sum "${numa}" | cut -d' ' -f1) != "${expected_numa_sha}" ||
      $(sha256sum "${harness}" | cut -d' ' -f1) != "${expected_harness_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${air}" "${reference}" "${graph_config}" "${func_config}" \
  "${controller_source}" "${controller_binary}" "${numa}" "${harness}" \
  "${protocol}" "${root}/run_real_qwen_p0.sh" \
  >"${raw}/artifact-integrity.log"

cd "${sample_dir}/output"
for n in 4 2 1; do
  timeout 1200 python "${harness}" --n "${n}" --warmup 3 --repetitions 10 \
    --air-path "${air}" --reference "${reference}" \
    --graph-config "${graph_config}" --func-config "${func_config}" \
    --output "${raw}/n${n}.json" >"${raw}/n${n}.stdout.log" 2>&1
  case_status=$?
  printf 'n%s\t%s\n' "${n}" "${case_status}" >>"${status_file}"
  if [[ "${case_status}" -ne 0 ]]; then
    tail -160 "${raw}/n${n}.stdout.log"
    exit "${case_status}"
  fi
done

sha256sum "${raw}/n1.json" "${raw}/n2.json" "${raw}/n4.json" \
  >"${raw}/result-integrity.log"
printf 'REAL_QWEN_P0_COMPLETE\n'
