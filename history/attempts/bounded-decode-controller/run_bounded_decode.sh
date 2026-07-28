#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-bounded-decode-20260722
source_root=/root/ascend-control-g2d-20260718
air=${source_root}/export/qwen_attention_kv.air
reference=${source_root}/export/reference.npz
graph_config=${source_root}/official-dflow/cpp/config/qwen_attention_graph.json
func_config=${root}/config/bounded_decode_controller_func.json
controller_source=${root}/controller/bounded_decode_controller.cpp
controller_cmake=${root}/controller/CMakeLists.txt
harness=${root}/bounded_decode_benchmark.py
analyzer=${root}/analyze_bounded_decode.py
protocol=${root}/protocol.md
numa=/root/ascend-control-g2g-20260719/numa_config.physical7.json
raw=${root}/raw-run3

if [[ -e "${raw}" ]]; then
  printf 'BOUNDED_DECODE_REFUSE_OVERWRITE raw=%s\n' "${raw}"
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
npu-smi info >"${raw}/npu-before.txt" 2>&1
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1
capture_after() {
  npu-smi info >"${raw}/npu-after.txt" 2>&1 || true
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT

sha256sum "${air}" "${reference}" "${graph_config}" "${func_config}" \
  "${controller_source}" "${controller_cmake}" "${harness}" "${analyzer}" \
  "${protocol}" "${numa}" "${root}/run_bounded_decode.sh" \
  >"${raw}/artifact-integrity.log"
printf 'artifact-integrity\t0\n' >>"${status_file}"

cd "${source_root}/official-dflow/cpp/output"
timeout 1800 python "${harness}" --warmup 3 --repetitions 10 \
  --air-path "${air}" --reference "${reference}" \
  --graph-config "${graph_config}" --func-config "${func_config}" \
  --output "${raw}/bounded-decode-results.json" \
  >"${raw}/benchmark.stdout.log" 2>&1
benchmark_status=$?
printf 'benchmark\t%s\n' "${benchmark_status}" >>"${status_file}"
if [[ "${benchmark_status}" -ne 0 ]]; then
  tail -200 "${raw}/benchmark.stdout.log"
  exit "${benchmark_status}"
fi

python "${analyzer}" "${raw}/bounded-decode-results.json" \
  --json-output "${root}/bounded-decode-summary.json" \
  --markdown-output "${root}/BOUNDED-DECODE-RESULTS.md"
analysis_status=$?
printf 'analysis\t%s\n' "${analysis_status}" >>"${status_file}"
if [[ "${analysis_status}" -ne 0 ]]; then
  exit "${analysis_status}"
fi
sha256sum "${raw}/bounded-decode-results.json" \
  "${root}/bounded-decode-summary.json" "${root}/BOUNDED-DECODE-RESULTS.md" \
  >"${raw}/result-integrity.log"
printf 'BOUNDED_DECODE_COMPLETE\n'
