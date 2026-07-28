#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-bounded-decode-20260722
source_root=/root/ascend-control-g2d-20260718
air=${source_root}/export/qwen_attention_kv.air
reference=${source_root}/export/reference.npz
graph_config=${source_root}/official-dflow/cpp/config/qwen_attention_graph.json
func_config=${root}/config/bounded_decode_controller_func.json
probe=${root}/capacity_guard_probe.py
numa=/root/ascend-control-g2g-20260719/numa_config.physical7.json
raw=${root}/capacity-probe-run1

if [[ -e "${raw}" ]]; then
  printf 'CAPACITY_GUARD_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2

printf 'case\texit_status\n' >"${raw}/status.tsv"
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1
sha256sum "${air}" "${reference}" "${graph_config}" "${func_config}" \
  "${root}/controller/bounded_decode_controller.cpp" "${probe}" \
  "${root}/run_capacity_guard_probe.sh" >"${raw}/artifact-integrity.log"
printf 'artifact-integrity\t0\n' >>"${raw}/status.tsv"

cd "${source_root}/official-dflow/cpp/output"
timeout 600 python "${probe}" --air-path "${air}" \
  --reference "${reference}" --graph-config "${graph_config}" \
  --func-config "${func_config}" --output "${raw}/capacity-guard-result.json" \
  >"${raw}/probe.stdout.log" 2>&1
probe_status=$?
printf 'capacity-guard\t%s\n' "${probe_status}" >>"${raw}/status.tsv"
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1
if [[ "${probe_status}" -ne 0 ]]; then
  tail -160 "${raw}/probe.stdout.log"
  exit "${probe_status}"
fi
sha256sum "${raw}/capacity-guard-result.json" \
  >"${raw}/result-integrity.log"
printf 'CAPACITY_GUARD_COMPLETE\n'
