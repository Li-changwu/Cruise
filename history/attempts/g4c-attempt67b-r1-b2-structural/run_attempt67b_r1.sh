#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt67b-r1-src
raw=${root}/raw-attempt67b-r1-b2-structural
validation=${root}/validation-attempt67b-r1-b2
export_dir=${root}/export-attempt67b-b2
graph=${export_dir}/dynamo.pbtxt
air=${export_dir}/qwen_b2_decoder_step_attempt67b.air

if [[ -e "${raw}" || -e "${validation}" ]]; then exit 97; fi
if [[ ! -f "${graph}" || ! -f "${air}" ]]; then exit 96; fi
mkdir -p "${raw}" "${validation}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"
sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${graph}" "${air}" \
  >"${raw}/input-integrity.log"

python "${src}/extract_b2_decoder_abi.py" --graph "${graph}" \
  --output "${validation}/abi.json" >"${raw}/abi.stdout.log" 2>&1
abi_status=$?
printf 'abi\t%s\n' "${abi_status}" >>"${status}"

python "${src}/inspect_b2_graph.py" --graph "${graph}" \
  --output "${validation}/graph-inspection.json" \
  >"${raw}/graph-inspection.stdout.log" 2>&1
graph_status=$?
printf 'graph-inspection\t%s\n' "${graph_status}" >>"${status}"

npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${validation}/abi.json" "${validation}/graph-inspection.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${validation}/abi.json"
python -c "import json; p=json.load(open('${validation}/graph-inspection.json')); print(json.dumps({k:p[k] for k in ('valid','observed_op_counts','active_mask_data_count','slot_mapping_data_count')}, indent=2))"
if [[ ${abi_status} -ne 0 ]]; then exit ${abi_status}; fi
if [[ ${graph_status} -ne 0 ]]; then exit ${graph_status}; fi
exit ${idle_status}
