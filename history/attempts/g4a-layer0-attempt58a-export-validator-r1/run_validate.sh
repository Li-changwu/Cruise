#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt58a-validator-r1-src
original_raw=${root}/raw-attempt58a-export
export_dir=${root}/export-attempt58a
raw=${root}/raw-attempt58a-export-validator-r1

if [[ -e "${raw}" ]]; then exit 97; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle\t0\n' >>"${status}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh \
  "${original_raw}/status.tsv" "${export_dir}/qwen_layer0_boundary_attempt58a.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/attempt58a-eager-reference.npz" \
  "${export_dir}/export-result.json" "${export_dir}/eager-comparison.json" \
  "${export_dir}/abi.json" >"${raw}/input-integrity.log"

python3 "${src}/inspect_graph.py" --graph "${export_dir}/dynamo.pbtxt" \
  --output "${raw}/graph-inspection-r1.json" >"${raw}/graph.stdout.log" 2>&1
s=$?
printf 'graph-inspection-r1\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/graph.stdout.log"; exit $s; fi

python3 "${src}/validate_export.py" --original-raw "${original_raw}" \
  --export-dir "${export_dir}" --graph-inspection "${raw}/graph-inspection-r1.json" \
  --output "${raw}/validation-result.json" >"${raw}/validation.stdout.log" 2>&1
s=$?
printf 'validation\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/validation.stdout.log"; exit $s; fi

npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  printf 'npu7-idle-after\t0\n' >>"${status}"
else
  printf 'npu7-idle-after\t96\n' >>"${status}"; exit 96
fi
sha256sum "${raw}/graph-inspection-r1.json" "${raw}/validation-result.json" \
  >"${raw}/result-integrity.log"
cat "${raw}/validation.stdout.log"
