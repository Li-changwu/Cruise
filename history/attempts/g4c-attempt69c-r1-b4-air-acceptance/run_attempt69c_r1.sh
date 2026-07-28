#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt69c-r1-src
source_raw=${root}/raw-attempt69c-b4-air
export_dir=${root}/export-attempt69c-b4
raw=${root}/raw-attempt69c-r1-b4-air-acceptance

if [[ -e "${raw}" ]]; then exit 97; fi
if [[ ! -f "${source_raw}/status.tsv" ||
      ! -f "${export_dir}/dedup-manifest.json" ]]; then exit 96; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 || true
if ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"
find "${src}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/source-integrity.log"

python3 "${src}/accept_attempt69c_r1.py" --source-raw "${source_raw}" \
  --export-dir "${export_dir}" \
  --output "${raw}/attempt69c-r1-acceptance.json" \
  >"${raw}/acceptance.stdout.log" 2>&1
acceptance_status=$?
printf 'acceptance\t%s\n' "${acceptance_status}" >>"${status}"

npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${raw}/attempt69c-r1-acceptance.json" "${raw}/status.tsv" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt69c-r1-acceptance.json" 2>/dev/null || true
[[ ${acceptance_status} -eq 0 ]] || exit ${acceptance_status}
exit ${idle_status}
