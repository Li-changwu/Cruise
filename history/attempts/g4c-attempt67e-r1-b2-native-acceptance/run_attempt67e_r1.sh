#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt67e-r1-src
source_raw=${root}/raw-attempt67e-b2-native
raw=${root}/raw-attempt67e-r1-b2-native-acceptance

if [[ -e "${raw}" ]]; then exit 97; fi
if [[ ! -f "${source_raw}/attempt67e-result.json" ||
      ! -f "${source_raw}/attempt67e-native-output.npz" ||
      ! -f "${source_raw}/native.stdout.log" ]]; then
  exit 96
fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"

npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 || true
if ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh \
  "${source_raw}/attempt67e-result.json" \
  "${source_raw}/attempt67e-native-output.npz" \
  "${source_raw}/native.stdout.log" \
  "${source_raw}/status.tsv" \
  "${source_raw}/exactqk-launch-metadata.txt" \
  "${source_raw}/barrier-launch-metadata.txt" \
  "${source_raw}/materialize-launch-metadata.txt" \
  "${source_raw}/linear-launches.json" \
  "${source_raw}/npu7-processes-before.txt" \
  "${source_raw}/npu7-processes-after.txt" \
  >"${raw}/input-integrity.log"
hash_status=$?
printf 'input-integrity\t%s\n' "${hash_status}" >>"${status}"
[[ ${hash_status} -eq 0 ]] || exit ${hash_status}

python3 "${src}/accept_attempt67e_r1.py" --source-raw "${source_raw}" \
  --output "${raw}/attempt67e-r1-acceptance.json" \
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

sha256sum "${raw}/attempt67e-r1-acceptance.json" "${raw}/status.tsv" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt67e-r1-acceptance.json" 2>/dev/null || true
[[ ${acceptance_status} -eq 0 ]] || exit ${acceptance_status}
exit ${idle_status}
