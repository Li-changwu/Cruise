#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
source_raw=${root}/raw-attempt42
raw=${root}/raw-attempt43
outputs=${source_raw}/native-outputs
launch=${source_raw}/exactqk-launch-metadata.txt
comparator=${root}/compare_g2g_attempt6.py
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
protocol=${root}/attempt43-protocol.md
python=/root/miniconda3/envs/vllm-hust-dev/bin/python

expected_native_log_sha=3782966d322afe7b1cd12021784e43fa4c9f140688207855fce6e663355d7ac0
expected_status_sha=28ec6236657de1162a91b92e5edc05469e78421a576940845dc8a8fd3a9a4f9b
expected_launch_sha=41064632f9908c06f84bbb36f4025dde6e34fb308ff59a39c3c6cb733b315f44
expected_output_tree_sha=8dce013bc213928ecea41eddf6d6dedc4db4bf9785cbd7afd6ce7970670c7d85
expected_comparator_sha=2fd2942433541c25bb1010a0d217c7354eb1236b0109379ed87ee0c47bffe455
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_protocol_sha=55ca8e33e1fc897c6feccb18e8deb0256f9b1f72d76b3178ab6f7842ba3ff901

if [[ -e "${raw}" ]]; then
  printf 'G2G_ATTEMPT43_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}"
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"

output_tree_sha=$(cd "${source_raw}" && \
  find native-outputs -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ $(sha256sum "${source_raw}/native.stdout.log" | cut -d' ' -f1) != "${expected_native_log_sha}" ||
      $(sha256sum "${source_raw}/status.tsv" | cut -d' ' -f1) != "${expected_status_sha}" ||
      $(sha256sum "${launch}" | cut -d' ' -f1) != "${expected_launch_sha}" ||
      "${output_tree_sha}" != "${expected_output_tree_sha}" ||
      $(sha256sum "${comparator}" | cut -d' ' -f1) != "${expected_comparator_sha}" ||
      $(sha256sum "${eager}" | cut -d' ' -f1) != "${expected_eager_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${source_raw}/native.stdout.log" "${source_raw}/status.tsv" \
  "${launch}" "${comparator}" "${eager}" "${protocol}" \
  >"${raw}/artifact-integrity.log"
printf '%s  attempt42-native-output-tree\n' "${output_tree_sha}" \
  >>"${raw}/artifact-integrity.log"

if ! awk -F '\t' '$1 == "native" && $2 == "0" { found=1 } END { exit !found }' \
  "${source_raw}/status.tsv"; then
  printf 'attempt42-native-success\t96\n' >>"${status_file}"
  exit 96
fi
printf 'attempt42-native-success\t0\n' >>"${status_file}"

if [[ $(wc -l <"${launch}") -ne 1 ]] ||
   ! grep -Fq 'te_exactqk' "${launch}" ||
   ! grep -Fq 'kernelType=0' "${launch}" ||
   ! grep -Fq 'coreDim=24' "${launch}" ||
   ! grep -Fq 'schemMode=1' "${launch}"; then
  printf 'launch-fields\t95\n' >>"${status_file}"
  exit 95
fi
cp "${launch}" "${raw}/validated-launch-metadata.txt"
printf 'launch-fields\t0\n' >>"${status_file}"

for step in 1 2 3 4; do
  path=${outputs}/step${step}_qk_scores.bin
  if [[ ! -f "${path}" || $(stat -c '%s' "${path}") -ne 896 ]]; then
    printf 'output-contract\t94\n' >>"${status_file}"
    exit 94
  fi
done
if [[ $(find "${outputs}" -maxdepth 1 -type f | wc -l) -ne 4 ]]; then
  printf 'output-contract\t94\n' >>"${status_file}"
  exit 94
fi
printf 'output-contract\t0\n' >>"${status_file}"

"${python}" "${comparator}" --native-dir "${outputs}" \
  --eager-reference "${eager}" --output-npz "${raw}/attempt43-output.npz" \
  --output "${raw}/attempt43-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status_file}"
if [[ "${compare_status}" -ne 0 ]]; then
  tail -160 "${raw}/compare.stdout.log"
  exit "${compare_status}"
fi
sha256sum "${raw}/attempt43-output.npz" "${raw}/attempt43-result.json" \
  >"${raw}/result-integrity.log"
printf 'G2G_ATTEMPT43_COMPLETE_NO_NPU\n'
