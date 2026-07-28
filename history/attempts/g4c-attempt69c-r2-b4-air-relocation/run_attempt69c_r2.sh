#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt69c-r2-src
raw=${root}/raw-attempt69c-r2-b4-air-relocation
build=${root}/build-attempt69c-r2-b4-air-relocation
output_dir=${root}/export-attempt69c-r2-b4
source_dir=${root}/export-attempt69c-b4
source_air=${source_dir}/qwen_b4_decoder_step_attempt69c.air
output_air=${output_dir}/qwen_b4_decoder_step_attempt69c_r2.air
old_prefix=/dev/shm/ascend-control-g4-20260724/attempt69c-b4-export-tmp
new_prefix=${source_dir}
expected_air_sha=de4a7bf439337970b343eb1fa91c3dd326545e81a88c94680c355234f96044bb

if [[ -e "${raw}" || -e "${build}" || -e "${output_dir}" ]]; then exit 97; fi
if [[ ! -f "${source_air}" || ! -f "${source_dir}/dedup-manifest.json" ]]; then
  exit 96
fi
actual_air_sha=$(sha256sum "${source_air}" | awk '{print $1}')
[[ "${actual_air_sha}" == "${expected_air_sha}" ]] || exit 95
mkdir -p "${raw}" "${build}" "${output_dir}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t94\n' >>"${status}"
  exit 94
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh
sha256sum "${src}"/*.cpp "${src}"/*.md "${src}"/*.sh "${source_air}" \
  "${source_dir}/dedup-manifest.json" >"${raw}/input-integrity.log"
g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -fPIC -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/relocate_air_paths.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" -Wl,--no-whole-archive \
  -o "${build}/relocate_air_paths" >"${raw}/compile.log" 2>&1
s=$?
printf 'compile\t%s\n' "${s}" >>"${status}"
[[ ${s} -eq 0 ]] || exit ${s}

"${build}/relocate_air_paths" "${source_air}" "${output_air}" \
  "${old_prefix}" "${new_prefix}" "${raw}/attempt69c-r2-result.json" \
  >"${raw}/relocate.stdout.log" 2>&1
s=$?
printf 'relocate-and-reload-audit\t%s\n' "${s}" >>"${status}"
[[ ${s} -eq 0 ]] || exit ${s}

old_count=$(grep -aoF "${old_prefix}/" "${output_air}" | wc -l)
new_count=$(grep -aoF "${new_prefix}/" "${output_air}" | wc -l)
if [[ ${old_count} -eq 0 && ${new_count} -eq 342 ]]; then
  binary_path_status=0
else
  binary_path_status=93
fi
printf '%s\t%s\n' "${old_count}" "${new_count}" \
  >"${raw}/binary-path-counts.tsv"
printf 'binary-path-audit\t%s\n' "${binary_path_status}" >>"${status}"

capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=94
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${source_air}" "${output_air}" \
  "${raw}/attempt69c-r2-result.json" >"${raw}/result-integrity.log"
cat "${raw}/attempt69c-r2-result.json"
if [[ ${binary_path_status} -ne 0 ]]; then exit ${binary_path_status}; fi
exit ${idle_status}
