#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt40
cache_dir=${root}/cache-attempt40
native_input_dir=${root}/raw-attempt28/native-inputs
native_output_dir=${raw}/native-outputs
package=${root}/custom-op-project/output/CANN-custom_ops--linux.aarch64.run
install_dir=${root}/install-attempt39
air=${root}/export-attempt28/exact_qk_minimal.air
host_source=${root}/native_exact_qk_air_host_isolated_cache.cpp
comparator=${root}/compare_g2g_attempt6.py
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
numa=${root}/numa_config.physical7.json
protocol=${root}/attempt40-protocol.md
expected_package_sha=f184a2d495e4ae52e96a9b4d35be7534d840322e6162b7dea8de41a693158f40
expected_air_sha=f02a3753a7a0a118bf27982d561d6eb7efc650e45ef3eb00e070072ca7e3478a
expected_host_sha=9cddba7bd5e5387718985ae738e17eef60e262988883dac5325c2dc5472cd063
expected_comparator_sha=2fd2942433541c25bb1010a0d217c7354eb1236b0109379ed87ee0c47bffe455
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_input_tree_sha=dc6fd7b690a497f14b9ca48c34a12361ad1fb1d432bc1cc66326cde37d485587
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_protocol_sha=500464c4daf6db5c7cb624e91648caa98e44e513a89eea2524e616156517cdfa

if [[ -e "${raw}" || -e "${cache_dir}" ]]; then
  printf 'G2G_ATTEMPT40_REFUSE_OVERWRITE raw=%s cache=%s\n' \
    "${raw}" "${cache_dir}"
  exit 97
fi
mkdir -p "${raw}" "${cache_dir}" "${native_output_dir}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${numa}
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT
npu-smi info >"${raw}/npu-before.txt" 2>&1

input_tree_sha=$(cd "${native_input_dir}" && \
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ $(sha256sum "${package}" | awk '{print $1}') != "${expected_package_sha}" ||
      $(sha256sum "${air}" | awk '{print $1}') != "${expected_air_sha}" ||
      $(sha256sum "${host_source}" | awk '{print $1}') != "${expected_host_sha}" ||
      $(sha256sum "${comparator}" | awk '{print $1}') != "${expected_comparator_sha}" ||
      $(sha256sum "${eager}" | awk '{print $1}') != "${expected_eager_sha}" ||
      "${input_tree_sha}" != "${expected_input_tree_sha}" ||
      $(sha256sum "${numa}" | awk '{print $1}') != "${expected_numa_sha}" ||
      $(sha256sum "${protocol}" | awk '{print $1}') != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${package}" "${air}" "${host_source}" "${comparator}" \
  "${eager}" "${numa}" "${protocol}" "${root}/run_g2g_attempt40.sh" \
  >"${raw}/artifact-integrity.log"
printf '%s  native-input-tree\n' "${input_tree_sha}" \
  >>"${raw}/artifact-integrity.log"

custom_set_env=$(find "${install_dir}" -type f -name set_env.bash | head -1)
if [[ -z "${custom_set_env}" ]]; then
  printf 'custom-set-env\t95\n' >>"${status_file}"
  exit 95
fi
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
printf 'custom-set-env\t0\n' >>"${status_file}"

compile_common=(
  -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv
  -fstack-protector-all -fPIC -I"${ASCEND_HOME_PATH}/include"
  -I"${ASCEND_HOME_PATH}/include/external"
)
g++ "${compile_common[@]}" "${host_source}" \
  -Wl,--whole-archive "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  -Wl,--no-whole-archive -o "${raw}/native_exact_qk_air_host" \
  >"${raw}/native-compile.log" 2>&1
compile_status=$?
printf 'native-compile\t%s\n' "${compile_status}" >>"${status_file}"
if [[ "${compile_status}" -ne 0 ]]; then
  tail -120 "${raw}/native-compile.log"
  exit "${compile_status}"
fi

timeout 900 "${raw}/native_exact_qk_air_host" "${air}" \
  "${native_input_dir}" "${native_output_dir}" "${cache_dir}" \
  >"${raw}/native.stdout.log" 2>&1
native_status=$?
printf 'native\t%s\n' "${native_status}" >>"${status_file}"
grep -E 'MatchCompileCache.*te_exactqk|cache dir.*te_exactqk|single op compile success.*te_exactqk' \
  "${raw}/native.stdout.log" >"${raw}/exactqk-cache-evidence.txt"
cache_evidence_status=$?
printf 'cache-evidence\t%s\n' "${cache_evidence_status}" >>"${status_file}"
grep 'LaunchKernel: kernel info.*te_exactqk' "${raw}/native.stdout.log" \
  >"${raw}/exactqk-launch-metadata.txt"
launch_status=$?
printf 'launch-metadata\t%s\n' "${launch_status}" >>"${status_file}"
grep 'LaunchKernel: kernel info.*te_exactqk.*schemMode=2' \
  "${raw}/native.stdout.log" >"${raw}/exactqk-schedule-mode-2.txt"
mode_status=$?
printf 'schedule-mode-2\t%s\n' "${mode_status}" >>"${status_file}"
if [[ "${native_status}" -ne 0 ]]; then
  tail -320 "${raw}/native.stdout.log"
  exit "${native_status}"
fi
if [[ "${launch_status}" -ne 0 || "${mode_status}" -ne 0 ]]; then
  exit 94
fi

python "${comparator}" --native-dir "${native_output_dir}" \
  --eager-reference "${eager}" --output-npz "${raw}/attempt40-output.npz" \
  --output "${raw}/attempt40-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status_file}"
if [[ "${compare_status}" -ne 0 ]]; then
  tail -160 "${raw}/compare.stdout.log"
  exit "${compare_status}"
fi
sha256sum "${air}" "${raw}/attempt40-output.npz" \
  "${raw}/attempt40-result.json" >"${raw}/result-integrity.log"
find "${cache_dir}" -type f \( -name 'te_exactqk*.json' -o \
  -name 'te_exactqk*.o' \) -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/exactqk-cache-integrity.log"
printf 'G2G_ATTEMPT40_COMPLETE\n'
