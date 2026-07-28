#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt42
cache_dir=${root}/cache-attempt42
native_input_dir=${root}/raw-attempt28/native-inputs
native_output_dir=${raw}/native-outputs
package=${root}/custom-op-project/output/CANN-custom_ops--linux.aarch64.run
install_dir=${root}/install-attempt41
air=${root}/export-attempt28/exact_qk_minimal.air
host_source=${root}/native_exact_qk_air_host_isolated_cache.cpp
comparator=${root}/compare_g2g_attempt6.py
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
numa=${root}/numa_config.physical7.json
protocol=${root}/attempt42-protocol.md

expected_package_sha=7b709fe0c5f700d6c3a82712fc8b538fed255e7e2d158e38e1d8eb679ec78124
expected_source_sha=3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc
expected_air_sha=f02a3753a7a0a118bf27982d561d6eb7efc650e45ef3eb00e070072ca7e3478a
expected_host_sha=9cddba7bd5e5387718985ae738e17eef60e262988883dac5325c2dc5472cd063
expected_comparator_sha=2fd2942433541c25bb1010a0d217c7354eb1236b0109379ed87ee0c47bffe455
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_input_tree_sha=dc6fd7b690a497f14b9ca48c34a12361ad1fb1d432bc1cc66326cde37d485587
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_protocol_sha=cebc7bbd999330802f2fb5f8a994fdc870865b0a9f7d8703682cd8f6ff463109

if [[ -e "${raw}" || -e "${cache_dir}" ]]; then
  printf 'G2G_ATTEMPT42_REFUSE_OVERWRITE raw=%s cache=%s\n' \
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

installed_source=$(find "${install_dir}" -type f \
  -path '*/ascendc/exact_qk/exact_qk.cpp' | head -1)
input_tree_sha=$(cd "${native_input_dir}" && \
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ $(sha256sum "${package}" | cut -d' ' -f1) != "${expected_package_sha}" ||
      -z "${installed_source}" ||
      $(sha256sum "${installed_source}" | cut -d' ' -f1) != "${expected_source_sha}" ||
      $(sha256sum "${air}" | cut -d' ' -f1) != "${expected_air_sha}" ||
      $(sha256sum "${host_source}" | cut -d' ' -f1) != "${expected_host_sha}" ||
      $(sha256sum "${comparator}" | cut -d' ' -f1) != "${expected_comparator_sha}" ||
      $(sha256sum "${eager}" | cut -d' ' -f1) != "${expected_eager_sha}" ||
      "${input_tree_sha}" != "${expected_input_tree_sha}" ||
      $(sha256sum "${numa}" | cut -d' ' -f1) != "${expected_numa_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]] ||
   grep -Fq 'gm_tiling_data->' "${installed_source}"; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${package}" "${installed_source}" "${air}" "${host_source}" \
  "${comparator}" "${eager}" "${numa}" "${protocol}" \
  "${root}/run_g2g_attempt42.sh" >"${raw}/artifact-integrity.log"
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
grep 'LaunchKernel: kernel info.*te_exactqk.*kernelType=0.*coreDim=24.*schemMode=1' \
  "${raw}/native.stdout.log" >"${raw}/exactqk-aic-mode1.txt"
mode_status=$?
printf 'aic-mode1\t%s\n' "${mode_status}" >>"${status_file}"
if [[ "${native_status}" -ne 0 ]]; then
  tail -320 "${raw}/native.stdout.log"
  exit "${native_status}"
fi
if [[ "${launch_status}" -ne 0 || "${mode_status}" -ne 0 ]]; then
  exit 94
fi

python "${comparator}" --native-dir "${native_output_dir}" \
  --eager-reference "${eager}" --output-npz "${raw}/attempt42-output.npz" \
  --output "${raw}/attempt42-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status_file}"
if [[ "${compare_status}" -ne 0 ]]; then
  tail -160 "${raw}/compare.stdout.log"
  exit "${compare_status}"
fi
sha256sum "${air}" "${raw}/attempt42-output.npz" \
  "${raw}/attempt42-result.json" >"${raw}/result-integrity.log"
find "${cache_dir}" -type f \( -name 'te_exactqk*.json' -o \
  -name 'te_exactqk*.o' \) -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/exactqk-cache-integrity.log"
printf 'G2G_ATTEMPT42_COMPLETE\n'
