#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt49
cache_dir=${root}/cache-attempt49
native_input_dir=${root}/raw-attempt48/native-inputs
native_output_dir=${raw}/native-outputs
package=${root}/custom-op-project/output/CANN-custom_ops--linux.aarch64.run
install_dir=${root}/install-attempt47
air=${root}/export-attempt48/exact_qk_explicit_tiling_minimal.air
host_source=${root}/native_exact_qk_input_tiling_air_host.cpp
comparator=${root}/compare_g2g_attempt6.py
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
numa=${root}/numa_config.physical7.json
protocol=${root}/attempt49-protocol.md

expected_package_sha=6e4435ce0c85b4c18ca672a957a023c77b0f83810c86aeca6ac60e8be6b7e18e
expected_air_sha=00f846fa80fc09f43a760f28e64927027b7be9dc8c54be7d7b0442532a4269b9
expected_input_tree_sha=11371e992d0634958b5583d425e77013dbbd701e4aeb8e3b7dfcae6f37246768
expected_tiling_sha=1fc40ec0d67e231128773a5448cdd3333bdd1b97ea8a67bb7ba881d43b0da51f
expected_host_sha=8ce3a52c16cfb80301f81565cee7b622c9a48e44b21f1679cd171615a5736e0c
expected_comparator_sha=2fd2942433541c25bb1010a0d217c7354eb1236b0109379ed87ee0c47bffe455
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_protocol_sha=b190fdf59c69163c465c5c085d3d288c6f6668776484e7f3a10a0850e4fb1b03

if [[ -e "${raw}" || -e "${cache_dir}" ]]; then
  printf 'G2G_ATTEMPT49_REFUSE_OVERWRITE raw=%s cache=%s\n' \
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
if [[ $(sha256sum "${package}" | cut -d' ' -f1) != "${expected_package_sha}" ||
      $(sha256sum "${air}" | cut -d' ' -f1) != "${expected_air_sha}" ||
      "${input_tree_sha}" != "${expected_input_tree_sha}" ||
      $(sha256sum "${native_input_dir}/tiling.bin" | cut -d' ' -f1) != "${expected_tiling_sha}" ||
      $(sha256sum "${host_source}" | cut -d' ' -f1) != "${expected_host_sha}" ||
      $(sha256sum "${comparator}" | cut -d' ' -f1) != "${expected_comparator_sha}" ||
      $(sha256sum "${eager}" | cut -d' ' -f1) != "${expected_eager_sha}" ||
      $(sha256sum "${numa}" | cut -d' ' -f1) != "${expected_numa_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${package}" "${air}" "${native_input_dir}/tiling.bin" \
  "${host_source}" "${comparator}" "${eager}" "${numa}" "${protocol}" \
  "${root}/run_g2g_attempt49.sh" >"${raw}/artifact-integrity.log"
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
  -Wl,--no-whole-archive -o "${raw}/native_exact_qk_input_tiling_air_host" \
  >"${raw}/native-compile.log" 2>&1
compile_status=$?
printf 'native-compile\t%s\n' "${compile_status}" >>"${status_file}"
if [[ "${compile_status}" -ne 0 ]]; then
  tail -160 "${raw}/native-compile.log"
  exit "${compile_status}"
fi

timeout 900 "${raw}/native_exact_qk_input_tiling_air_host" "${air}" \
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
if [[ $(wc -l <"${raw}/exactqk-launch-metadata.txt") -eq 1 ]] &&
   grep -Fq 'arg_size=48' "${raw}/exactqk-launch-metadata.txt" &&
   grep -Fq 'kernelType=0' "${raw}/exactqk-launch-metadata.txt" &&
   grep -Fq 'coreDim=24' "${raw}/exactqk-launch-metadata.txt" &&
   grep -Fq 'schemMode=0' "${raw}/exactqk-launch-metadata.txt"; then
  abi_status=0
else
  abi_status=1
fi
printf 'explicit-input-launch-abi\t%s\n' "${abi_status}" >>"${status_file}"
if [[ "${native_status}" -ne 0 ]]; then
  tail -360 "${raw}/native.stdout.log"
  exit "${native_status}"
fi
if [[ "${launch_status}" -ne 0 || "${abi_status}" -ne 0 ]]; then
  exit 94
fi

python "${comparator}" --native-dir "${native_output_dir}" \
  --eager-reference "${eager}" --output-npz "${raw}/attempt49-output.npz" \
  --output "${raw}/attempt49-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status_file}"
if [[ "${compare_status}" -ne 0 ]]; then
  tail -160 "${raw}/compare.stdout.log"
  exit "${compare_status}"
fi
sha256sum "${air}" "${raw}/attempt49-output.npz" \
  "${raw}/attempt49-result.json" >"${raw}/result-integrity.log"
find "${cache_dir}" -type f \( -name 'te_exactqk*.json' -o \
  -name 'te_exactqk*.o' \) -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/exactqk-cache-integrity.log"
printf 'G2G_ATTEMPT49_COMPLETE\n'
