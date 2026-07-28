#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt69d-src
raw=${root}/raw-attempt69d-b4-native
build=${root}/build-attempt69d-b4-native
inputs=${raw}/native-inputs
outputs=${raw}/native-outputs
export_dir=${root}/export-attempt69c-b4
air=${export_dir}/qwen_b4_decoder_step_attempt69c.air
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${root}/raw-attempt69c-r1-b4-air-acceptance/attempt69c-r1-acceptance.json

if [[ -e "${raw}" || -e "${build}" ]]; then exit 97; fi
if [[ ! -f "${air}" || ! -f "${reference}" || ! -f "${acceptance}" ]] ||
   ! grep -q '"valid": true' "${export_dir}/abi.json" ||
   ! grep -q '"valid": true' "${export_dir}/graph-inspection.json" ||
   ! grep -q '"pass": true' "${acceptance}"; then
  exit 96
fi
mkdir -p "${raw}" "${build}" "${outputs}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH

sha256sum "${src}"/*.py "${src}"/*.cpp "${src}"/*.md "${src}"/*.sh \
  "${air}" "${reference}" "${export_dir}/abi.json" \
  "${export_dir}/graph-inspection.json" "${acceptance}" \
  "${root}/raw-attempt69a-b4-barrier-build/result-integrity.log" \
  >"${raw}/input-integrity.log"

python "${src}/prepare_b4_inputs.py" --reference "${reference}" \
  --output-dir "${inputs}" --manifest "${raw}/native-input-manifest.json" \
  >"${raw}/prepare.stdout.log" 2>&1
s=$?
printf 'prepare\t%s\n' "${s}" >>"${status}"
[[ ${s} -eq 0 ]] || exit ${s}

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv \
  -fstack-protector-all -fPIC -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_b4_decoder_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${build}/native_b4_decoder_host" >"${raw}/compile.log" 2>&1
s=$?
printf 'compile\t%s\n' "${s}" >>"${status}"
[[ ${s} -eq 0 ]] || exit ${s}

timeout 3600 "${build}/native_b4_decoder_host" "${air}" "${inputs}" \
  "${outputs}" >"${raw}/native.stdout.log" 2>&1
native_status=$?
printf 'native\t%s\n' "${native_status}" >>"${status}"
capture_after
if [[ ${native_status} -ne 0 ]]; then
  tail -240 "${raw}/native.stdout.log"
  exit ${native_status}
fi

python "${src}/compare_b4_native.py" --native-dir "${outputs}" \
  --reference "${reference}" --output-npz "${raw}/attempt69d-native-output.npz" \
  --output "${raw}/attempt69d-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status}"

grep 'LaunchKernel: kernel info.*te_exactqk' "${raw}/native.stdout.log" \
  >"${raw}/exactqk-launch-metadata.txt"
exact_count=$(wc -l <"${raw}/exactqk-launch-metadata.txt")
if [[ ${exact_count} -eq 112 ]]; then exact_status=0; else exact_status=93; fi
printf '%s\n' "${exact_count}" >"${raw}/exactqk-launch-count.txt"
printf 'exactqk-launch\t%s\n' "${exact_status}" >>"${status}"

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16barrier_' \
  "${raw}/native.stdout.log" >"${raw}/barrier-launch-metadata.txt"
barrier_count=$(wc -l <"${raw}/barrier-launch-metadata.txt")
if [[ ${barrier_count} -eq 28 ]]; then barrier_status=0; else barrier_status=92; fi
printf '%s\n' "${barrier_count}" >"${raw}/barrier-launch-count.txt"
printf 'barrier-launch\t%s\n' "${barrier_status}" >>"${status}"

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16materialize_' \
  "${raw}/native.stdout.log" >"${raw}/materialize-launch-metadata.txt"
materialize_count=$(wc -l <"${raw}/materialize-launch-metadata.txt")
if [[ ${materialize_count} -eq 56 ]]; then materialize_status=0; else materialize_status=91; fi
printf '%s\n' "${materialize_count}" >"${raw}/materialize-launch-count.txt"
printf 'materialize-launch\t%s\n' "${materialize_status}" >>"${status}"

python "${src}/extract_linear_launches.py" --log "${raw}/native.stdout.log" \
  --expected 197 --output "${raw}/linear-launches.json" \
  >"${raw}/linear-launches.stdout.log" 2>&1
linear_status=$?
printf 'linear-launches\t%s\n' "${linear_status}" >>"${status}"

if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${air}" "${raw}/attempt69d-native-output.npz" \
  "${raw}/attempt69d-result.json" "${raw}/linear-launches.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt69d-result.json" 2>/dev/null || true
cat "${raw}/linear-launches.json" 2>/dev/null || true
if [[ ${compare_status} -ne 0 ]]; then exit ${compare_status}; fi
if [[ ${exact_status} -ne 0 ]]; then exit ${exact_status}; fi
if [[ ${barrier_status} -ne 0 ]]; then exit ${barrier_status}; fi
if [[ ${materialize_status} -ne 0 ]]; then exit ${materialize_status}; fi
if [[ ${linear_status} -ne 0 ]]; then exit ${linear_status}; fi
exit ${idle_status}
