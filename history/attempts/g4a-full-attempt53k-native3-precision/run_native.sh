#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt53k-native3-src
export_dir=${root}/export-attempt53k
raw=${root}/raw-attempt53k-native3
cache=${root}/cache-attempt53k-native3
inputs=${raw}/native-inputs
outputs=${raw}/native-outputs
air=${export_dir}/qwen_full_decoder_step_attempt53k.air
reference=${root}/raw-attempt53k-eager/attempt53k-eager-reference.npz

if [[ -e "${raw}" || -e "${cache}" ]]; then exit 97; fi
mkdir -p "${raw}" "${cache}" "${inputs}" "${outputs}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
unset RESOURCE_CONFIG_PATH
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" ]] || exit 95
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
[[ -n "${barrier_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"

sha256sum "${src}"/* "${air}" "${reference}" "${export_dir}/abi.json" \
  >"${raw}/input-integrity.log"

python "${src}/prepare_native_inputs.py" --reference "${reference}" \
  --output-dir "${inputs}" --manifest "${raw}/native-input-manifest.json" \
  >"${raw}/prepare.stdout.log" 2>&1
s=$?
printf 'prepare\t%s\n' "$s" >>"${status}"
[[ $s -eq 0 ]] || exit $s

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_full_decoder_attempt53k_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_full_decoder_attempt53k3_host" >"${raw}/compile.log" 2>&1
s=$?
printf 'compile\t%s\n' "$s" >>"${status}"
[[ $s -eq 0 ]] || exit $s

timeout 3600 "${raw}/native_full_decoder_attempt53k3_host" "${air}" "${inputs}" \
  "${outputs}" "${cache}" >"${raw}/native.stdout.log" 2>&1
s=$?
printf 'native\t%s\n' "$s" >>"${status}"
capture_after
if [[ $s -ne 0 ]]; then tail -240 "${raw}/native.stdout.log"; exit $s; fi

python "${src}/compare_native.py" --native-dir "${outputs}" --reference "${reference}" \
  --initial-input-dir "${inputs}" --output-npz "${raw}/attempt53k-native3-output.npz" \
  --output "${raw}/attempt53k-native3-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status}"

grep 'LaunchKernel: kernel info.*te_exactqk' "${raw}/native.stdout.log" \
  >"${raw}/exactqk-launch-metadata.txt"
exactqk_status=$?
printf 'exactqk-launch\t%s\n' "${exactqk_status}" >>"${status}"

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16barrier_' "${raw}/native.stdout.log" \
  >"${raw}/barrier-launch-metadata.txt"
barrier_status=$?
printf 'barrier-launch\t%s\n' "${barrier_status}" >>"${status}"

sha256sum "${air}" "${raw}/attempt53k-native3-output.npz" \
  "${raw}/attempt53k-native3-result.json" >"${raw}/result-integrity.log" 2>/dev/null || true
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  printf 'npu7-idle-after\t0\n' >>"${status}"
  idle_status=0
else
  printf 'npu7-idle-after\t96\n' >>"${status}"
  idle_status=96
fi
cat "${raw}/attempt53k-native3-result.json" 2>/dev/null || true
if [[ ${compare_status} -ne 0 ]]; then exit ${compare_status}; fi
if [[ ${exactqk_status} -ne 0 ]]; then exit 94; fi
if [[ ${barrier_status} -ne 0 ]]; then exit 93; fi
exit ${idle_status}
