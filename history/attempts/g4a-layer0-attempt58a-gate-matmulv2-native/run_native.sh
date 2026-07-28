#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt58a-native-src
export_dir=${root}/export-attempt58a
raw=${root}/raw-attempt58a-native
cache=${root}/cache-attempt58a-native
inputs=${raw}/native-inputs
outputs=${raw}/native-outputs
air=${export_dir}/qwen_layer0_boundary_attempt58a.air
reference=${export_dir}/attempt58a-eager-reference.npz
baseline_reference=${root}/export-attempt57a/attempt57a-eager-reference.npz
baseline_native=${root}/raw-attempt57a-native/attempt57a-native-output.npz

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
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.cpp "${src}"/*.sh "${src}"/*.md \
  "${air}" "${reference}" "${export_dir}/eager-screen.json" "${export_dir}/abi.json" \
  "${baseline_reference}" "${baseline_native}" \
  >"${raw}/input-integrity.log"
python "${src}/prepare_inputs.py" --reference "${reference}" --output-dir "${inputs}" \
  --manifest "${raw}/native-input-manifest.json" >"${raw}/prepare.stdout.log" 2>&1
s=$?; printf 'prepare\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_layer0_attempt58a_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_layer0_attempt58a_host" >"${raw}/compile.log" 2>&1
s=$?; printf 'compile\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

timeout 1200 "${raw}/native_layer0_attempt58a_host" "${air}" "${inputs}" "${outputs}" \
  >"${raw}/native.stdout.log" 2>&1
native_status=$?
printf 'native\t%s\n' "${native_status}" >>"${status}"
capture_after
if [[ ${native_status} -ne 0 ]]; then tail -240 "${raw}/native.stdout.log"; exit ${native_status}; fi

python "${src}/compare_native.py" --native-dir "${outputs}" --reference "${reference}" \
  --eager-screen "${export_dir}/eager-screen.json" --initial-input-dir "${inputs}" \
  --output-npz "${raw}/attempt58a-native-output.npz" \
  --output "${raw}/attempt58a-result.json" >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status}"

python "${src}/compare_boundary.py" --baseline-native "${baseline_native}" \
  --baseline-reference "${baseline_reference}" \
  --candidate-native "${raw}/attempt58a-native-output.npz" \
  --candidate-reference "${reference}" \
  --output "${raw}/attempt58a-mechanism.json" >"${raw}/mechanism.stdout.log" 2>&1
localization_status=$?
printf 'gate-matmulv2\t%s\n' "${localization_status}" >>"${status}"

grep 'LaunchKernel: kernel info.*te_exactqk' "${raw}/native.stdout.log" \
  >"${raw}/exactqk-launch-metadata.txt"
exact_status=$?
printf 'exactqk-launch\t%s\n' "${exact_status}" >>"${status}"
grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16barrier_' "${raw}/native.stdout.log" \
  >"${raw}/barrier-launch-metadata.txt"
barrier_status=$?
printf 'barrier-launch\t%s\n' "${barrier_status}" >>"${status}"
grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16materialize_' \
  "${raw}/native.stdout.log" >"${raw}/materialize-launch-metadata.txt"
materialize_status=$?
materialize_count=$(wc -l <"${raw}/materialize-launch-metadata.txt")
printf '%s\n' "${materialize_count}" >"${raw}/materialize-launch-count.txt"
if [[ ${materialize_status} -eq 0 && ${materialize_count} -eq 1 ]]; then
  materialize_gate=0
else
  materialize_gate=92
fi
printf 'materialize-launch\t%s\n' "${materialize_gate}" >>"${status}"

if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${air}" "${raw}/attempt58a-native-output.npz" \
  "${raw}/attempt58a-result.json" "${raw}/attempt58a-mechanism.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/compare.stdout.log"
cat "${raw}/mechanism.stdout.log"
if [[ ${compare_status} -ne 0 ]]; then exit ${compare_status}; fi
if [[ ${exact_status} -ne 0 ]]; then exit 94; fi
if [[ ${barrier_status} -ne 0 ]]; then exit 93; fi
if [[ ${materialize_gate} -ne 0 ]]; then exit ${materialize_gate}; fi
if [[ ${localization_status} -ne 0 ]]; then exit ${localization_status}; fi
exit ${idle_status}
