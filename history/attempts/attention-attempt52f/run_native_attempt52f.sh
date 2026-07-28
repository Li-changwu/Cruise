#!/usr/bin/env bash
set -uo pipefail

g2e=/root/ascend-control-g2e-20260718
g2g=/root/ascend-control-g2g-20260719
src=${g2e}/attempt52f-src
export_dir=${g2e}/attempt52f-export
raw=${g2e}/raw-attempt52f-native
cache=${g2e}/cache-attempt52f-native
inputs=${g2e}/raw-attempt3/native-inputs
outputs=${raw}/native-outputs
air=${export_dir}/qwen_attention_attempt52.air
tiling=${export_dir}/tiling.bin
candidate=${export_dir}/attempt52-eager-reference.npz
frozen=${g2e}/attempt7-export/attempt7-eager-reference.npz

if [[ -e "${raw}" || -e "${cache}" ]]; then exit 97; fi
mkdir -p "${raw}" "${cache}" "${outputs}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle\t0\n' >>"${status}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" ]] || exit 95
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
[[ -n "${barrier_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
sha256sum "${src}"/* "${air}" "${tiling}" "${candidate}" "${frozen}" \
  "${export_dir}/abi.json" >"${raw}/input-integrity.log"

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_attention_attempt52_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_attention_attempt52_host" >"${raw}/compile.log" 2>&1
s=$?; printf 'compile\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

timeout 900 "${raw}/native_attention_attempt52_host" "${air}" "${inputs}" \
  "${tiling}" "${outputs}" "${cache}" >"${raw}/native.stdout.log" 2>&1
s=$?; printf 'native\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then tail -240 "${raw}/native.stdout.log"; exit $s; fi

grep 'LaunchKernel: kernel info.*te_exactqk' "${raw}/native.stdout.log" \
  >"${raw}/exactqk-launch-metadata.txt"
s=$?; printf 'exactqk-launch\t%s\n' "$s" >>"${status}"
[[ $s -eq 0 ]] || exit 94

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16barrier_' "${raw}/native.stdout.log" \
  >"${raw}/barrier-launch-metadata.txt"
s=$?; printf 'barrier-launch\t%s\n' "$s" >>"${status}"
[[ $s -eq 0 ]] || exit 93
barrier_launch_count=$(wc -l <"${raw}/barrier-launch-metadata.txt")
printf '%s\n' "${barrier_launch_count}" >"${raw}/barrier-launch-count.txt"
if [[ ${barrier_launch_count} -lt 2 ]]; then
  printf 'barrier-launch-count\t92\n' >>"${status}"; exit 92
fi
printf 'barrier-launch-count\t0\n' >>"${status}"

python "${src}/compare_attention_attempt52.py" --native-dir "${outputs}" \
  --candidate-reference "${candidate}" --frozen-reference "${frozen}" \
  --output-npz "${raw}/attempt52-native-output.npz" \
  --output "${raw}/attempt52-result.json" >"${raw}/compare.stdout.log" 2>&1
s=$?; printf 'compare\t%s\n' "$s" >>"${status}"
sha256sum "${air}" "${raw}/attempt52-native-output.npz" "${raw}/attempt52-result.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1
idle=$?
if [[ $idle -eq 0 ]] && grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  printf 'npu7-idle-after\t0\n' >>"${status}"
else
  printf 'npu7-idle-after\t96\n' >>"${status}"
  exit 96
fi
cat "${raw}/attempt52-result.json" 2>/dev/null || true
exit $s
