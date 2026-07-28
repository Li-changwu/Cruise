#!/usr/bin/env bash
set -uo pipefail

g2e=/root/ascend-control-g2e-20260718
g2g=/root/ascend-control-g2g-20260719
src=${g2g}/attempt54b-probe-src
input=${g2e}/raw-attempt52c-native/native-outputs
export_dir=${g2g}/attempt54b-export
raw=${g2g}/raw-attempt54b-probe
cache=${g2g}/cache-attempt54b-probe
outputs=${raw}/native-outputs

if [[ -e "${export_dir}" || -e "${raw}" || -e "${cache}" ]]; then exit 97; fi
mkdir -p "${export_dir}" "${raw}" "${cache}" "${outputs}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle-before\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle-before\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
[[ -n "${barrier_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${barrier_set_env}"

python "${src}/export_barrier_probe.py" --input-dir "${input}" --output-dir "${export_dir}" \
  >"${raw}/export.stdout.log" 2>&1
s=$?; printf 'export\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_barrier_probe_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_barrier_probe_host" >"${raw}/compile.log" 2>&1
s=$?; printf 'compile\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

timeout 900 "${raw}/native_barrier_probe_host" "${export_dir}/bf16_barrier_probe.air" \
  "${input}" "${outputs}" "${cache}" >"${raw}/native.stdout.log" 2>&1
s=$?; printf 'native\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1
s=$?; if [[ $s -eq 0 ]] && grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  printf 'npu7-idle-after\t0\n' >>"${status}"
else
  printf 'npu7-idle-after\t96\n' >>"${status}"; exit 96
fi

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16barrier_' "${raw}/native.stdout.log" \
  >"${raw}/barrier-launch-metadata.txt"
s=$?; printf 'barrier-launch\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit 94

python "${src}/compare_barrier.py" --input-dir "${input}" --output-dir "${outputs}" \
  --output "${raw}/barrier-result.json" >"${raw}/compare.stdout.log" 2>&1
s=$?; printf 'compare\t%s\n' "$s" >>"${status}"
sha256sum "${src}"/* "${export_dir}/bf16_barrier_probe.air" "${raw}/barrier-result.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/barrier-result.json" 2>/dev/null || true
exit $s
