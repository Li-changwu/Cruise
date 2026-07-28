#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt56-materialize-src
export_dir=${root}/export-attempt56-materialize
raw=${root}/raw-attempt56-materialize-probe
cache=${root}/cache-attempt56-materialize-probe
air=${export_dir}/bf16_materialize_probe.air

if [[ -e "${export_dir}" || -e "${raw}" || -e "${cache}" ]]; then exit 97; fi
mkdir -p "${export_dir}" "${raw}" "${cache}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle-before\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle-before\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
unset RESOURCE_CONFIG_PATH
materialize_set_env=$(find "${root}/install-attempt56" -type f -name set_env.bash | head -1)
[[ -n "${materialize_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${materialize_set_env}"

sha256sum "${src}"/* >"${raw}/input-integrity.log"
python "${src}/export_probe.py" --output-dir "${export_dir}" \
  >"${raw}/export.stdout.log" 2>&1
s=$?; printf 'export\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
node_count=$(grep -c 'op: "Bf16Materialize"' "${export_dir}/dynamo.pbtxt")
if [[ ${node_count} -eq 1 ]]; then node_status=0; else node_status=94; fi
printf 'materialize-node\t%s\n' "${node_status}" >>"${status}"
[[ ${node_status} -eq 0 ]] || exit ${node_status}

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_probe_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_probe_host" >"${raw}/compile.log" 2>&1
s=$?; printf 'compile\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

timeout 900 "${raw}/native_probe_host" "${air}" "${export_dir}/input.bin" "${raw}/output.bin" \
  >"${raw}/native.stdout.log" 2>&1
s=$?; printf 'native\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
capture_after

grep -i 'LaunchKernel: kernel info.*kernel_name=te_bf16materialize_' "${raw}/native.stdout.log" \
  >"${raw}/materialize-launch-metadata.txt"
s=$?; printf 'materialize-launch\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit 93

python "${src}/compare_probe.py" --input "${export_dir}/input.bin" \
  --output "${raw}/output.bin" --result "${raw}/materialize-result.json" \
  >"${raw}/compare.stdout.log" 2>&1
compare_status=$?; printf 'compare\t%s\n' "${compare_status}" >>"${status}"
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${air}" "${export_dir}/input.bin" "${raw}/output.bin" \
  "${raw}/materialize-result.json" >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/compare.stdout.log"
if [[ ${compare_status} -ne 0 ]]; then exit ${compare_status}; fi
exit ${idle_status}
