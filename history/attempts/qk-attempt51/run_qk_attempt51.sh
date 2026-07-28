#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
src=${root}/attempt51-src
raw=${root}/raw-attempt51
export_dir=${root}/export-attempt51
cache_dir=${root}/cache-attempt51
inputs=${raw}/native-inputs
outputs=${raw}/native-outputs
attempt8=/root/ascend-control-g2e-20260718/raw-attempt8/attempt8-native-output.npz
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
direct=${root}/raw-attempt44/attempt44-output.npz
install_dir=${root}/install-attempt47

if [[ -e "${raw}" || -e "${export_dir}" || -e "${cache_dir}" ]]; then exit 97; fi
mkdir -p "${raw}" "${export_dir}" "${cache_dir}" "${inputs}" "${outputs}"
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
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
custom_set_env=$(find "${install_dir}" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
sha256sum "${src}"/* "${attempt8}" "${eager}" "${direct}" >"${raw}/input-integrity.log"

python "${src}/export_qk_attempt51_air.py" --attempt8-output "${attempt8}" \
  --eager-reference "${eager}" --output-dir "${export_dir}" \
  --native-input-dir "${inputs}" >"${raw}/export.stdout.log" 2>&1
s=$?; printf 'export\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || { tail -160 "${raw}/export.stdout.log"; exit $s; }

g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 -ftrapv -fstack-protector-all \
  -fPIC -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/include/external" \
  "${src}/native_qk_attempt51_host.cpp" -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" -Wl,--no-whole-archive \
  -o "${raw}/native_qk_attempt51_host" >"${raw}/native-compile.log" 2>&1
s=$?; printf 'native-compile\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s

timeout 900 "${raw}/native_qk_attempt51_host" "${export_dir}/qk_bf16_scaling_probe.air" \
  "${inputs}" "${outputs}" "${cache_dir}" >"${raw}/native.stdout.log" 2>&1
s=$?; printf 'native\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || { tail -200 "${raw}/native.stdout.log"; exit $s; }

python "${src}/compare_qk_attempt51.py" --native-dir "${outputs}" \
  --eager-reference "${eager}" --direct-reference "${direct}" \
  --output "${raw}/attempt51-result.json" >"${raw}/compare.stdout.log" 2>&1
s=$?; printf 'compare\t%s\n' "$s" >>"${status}"
sha256sum "${export_dir}/qk_bf16_scaling_probe.air" "${raw}/attempt51-result.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
cat "${raw}/attempt51-result.json" 2>/dev/null || true
exit $s

