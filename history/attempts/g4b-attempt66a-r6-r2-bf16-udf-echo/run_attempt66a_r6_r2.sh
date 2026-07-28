#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt66a-r6-r2-src
raw=${root}/raw-attempt66a-r6-r2-bf16-udf-echo
cache=${root}/cache-attempt66a-r6-r2-bf16-udf-echo
build=${root}/build-attempt66a-r6-r2-bf16-udf-echo

if [[ -e "${raw}" || -e "${cache}" || -e "${build}" ]]; then exit 97; fi
mkdir -p "${raw}" "${cache}" "${build}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"; exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

find "${src}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/input-integrity.log"
cmake -S "${src}/host" -B "${build}" >"${raw}/cmake.stdout.log" 2>&1
cmake_status=$?
printf 'cmake\t%s\n' "${cmake_status}" >>"${status}"
if [[ ${cmake_status} -ne 0 ]]; then exit ${cmake_status}; fi
cmake --build "${build}" --parallel 2 >"${raw}/build.stdout.log" 2>&1
build_status=$?
printf 'build\t%s\n' "${build_status}" >>"${status}"
if [[ ${build_status} -ne 0 ]]; then exit ${build_status}; fi

if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-pre-run.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-pre-run.txt"; then
  printf 'npu7-idle-pre-run\t95\n' >>"${status}"; exit 95
fi
printf 'npu7-idle-pre-run\t0\n' >>"${status}"
timeout 1800 "${build}/bf16_udf_echo_smoke" \
  "${src}/config/bf16_echo_func.json" >"${raw}/smoke.stdout.log" 2>&1
smoke_status=$?
printf 'bf16-udf-echo\t%s\n' "${smoke_status}" >>"${status}"
capture_after
if grep -q No.process.in.device "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
tail -160 "${raw}/smoke.stdout.log"
if [[ ${smoke_status} -ne 0 ]]; then exit ${smoke_status}; fi
exit ${idle_status}
