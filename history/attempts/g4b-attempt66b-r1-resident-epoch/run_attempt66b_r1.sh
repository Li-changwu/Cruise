#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt66b-r1-src
raw=${root}/raw-attempt66b-r1
cache=${root}/cache-attempt66b-r1
build=${root}/build-attempt66b-r1
air=${root}/export-attempt65/qwen_full_decoder_step_attempt65.air
reference=${root}/raw-attempt65-eager/attempt65-eager-reference.npz
inputs=${raw}/inputs
host_outputs=${raw}/host
device_outputs=${raw}/device

if [[ -e "${raw}" || -e "${cache}" || -e "${build}" ]]; then exit 97; fi
if [[ ! -f "${air}" || ! -f "${reference}" ]]; then exit 96; fi
mkdir -p "${raw}" "${cache}" "${build}" \
  "${host_outputs}"/{k1,k2,k4,k8,early-eos} \
  "${device_outputs}"/{k1,k2,k4,k8,early-eos}
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

find "${src}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/source-integrity.log"
sha256sum "${air}" "${reference}" >"${raw}/frozen-artifact-integrity.log"
python "${src}/prepare_inputs.py" --reference "${reference}" \
  --output-dir "${inputs}" >"${raw}/prepare.stdout.log" 2>&1
prepare_status=$?
printf 'prepare-inputs\t%s\n' "${prepare_status}" >>"${status}"
if [[ ${prepare_status} -ne 0 ]]; then exit ${prepare_status}; fi

cmake -S "${src}/host" -B "${build}" >"${raw}/cmake.stdout.log" 2>&1
cmake_status=$?
printf 'cmake\t%s\n' "${cmake_status}" >>"${status}"
if [[ ${cmake_status} -ne 0 ]]; then exit ${cmake_status}; fi
cmake --build "${build}" --parallel 2 >"${raw}/build.stdout.log" 2>&1
build_status=$?
printf 'build\t%s\n' "${build_status}" >>"${status}"
if [[ ${build_status} -ne 0 ]]; then exit ${build_status}; fi

if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-pre-host.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-pre-host.txt"; then
  printf 'npu7-idle-pre-host\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle-pre-host\t0\n' >>"${status}"
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH
timeout 7200 "${build}/g4b_epoch_runner" host "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4b_resident_epoch_func.json" \
  "${inputs}" "${host_outputs}" -1 >"${raw}/host.stdout.log" 2>&1
host_status=$?
printf 'host-epoch-suite\t%s\n' "${host_status}" >>"${status}"
if [[ ${host_status} -ne 0 ]]; then
  tail -240 "${raw}/host.stdout.log"
  exit ${host_status}
fi
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after-host.txt" 2>&1 || true
if grep -q No.process.in.device "${raw}/npu7-processes-after-host.txt"; then
  host_idle_status=0
else
  host_idle_status=95
fi
printf 'npu7-idle-after-host\t%s\n' "${host_idle_status}" >>"${status}"
if [[ ${host_idle_status} -ne 0 ]]; then exit ${host_idle_status}; fi

early_eos=$(tr -d '[:space:]' <"${host_outputs}/early_eos_token.txt")
if ! grep -Eq '^[0-9]+$' <<<"${early_eos}"; then
  printf 'early-eos-token\t93\n' >>"${status}"
  exit 93
fi
printf 'early-eos-token\t0\n' >>"${status}"

if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-pre-device.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-pre-device.txt"; then
  printf 'npu7-idle-pre-device\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle-pre-device\t0\n' >>"${status}"
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
timeout 7200 "${build}/g4b_epoch_runner" device "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4b_resident_epoch_func.json" \
  "${inputs}" "${device_outputs}" "${early_eos}" \
  >"${raw}/device.stdout.log" 2>&1
device_status=$?
printf 'device-epoch-suite\t%s\n' "${device_status}" >>"${status}"
if [[ ${device_status} -ne 0 ]]; then
  tail -320 "${raw}/device.stdout.log"
  exit ${device_status}
fi

python "${src}/compare_epochs.py" --host-dir "${host_outputs}" \
  --device-dir "${device_outputs}" --input-dir "${inputs}" \
  --eager-reference "${reference}" --output "${raw}/attempt66b-r1-result.json" \
  >"${raw}/compare.stdout.log" 2>&1
compare_status=$?
printf 'compare\t%s\n' "${compare_status}" >>"${status}"
capture_after
if grep -q No.process.in.device "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${raw}/attempt66b-r1-result.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt66b-r1-result.json" 2>/dev/null || \
  tail -240 "${raw}/compare.stdout.log"
if [[ ${compare_status} -ne 0 ]]; then exit ${compare_status}; fi
exit ${idle_status}
