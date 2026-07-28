#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt69e-src
raw=${root}/raw-attempt69e-b4-resident-epoch
cache=/dev/shm/ascend-control-g4-20260724/cache-attempt69e-b4-resident-epoch
build=${root}/build-attempt69e-b4-resident-epoch
weights=/dev/shm/ascend-control-g4-20260724/external-weights-attempt69e
air=${root}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${root}/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json
inputs=${raw}/inputs
host_outputs=${raw}/host
device_outputs=${raw}/device
cases=(k1-heterogeneous k2-heterogeneous k4-heterogeneous k8-all-active \
       active-empty-alternating finished-active-empty-active \
       independent-early-eos)

if [[ -e "${raw}" || -e "${cache}" || -e "${build}" ||
      -e "${weights}" ]]; then exit 97; fi
if [[ ! -f "${air}" || ! -f "${reference}" || ! -f "${acceptance}" ]] ||
   ! grep -q '"pass": true' "${acceptance}"; then
  exit 96
fi
mkdir -p "${raw}" "${cache}" "${build}" "${weights}" \
  "${host_outputs}" "${device_outputs}"
for case_name in "${cases[@]}"; do
  mkdir -p "${host_outputs}/${case_name}" "${device_outputs}/${case_name}"
done
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
  du -sh "${weights}" >"${raw}/external-weights-size.txt" 2>&1 || true
  find "${weights}" -type f -printf '%p\t%s bytes\n' \
    | sort >"${raw}/external-weights-files.txt" 2>&1 || true
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
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" &&
   -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"
unset RESOURCE_CONFIG_PATH
unset ASCEND_CACHE_PATH

find "${src}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${raw}/source-integrity.log"
sha256sum "${air}" "${reference}" "${acceptance}" \
  >"${raw}/frozen-artifact-integrity.log"

python3 "${src}/prepare_inputs.py" --reference "${reference}" \
  --output-dir "${inputs}" >"${raw}/prepare.stdout.log" 2>&1
prepare_status=$?
printf 'prepare-inputs\t%s\n' "${prepare_status}" >>"${status}"
[[ ${prepare_status} -eq 0 ]] || exit ${prepare_status}

cmake -S "${src}/host" -B "${build}" >"${raw}/cmake.stdout.log" 2>&1
cmake_status=$?
printf 'cmake\t%s\n' "${cmake_status}" >>"${status}"
[[ ${cmake_status} -eq 0 ]] || exit ${cmake_status}
cmake --build "${build}" --parallel 2 >"${raw}/build.stdout.log" 2>&1
build_status=$?
printf 'build\t%s\n' "${build_status}" >>"${status}"
[[ ${build_status} -eq 0 ]] || exit ${build_status}

if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-pre-host.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-pre-host.txt"; then
  printf 'npu7-idle-pre-host\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle-pre-host\t0\n' >>"${status}"
timeout 10800 "${build}/g4c_b4_epoch_runner" host "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4c_b4_resident_epoch_func.json" \
  "${inputs}" "${host_outputs}" -1 -1 -1 -1 \
  >"${raw}/host.stdout.log" 2>&1
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
[[ ${host_idle_status} -eq 0 ]] || exit ${host_idle_status}

mapfile -t early_eos <"${host_outputs}/early_eos_tokens.txt"
if [[ ${#early_eos[@]} -ne 4 ]]; then
  printf 'early-eos-tokens\t93\n' >>"${status}"
  exit 93
fi
for eos_value in "${early_eos[@]}"; do
  if ! [[ ${eos_value} =~ ^[0-9]+$ ]]; then
    printf 'early-eos-tokens\t93\n' >>"${status}"
    exit 93
  fi
done
printf 'early-eos-tokens\t0\n' >>"${status}"

if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-pre-device.txt" 2>&1 ||
   ! grep -q No.process.in.device "${raw}/npu7-processes-pre-device.txt"; then
  printf 'npu7-idle-pre-device\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle-pre-device\t0\n' >>"${status}"
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
timeout 10800 "${build}/g4c_b4_epoch_runner" device "${air}" \
  "${src}/config/graph_config.json" \
  "${src}/config/g4c_b4_resident_epoch_func.json" \
  "${inputs}" "${device_outputs}" "${early_eos[@]}" \
  >"${raw}/device.stdout.log" 2>&1
device_status=$?
printf 'device-epoch-suite\t%s\n' "${device_status}" >>"${status}"
if [[ ${device_status} -ne 0 ]]; then
  tail -320 "${raw}/device.stdout.log"
  exit ${device_status}
fi

python3 "${src}/compare_epochs.py" --host-dir "${host_outputs}" \
  --device-dir "${device_outputs}" --input-dir "${inputs}" \
  --eager-reference "${reference}" --output "${raw}/attempt69e-result.json" \
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
sha256sum "${raw}/attempt69e-result.json" "${raw}/status.tsv" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt69e-result.json" 2>/dev/null || \
  tail -240 "${raw}/compare.stdout.log"
[[ ${compare_status} -eq 0 ]] || exit ${compare_status}
exit ${idle_status}
