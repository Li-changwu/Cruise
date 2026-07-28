#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-p0-20260722
raw=${root}/ge-device-sweep
g2b=/root/ascend-control-g2b-20260718
sample_dir=${g2b}/official-dflow/cpp
host_binary=${sample_dir}/output/sample_host_token_loop
device_binary=${sample_dir}/output/sample_device_token_loop
numa=${root}/numa_config.physical7.json
protocol=${root}/p0-protocol.md
expected_host_sha=7460f558b4daa401ec4272e371b3926532a1df52e6d4d228a601839f64612907
expected_device_sha=9159dbb130629ebedfda9a465395b65f59e6cbcfb4988ada19bc0f2b2f482730
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_protocol_sha=946919833c0b6bd7ada6c56d00f725c9fb04a9af60ba97de562985928fca6606

if [[ -e "${raw}" ]]; then
  printf 'P0_G2B_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT
npu-smi info >"${raw}/npu-before.txt" 2>&1

if [[ $(sha256sum "${host_binary}" | awk '{print $1}') != "${expected_host_sha}" ||
      $(sha256sum "${device_binary}" | awk '{print $1}') != "${expected_device_sha}" ||
      $(sha256sum "${numa}" | awk '{print $1}') != "${expected_numa_sha}" ||
      $(sha256sum "${protocol}" | awk '{print $1}') != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${host_binary}" "${device_binary}" "${numa}" "${protocol}" \
  "${root}/run_p0_g2b_sweep.sh" >"${raw}/artifact-integrity.log"

run_case() {
  local case_name=$1
  shift
  timeout 900 "$@" >"${raw}/${case_name}.stdout.log" 2>&1
  local status=$?
  printf '%s\t%s\n' "${case_name}" "${status}" >>"${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    tail -160 "${raw}/${case_name}.stdout.log"
    exit "${status}"
  fi
}

cd "${sample_dir}/output"
for n in 1 2 4 8 16 32; do
  run_case "host-ge-n${n}" "${host_binary}" "${n}" 100000 1 0 20
  run_case "device-udf-n${n}" "${device_binary}" "${n}" 100000 1 20
done
printf 'P0_G2B_SWEEP_COMPLETE\n'
