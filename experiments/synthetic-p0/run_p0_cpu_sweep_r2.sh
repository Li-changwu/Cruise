#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-p0-20260722
raw=${root}/cpu-sweep-r2
sample_dir=/root/ascend-control-g2b-20260718/official-dflow/cpp
host_binary=${root}/cpu-sweep/bin/sample_host_token_loop_cpu
device_binary=${root}/cpu-sweep/bin/sample_device_token_loop_cpu
numa=${root}/numa_config.physical7.json
expected_host_sha=a154896e55857d04544b39b5dc574829520bfe59cb661a461ca1b47544811adb
expected_device_sha=c9f53eaf8c103f2841360b8b7e9bc3a64652cec33071d4bc6c4371bb0993f819
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf

if [[ -e "${raw}" ]]; then
  printf 'P0_CPU_R2_REFUSE_OVERWRITE raw=%s\n' "${raw}"
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
npu-smi info >"${raw}/npu-before.txt" 2>&1
uptime >"${raw}/host-before.txt"
top -b -n1 | head -8 >>"${raw}/host-before.txt"
capture_after() {
  npu-smi info >"${raw}/npu-after.txt" 2>&1 || true
  uptime >"${raw}/host-after.txt"
  top -b -n1 | head -8 >>"${raw}/host-after.txt"
}
trap capture_after EXIT

if [[ $(sha256sum "${host_binary}" | awk '{print $1}') != "${expected_host_sha}" ||
      $(sha256sum "${device_binary}" | awk '{print $1}') != "${expected_device_sha}" ||
      $(sha256sum "${numa}" | awk '{print $1}') != "${expected_numa_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${host_binary}" "${device_binary}" "${numa}" \
  "${root}/run_p0_cpu_sweep_r2.sh" >"${raw}/artifact-integrity.log"

run_case() {
  local case_name=$1
  shift
  npu-smi info -t proc-mem -i 7 >"${raw}/${case_name}.npu-before.txt" 2>&1
  timeout 900 "$@" >"${raw}/${case_name}.stdout.log" 2>&1
  local status=$?
  printf '%s\t%s\n' "${case_name}" "${status}" >>"${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    tail -160 "${raw}/${case_name}.stdout.log"
    exit "${status}"
  fi
}

cd "${sample_dir}/output"
index=0
for n in 32 16 8 4 2 1; do
  if (( index % 2 == 0 )); then
    run_case "host-ge-n${n}" "${host_binary}" "${n}" 100000 5 0 30
    run_case "device-udf-n${n}" "${device_binary}" "${n}" 100000 5 30
  else
    run_case "device-udf-n${n}" "${device_binary}" "${n}" 100000 5 30
    run_case "host-ge-n${n}" "${host_binary}" "${n}" 100000 5 0 30
  fi
  index=$((index + 1))
done
printf 'P0_CPU_SWEEP_R2_COMPLETE\n'
