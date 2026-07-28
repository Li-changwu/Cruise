#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-p0-20260722
profiles=${root}/profiles
g2b=/root/ascend-control-g2b-20260718
sample_dir=${g2b}/official-dflow/cpp
host_binary=${sample_dir}/output/sample_host_token_loop
device_binary=${sample_dir}/output/sample_device_token_loop
numa=${root}/numa_config.physical7.json
expected_host_sha=7460f558b4daa401ec4272e371b3926532a1df52e6d4d228a601839f64612907
expected_device_sha=9159dbb130629ebedfda9a465395b65f59e6cbcfb4988ada19bc0f2b2f482730
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf

if [[ -e "${profiles}" ]]; then
  printf 'P0_PROFILE_REFUSE_OVERWRITE profiles=%s\n' "${profiles}"
  exit 97
fi
mkdir -p "${profiles}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2
status_file=${profiles}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
npu-smi info >"${profiles}/npu-before.txt" 2>&1
capture_after() { npu-smi info >"${profiles}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT

if [[ $(sha256sum "${host_binary}" | awk '{print $1}') != "${expected_host_sha}" ||
      $(sha256sum "${device_binary}" | awk '{print $1}') != "${expected_device_sha}" ||
      $(sha256sum "${numa}" | awk '{print $1}') != "${expected_numa_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${host_binary}" "${device_binary}" "${numa}" \
  "${root}/profile_p0_routes.sh" >"${profiles}/artifact-integrity.log"

wait_for_first_result() {
  local pid=$1
  local log=$2
  local marker=$3
  for _ in $(seq 1 1200); do
    if grep -Fq "${marker}" "${log}" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"
      return $?
    fi
    sleep 0.5
  done
  return 124
}

run_system_profile() {
  local name=$1
  local marker=$2
  shift 2
  local workload_log=${profiles}/${name}-steady-workload.log
  "$@" >"${workload_log}" 2>&1 &
  local workload_pid=$!
  wait_for_first_result "${workload_pid}" "${workload_log}" "${marker}"
  local ready_status=$?
  printf '%s-ready\t%s\n' "${name}" "${ready_status}" >>"${status_file}"
  if [[ "${ready_status}" -ne 0 ]]; then
    tail -160 "${workload_log}"
    exit "${ready_status}"
  fi
  msprof --output="${profiles}/${name}-sys" --sys-devices=7 \
    --sys-period=10 --sys-cpu-profiling=on \
    >"${profiles}/${name}-sys-msprof.log" 2>&1
  local profile_status=$?
  printf '%s-sys-profile\t%s\n' "${name}" "${profile_status}" \
    >>"${status_file}"
  wait "${workload_pid}"
  local workload_status=$?
  printf '%s-workload\t%s\n' "${name}" "${workload_status}" \
    >>"${status_file}"
  if [[ "${profile_status}" -ne 0 || "${workload_status}" -ne 0 ]]; then
    tail -160 "${profiles}/${name}-sys-msprof.log"
    tail -160 "${workload_log}"
    return 1
  fi
}

cd "${sample_dir}/output"
run_system_profile host-ge HOST_TOKEN_LOOP_RESULT \
  "${host_binary}" 1024 100000 1 0 40
run_system_profile device-udf DEVICE_TOKEN_LOOP_RESULT \
  "${device_binary}" 1024 100000 1 100

run_app_profile() {
  local name=$1
  shift
  timeout 900 msprof --output="${profiles}/${name}-app" \
    --runtime-api=on --ge-api=l0 --task-time=l1 --ai-core=on \
    --aic-metrics=PipeUtilization "$@" \
    >"${profiles}/${name}-app-msprof.log" 2>&1
  local status=$?
  printf '%s-app-profile\t%s\n' "${name}" "${status}" >>"${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    tail -240 "${profiles}/${name}-app-msprof.log"
    return "${status}"
  fi
}

run_app_profile host-ge "${host_binary}" 32 100000 1 0 5
run_app_profile device-udf "${device_binary}" 32 100000 1 5
printf 'P0_ROUTE_PROFILING_COMPLETE\n'
