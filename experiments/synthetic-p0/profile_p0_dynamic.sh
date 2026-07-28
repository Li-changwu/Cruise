#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-p0-20260722
out=${root}/dynamic-profiles-v3
g2b=/root/ascend-control-g2b-20260718
sample_dir=${g2b}/official-dflow/cpp
host_binary=${sample_dir}/output/sample_host_token_loop
device_binary=${sample_dir}/output/sample_device_token_loop
numa=${root}/numa_config.physical7.json

if [[ -e "${out}" ]]; then
  printf 'P0_DYNAMIC_REFUSE_OVERWRITE out=%s\n' "${out}"
  exit 97
fi
mkdir -p "${out}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${numa}
export ASCEND_GLOBAL_LOG_LEVEL=2
status_file=${out}/status.tsv
printf 'case\texit_status\n' >"${status_file}"

wait_for_marker() {
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

descendants() {
  local frontier=$1
  local all=""
  for _ in $(seq 1 6); do
    local next=""
    for pid in ${frontier}; do
      next="${next} $(pgrep -P "${pid}" 2>/dev/null || true)"
    done
    next=$(echo "${next}" | xargs)
    if [[ -z "${next}" ]]; then
      break
    fi
    all="${all} ${next}"
    frontier=${next}
  done
  echo "${all}" | xargs
}

run_dynamic() {
  local name=$1
  local marker=$2
  shift 2
  local log=${out}/${name}-workload.log
  "$@" >"${log}" 2>&1 &
  local workload_pid=$!
  wait_for_marker "${workload_pid}" "${log}" "${marker}"
  local ready_status=$?
  printf '%s-ready\t%s\n' "${name}" "${ready_status}" >>"${status_file}"
  if [[ "${ready_status}" -ne 0 ]]; then
    tail -160 "${log}"
    return "${ready_status}"
  fi

  ps -eo pid,ppid,comm,args >"${out}/${name}-process-tree.txt"
  local executor_pid=""
  for pid in $(descendants "${workload_pid}"); do
    if [[ $(ps -p "${pid}" -o comm= | xargs) == npu_executor* ]]; then
      executor_pid=${pid}
      break
    fi
  done
  if [[ -z "${executor_pid}" ]]; then
    printf '%s-find-executor\t96\n' "${name}" >>"${status_file}"
    wait "${workload_pid}"
    return 96
  fi
  printf '%s\n' "${executor_pid}" >"${out}/${name}-executor-pid.txt"
  printf '%s-find-executor\t0\n' "${name}" >>"${status_file}"
  printf '%s\n' "${workload_pid}" >"${out}/${name}-profile-target-pid.txt"

  msprof --output="${out}/${name}" --dynamic=on --pid="${workload_pid}" \
    --duration=10 \
    --runtime-api=on --task-time=l1 --ai-core=on \
    --aic-metrics=PipeUtilization >"${out}/${name}-msprof.log" 2>&1
  local profile_status=$?
  printf '%s-profile\t%s\n' "${name}" "${profile_status}" >>"${status_file}"
  wait "${workload_pid}"
  local workload_status=$?
  printf '%s-workload\t%s\n' "${name}" "${workload_status}" >>"${status_file}"
  if [[ "${profile_status}" -ne 0 || "${workload_status}" -ne 0 ]]; then
    tail -200 "${out}/${name}-msprof.log"
    tail -160 "${log}"
    return 1
  fi
}

cd "${sample_dir}/output"
run_dynamic host-ge HOST_TOKEN_LOOP_RESULT \
  "${host_binary}" 1024 100000 1 0 20 || exit $?
run_dynamic device-udf DEVICE_TOKEN_LOOP_RESULT \
  "${device_binary}" 1024 100000 1 100 || exit $?
printf 'P0_DYNAMIC_PROFILING_COMPLETE\n'
