#!/usr/bin/env bash
set -euo pipefail

control_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
supervisor=${control_dir}/storage_watchdog_supervisor.sh
state_dir=${control_dir}/state
lock=/dev/shm/ascend-storage-watchdog-supervisor-start.lock

[[ -x "${supervisor}" ]] || exit 65
supervisor_alive() {
  local pid=
  [[ -f "${state_dir}/supervisor.pid" ]] && pid=$(<"${state_dir}/supervisor.pid")
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null &&
    tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null |
      grep -Fq "${supervisor}"
}

supervisor_alive && exit 0
exec 8>"${lock}"
flock -w 5 8 || exit 66
supervisor_alive && exit 0

mkdir -p "${state_dir}"
nohup setsid "${supervisor}" 8>&- </dev/null >/dev/null 2>&1 &
new_pid=$!
for _ in 1 2 3 4 5; do
  sleep 1
  if [[ -s "${state_dir}/supervisor.pid" ]] && kill -0 "${new_pid}" 2>/dev/null; then
    exit 0
  fi
done
exit 67
