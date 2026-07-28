#!/usr/bin/env bash
set -uo pipefail

control_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
watchdog=${control_dir}/storage_watchdog.sh
state_dir=${STORAGE_WATCHDOG_STATE_DIR:-${control_dir}/state}
lock_path=${STORAGE_SUPERVISOR_LOCK_PATH:-/dev/shm/ascend-storage-watchdog-supervisor.lock}
check_seconds=${STORAGE_SUPERVISOR_CHECK_SECONDS:-10}
restart_seconds=${STORAGE_SUPERVISOR_RESTART_SECONDS:-5}
once=${STORAGE_SUPERVISOR_ONCE:-0}

[[ "${check_seconds}" =~ ^[1-9][0-9]*$ &&
   "${restart_seconds}" =~ ^[1-9][0-9]*$ && -x "${watchdog}" ]] || exit 64
mkdir -p "${state_dir}" || exit 65
exec 8>"${lock_path}" || exit 66
flock -n 8 || exit 0
printf '%s\n' "$$" >"${state_dir}/supervisor.pid"

watchdog_alive() {
  local pid=
  [[ -f "${state_dir}/watchdog.pid" ]] && pid=$(<"${state_dir}/watchdog.pid")
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null &&
    tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null |
      grep -Fq "${watchdog}"
}

while true; do
  if watchdog_alive; then
    (( once != 0 )) && exit 0
    sleep "${check_seconds}" || exit 67
    continue
  fi
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "${watchdog}"
  child_status=$?
  printf 'utc\t%s\nwatchdog_exit_status\t%s\n' \
    "${started}" "${child_status}" >"${state_dir}/last-watchdog-exit.tsv"
  (( once != 0 )) && exit "${child_status}"
  sleep "${restart_seconds}" || exit 67
done
