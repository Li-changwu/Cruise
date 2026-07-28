#!/usr/bin/env bash
set -uo pipefail

control_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit=${control_dir}/audit_storage.sh
rotate=${control_dir}/rotate_ascend_logs.sh
reap=${control_dir}/reap_finalized_scratch.sh
state_dir=${STORAGE_WATCHDOG_STATE_DIR:-${control_dir}/state}
interval_seconds=${STORAGE_WATCHDOG_INTERVAL_SECONDS:-60}
rotation_interval_seconds=${STORAGE_WATCHDOG_ROTATION_INTERVAL_SECONDS:-3600}
reap_interval_seconds=${STORAGE_WATCHDOG_REAP_INTERVAL_SECONDS:-3600}
once=${STORAGE_WATCHDOG_ONCE:-0}
lock_path=${STORAGE_WATCHDOG_LOCK_PATH:-/dev/shm/ascend-storage-watchdog.lock}

[[ "${interval_seconds}" =~ ^[1-9][0-9]*$ &&
   "${rotation_interval_seconds}" =~ ^[1-9][0-9]*$ &&
   "${reap_interval_seconds}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ -x "${audit}" && -x "${rotate}" && -x "${reap}" ]] || exit 65
mkdir -p "${state_dir}" || exit 66

exec 9>"${lock_path}" || exit 67
flock -n 9 || exit 0
printf '%s\n' "$$" >"${state_dir}/watchdog.pid"

append_transition() {
  local status=$1 reason=$2 now=$3 log=${state_dir}/transitions.tsv
  if [[ -f "${log}" && $(stat -c %s "${log}") -gt 1048576 ]]; then
    mv -f -- "${log}" "${log}.previous"
  fi
  [[ -f "${log}" ]] || printf 'utc\tstatus\treason\n' >"${log}"
  printf '%s\t%s\t%s\n' "${now}" "${status}" "${reason}" >>"${log}"
}

last_status=
last_reason=
last_rotation=0
last_reap=0
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  now_epoch=$(date +%s)
  tmp=${state_dir}/.current.$$.tsv
  if AUDIT_ROOT_ATTRIBUTION=0 "${audit}" >"${tmp}" 2>&1; then
    audit_exit=0
  else
    audit_exit=$?
  fi
  mv -f -- "${tmp}" "${state_dir}/current.tsv"
  status=$(awk -F '\t' '$1 == "status" {print $2}' "${state_dir}/current.tsv")
  reason=$(awk -F '\t' '$1 == "reason" {print $2}' "${state_dir}/current.tsv")
  [[ -n "${status}" ]] || status=ERROR
  [[ -n "${reason}" ]] || reason=audit-exit-${audit_exit}
  if [[ "${status}" != "${last_status}" || "${reason}" != "${last_reason}" ]]; then
    append_transition "${status}" "${reason}" "${now}"
    last_status=${status}
    last_reason=${reason}
  fi
  if [[ "${status}" == PASS ]]; then
    rm -f -- "${state_dir}/BLOCK"
  else
    printf 'utc\t%s\nstatus\t%s\nreason\t%s\n' \
      "${now}" "${status}" "${reason}" >"${state_dir}/BLOCK"
  fi

  if (( now_epoch - last_rotation >= rotation_interval_seconds )); then
    rotation_tmp=${state_dir}/.last-log-rotation.$$.tsv
    if "${rotate}" --apply >"${rotation_tmp}" 2>&1; then
      :
    else
      printf 'rotation_exit_status\t%s\n' "$?" >>"${rotation_tmp}"
    fi
    mv -f -- "${rotation_tmp}" "${state_dir}/last-log-rotation.tsv"
    last_rotation=${now_epoch}
  fi

  if (( now_epoch - last_reap >= reap_interval_seconds )); then
    reap_tmp=${state_dir}/.last-scratch-reap.$$.tsv
    if "${reap}" --apply >"${reap_tmp}" 2>&1; then
      :
    else
      printf 'reap_exit_status\t%s\n' "$?" >>"${reap_tmp}"
    fi
    mv -f -- "${reap_tmp}" "${state_dir}/last-scratch-reap.tsv"
    last_reap=${now_epoch}
  fi

  (( once != 0 )) && break
  sleep "${interval_seconds}" || exit 69
done
