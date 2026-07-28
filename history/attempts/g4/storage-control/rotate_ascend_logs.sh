#!/usr/bin/env bash
set -euo pipefail

mode=${1:---dry-run}
[[ "${mode}" == --dry-run || "${mode}" == --apply ]] || {
  printf 'usage: %s [--dry-run|--apply]\n' "$0" >&2
  exit 64
}

log_root=${ASCEND_LOG_ROOT:-/root/ascend/log}
retention_days=${ASCEND_LOG_RETENTION_DAYS:-3}
trigger_gib=${ASCEND_LOG_TRIGGER_GIB:-2}
target_gib=${ASCEND_LOG_TARGET_GIB:-1}
pause_marker=${log_root}/.storage-retention-pause

[[ "${retention_days}" =~ ^[0-9]+$ &&
   "${trigger_gib}" =~ ^[1-9][0-9]*$ &&
   "${target_gib}" =~ ^[0-9]+$ &&
   ${target_gib} -lt ${trigger_gib} ]] || {
  printf 'LOG_ROTATION_ERROR\tinvalid retention or size setting\n' >&2
  exit 65
}

[[ -d "${log_root}" ]] || {
  printf 'mode\t%s\nstatus\tNO_LOG_ROOT\n' "${mode}"
  exit 0
}

log_root=$(realpath -e -- "${log_root}")
case "${log_root}" in
  /root/ascend/log|/dev/shm/ascend-log-rotation-test-*) ;;
  *)
    printf 'LOG_ROTATION_ERROR\tunsafe log root: %s\n' "${log_root}" >&2
    exit 66
    ;;
esac

if [[ -e "${pause_marker}" ]]; then
  printf 'mode\t%s\nstatus\tPAUSED\npause_marker\t%s\n' \
    "${mode}" "${pause_marker}"
  exit 0
fi

used_bytes=$(du -sxB1 -- "${log_root}" | awk '{print $1}')
trigger_bytes=${ASCEND_LOG_TRIGGER_BYTES:-$((trigger_gib * 1024 * 1024 * 1024))}
target_bytes=${ASCEND_LOG_TARGET_BYTES:-$((target_gib * 1024 * 1024 * 1024))}
[[ "${trigger_bytes}" =~ ^[1-9][0-9]*$ &&
   "${target_bytes}" =~ ^[0-9]+$ &&
   ${target_bytes} -lt ${trigger_bytes} ]] || {
  printf 'LOG_ROTATION_ERROR\tinvalid byte threshold override\n' >&2
  exit 65
}
if (( used_bytes <= trigger_bytes )); then
  printf 'mode\t%s\nstatus\tBELOW_TRIGGER\nlog_bytes_before\t%s\n' \
    "${mode}" "${used_bytes}"
  exit 0
fi

if find /proc/[0-9]*/fd -lname "${log_root}/*" -print -quit \
    2>/dev/null | grep -q .; then
  printf 'LOG_ROTATION_ERROR\tlog tree has an open file descriptor\n' >&2
  exit 67
fi

candidate_count=0
candidate_bytes=0
deleted_count=0
deleted_bytes=0
remaining_bytes=${used_bytes}
while IFS=$'\t' read -r -d '' mtime size path; do
  [[ -n "${path}" && "${path}" == "${log_root}/"* ]] || {
    printf 'LOG_ROTATION_ERROR\tunsafe candidate: %s\n' "${path}" >&2
    exit 68
  }
  candidate_count=$((candidate_count + 1))
  candidate_bytes=$((candidate_bytes + size))
  if [[ "${mode}" == --apply && ${remaining_bytes} -gt ${target_bytes} ]]; then
    rm -- "${path}"
    deleted_count=$((deleted_count + 1))
    deleted_bytes=$((deleted_bytes + size))
    remaining_bytes=$((remaining_bytes - size))
  fi
done < <(find "${log_root}" -xdev -type f ! -name '.storage-retention-pause' \
  -mtime "+${retention_days}" -printf '%T@\t%s\t%p\0' | sort -z -n)

if [[ "${mode}" == --apply ]]; then
  find "${log_root}" -xdev -depth -mindepth 1 -type d -empty -delete
  remaining_bytes=$(du -sxB1 -- "${log_root}" | awk '{print $1}')
fi

printf 'mode\t%s\n' "${mode}"
printf 'status\t%s\n' "$([[ "${mode}" == --apply ]] && printf APPLIED || printf DRY_RUN)"
printf 'log_bytes_before\t%s\n' "${used_bytes}"
printf 'log_bytes_after\t%s\n' "${remaining_bytes}"
printf 'candidate_count\t%s\n' "${candidate_count}"
printf 'candidate_bytes\t%s\n' "${candidate_bytes}"
printf 'deleted_count\t%s\n' "${deleted_count}"
printf 'deleted_bytes\t%s\n' "${deleted_bytes}"
