#!/usr/bin/env bash
set -euo pipefail

project_root=${G4_PROJECT_ROOT:-/root/ascend-control-g4-20260723}
root_home=${STORAGE_ROOT_HOME:-/root}
ascend_log_root=${ASCEND_LOG_ROOT:-/root/ascend/log}
shm_parent=${G4_SHM_PARENT:-/dev/shm}
retention_manifest=${STORAGE_RETENTION_MANIFEST:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/retention-manifest.tsv}
root_retention_manifest=${STORAGE_ROOT_RETENTION_MANIFEST:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/root-retention-manifest.tsv}
min_root_free_gib=${MIN_ROOT_FREE_GIB:-100}
min_shm_free_gib=${MIN_SHM_FREE_GIB:-128}
max_project_gib=${MAX_PROJECT_GIB:-24}
max_g4_shm_gib=${MAX_G4_SHM_GIB:-96}
max_ascend_logs_gib=${MAX_ASCEND_LOGS_GIB:-2}
max_unlisted_dir_gib=${MAX_UNLISTED_DIR_GIB:-2}
max_root_home_gib=${MAX_ROOT_HOME_GIB:-48}
max_control_roots_gib=${MAX_CONTROL_ROOTS_GIB:-16}
max_unlisted_root_dir_gib=${MAX_UNLISTED_ROOT_DIR_GIB:-2}
audit_root_attribution=${AUDIT_ROOT_ATTRIBUTION:-1}
audit_root_home=${AUDIT_ROOT_HOME:-1}

bytes_for_gib() {
  printf '%s\n' "$(( $1 * 1024 * 1024 * 1024 ))"
}

available_bytes() {
  df -PB1 -- "$1" | awk 'NR == 2 {print $4}'
}

used_bytes() {
  if [[ -e "$1" ]]; then
    du -sxB1 -- "$1" | awk '{print $1}'
  else
    printf '0\n'
  fi
}

matching_dir_bytes() {
  local total=0 path bytes
  while IFS= read -r -d '' path; do
    bytes=$(used_bytes "${path}")
    total=$((total + bytes))
  done < <(find "$1" -xdev -mindepth 1 -maxdepth 1 -type d \
    -name 'ascend-control-g4-20*' -print0)
  printf '%s\n' "${total}"
}

declare -A retained_max_bytes=()
while IFS=$'\t' read -r name max_bytes role disposition; do
  [[ -z "${name}" || "${name}" == \#* || "${name}" == name ]] && continue
  [[ "${name}" != */* && "${max_bytes}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'STORAGE_AUDIT_ERROR\tinvalid retention manifest row: %s\n' \
      "${name:-empty}" >&2
    exit 65
  }
  retained_max_bytes["${name}"]=${max_bytes}
done <"${retention_manifest}"

declare -A root_retained_max_bytes=()
while IFS=$'\t' read -r name max_bytes role disposition; do
  [[ -z "${name}" || "${name}" == \#* || "${name}" == name ]] && continue
  [[ "${name}" != */* && "${max_bytes}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'STORAGE_AUDIT_ERROR\tinvalid root retention manifest row: %s\n' \
      "${name:-empty}" >&2
    exit 65
  }
  root_retained_max_bytes["${name}"]=${max_bytes}
done <"${root_retention_manifest}"

read -r fs_total fs_used fs_available < <(
  df -PB1 / | awk 'NR == 2 {print $2, $3, $4}'
)
project_bytes=$(used_bytes "${project_root}")
ascend_log_bytes=$(used_bytes "${ascend_log_root}")
shm_available=$(available_bytes "${shm_parent}")
g4_shm_bytes=$(matching_dir_bytes "${shm_parent}")
unapproved_large_dir=none
unapproved_large_dir_bytes=0
oversized_retained_dir=none
oversized_retained_dir_bytes=0
while IFS=$'\t' read -r dir_bytes dir; do
  [[ "${dir}" == "${project_root}" ]] && continue
  name=${dir##*/}
  if (( dir_bytes > $(bytes_for_gib "${max_unlisted_dir_gib}") )); then
    if [[ -z ${retained_max_bytes["${name}"]+x} ]]; then
      unapproved_large_dir=${dir}
      unapproved_large_dir_bytes=${dir_bytes}
      break
    fi
    if (( dir_bytes > retained_max_bytes["${name}"] )); then
      oversized_retained_dir=${dir}
      oversized_retained_dir_bytes=${dir_bytes}
      break
    fi
  fi
done < <(du -x -B1 --max-depth=1 -- "${project_root}" | sort -n)
root_home_bytes=-1
control_roots_bytes=-1
unapproved_large_root_dir=none
unapproved_large_root_dir_bytes=0
oversized_root_retained_dir=none
oversized_root_retained_dir_bytes=0
if (( audit_root_home != 0 )); then
  root_home_bytes=$(used_bytes "${root_home}")
  control_roots_bytes=0
  while IFS=$'\t' read -r dir_bytes dir; do
    [[ "${dir}" == "${root_home}" ]] && continue
    name=${dir##*/}
    [[ "${name}" == ascend-control-* ]] && \
      control_roots_bytes=$((control_roots_bytes + dir_bytes))
    if (( dir_bytes > $(bytes_for_gib "${max_unlisted_root_dir_gib}") )); then
      if [[ -z ${root_retained_max_bytes["${name}"]+x} ]]; then
        unapproved_large_root_dir=${dir}
        unapproved_large_root_dir_bytes=${dir_bytes}
        break
      fi
      if (( dir_bytes > root_retained_max_bytes["${name}"] )); then
        oversized_root_retained_dir=${dir}
        oversized_root_retained_dir_bytes=${dir_bytes}
        break
      fi
    fi
  done < <(du -x -B1 --max-depth=1 -- "${root_home}" | sort -n)
fi
if (( audit_root_attribution != 0 )); then
  visible_root_bytes=$(used_bytes /)
  backing_not_visible=$((fs_used - visible_root_bytes))
  (( backing_not_visible < 0 )) && backing_not_visible=0
else
  visible_root_bytes=-1
  backing_not_visible=-1
fi

status=PASS
reason=none
if (( fs_available < $(bytes_for_gib "${min_root_free_gib}") )); then
  status=BLOCK
  reason=root-reserve
elif (( shm_available < $(bytes_for_gib "${min_shm_free_gib}") )); then
  status=BLOCK
  reason=shm-reserve
elif (( audit_root_home != 0 && root_home_bytes > $(bytes_for_gib "${max_root_home_gib}") )); then
  status=BLOCK
  reason=root-home-budget
elif (( audit_root_home != 0 && control_roots_bytes > $(bytes_for_gib "${max_control_roots_gib}") )); then
  status=BLOCK
  reason=control-roots-budget
elif (( project_bytes > $(bytes_for_gib "${max_project_gib}") )); then
  status=BLOCK
  reason=project-budget
elif [[ "${unapproved_large_root_dir}" != none ]]; then
  status=BLOCK
  reason=unapproved-large-root-artifact
elif [[ "${oversized_root_retained_dir}" != none ]]; then
  status=BLOCK
  reason=root-retained-artifact-budget
elif [[ "${unapproved_large_dir}" != none ]]; then
  status=BLOCK
  reason=unapproved-large-artifact
elif [[ "${oversized_retained_dir}" != none ]]; then
  status=BLOCK
  reason=retained-artifact-budget
elif (( g4_shm_bytes > $(bytes_for_gib "${max_g4_shm_gib}") )); then
  status=BLOCK
  reason=g4-shm-budget
elif (( ascend_log_bytes > $(bytes_for_gib "${max_ascend_logs_gib}") )); then
  status=BLOCK
  reason=ascend-log-budget
fi

printf 'metric\tvalue\n'
printf 'utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'status\t%s\n' "${status}"
printf 'reason\t%s\n' "${reason}"
printf 'root_fs_total_bytes\t%s\n' "${fs_total}"
printf 'root_fs_used_bytes\t%s\n' "${fs_used}"
printf 'root_fs_available_bytes\t%s\n' "${fs_available}"
printf 'root_namespace_visible_bytes\t%s\n' "${visible_root_bytes}"
printf 'root_backing_store_not_visible_bytes\t%s\n' "${backing_not_visible}"
printf 'root_home_bytes\t%s\n' "${root_home_bytes}"
printf 'control_roots_bytes\t%s\n' "${control_roots_bytes}"
printf 'shm_available_bytes\t%s\n' "${shm_available}"
printf 'g4_shm_bytes\t%s\n' "${g4_shm_bytes}"
printf 'g4_project_bytes\t%s\n' "${project_bytes}"
printf 'ascend_log_bytes\t%s\n' "${ascend_log_bytes}"
printf 'unapproved_large_dir\t%s\n' "${unapproved_large_dir}"
printf 'unapproved_large_dir_bytes\t%s\n' "${unapproved_large_dir_bytes}"
printf 'oversized_retained_dir\t%s\n' "${oversized_retained_dir}"
printf 'oversized_retained_dir_bytes\t%s\n' "${oversized_retained_dir_bytes}"
printf 'unapproved_large_root_dir\t%s\n' "${unapproved_large_root_dir}"
printf 'unapproved_large_root_dir_bytes\t%s\n' "${unapproved_large_root_dir_bytes}"
printf 'oversized_root_retained_dir\t%s\n' "${oversized_root_retained_dir}"
printf 'oversized_root_retained_dir_bytes\t%s\n' "${oversized_root_retained_dir_bytes}"
printf 'min_root_free_gib\t%s\n' "${min_root_free_gib}"
printf 'min_shm_free_gib\t%s\n' "${min_shm_free_gib}"
printf 'max_project_gib\t%s\n' "${max_project_gib}"
printf 'max_g4_shm_gib\t%s\n' "${max_g4_shm_gib}"
printf 'max_ascend_logs_gib\t%s\n' "${max_ascend_logs_gib}"
printf 'max_unlisted_dir_gib\t%s\n' "${max_unlisted_dir_gib}"
printf 'max_root_home_gib\t%s\n' "${max_root_home_gib}"
printf 'max_control_roots_gib\t%s\n' "${max_control_roots_gib}"
printf 'max_unlisted_root_dir_gib\t%s\n' "${max_unlisted_root_dir_gib}"
printf 'root_attribution_collected\t%s\n' "${audit_root_attribution}"
printf 'root_home_audit_collected\t%s\n' "${audit_root_home}"

[[ "${status}" == PASS ]]
