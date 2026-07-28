#!/usr/bin/env bash
set -euo pipefail

control_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test_root=/dev/shm/ascend-storage-control-selftest-$$
log_root=/dev/shm/ascend-log-rotation-test-$$
state_dir=${test_root}/state
shm_parent=${test_root}/shm
root_home=${test_root}/root-home
root_manifest=${test_root}/root-retention-manifest.tsv
cleanup() {
  rm -rf -- "${test_root}" "${log_root}"
}
trap cleanup EXIT
mkdir -p "${test_root}/project" "${log_root}/old" "${state_dir}" \
  "${shm_parent}" "${root_home}"
printf 'name\tmax_bytes\trole\tdisposition\n' >"${root_manifest}"
export STORAGE_ROOT_HOME=${root_home}
export STORAGE_ROOT_RETENTION_MANIFEST=${root_manifest}
export MAX_ROOT_HOME_GIB=1
export MAX_CONTROL_ROOTS_GIB=1

dd if=/dev/zero of="${log_root}/old/delete.log" bs=1024 count=80 status=none
dd if=/dev/zero of="${log_root}/keep.log" bs=1024 count=80 status=none
touch -d '5 days ago' "${log_root}/old/delete.log"
ASCEND_LOG_ROOT=${log_root} ASCEND_LOG_TRIGGER_BYTES=100000 \
  ASCEND_LOG_TARGET_BYTES=90000 ASCEND_LOG_RETENTION_DAYS=3 \
  "${control_dir}/rotate_ascend_logs.sh" --apply \
  >"${test_root}/rotation.tsv"
[[ ! -e "${log_root}/old/delete.log" && -f "${log_root}/keep.log" ]]
grep -Fqx $'deleted_count\t1' "${test_root}/rotation.tsv"

dd if=/dev/zero of="${log_root}/paused.log" bs=1024 count=80 status=none
touch -d '5 days ago' "${log_root}/paused.log"
touch "${log_root}/.storage-retention-pause"
ASCEND_LOG_ROOT=${log_root} ASCEND_LOG_TRIGGER_BYTES=100000 \
  ASCEND_LOG_TARGET_BYTES=90000 ASCEND_LOG_RETENTION_DAYS=3 \
  "${control_dir}/rotate_ascend_logs.sh" --apply \
  >"${test_root}/paused.tsv"
[[ -f "${log_root}/paused.log" ]]
grep -Fqx $'status\tPAUSED' "${test_root}/paused.tsv"

STORAGE_WATCHDOG_STATE_DIR=${state_dir} \
STORAGE_WATCHDOG_LOCK_PATH=${test_root}/watchdog.lock \
STORAGE_WATCHDOG_INTERVAL_SECONDS=1 \
STORAGE_WATCHDOG_ROTATION_INTERVAL_SECONDS=1 \
STORAGE_WATCHDOG_ONCE=1 \
G4_PROJECT_ROOT=${test_root}/project \
G4_SHM_PARENT=${shm_parent} \
ASCEND_LOG_ROOT=${log_root} \
MIN_ROOT_FREE_GIB=0 MIN_SHM_FREE_GIB=0 MAX_PROJECT_GIB=1 \
MAX_G4_SHM_GIB=1 \
MAX_ASCEND_LOGS_GIB=1 \
  "${control_dir}/storage_watchdog.sh"
grep -Fqx $'status\tPASS' "${state_dir}/current.tsv"
[[ ! -e "${state_dir}/BLOCK" ]]

unlisted_project=${test_root}/unlisted-project
mkdir -p "${unlisted_project}/unexpected-export"
printf 'x\n' >"${unlisted_project}/unexpected-export/payload"
if G4_PROJECT_ROOT=${unlisted_project} G4_SHM_PARENT=${shm_parent} \
   ASCEND_LOG_ROOT=${log_root} MIN_ROOT_FREE_GIB=0 MIN_SHM_FREE_GIB=0 \
   MAX_PROJECT_GIB=1 MAX_G4_SHM_GIB=1 MAX_ASCEND_LOGS_GIB=1 \
   MAX_UNLISTED_DIR_GIB=0 \
     "${control_dir}/audit_storage.sh" >"${test_root}/unlisted.tsv"; then
  exit 80
fi
grep -Fqx $'status\tBLOCK' "${test_root}/unlisted.tsv"
grep -Fqx $'reason\tunapproved-large-artifact' "${test_root}/unlisted.tsv"

mkdir -p "${root_home}/unexpected-root-artifact"
printf 'x\n' >"${root_home}/unexpected-root-artifact/payload"
if G4_PROJECT_ROOT=${test_root}/project G4_SHM_PARENT=${shm_parent} \
   ASCEND_LOG_ROOT=${log_root} MIN_ROOT_FREE_GIB=0 MIN_SHM_FREE_GIB=0 \
   MAX_PROJECT_GIB=1 MAX_G4_SHM_GIB=1 MAX_ASCEND_LOGS_GIB=1 \
   MAX_UNLISTED_ROOT_DIR_GIB=0 \
     "${control_dir}/audit_storage.sh" >"${test_root}/unlisted-root.tsv"; then
  exit 82
fi
grep -Fqx $'status\tBLOCK' "${test_root}/unlisted-root.tsv"
grep -Fqx $'reason\tunapproved-large-root-artifact' \
  "${test_root}/unlisted-root.tsv"
rm -rf -- "${root_home}/unexpected-root-artifact"

evidence=${test_root}/project/evidence-attempt-test
scratch=${shm_parent}/ascend-control-g4-20260726/attempt-clean
mkdir -p "${evidence}" "${scratch}"
printf '{"pass": true}\n' >"${evidence}/attempt-result.json"
printf 'case\texit_status\ncase-a\t0\n' >"${evidence}/status.tsv"
find "${evidence}" -type f ! -name evidence-integrity.log -print0 | \
  sort -z | xargs -0 sha256sum \
  >"${evidence}/evidence-integrity.log"
printf 'format\tstorage-guard-scratch-v1\nevidence_dir\t%s\n' "${evidence}" \
  >"${scratch}/.storage-guard-scratch.tsv"
printf 'evidence_dir\t%s\n' "${evidence}" \
  >"${scratch}/.storage-guard-finalized.tsv"
touch -d '20 minutes ago' "${scratch}/.storage-guard-finalized.tsv"
G4_SHM_PARENT=${shm_parent} G4_PROJECT_ROOT=${test_root}/project \
SCRATCH_REAP_GRACE_MINUTES=10 \
  "${control_dir}/reap_finalized_scratch.sh" --apply \
  >"${test_root}/reap.tsv"
[[ ! -e "${scratch}" ]]
grep -Fqx $'deleted_count\t1' "${test_root}/reap.tsv"

unresolved=${shm_parent}/ascend-control-g4-20260726/attempt-unresolved
mkdir -p "${unresolved}"
printf 'evidence_dir\t%s\n' "${evidence}" \
  >"${unresolved}/.storage-guard-finalized.tsv"
touch -d '20 minutes ago' "${unresolved}/.storage-guard-finalized.tsv"
G4_SHM_PARENT=${shm_parent} G4_PROJECT_ROOT=${test_root}/project \
SCRATCH_REAP_GRACE_MINUTES=10 \
  "${control_dir}/reap_finalized_scratch.sh" --apply \
  >"${test_root}/reap-unresolved.tsv"
[[ -d "${unresolved}" ]]
grep -Fqx $'deleted_count\t0' "${test_root}/reap-unresolved.tsv"

if G4_PROJECT_ROOT=${test_root}/project G4_SHM_PARENT=${shm_parent} \
   ASCEND_LOG_ROOT=${log_root} MIN_ROOT_FREE_GIB=0 MIN_SHM_FREE_GIB=0 \
   MAX_PROJECT_GIB=1 MAX_G4_SHM_GIB=0 MAX_ASCEND_LOGS_GIB=1 \
     "${control_dir}/audit_storage.sh" >"${test_root}/shm-budget.tsv"; then
  exit 81
fi
grep -Fqx $'reason\tg4-shm-budget' "${test_root}/shm-budget.tsv"

STORAGE_WATCHDOG_STATE_DIR=${test_root}/supervisor-state \
STORAGE_WATCHDOG_LOCK_PATH=${test_root}/supervised-watchdog.lock \
STORAGE_SUPERVISOR_LOCK_PATH=${test_root}/supervisor.lock \
STORAGE_SUPERVISOR_ONCE=1 STORAGE_WATCHDOG_ONCE=1 \
G4_PROJECT_ROOT=${test_root}/project G4_SHM_PARENT=${shm_parent} \
ASCEND_LOG_ROOT=${log_root} \
MIN_ROOT_FREE_GIB=0 MIN_SHM_FREE_GIB=0 MAX_PROJECT_GIB=1 \
MAX_G4_SHM_GIB=1 MAX_ASCEND_LOGS_GIB=1 \
  "${control_dir}/storage_watchdog_supervisor.sh"
grep -Fqx $'status\tPASS' "${test_root}/supervisor-state/current.tsv"

printf 'storage-control-selftest\tPASS\n'
