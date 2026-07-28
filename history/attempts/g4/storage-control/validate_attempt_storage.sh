#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  printf 'usage: %s ATTEMPT_SOURCE_DIR\n' "$0" >&2
  exit 64
fi

src=$(realpath -m -- "$1")
run_script=$(find "${src}" -maxdepth 1 -type f -name 'run_attempt*.sh' -print -quit)
[[ -n "${run_script}" ]] || {
  printf 'STORAGE_VALIDATION_ERROR\tmissing run_attempt*.sh\n' >&2
  exit 65
}

required_patterns=(
  'storage_guard_preflight "${root}" "${evidence}" "${scratch}" 7 100 24 128'
  'export STORAGE_GUARD_MAX_SCRATCH_GIB=64'
  'export STORAGE_GUARD_NPU_STABLE_SAMPLES=3'
  'export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=5'
  'export ASCEND_PROCESS_LOG_PATH=${cann_logs}'
  'storage_guard_run_log'
  'storage_guard_finalize'
  'storage_guard_cleanup_scratch'
  'trap on_exit EXIT'
  'scratch=/dev/shm/'
)

for pattern in "${required_patterns[@]}"; do
  if ! grep -Fq "${pattern}" "${run_script}"; then
    printf 'STORAGE_VALIDATION_ERROR\t%s\tmissing: %s\n' \
      "${run_script}" "${pattern}" >&2
    exit 66
  fi
done

declare -A expected_guard_hashes=(
  [storage_guard.sh]=0281492db54cc1088394b4ff71dd77ebd90aef203ad40f6fea0a586a93706bea
  [bounded_log.py]=503c25465b09675d048ccea619c39e7607dee387f08ab3b85699d6400a9175be
)
for name in "${!expected_guard_hashes[@]}"; do
  path=${src}/storage_guard/${name}
  [[ -f "${path}" ]] || {
    printf 'STORAGE_VALIDATION_ERROR\tmissing pinned guard file: %s\n' \
      "${path}" >&2
    exit 67
  }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected_guard_hashes[${name}]}" ]]; then
    printf 'STORAGE_VALIDATION_ERROR\tguard hash mismatch: %s\n' \
      "${path}" >&2
    exit 68
  fi
done

bash -n "${run_script}"
bash -n "${src}/storage_guard/storage_guard.sh"
python3 -m py_compile "${src}/storage_guard/bounded_log.py"
printf 'storage-attempt-validation\tPASS\t%s\n' "${src}"
