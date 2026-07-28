#!/usr/bin/env bash

STORAGE_GUARD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STORAGE_GUARD_BOUNDED_LOG=${STORAGE_GUARD_DIR}/bounded_log.py
STORAGE_GUARD_LOCK_FD=
STORAGE_GUARD_PERSISTENT_ROOT=
STORAGE_GUARD_EVIDENCE_DIR=
STORAGE_GUARD_SCRATCH_DIR=
STORAGE_GUARD_MAX_PROJECT_GIB=
STORAGE_GUARD_MIN_ROOT_FREE_BYTES=
STORAGE_GUARD_MIN_SHM_FREE_BYTES=
STORAGE_GUARD_MAX_EVIDENCE_BYTES=${STORAGE_GUARD_MAX_EVIDENCE_BYTES:-$((512 * 1024 * 1024))}
STORAGE_GUARD_MAX_SCRATCH_BYTES=
STORAGE_GUARD_PERSISTENT_BASELINE=
STORAGE_GUARD_MAX_UNEXPECTED_GROWTH_BYTES=${STORAGE_GUARD_MAX_UNEXPECTED_GROWTH_BYTES:-$((64 * 1024 * 1024))}
STORAGE_GUARD_WATCH_INTERVAL_SECONDS=${STORAGE_GUARD_WATCH_INTERVAL_SECONDS:-1}
STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=${STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS:-30}

storage_guard_error() {
  printf 'STORAGE_GUARD_ERROR\t%s\n' "$*" >&2
}

storage_guard_realpath() {
  realpath -m -- "$1"
}

storage_guard_require_child() {
  local path root path_real root_real
  path=$1
  root=$2
  path_real=$(storage_guard_realpath "${path}") || return 90
  root_real=$(storage_guard_realpath "${root}") || return 90
  if [[ "${path_real}" != "${root_real}/"* ]]; then
    storage_guard_error "path escapes allowed root: ${path_real} not under ${root_real}"
    return 90
  fi
}

storage_guard_available_bytes() {
  df -PB1 -- "$1" | awk 'NR == 2 {print $4}'
}

storage_guard_used_bytes() {
  du -sxB1 -- "$1" | awk '{print $1}'
}

storage_guard_npu_state() {
  local physical_npu=$1
  local npu_tool=${STORAGE_GUARD_NPU_SMI:-npu-smi}
  "${npu_tool}" info -t proc-mem -i "${physical_npu}"
}

storage_guard_npu_usage_state() {
  local physical_npu=$1
  local npu_tool=${STORAGE_GUARD_NPU_SMI:-npu-smi}
  "${npu_tool}" info -t usages -i "${physical_npu}"
}

storage_guard_parse_hbm_usage_percent() {
  awk -F ':' '/HBM Usage Rate\(%\)/ {
    gsub(/[[:space:]]/, "", $2)
    print $2
    exit
  }'
}

storage_guard_require_npu_idle() {
  local physical_npu=$1 state
  if ! state=$(storage_guard_npu_state "${physical_npu}" 2>&1); then
    storage_guard_error "npu-smi failed for physical NPU ${physical_npu}"
    printf '%s\n' "${state}" >&2
    return 95
  fi
  if ! grep -Fq 'No process in device.' <<<"${state}"; then
    storage_guard_error "physical NPU ${physical_npu} is not idle"
    printf '%s\n' "${state}" >&2
    return 95
  fi
}

storage_guard_require_npu_ready() {
  local physical_npu=$1 state usage hbm
  local max_hbm=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-5}
  state=$(storage_guard_npu_state "${physical_npu}" 2>&1) || return 95
  usage=$(storage_guard_npu_usage_state "${physical_npu}" 2>&1) || return 95
  hbm=$(storage_guard_parse_hbm_usage_percent <<<"${usage}")
  if ! grep -Fq 'No process in device.' <<<"${state}" ||
     [[ ! ${hbm} =~ ^[0-9]+$ ]] || (( hbm > max_hbm )); then
    storage_guard_error \
      "physical NPU ${physical_npu} is not ready (HBM usage=${hbm:-unknown}%, max=${max_hbm}%)"
    printf '%s\n%s\n' "${state}" "${usage}" >&2
    return 95
  fi
  STORAGE_GUARD_WAITED_NPU_STATE=${state}
  STORAGE_GUARD_WAITED_NPU_USAGE=${usage}
}

storage_guard_wait_for_npu_idle() {
  if [[ $# -ne 6 ]]; then
    storage_guard_error \
      'wait_for_npu_idle requires: physical_npu wait_seconds persistent_root max_project_gib min_root_bytes min_shm_bytes'
    return 90
  fi
  local physical_npu=$1 wait_seconds=$2 persistent_root=$3 max_project_gib=$4
  local min_root_bytes=$5 min_shm_bytes=$6
  local start now last_audit root_free shm_free state='' usage='' hbm=''
  local poll_seconds=${STORAGE_GUARD_NPU_POLL_SECONDS:-5}
  local reaudit_seconds=${STORAGE_GUARD_REAUDIT_SECONDS:-300}
  local required_samples=${STORAGE_GUARD_NPU_STABLE_SAMPLES:-3}
  local max_hbm=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-5}
  local ready_samples=0
  if [[ ! ${wait_seconds} =~ ^[0-9]+$ ||
        ! ${reaudit_seconds} =~ ^[1-9][0-9]*$ ||
        ! ${required_samples} =~ ^[1-9][0-9]*$ ||
        ! ${max_hbm} =~ ^[0-9]+$ || ${max_hbm} -gt 100 ]]; then
    storage_guard_error 'NPU wait, stability, HBM, or reaudit setting is invalid'
    return 90
  fi

  start=$(date +%s)
  last_audit=${start}
  while true; do
    state=$(storage_guard_npu_state "${physical_npu}" 2>&1) || true
    usage=$(storage_guard_npu_usage_state "${physical_npu}" 2>&1) || true
    hbm=$(storage_guard_parse_hbm_usage_percent <<<"${usage}")
    if grep -Fq 'No process in device.' <<<"${state}" &&
       [[ ${hbm} =~ ^[0-9]+$ ]] && (( hbm <= max_hbm )); then
      ready_samples=$((ready_samples + 1))
      if (( ready_samples >= required_samples )); then
        STORAGE_GUARD_WAITED_NPU_STATE=${state}
        STORAGE_GUARD_WAITED_NPU_USAGE=${usage}
        return 0
      fi
    else
      ready_samples=0
    fi

    now=$(date +%s)
    if (( now - start >= wait_seconds )); then
      storage_guard_error \
        "physical NPU ${physical_npu} did not become ready within ${wait_seconds} seconds"
      printf '%s\n%s\n' "${state}" "${usage}" >&2
      return 95
    fi
    root_free=$(storage_guard_available_bytes "${persistent_root}") || return 91
    shm_free=$(storage_guard_available_bytes /dev/shm) || return 91
    if (( root_free < min_root_bytes || shm_free < min_shm_bytes )); then
      storage_guard_error 'filesystem reserve was crossed while waiting for NPU'
      return 91
    fi
    if (( now - last_audit >= reaudit_seconds )); then
      storage_guard_audit_persistent "${persistent_root}" "${max_project_gib}" || return $?
      last_audit=${now}
    fi
    sleep "${poll_seconds}" || return 90
  done
}

storage_guard_large_dir_allowed() {
  local candidate=$1 entry
  IFS=':' read -r -a entries <<<"${STORAGE_GUARD_LARGE_ALLOWLIST:-}"
  for entry in "${entries[@]}"; do
    [[ -n "${entry}" && "${candidate}" == "${entry}" ]] && return 0
  done
  return 1
}

storage_guard_audit_persistent() {
  local persistent_root=$1 max_project_gib=$2
  local max_project_bytes project_bytes dir dir_bytes base large_log
  max_project_bytes=$((max_project_gib * 1024 * 1024 * 1024))
  project_bytes=$(storage_guard_used_bytes "${persistent_root}") || return 92
  if (( project_bytes > max_project_bytes )); then
    storage_guard_error \
      "project uses ${project_bytes} bytes; budget is ${max_project_bytes}"
    return 92
  fi

  while IFS= read -r -d '' dir; do
    dir_bytes=$(storage_guard_used_bytes "${dir}") || return 92
    if (( dir_bytes > 2 * 1024 * 1024 * 1024 )); then
      base=${dir##*/}
      if ! storage_guard_large_dir_allowed "${base}"; then
        storage_guard_error \
          "unapproved persistent directory exceeds 2 GiB: ${dir} (${dir_bytes})"
        return 92
      fi
    fi
  done < <(find "${persistent_root}" -mindepth 1 -maxdepth 1 -type d -print0)

  large_log=$(find "${persistent_root}" -type f -name '*.log' \
    -size +134217728c -print -quit)
  if [[ -n "${large_log}" ]]; then
    storage_guard_error "persistent log exceeds 128 MiB: ${large_log}"
    return 92
  fi
}

storage_guard_capture_persistent_baseline() {
  local path bytes
  STORAGE_GUARD_PERSISTENT_BASELINE=${STORAGE_GUARD_SCRATCH_DIR}/.persistent-baseline.tsv
  : >"${STORAGE_GUARD_PERSISTENT_BASELINE}" || return 93
  while IFS= read -r -d '' path; do
    bytes=$(storage_guard_used_bytes "${path}") || return 92
    printf '%s\t%s\n' "${bytes}" "${path}" \
      >>"${STORAGE_GUARD_PERSISTENT_BASELINE}"
  done < <(find "${STORAGE_GUARD_PERSISTENT_ROOT}" -mindepth 1 -maxdepth 1 \
    -print0 | sort -z)
  return 0
}

storage_guard_persistent_growth_ok() {
  local path bytes baseline growth
  local -A initial_bytes=()
  while IFS=$'\t' read -r baseline path; do
    [[ -n "${path}" ]] && initial_bytes["${path}"]=${baseline}
  done <"${STORAGE_GUARD_PERSISTENT_BASELINE}"

  while IFS= read -r -d '' path; do
    [[ "${path}" == "${STORAGE_GUARD_EVIDENCE_DIR}" ]] && continue
    bytes=$(storage_guard_used_bytes "${path}") || return 92
    baseline=${initial_bytes["${path}"]:-0}
    growth=$((bytes - baseline))
    if (( growth > STORAGE_GUARD_MAX_UNEXPECTED_GROWTH_BYTES )); then
      storage_guard_error \
        "unexpected persistent growth ${growth} bytes in ${path}; heavy outputs must use /dev/shm"
      return 92
    fi
  done < <(find "${STORAGE_GUARD_PERSISTENT_ROOT}" -mindepth 1 -maxdepth 1 \
    -print0 | sort -z)
  return 0
}

storage_guard_snapshot() {
  local label=$1 output=$2
  local root_free shm_free project_used scratch_used evidence_used
  root_free=$(storage_guard_available_bytes "${STORAGE_GUARD_PERSISTENT_ROOT}")
  shm_free=$(storage_guard_available_bytes /dev/shm)
  project_used=$(storage_guard_used_bytes "${STORAGE_GUARD_PERSISTENT_ROOT}")
  scratch_used=$(storage_guard_used_bytes "${STORAGE_GUARD_SCRATCH_DIR}")
  evidence_used=$(storage_guard_used_bytes "${STORAGE_GUARD_EVIDENCE_DIR}")
  {
    printf 'metric\tvalue\n'
    printf 'label\t%s\n' "${label}"
    printf 'utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'root_free_bytes\t%s\n' "${root_free}"
    printf 'shm_free_bytes\t%s\n' "${shm_free}"
    printf 'project_used_bytes\t%s\n' "${project_used}"
    printf 'scratch_used_bytes\t%s\n' "${scratch_used}"
    printf 'evidence_used_bytes\t%s\n' "${evidence_used}"
  } >"${output}"
}

storage_guard_preflight() {
  if [[ $# -ne 7 ]]; then
    storage_guard_error \
      'preflight requires: persistent_root evidence_dir scratch_dir physical_npu min_root_free_gib max_project_gib min_shm_free_gib'
    return 90
  fi
  local persistent_root=$1 evidence_dir=$2 scratch_dir=$3 physical_npu=$4
  local min_root_free_gib=$5 max_project_gib=$6 min_shm_free_gib=$7
  local root_free shm_free min_root_bytes min_shm_bytes lock_dir lock_path npu_state
  local max_scratch_gib=${STORAGE_GUARD_MAX_SCRATCH_GIB:-256}
  local npu_wait_seconds=${STORAGE_GUARD_NPU_WAIT_SECONDS:-0}

  [[ -d "${persistent_root}" && -d /dev/shm ]] || {
    storage_guard_error 'persistent root or /dev/shm is unavailable'
    return 90
  }
  storage_guard_require_child "${evidence_dir}" "${persistent_root}" || return $?
  storage_guard_require_child "${scratch_dir}" /dev/shm || return $?
  if [[ ! ${max_scratch_gib} =~ ^[1-9][0-9]*$ ]]; then
    storage_guard_error 'STORAGE_GUARD_MAX_SCRATCH_GIB must be a positive integer'
    return 90
  fi
  if [[ -e "${evidence_dir}" || -e "${scratch_dir}" ]]; then
    storage_guard_error 'evidence or scratch target already exists; overwrite is forbidden'
    return 97
  fi

  min_root_bytes=$((min_root_free_gib * 1024 * 1024 * 1024))
  min_shm_bytes=$((min_shm_free_gib * 1024 * 1024 * 1024))
  root_free=$(storage_guard_available_bytes "${persistent_root}") || return 91
  shm_free=$(storage_guard_available_bytes /dev/shm) || return 91
  if (( root_free < min_root_bytes )); then
    storage_guard_error \
      "root free space ${root_free} is below ${min_root_bytes} bytes"
    return 91
  fi
  if (( shm_free < min_shm_bytes )); then
    storage_guard_error \
      "/dev/shm free space ${shm_free} is below ${min_shm_bytes} bytes"
    return 91
  fi
  storage_guard_audit_persistent "${persistent_root}" "${max_project_gib}" || return $?
  storage_guard_wait_for_npu_idle "${physical_npu}" "${npu_wait_seconds}" \
    "${persistent_root}" "${max_project_gib}" "${min_root_bytes}" \
    "${min_shm_bytes}" || return $?

  lock_dir=/dev/shm/ascend-control-g4-locks
  mkdir -p "${lock_dir}" || return 93
  lock_path=${lock_dir}/physical-npu-${physical_npu}.lock
  exec {STORAGE_GUARD_LOCK_FD}>"${lock_path}" || return 93
  if ! flock -n "${STORAGE_GUARD_LOCK_FD}"; then
    storage_guard_error "another guarded experiment holds ${lock_path}"
    return 95
  fi
  storage_guard_require_npu_ready "${physical_npu}" || return $?
  npu_state=${STORAGE_GUARD_WAITED_NPU_STATE}

  mkdir -p "${evidence_dir}" "${scratch_dir}" || return 93
  STORAGE_GUARD_PERSISTENT_ROOT=$(storage_guard_realpath "${persistent_root}")
  STORAGE_GUARD_EVIDENCE_DIR=$(storage_guard_realpath "${evidence_dir}")
  STORAGE_GUARD_SCRATCH_DIR=$(storage_guard_realpath "${scratch_dir}")
  STORAGE_GUARD_MAX_PROJECT_GIB=${max_project_gib}
  STORAGE_GUARD_MIN_ROOT_FREE_BYTES=${min_root_bytes}
  STORAGE_GUARD_MIN_SHM_FREE_BYTES=${min_shm_bytes}
  STORAGE_GUARD_MAX_SCRATCH_BYTES=$((max_scratch_gib * 1024 * 1024 * 1024))
  printf '%s\n' "${npu_state}" >"${evidence_dir}/npu-processes-preflight.txt"
  printf '%s\n' "${STORAGE_GUARD_WAITED_NPU_USAGE}" \
    >"${evidence_dir}/npu-usages-preflight.txt"
  {
    printf 'format\tstorage-guard-scratch-v1\n'
    printf 'evidence_dir\t%s\n' "${STORAGE_GUARD_EVIDENCE_DIR}"
    printf 'created_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${scratch_dir}/.storage-guard-scratch.tsv"
  storage_guard_capture_persistent_baseline || return $?
  storage_guard_snapshot preflight "${evidence_dir}/storage-preflight.tsv"
}

storage_guard_assert_scratch_path() {
  storage_guard_require_child "$1" "${STORAGE_GUARD_SCRATCH_DIR}"
}

storage_guard_runtime_budget_ok() {
  local check_project=${1:-0}
  local root_free shm_free evidence_bytes scratch_bytes project_bytes
  root_free=$(storage_guard_available_bytes "${STORAGE_GUARD_PERSISTENT_ROOT}") || return 92
  shm_free=$(storage_guard_available_bytes /dev/shm) || return 92
  evidence_bytes=$(storage_guard_used_bytes "${STORAGE_GUARD_EVIDENCE_DIR}") || return 92
  scratch_bytes=$(storage_guard_used_bytes "${STORAGE_GUARD_SCRATCH_DIR}") || return 92
  if (( root_free < STORAGE_GUARD_MIN_ROOT_FREE_BYTES )); then
    storage_guard_error \
      "runtime root free space ${root_free} fell below ${STORAGE_GUARD_MIN_ROOT_FREE_BYTES} bytes"
    return 92
  fi
  if (( shm_free < STORAGE_GUARD_MIN_SHM_FREE_BYTES )); then
    storage_guard_error \
      "runtime /dev/shm free space ${shm_free} fell below ${STORAGE_GUARD_MIN_SHM_FREE_BYTES} bytes"
    return 92
  fi
  if (( evidence_bytes > STORAGE_GUARD_MAX_EVIDENCE_BYTES )); then
    storage_guard_error \
      "runtime evidence size ${evidence_bytes} exceeded ${STORAGE_GUARD_MAX_EVIDENCE_BYTES} bytes"
    return 92
  fi
  if (( scratch_bytes > STORAGE_GUARD_MAX_SCRATCH_BYTES )); then
    storage_guard_error \
      "runtime scratch size ${scratch_bytes} exceeded ${STORAGE_GUARD_MAX_SCRATCH_BYTES} bytes"
    return 92
  fi
  if (( check_project != 0 )); then
    storage_guard_audit_persistent "${STORAGE_GUARD_PERSISTENT_ROOT}" \
      "${STORAGE_GUARD_MAX_PROJECT_GIB}" || return $?
    storage_guard_persistent_growth_ok || return $?
  fi
}

storage_guard_run_monitored() {
  local timeout_value=$1
  shift
  local command_pid command_status=0 budget_status=0 elapsed=0 check_project=0

  setsid timeout --signal=TERM --kill-after=30s "${timeout_value}" "$@" &
  command_pid=$!
  while kill -0 "${command_pid}" 2>/dev/null; do
    sleep "${STORAGE_GUARD_WATCH_INTERVAL_SECONDS}"
    kill -0 "${command_pid}" 2>/dev/null || break
    elapsed=$((elapsed + 1))
    if (( elapsed >= STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS )); then
      check_project=1
      elapsed=0
    else
      check_project=0
    fi
    if storage_guard_runtime_budget_ok "${check_project}"; then
      :
    else
      budget_status=$?
      kill -TERM -- "-${command_pid}" 2>/dev/null || kill -TERM "${command_pid}" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-${command_pid}" 2>/dev/null || true
      break
    fi
  done
  wait "${command_pid}" || command_status=$?
  if (( budget_status != 0 )); then
    return "${budget_status}"
  fi
  return "${command_status}"
}

storage_guard_run_log() {
  if [[ $# -lt 5 ]]; then
    storage_guard_error \
      'run_log requires: output metadata timeout -- command [args...]'
    return 90
  fi
  local output=$1 metadata=$2 timeout_value=$3
  shift 3
  [[ "${1:-}" == '--' ]] || {
    storage_guard_error 'run_log command must follow --'
    return 90
  }
  shift
  [[ ! -e "${output}" && ! -e "${metadata}" ]] || {
    storage_guard_error "bounded log refuses overwrite: ${output}"
    return 97
  }

  storage_guard_run_monitored "${timeout_value}" "$@" 2>&1 |
    python3 "${STORAGE_GUARD_BOUNDED_LOG}" \
      --output "${output}" --metadata "${metadata}" \
      --head-bytes $((32 * 1024 * 1024)) \
      --tail-bytes $((32 * 1024 * 1024))
  local statuses=("${PIPESTATUS[@]}")
  if [[ ${statuses[1]} -ne 0 ]]; then
    storage_guard_error "bounded logger failed with status ${statuses[1]}"
    return 98
  fi
  return "${statuses[0]}"
}

storage_guard_finalize() {
  local evidence_dir=${STORAGE_GUARD_EVIDENCE_DIR}
  local scratch_dir=${STORAGE_GUARD_SCRATCH_DIR}
  local evidence_bytes project_bytes max_project_bytes large_log
  find "${scratch_dir}" -type f -printf '%P\t%s\n' | sort \
    >"${evidence_dir}/scratch-files.tsv"
  storage_guard_snapshot final "${evidence_dir}/storage-final.tsv"

  evidence_bytes=$(storage_guard_used_bytes "${evidence_dir}") || return 92
  if (( evidence_bytes > STORAGE_GUARD_MAX_EVIDENCE_BYTES )); then
    storage_guard_error "evidence directory exceeds 1 GiB: ${evidence_bytes}"
    return 92
  fi
  large_log=$(find "${evidence_dir}" -type f -name '*.log' \
    -size +68157440c -print -quit)
  if [[ -n "${large_log}" ]]; then
    storage_guard_error "bounded evidence log exceeds 65 MiB: ${large_log}"
    return 92
  fi
  max_project_bytes=$((STORAGE_GUARD_MAX_PROJECT_GIB * 1024 * 1024 * 1024))
  project_bytes=$(storage_guard_used_bytes "${STORAGE_GUARD_PERSISTENT_ROOT}")
  if (( project_bytes > max_project_bytes )); then
    storage_guard_error "post-run project budget exceeded: ${project_bytes}"
    return 92
  fi
  find "${evidence_dir}" -type f ! -name evidence-integrity.log -print0 |
    sort -z | xargs -0 sha256sum >"${evidence_dir}/evidence-integrity.log"
  {
    printf 'format\tstorage-guard-finalized-v1\n'
    printf 'evidence_dir\t%s\n' "${evidence_dir}"
    printf 'evidence_integrity_sha256\t%s\n' \
      "$(sha256sum "${evidence_dir}/evidence-integrity.log" | awk '{print $1}')"
  } >"${scratch_dir}/.storage-guard-finalized.tsv"
}

storage_guard_cleanup_scratch() {
  local scratch_real marker_evidence
  scratch_real=$(storage_guard_realpath "${STORAGE_GUARD_SCRATCH_DIR}") || return 90
  storage_guard_require_child "${scratch_real}" /dev/shm || return $?
  [[ -f "${scratch_real}/.storage-guard-scratch.tsv" &&
     -f "${scratch_real}/.storage-guard-finalized.tsv" &&
     ! -e "${scratch_real}/.storage-guard-keep" ]] || {
    storage_guard_error 'scratch cleanup requires guard markers and refuses pinned trees'
    return 99
  }
  marker_evidence=$(awk -F '\t' '$1 == "evidence_dir" {print $2}' \
    "${scratch_real}/.storage-guard-finalized.tsv")
  [[ "${marker_evidence}" == "${STORAGE_GUARD_EVIDENCE_DIR}" ]] || {
    storage_guard_error 'scratch finalization marker does not match current evidence'
    return 99
  }
  sha256sum -c "${STORAGE_GUARD_EVIDENCE_DIR}/evidence-integrity.log" \
    >/dev/null || return 99
  rm -rf --one-file-system -- "${scratch_real}"
}
