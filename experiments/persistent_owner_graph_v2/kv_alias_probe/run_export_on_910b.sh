#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../../.." && pwd)
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-owner-v2-kv-alias-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${run_root}/evidence
scratch=${CRUISE_V2_KV_ALIAS_SCRATCH:-/dev/shm/cruise-v2-kv-alias-${physical_npu}-$$}
python_bin=${CRUISE_V2_EXPORT_PYTHON:-$(command -v python3)}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
exporter=${script_dir}/export_kv_alias.py
inspector=${script_dir}/inspect_kv_alias.py

for required in "${python_bin}" "${cann_set_env}" "${guard}" \
  "${lifecycle_tool}" "${exporter}" "${inspector}"; do
  [[ -f "${required}" ]] || {
    printf 'missing V2 KV alias input: %s\n' "${required}" >&2
    exit 96
  }
done
[[ "${scratch}" == /dev/shm/cruise-v2-kv-alias-* ]] || {
  printf 'V2 KV alias scratch must be PID-scoped under /dev/shm: %s\n' \
    "${scratch}" >&2
  exit 96
}

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=1
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((128 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=60
export STORAGE_GUARD_NPU_STABLE_SAMPLES=3
# A small driver/runtime reservation is normal on this host. V2 rejects live
# processes and unstable/growing HBM instead of using the historical 5% line.
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
  --assets-root /workspace/cruise-assets --summary-only
python3 "${lifecycle_tool}" shm-audit --summary-only
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 1
python3 "${lifecycle_tool}" mark --runs-root "${persistent_root}" \
  --run-dir "${run_root}" --retention-class diagnostic --retention-days 7 \
  --max-run-gib 1

export_dir=${scratch}/export
driver_logs=${scratch}/driver-logs
cache=${scratch}/cache
tmp=${scratch}/tmp
mkdir -p "${driver_logs}" "${cache}" "${tmp}"

finalize() {
  local command_status=$? finalize_status=0 cleanup_status=0 lifecycle_status=0
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >"${evidence}/status.tsv"
  if [[ -d "${driver_logs}" && ${command_status} -ne 0 ]]; then
    mkdir -p "${evidence}/failure-driver-logs"
    while IFS= read -r log; do
      tail -c $((512 * 1024)) -- "${log}" \
        >"${evidence}/failure-driver-logs/$(basename -- "${log}")"
    done < <(find "${driver_logs}" -type f -print | sort | head -n 24)
  fi
  storage_guard_finalize
  finalize_status=$?
  if [[ ${finalize_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${finalize_status} -eq 0 && ${cleanup_status} -eq 0 ]]; then
    python3 "${lifecycle_tool}" finalize --runs-root "${persistent_root}" \
      --run-dir "${run_root}" --retention-class diagnostic --retention-days 7
    lifecycle_status=$?
  fi
  if [[ ${command_status} -ne 0 ]]; then exit "${command_status}"; fi
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

source "${cann_set_env}"
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_EXPORT_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_PROCESS_LOG_PATH=${driver_logs}
export ASCEND_CACHE_PATH=${cache}
export TMPDIR=${tmp}
export PYTHONDONTWRITEBYTECODE=1

sha256sum "${exporter}" "${inspector}" "${script_dir}/run_export_on_910b.sh" \
  >"${evidence}/source-identity.sha256"
git -C "${source_dir}" status --porcelain=v1 \
  >"${evidence}/source-worktree-status.txt"
git -C "${source_dir}" rev-parse HEAD >"${evidence}/source-commit.txt"
"${python_bin}" - <<'PY' >"${evidence}/package-identity.txt"
import importlib.metadata
for name in ("torch", "torch-npu"):
    print(f"{name}=={importlib.metadata.version(name)}")
PY

"${python_bin}" "${exporter}" --output-dir "${export_dir}"
cp "${export_dir}/export-result.json" "${evidence}/"
cp "${export_dir}/graph-structure.json" "${evidence}/"
cp "${export_dir}/dynamo.pbtxt" "${evidence}/"
printf 'run_root\t%s\n' "${run_root}"
