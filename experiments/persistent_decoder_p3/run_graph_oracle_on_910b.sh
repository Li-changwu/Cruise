#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${script_dir}/../.." && pwd)
scenario=${1:-${CRUISE_P3_SCENARIO:-b1-short}}
physical_npu=${CRUISE_PHYSICAL_NPU:-0}
run_id=${CRUISE_RUN_ID:-persistent-decoder-p3-oracle-${scenario}-$(date -u +%Y%m%dT%H%M%SZ)}
persistent_root=${CRUISE_PERSISTENT_ROOT:-/workspace/cruise-runs}
run_root=${persistent_root}/${run_id}
evidence=${CRUISE_EVIDENCE_DIR:-${run_root}/evidence}
scratch=${CRUISE_SCRATCH_DIR:-/dev/shm/cruise-p3-oracle-${physical_npu}-$$}
asset_root=${CRUISE_P3_ASSET_ROOT:-/workspace/cruise-assets}
air=${CRUISE_P3_AIR:-${asset_root}/p3-air384-fia-37bd7557850a72b4/qwen_b4_p3_decoder_step.runtime.air}
external_weights=${CRUISE_P3_EXTERNAL_WEIGHT_DIR:-${asset_root}/runtime-weights/p3-ge-external-420a16406d4f8723}
conda_sh=${CRUISE_CONDA_SH:-/home/changwu/miniconda3/etc/profile.d/conda.sh}
conda_env=${CRUISE_CONDA_ENV:-}
cann_set_env=${CRUISE_CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
guard=${source_dir}/storage_guard/storage_guard.sh
lifecycle_tool=${CRUISE_STORAGE_TOOL:-/workspace/Cruise/scripts/manage_workspace_storage.py}
oracle_source=${script_dir}/p3_graph_oracle.cpp
external_view_tool=${script_dir}/manage_ge_external_view.py
external_view=${run_root}/.ge-external-view
external_capture=${CRUISE_P3_EXTERNAL_CAPTURE_DIR:-}
reuse_external_capture=${CRUISE_P3_REUSE_EXTERNAL_CAPTURE:-0}
runtime_external_weights=${external_view}

for required in "${cann_set_env}" "${guard}" \
  "${lifecycle_tool}" "${oracle_source}" "${external_view_tool}" "${air}" \
  "${cann_python_env}/bin/python" \
  "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json"; do
  [[ -f "${required}" ]] || {
    printf 'missing required input: %s\n' "${required}" >&2
    exit 96
  }
done

python3 - "${external_weights}" <<'PY'
import json
import stat
import sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve(strict=True)
meta_path = bundle / "meta.json"
manifest_path = bundle / "dedup-manifest.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not manifest.get("valid"):
    raise SystemExit("external-weight bundle manifest is not valid")

for path in (meta_path, manifest_path):
    if not stat.S_IMODE(path.stat().st_mode) & stat.S_IROTH:
        raise SystemExit(f"external-weight runtime file is not world-readable: {path}")
for parent in (bundle, *bundle.parents):
    if not stat.S_IMODE(parent.stat().st_mode) & stat.S_IXOTH:
        raise SystemExit(f"external-weight runtime path is not traversable: {parent}")

mapping = meta.get("hash_to_weight_file")
if not isinstance(mapping, dict) or len(mapping) != manifest.get("logical_file_count"):
    raise SystemExit("external-weight meta and manifest counts disagree")
for raw_path in mapping.values():
    path = Path(raw_path).resolve(strict=True)
    if path.parent != bundle or not path.is_file():
        raise SystemExit(f"external-weight meta path escapes bundle: {path}")
    if not stat.S_IMODE(path.stat().st_mode) & stat.S_IROTH:
        raise SystemExit(f"external-weight runtime file is not world-readable: {path}")
PY
[[ "${physical_npu}" =~ ^[0-9]+$ ]]
[[ "${scratch}" == /dev/shm/* ]]
[[ "${external_weights}" == "${asset_root}"/* ]]
if [[ -n "${external_capture}" ]]; then
  [[ "$(dirname -- "${external_capture}")" == "${asset_root}" ]]
  [[ "$(basename -- "${external_capture}")" == .p3-ge-external-capture-* ]]
  if [[ "${reuse_external_capture}" == 1 ]]; then
    [[ -f "${external_capture}/meta.json" ]]
    [[ -f "${external_capture}/.cruise-ge-external-capture.json" ]]
  else
    [[ ! -e "${external_capture}" && ! -L "${external_capture}" ]]
  fi
  runtime_external_weights=${external_capture}
fi

source "${guard}"
export STORAGE_GUARD_MAX_SCRATCH_GIB=2
export STORAGE_GUARD_MAX_EVIDENCE_BYTES=$((64 * 1024 * 1024))
export STORAGE_GUARD_NPU_WAIT_SECONDS=${CRUISE_P3_NPU_WAIT_SECONDS:-60}
export STORAGE_GUARD_NPU_STABLE_SAMPLES=1
export STORAGE_GUARD_MAX_IDLE_HBM_PERCENT=${STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65}
export STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=2
python3 "${lifecycle_tool}" audit --runs-root "${persistent_root}" \
  --assets-root "${asset_root}" --max-runs-gib 20 --max-run-gib 2 \
  --summary-only
python3 "${lifecycle_tool}" shm-audit --summary-only
storage_guard_preflight "${persistent_root}" "${evidence}" "${scratch}" \
  "${physical_npu}" 4 100 2
python3 "${lifecycle_tool}" mark --runs-root "${persistent_root}" \
  --run-dir "${run_root}" --retention-class evidence --retention-days 30 \
  --max-run-gib 2
sha256sum "${oracle_source}" "${external_view_tool}" "${air}" \
  "${external_weights}/meta.json" \
  "${external_weights}/dedup-manifest.json" \
  >"${evidence}/source-identity.sha256"

status=${evidence}/status.tsv
build=${scratch}/build
cache=${scratch}/cache
driver_logs=${scratch}/driver-logs
tmp=${scratch}/tmp
mkdir -p "${build}" "${cache}" "${driver_logs}" "${tmp}"
for path in "${build}" "${cache}" "${driver_logs}" "${tmp}"; do
  storage_guard_assert_scratch_path "${path}"
done

extract_failure_logs() {
  local source_file relative target
  [[ -d "${driver_logs}" ]] || return 0
  mkdir -p "${evidence}/failure-driver-logs"
  while IFS= read -r -d '' source_file; do
    relative=${source_file#"${driver_logs}"/}
    target=${evidence}/failure-driver-logs/${relative}
    mkdir -p "$(dirname -- "${target}")"
    tail -c $((512 * 1024)) -- "${source_file}" >"${target}"
  done < <(find "${driver_logs}" -type f -print0 | sort -z | head -z -n 32)
}

finalize() {
  local command_status=$? view_status=0 finalize_status=0 cleanup_status=0 lifecycle_status=0
  local retention_class=evidence retention_days=30
  trap - EXIT
  set +e
  printf 'driver-exit\t%s\n' "${command_status}" >>"${status}"
  if [[ -f "${scratch}/summary.json" && ! -f "${evidence}/summary.json" ]]; then
    cp --reflink=auto "${scratch}/summary.json" "${evidence}/summary.json"
  fi
  if [[ -n "${external_capture}" && -f "${external_capture}/meta.json" && \
        ${command_status} -ne 0 ]]; then
    cp --reflink=auto "${external_capture}/meta.json" \
      "${evidence}/failed-capture-meta.json"
  fi
  if [[ ${command_status} -ne 0 ]]; then
    extract_failure_logs
  fi
  if [[ -n "${external_capture}" && -d "${external_capture}" && \
        ${command_status} -ne 0 && "${reuse_external_capture}" != 1 ]]; then
    python3 "${external_view_tool}" capture-cleanup \
      --assets-root "${asset_root}" --run-root "${run_root}" \
      --capture "${external_capture}" \
      --receipt "${evidence}/ge-external-capture-cleanup.json"
    view_status=$?
  elif [[ -d "${external_view}" ]]; then
    python3 "${external_view_tool}" cleanup --run-root "${run_root}" \
      --view "${external_view}" \
      --receipt "${evidence}/ge-external-view-cleanup.json"
    view_status=$?
  fi
  storage_guard_finalize
  finalize_status=$?
  if [[ ${finalize_status} -eq 0 ]]; then
    storage_guard_cleanup_scratch
    cleanup_status=$?
  fi
  if [[ ${finalize_status} -eq 0 && ${cleanup_status} -eq 0 ]]; then
    if [[ ${command_status} -ne 0 ]]; then
      retention_class=diagnostic
      retention_days=7
    fi
    python3 "${lifecycle_tool}" finalize --runs-root "${persistent_root}" \
      --run-dir "${run_root}" --retention-class "${retention_class}" \
      --retention-days "${retention_days}"
    lifecycle_status=$?
  fi
  if [[ ${command_status} -ne 0 ]]; then exit "${command_status}"; fi
  if [[ ${view_status} -ne 0 ]]; then exit "${view_status}"; fi
  if [[ ${finalize_status} -ne 0 ]]; then exit "${finalize_status}"; fi
  if [[ ${cleanup_status} -ne 0 ]]; then exit "${cleanup_status}"; fi
  exit "${lifecycle_status}"
}
trap finalize EXIT

if [[ -n "${conda_env}" ]]; then
  [[ -f "${conda_sh}" ]] || {
    printf 'missing Conda activation script: %s\n' "${conda_sh}" >&2
    exit 96
  }
  source "${conda_sh}"
  conda activate "${conda_env}"
fi
source "${cann_set_env}"
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
export ASCEND_RT_VISIBLE_DEVICES=${physical_npu}
export ASCEND_GLOBAL_LOG_LEVEL=${CRUISE_ASCEND_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export PYTHONDONTWRITEBYTECODE=1

oracle=${build}/p3_graph_oracle
g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++11 \
  -fstack-protector-all -pthread \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/external" \
  "${oracle_source}" \
  -Wl,--whole-archive \
  "${ASCEND_HOME_PATH}/lib64/libgraph.so" \
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so" \
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so" \
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so" \
  -Wl,--no-whole-archive -o "${oracle}" \
  >"${scratch}/oracle-compile.log" 2>&1
storage_guard_runtime_budget_ok 0

summary=${scratch}/summary.json
if [[ -n "${external_capture}" ]]; then
  if [[ "${reuse_external_capture}" != 1 ]]; then
    python3 "${external_view_tool}" capture-create --source "${external_weights}" \
      --assets-root "${asset_root}" --capture "${external_capture}"
  fi
else
  python3 "${external_view_tool}" create --source "${external_weights}" \
    --run-root "${run_root}" --view "${external_view}"
fi
storage_guard_run_log "${evidence}/oracle.stdout.log" \
  "${evidence}/oracle.stdout.meta.json" \
  "${CRUISE_P3_TIMEOUT:-1800}s" -- env \
  ASCEND_PROCESS_LOG_PATH="${driver_logs}" \
  ASCEND_CACHE_PATH="${cache}" \
  TMPDIR="${tmp}" \
  CRUISE_P3_ASSET_ROOT="${asset_root}" \
  CRUISE_P3_RUN_ROOT="${run_root}" \
  CRUISE_P3_EXTERNAL_WEIGHT_DIR="${runtime_external_weights}" \
  "${oracle}" "${air}" "${scenario}" "${summary}"
cp --reflink=auto "${summary}" "${evidence}/summary.json"
if [[ -n "${external_capture}" ]]; then
  cp --reflink=auto "${external_capture}/meta.json" \
    "${evidence}/capture-meta.json"
  cp --reflink=auto "${external_capture}/.cruise-ge-external-capture.json" \
    "${evidence}/capture-marker.json"
  printf 'external-capture-preserved\t%s\n' "${external_capture}" >>"${status}"
fi
storage_guard_snapshot after-oracle "${evidence}/storage-after-oracle.tsv"
find "${driver_logs}" -type f -size +16M -delete
printf 'P3_GRAPH_ORACLE_COMPLETE scenario=%s evidence=%s\n' \
  "${scenario}" "${evidence}"
