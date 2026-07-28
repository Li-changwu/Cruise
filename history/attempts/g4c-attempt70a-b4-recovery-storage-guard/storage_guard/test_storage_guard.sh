#!/usr/bin/env bash
set -uo pipefail

guard_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${guard_dir}/storage_guard.sh"

test_root=/dev/shm/storage-guard-selftest-$$
persistent=${test_root}/persistent
scratch_parent=${test_root}/scratch
fake_npu=${test_root}/fake-npu-smi
fake_busy_npu=${test_root}/fake-busy-npu-smi
fake_delayed_npu=${test_root}/fake-delayed-npu-smi
fake_npu_counter=${test_root}/fake-npu-counter
cleanup() {
  rm -rf -- "${test_root}"
}
trap cleanup EXIT
mkdir -p "${persistent}" "${scratch_parent}"
cat >"${fake_npu}" <<'EOF'
#!/usr/bin/env bash
printf '\tNo process in device.\n'
EOF
chmod 700 "${fake_npu}"
cat >"${fake_busy_npu}" <<'EOF'
#!/usr/bin/env bash
printf 'Process id:123 Process name:test Process memory(MB):1\n'
EOF
chmod 700 "${fake_busy_npu}"
cat >"${fake_delayed_npu}" <<'EOF'
#!/usr/bin/env bash
count=0
[[ -f "${FAKE_NPU_COUNTER}" ]] && count=$(<"${FAKE_NPU_COUNTER}")
count=$((count + 1))
printf '%s\n' "${count}" >"${FAKE_NPU_COUNTER}"
if (( count < 3 )); then
  printf 'Process id:123 Process name:test Process memory(MB):1\n'
else
  printf '\tNo process in device.\n'
fi
EOF
chmod 700 "${fake_delayed_npu}"
export FAKE_NPU_COUNTER=${fake_npu_counter}
export STORAGE_GUARD_NPU_SMI=${fake_npu}
export STORAGE_GUARD_LARGE_ALLOWLIST=allowed-large
export STORAGE_GUARD_WATCH_INTERVAL_SECONDS=0.1
export STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=2

printf 'small-log\n' |
  python3 "${guard_dir}/bounded_log.py" \
    --output "${test_root}/small.log" \
    --metadata "${test_root}/small.json" --head-bytes 16 --tail-bytes 16
cmp -s "${test_root}/small.log" <(printf 'small-log\n') || exit 11
grep -q '"truncated": false' "${test_root}/small.json" || exit 12

head -c 4096 /dev/zero |
  python3 "${guard_dir}/bounded_log.py" \
    --output "${test_root}/large.log" \
    --metadata "${test_root}/large.json" --head-bytes 128 --tail-bytes 64
grep -q '"truncated": true' "${test_root}/large.json" || exit 13
[[ $(stat -c %s "${test_root}/large.log") -lt 512 ]] || exit 14

if storage_guard_require_child /root/escape "${persistent}"; then exit 15; fi

if storage_guard_preflight "${persistent}" "${persistent}/too-low" \
  "${scratch_parent}/too-low" 99 999999 1 0; then
  exit 16
elif [[ $? -ne 91 ]]; then
  exit 17
fi

mkdir "${persistent}/already-exists"
if storage_guard_preflight "${persistent}" "${persistent}/already-exists" \
  "${scratch_parent}/existing-test" 99 0 1 0; then
  exit 18
elif [[ $? -ne 97 ]]; then
  exit 19
fi

export STORAGE_GUARD_NPU_SMI=${fake_busy_npu}
if storage_guard_preflight "${persistent}" "${persistent}/busy-npu" \
  "${scratch_parent}/busy-npu" 99 0 1 0; then
  exit 191
elif [[ $? -ne 95 ]]; then
  exit 192
fi
[[ ! -e "${persistent}/busy-npu" && ! -e "${scratch_parent}/busy-npu" ]] || exit 193

export STORAGE_GUARD_NPU_SMI=${fake_delayed_npu}
export STORAGE_GUARD_NPU_WAIT_SECONDS=5
export STORAGE_GUARD_NPU_POLL_SECONDS=0.1
storage_guard_preflight "${persistent}" "${persistent}/delayed-npu" \
  "${scratch_parent}/delayed-npu" 98 0 1 0 || exit 194
[[ $(<"${fake_npu_counter}") -ge 3 ]] || exit 195
[[ -d "${persistent}/delayed-npu" && -d "${scratch_parent}/delayed-npu" ]] || exit 196

export STORAGE_GUARD_NPU_SMI=${fake_npu}
export STORAGE_GUARD_NPU_WAIT_SECONDS=0

evidence=${persistent}/valid-evidence
scratch=${scratch_parent}/valid-scratch
storage_guard_preflight "${persistent}" "${evidence}" "${scratch}" 99 0 1 0 || exit 20
storage_guard_assert_scratch_path "${scratch}/outputs" || exit 21
if storage_guard_assert_scratch_path "${persistent}/bad-output"; then exit 22; fi

mkdir "${persistent}/unexpected-growth"
dd if=/dev/zero of="${persistent}/unexpected-growth/payload" \
  bs=1M count=65 status=none
if storage_guard_persistent_growth_ok; then exit 220; fi
rm -rf -- "${persistent}/unexpected-growth"

set +e
storage_guard_run_log "${evidence}/exit7.log" "${evidence}/exit7.json" 10s -- \
  bash -c 'head -c 1048576 /dev/zero; exit 7'
run_status=$?
set -e
[[ ${run_status} -eq 7 ]] || exit 23
grep -q '"truncated": false' "${evidence}/exit7.json" || exit 24

saved_scratch_limit=${STORAGE_GUARD_MAX_SCRATCH_BYTES}
STORAGE_GUARD_MAX_SCRATCH_BYTES=1
set +e
storage_guard_run_log "${evidence}/watchdog.log" \
  "${evidence}/watchdog.json" 30s -- bash -c 'sleep 30'
watchdog_status=$?
set -e
[[ ${watchdog_status} -eq 92 ]] || exit 240
grep -q 'runtime scratch size' "${evidence}/watchdog.log" || exit 241
STORAGE_GUARD_MAX_SCRATCH_BYTES=${saved_scratch_limit}

storage_guard_run_log "${evidence}/bounded-80m.log" \
  "${evidence}/bounded-80m.json" 30s -- \
  bash -c 'head -c 83886080 /dev/zero' || exit 242
grep -q '"truncated": true' "${evidence}/bounded-80m.json" || exit 243
[[ $(stat -c %s "${evidence}/bounded-80m.log") -lt 68157440 ]] || exit 244

storage_guard_finalize || exit 25
[[ -s "${evidence}/evidence-integrity.log" ]] || exit 26
storage_guard_cleanup_scratch || exit 27
[[ ! -e "${scratch}" ]] || exit 28
printf 'storage-guard-selftest\tPASS\n'
