#!/usr/bin/env bash
set -euo pipefail

mode=${1:---dry-run}
[[ "${mode}" == --dry-run || "${mode}" == --apply ]] || {
  printf 'usage: %s [--dry-run|--apply]\n' "$0" >&2
  exit 64
}

project=/root/ascend-control-g4-20260723
log_root=/root/ascend/log
reference=${project}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
acceptance=${project}/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json
air=${project}/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air
result=${project}/evidence-attempt69e-r5-b4-resident-epoch/attempt69e-r5-result.json

declare -A expected=(
  ["${reference}"]=a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8
  ["${acceptance}"]=72e73a176097d9368171b866a384bf70c33616adcdd4cfccc53a015db0681605
  ["${air}"]=263b2acf291e13f6a84042ded53c8dccabb1fa847dcdcbbbe0ece418610ad1e3
  ["${result}"]=9a64b4a5488f0c38f5542825971b3ae2f75ed5084238ee16d43240cb52a3656c
)

verify_dependencies() {
  local path actual
  for path in "${!expected[@]}"; do
    [[ -f "${path}" ]] || {
      printf 'PRUNE_ERROR\tmissing dependency\t%s\n' "${path}" >&2
      return 90
    }
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected[${path}]}" ]] || {
      printf 'PRUNE_ERROR\thash mismatch\t%s\n' "${path}" >&2
      return 90
    }
  done
}

if find /proc/[0-9]*/fd \
    \( -lname "${project}/raw-*/*" -o -lname "${log_root}/*" \) \
    -print -quit 2>/dev/null | grep -q .; then
  printf 'PRUNE_ERROR\ta candidate tree has an open file descriptor\n' >&2
  exit 91
fi

verify_dependencies
raw_count=0
raw_bytes=0
while IFS= read -r -d '' size && IFS= read -r -d '' path; do
  case "${path}" in
    "${project}"/raw-*/*) ;;
    *) continue ;;
  esac
  [[ "${path}" == "${reference}" ]] && continue
  raw_count=$((raw_count + 1))
  raw_bytes=$((raw_bytes + size))
  [[ "${mode}" == --dry-run ]] || rm -- "${path}"
done < <(find "${project}" -xdev -type f -size +33554432c \
  -printf '%s\0%p\0')

log_count=0
log_bytes=0
while IFS= read -r -d '' size && IFS= read -r -d '' path; do
  case "${path}" in
    "${log_root}"/*) ;;
    *) printf 'PRUNE_ERROR\tunsafe log candidate\t%s\n' "${path}" >&2; exit 92 ;;
  esac
  log_count=$((log_count + 1))
  log_bytes=$((log_bytes + size))
  [[ "${mode}" == --dry-run ]] || rm -- "${path}"
done < <(find "${log_root}" -xdev -type f -mtime +3 -printf '%s\0%p\0')

if [[ "${mode}" == --apply ]]; then
  find "${log_root}" -xdev -depth -mindepth 1 -type d -empty -delete
  verify_dependencies
fi

printf 'mode\t%s\n' "${mode}"
printf 'raw_file_count\t%s\n' "${raw_count}"
printf 'raw_file_bytes\t%s\n' "${raw_bytes}"
printf 'old_log_count\t%s\n' "${log_count}"
printf 'old_log_bytes\t%s\n' "${log_bytes}"
