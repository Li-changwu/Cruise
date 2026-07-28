#!/usr/bin/env bash
set -euo pipefail

mode=${1:---dry-run}
[[ "${mode}" == --dry-run || "${mode}" == --apply ]] || {
  printf 'usage: %s [--dry-run|--apply]\n' "$0" >&2
  exit 64
}

project=/root/ascend-control-g4-20260723
completion=${project}/storage-control/g4-completion.json
performance_evidence=${project}/evidence-attempt70b-r3-r1-b4-performance
epoch_evidence=${project}/evidence-attempt69e-r5-b4-resident-epoch
weight_base=${project}/export-attempt67b-b2
weight_view=${project}/export-attempt69c-b4
epoch_scratch=/dev/shm/ascend-control-g4-20260726/attempt69e-r5-b4-resident-epoch

verify_hash() {
  local expected=$1 path=$2 actual
  [[ -f "${path}" ]] || {
    printf 'G4_PRUNE_ERROR\tmissing prerequisite\t%s\n' "${path}" >&2
    exit 80
  }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    printf 'G4_PRUNE_ERROR\thash mismatch\t%s\n' "${path}" >&2
    exit 81
  }
}

verify_pass_json() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
if data.get("pass") is not True:
    raise SystemExit(1)
PY
}

has_open_file() {
  find /proc/[0-9]*/fd -lname "$1/*" -print -quit 2>/dev/null | grep -q .
}

has_reference_link() {
  find "${project}" /dev/shm/ascend-control-g4-20260726 -xdev -type l \
    -lname "$1/*" -print -quit 2>/dev/null | grep -q .
}

verify_hash 8804b48c7382df14b8c74326756412224f3226057c7fa03c0379d46905f2d857 \
  "${completion}"
verify_hash a934793ae6a2c91a5935261a648cdd79f73ae96ea57ccea23c31203831fdbd50 \
  "${performance_evidence}/attempt70b-r3-r1-result.json"
verify_hash b086346f88f33e75bad921cbfed78a0bd6d13be31336d9bd53667e6069a1a9ac \
  "${performance_evidence}/evidence-integrity.log"
verify_hash 9a64b4a5488f0c38f5542825971b3ae2f75ed5084238ee16d43240cb52a3656c \
  "${epoch_evidence}/attempt69e-r5-result.json"
verify_hash f343c2c7695ed631f6c7dec4dfa31ca0d267e36a234631b7c79c54465cfd75d0 \
  "${epoch_evidence}/evidence-integrity.log"
verify_pass_json "${completion}"
verify_pass_json "${performance_evidence}/attempt70b-r3-r1-result.json"
verify_pass_json "${epoch_evidence}/attempt69e-r5-result.json"
sha256sum -c "${performance_evidence}/evidence-integrity.log" >/dev/null
sha256sum -c "${epoch_evidence}/evidence-integrity.log" >/dev/null

for target in "${weight_base}" "${weight_view}" "${epoch_scratch}"; do
  resolved=$(realpath -e -- "${target}")
  [[ "${resolved}" == "${target}" ]] || {
    printf 'G4_PRUNE_ERROR\tunexpected resolved path\t%s\t%s\n' \
      "${target}" "${resolved}" >&2
    exit 82
  }
  if has_open_file "${target}"; then
    printf 'G4_PRUNE_ERROR\topen file descriptor\t%s\n' "${target}" >&2
    exit 83
  fi
  if has_reference_link "${target}"; then
    printf 'G4_PRUNE_ERROR\texternal symbolic-link reference\t%s\n' \
      "${target}" >&2
    exit 83
  fi
done

marker_evidence=$(awk -F '\t' '$1 == "evidence_dir" {print $2}' \
  "${epoch_scratch}/.storage-guard-scratch.tsv")
[[ "${marker_evidence}" == "${epoch_evidence}" &&
   -f "${epoch_scratch}/.storage-guard-keep" ]] || {
  printf 'G4_PRUNE_ERROR\tscratch marker mismatch\t%s\n' "${epoch_scratch}" >&2
  exit 84
}

weight_bytes=$(du -scxB1 -- "${weight_base}" "${weight_view}" | \
  awk '$2 == "total" {print $1}')
scratch_bytes=$(du -sxB1 -- "${epoch_scratch}" | awk '{print $1}')
root_free_before=$(df -PB1 / | awk 'NR == 2 {print $4}')

if [[ "${mode}" == --apply ]]; then
  rm -rf --one-file-system -- "${weight_base}" "${weight_view}"
  rm -rf --one-file-system -- "${epoch_scratch}"
  verify_pass_json "${completion}"
  sha256sum -c "${performance_evidence}/evidence-integrity.log" >/dev/null
  sha256sum -c "${epoch_evidence}/evidence-integrity.log" >/dev/null
fi

root_free_after=$(df -PB1 / | awk 'NR == 2 {print $4}')
printf 'mode\t%s\n' "${mode}"
printf 'status\t%s\n' "$([[ "${mode}" == --apply ]] && printf APPLIED || printf DRY_RUN)"
printf 'weight_bytes\t%s\n' "${weight_bytes}"
printf 'scratch_bytes\t%s\n' "${scratch_bytes}"
printf 'root_free_before_bytes\t%s\n' "${root_free_before}"
printf 'root_free_after_bytes\t%s\n' "${root_free_after}"
