#!/usr/bin/env bash
set -euo pipefail

mode=${1:---dry-run}
[[ "${mode}" == --dry-run || "${mode}" == --apply ]] || {
  printf 'usage: %s [--dry-run|--apply]\n' "$0" >&2
  exit 64
}

shm_parent=${G4_SHM_PARENT:-/dev/shm}
project_root=${G4_PROJECT_ROOT:-/root/ascend-control-g4-20260723}
grace_minutes=${SCRATCH_REAP_GRACE_MINUTES:-10}
[[ "${grace_minutes}" =~ ^[0-9]+$ ]] || exit 65
shm_parent=$(realpath -e -- "${shm_parent}")
project_root=$(realpath -e -- "${project_root}")

candidate_count=0
eligible_count=0
eligible_bytes=0
deleted_count=0
deleted_bytes=0
skipped_count=0

has_open_file() {
  find /proc/[0-9]*/fd -lname "$1/*" -print -quit 2>/dev/null | grep -q .
}

status_is_clean() {
  awk -F '\t' '
    NR == 1 {next}
    NF >= 2 {seen = 1; if ($2 != 0) bad = 1}
    END {exit !(seen && !bad)}
  ' "$1"
}

while IFS= read -r -d '' finalized; do
  scratch=${finalized%/.storage-guard-finalized.tsv}
  candidate_count=$((candidate_count + 1))
  case "${scratch}" in
    "${shm_parent}"/ascend-control-g4-20*/attempt*) ;;
    *) printf 'SKIP\tunsafe-path\t%s\n' "${scratch}"; skipped_count=$((skipped_count + 1)); continue ;;
  esac
  if [[ -e "${scratch}/.storage-guard-keep" ||
        ! -f "${scratch}/.storage-guard-scratch.tsv" ||
        ! -f "${scratch}/.storage-guard-finalized.tsv" ]]; then
    printf 'SKIP\tmissing-marker-or-pinned\t%s\n' "${scratch}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if ! find "${finalized}" -mmin "+${grace_minutes}" -print -quit | grep -q .; then
    printf 'SKIP\tgrace-period\t%s\n' "${scratch}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  evidence_marker=$(awk -F '\t' '$1 == "evidence_dir" {print $2}' \
    "${scratch}/.storage-guard-finalized.tsv")
  scratch_evidence=$(awk -F '\t' '$1 == "evidence_dir" {print $2}' \
    "${scratch}/.storage-guard-scratch.tsv")
  if [[ "${evidence_marker}" != "${scratch_evidence}" ]] ||
     ! evidence=$(realpath -e -- "${evidence_marker}"); then
    printf 'SKIP\tevidence-marker-mismatch\t%s\n' "${scratch}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  case "${evidence}" in
    "${project_root}"/evidence-attempt*) ;;
    *) printf 'SKIP\tunsafe-evidence\t%s\n' "${scratch}"; skipped_count=$((skipped_count + 1)); continue ;;
  esac
  if [[ ! -f "${evidence}/evidence-integrity.log" ||
        ! -f "${evidence}/status.tsv" ]] ||
     ! sha256sum -c "${evidence}/evidence-integrity.log" >/dev/null 2>&1 ||
     ! status_is_clean "${evidence}/status.tsv" ||
     ! grep -RIl --include='*result.json' '"pass"[[:space:]]*:[[:space:]]*true' \
          "${evidence}" >/dev/null 2>&1 ||
     has_open_file "${scratch}"; then
    printf 'SKIP\tunverified-or-open\t%s\n' "${scratch}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  bytes=$(du -sxB1 -- "${scratch}" | awk '{print $1}')
  eligible_count=$((eligible_count + 1))
  eligible_bytes=$((eligible_bytes + bytes))
  printf 'ELIGIBLE\t%s\t%s\n' "${bytes}" "${scratch}"
  if [[ "${mode}" == --apply ]]; then
    rm -rf --one-file-system -- "${scratch}"
    deleted_count=$((deleted_count + 1))
    deleted_bytes=$((deleted_bytes + bytes))
  fi
done < <(find "${shm_parent}" -xdev -mindepth 3 -maxdepth 3 -type f \
  -path "${shm_parent}/ascend-control-g4-20*/attempt*/.storage-guard-finalized.tsv" \
  -print0)

printf 'mode\t%s\n' "${mode}"
printf 'candidate_count\t%s\n' "${candidate_count}"
printf 'eligible_count\t%s\n' "${eligible_count}"
printf 'eligible_bytes\t%s\n' "${eligible_bytes}"
printf 'deleted_count\t%s\n' "${deleted_count}"
printf 'deleted_bytes\t%s\n' "${deleted_bytes}"
printf 'skipped_count\t%s\n' "${skipped_count}"
