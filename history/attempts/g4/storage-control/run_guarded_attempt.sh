#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  printf 'usage: %s ATTEMPT_SOURCE_DIR\n' "$0" >&2
  exit 64
fi

control_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath -e -- "$1")
case "${src}" in
  /root/ascend-control-g4-20260723/attempt*-src) ;;
  *)
    printf 'GUARDED_ATTEMPT_ERROR\tunsafe source directory: %s\n' "${src}" >&2
    exit 65
    ;;
esac

mkdir -p "${control_dir}/state"
"${control_dir}/audit_storage.sh" >"${control_dir}/state/launch-preflight.tsv"
"${control_dir}/validate_attempt_storage.sh" "${src}"
run_script=$(find "${src}" -maxdepth 1 -type f -name 'run_attempt*.sh' -print -quit)
exec bash "${run_script}"
