#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=
work_root=
install_root=
evidence=
cann_home=${CRUISE_CANN_HOME:-/usr/local/Ascend/cann-9.0.0}
cann_python_env=${CRUISE_CANN_PYTHON_ENV:-/workspace/cruise-assets/python-envs/cann9-py311}
expected_commit=${CRUISE_FIA_OPS_COMMIT:-afe72144f9f2ac8441929035795db88a111b30c5}
patch_file=${script_dir}/preserve_op_kernel_hierarchy.patch
auditor=${script_dir}/audit_isolated_opp.py

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_root=$2; shift 2 ;;
    --work-root) work_root=$2; shift 2 ;;
    --install-root) install_root=$2; shift 2 ;;
    --evidence) evidence=$2; shift 2 ;;
    *) printf 'unknown FIA OPP builder argument: %s\n' "$1" >&2; exit 96 ;;
  esac
done

for value in "${source_root}" "${work_root}" "${install_root}" "${evidence}"; do
  [[ -n "${value}" ]] || { printf 'missing FIA OPP builder argument\n' >&2; exit 96; }
done
for required in "${source_root}/build.sh" "${source_root}/cmake/custom_build.cmake" \
  "${source_root}/third_party" "${cann_home}/set_env.sh" \
  "${cann_python_env}/bin/python" "${patch_file}" "${auditor}"; do
  [[ -e "${required}" ]] || { printf 'missing FIA OPP input: %s\n' "${required}" >&2; exit 96; }
done
[[ "${work_root}" == /dev/shm/* && "${install_root}" == /dev/shm/* ]] || {
  printf 'FIA OPP build and install roots must be isolated under /dev/shm\n' >&2
  exit 96
}
[[ ! -e "${work_root}" && ! -e "${install_root}" ]] || {
  printf 'FIA OPP build/install root already exists\n' >&2
  exit 96
}
mkdir -p "${evidence}"

source_commit=$(git -C "${source_root}" rev-parse HEAD)
[[ "${source_commit}" == "${expected_commit}" ]] || {
  printf 'unexpected ops-transformer commit: %s\n' "${source_commit}" >&2
  exit 94
}
git -C "${source_root}" status --porcelain=v1 --untracked-files=no \
  >"${evidence}/fia-ops-source-status.txt"
[[ ! -s "${evidence}/fia-ops-source-status.txt" ]] || {
  printf 'tracked ops-transformer source is dirty\n' >&2
  exit 94
}
printf '%s\n' "${source_commit}" >"${evidence}/fia-ops-source-commit.txt"
git -C "${source_root}" remote get-url origin \
  >"${evidence}/fia-ops-source-remote.txt"
sha256sum "${patch_file}" >"${evidence}/fia-hierarchy-patch.sha256"

mkdir -p "${work_root}"
git -C "${source_root}" archive --format=tar HEAD | tar -xf - -C "${work_root}"
ln -s "${source_root}/third_party" "${work_root}/third_party"
patch --directory="${work_root}" -p1 --input="${patch_file}" \
  >"${evidence}/fia-hierarchy-patch.log"

source "${cann_home}/set_env.sh"
export PATH=${cann_python_env}/bin:${PATH}
"${cann_python_env}/bin/python" -c 'import numpy, te, tbe'
(
  cd "${work_root}"
  bash build.sh --jit --soc=ascend910b \
    --ops=fused_infer_attention_score \
    --vendor_name=cruise_fia_graph_v2 -j8
) >"${evidence}/fia-opp-build.log" 2>&1

mapfile -t run_packages < <(
  find "${work_root}/build" -maxdepth 1 -type f -name '*.run' -print | sort
)
if [[ ${#run_packages[@]} -eq 1 ]]; then
  package=${run_packages[0]}
  printf 'run\n' >"${evidence}/fia-opp-package-mode.txt"
  sha256sum "${package}" >"${evidence}/fia-opp-package.sha256"
  "${package}" --quiet --install-path="${install_root}" \
    >"${evidence}/fia-opp-install.log" 2>&1
elif [[ ${#run_packages[@]} -eq 0 ]]; then
  mapfile -t staging_roots < <(
    find "${work_root}/build/_CPack_Packages/Linux/External" \
      -mindepth 1 -maxdepth 1 -type d \
      -name 'cann-ops-transformer-*_linux-*.run' -print | sort
  )
  [[ ${#staging_roots[@]} -eq 1 ]] || {
    printf 'expected one isolated FIA CPack staging root, found %s\n' \
      "${#staging_roots[@]}" >&2
    exit 95
  }
  staging=${staging_roots[0]}
  for required in "${staging}/install.sh" "${staging}/packages/vendors"; do
    [[ -e "${required}" ]] || {
      printf 'incomplete isolated FIA CPack staging root: %s\n' "${required}" >&2
      exit 95
    }
  done
  printf 'cpack-staging\n' >"${evidence}/fia-opp-package-mode.txt"
  (
    cd "${staging}"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) >"${evidence}/fia-opp-staging-files.sha256"
  sha256sum "${evidence}/fia-opp-staging-files.sha256" \
    >"${evidence}/fia-opp-package.sha256"
  descriptor=${staging}.json
  [[ -f "${descriptor}" ]] || {
    printf 'missing isolated FIA CPack descriptor: %s\n' "${descriptor}" >&2
    exit 95
  }
  sha256sum "${descriptor}" >"${evidence}/fia-opp-package-descriptor.sha256"
  (
    cd "${staging}"
    bash ./install.sh --quiet --install-path="${install_root}"
  ) >"${evidence}/fia-opp-install.log" 2>&1
else
  printf 'multiple isolated FIA run packages found\n' >&2
  exit 95
fi
"${cann_python_env}/bin/python" "${auditor}" \
  --source "${source_root}" --install-root "${install_root}" \
  --patch "${patch_file}" --expected-commit "${expected_commit}" \
  --output "${evidence}/fia-opp-layout.json" \
  >"${evidence}/fia-opp-layout.log" 2>&1

custom_set_env=$(find "${install_root}" -type f -name set_env.bash -print -quit)
[[ -f "${custom_set_env}" ]] || {
  printf 'isolated FIA package did not install set_env.bash\n' >&2
  exit 95
}
printf '%s\n' "${custom_set_env}"
