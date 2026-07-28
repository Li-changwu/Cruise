#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
src=${root}/attempt54-src
base=${root}/custom-op-project
project=${root}/custom-op-project-attempt54
op=${project}/bf16_barrier
raw=${root}/raw-attempt54-build
install=${root}/install-attempt54
package=${project}/output/CANN-custom_ops--linux.aarch64.run

if [[ -e "${project}" || -e "${raw}" || -e "${install}" ]]; then exit 97; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
cp -a "${base}" "${project}"
s=$?; printf 'copy-project\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
mkdir -p "${op}/op_kernel" "${op}/op_host"
cp "${src}/bf16_barrier.cpp" "${op}/op_kernel/bf16_barrier.cpp"
cp "${src}/bf16_barrier_def.cpp" "${op}/op_host/bf16_barrier_def.cpp"
cp "${src}/bf16_barrier_infershape.cpp" "${op}/op_host/bf16_barrier_infershape.cpp"
cp "${src}/bf16_barrier_tiling.cpp" "${op}/op_host/bf16_barrier_tiling.cpp"
cp "${src}/CMakeLists.txt" "${op}/op_host/CMakeLists.txt"
sha256sum "${src}"/* "${op}/op_kernel/bf16_barrier.cpp" "${op}/op_host"/* \
  >"${raw}/source-integrity.log"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHON_EXECUTABLE=$(command -v python) bash "${project}/build.sh" -n bf16_barrier -c ascend910b \
  >"${raw}/build.stdout.log" 2>&1
s=$?; printf 'build\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then tail -200 "${raw}/build.stdout.log"; exit $s; fi
"${package}" --quiet --install-path="${install}" >"${raw}/install.stdout.log" 2>&1
s=$?; printf 'install\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
installed=$(find "${install}" -type f -path '*/ascendc/bf16_barrier/bf16_barrier.cpp' | head -1)
[[ -n "${installed}" ]] || { printf 'installed-source\t95\n' >>"${status}"; exit 95; }
sha256sum "${package}" "${installed}" >"${raw}/result-integrity.log"
printf 'installed-source\t0\n' >>"${status}"
cat "${status}"

