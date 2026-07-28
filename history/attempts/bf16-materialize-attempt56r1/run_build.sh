#!/usr/bin/env bash
set -uo pipefail

g4=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${g4}/attempt56r1-materialize-src
base=${g2g}/custom-op-project
project=${g4}/custom-op-project-attempt56r1
op=${project}/bf16_materialize
raw=${g4}/raw-attempt56r1-materialize-build
install=${g4}/install-attempt56r1
package=${project}/output/CANN-custom_ops--linux.aarch64.run

if [[ -e "${project}" || -e "${raw}" || -e "${install}" ]]; then exit 97; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
cp -a "${base}" "${project}"
s=$?; printf 'copy-project\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
mkdir -p "${op}/op_kernel" "${op}/op_host"
cp "${src}/bf16_materialize.cpp" "${op}/op_kernel/bf16_materialize.cpp"
cp "${src}/bf16_materialize_def.cpp" "${op}/op_host/bf16_materialize_def.cpp"
cp "${src}/bf16_materialize_infershape.cpp" "${op}/op_host/bf16_materialize_infershape.cpp"
cp "${src}/bf16_materialize_tiling.cpp" "${op}/op_host/bf16_materialize_tiling.cpp"
cp "${src}/CMakeLists.txt" "${op}/op_host/CMakeLists.txt"
sha256sum "${src}/CMakeLists.txt" "${src}"/*.cpp "${src}"/*.py "${src}"/*.sh \
  "${src}"/*.md "${op}/op_kernel/bf16_materialize.cpp" "${op}/op_host"/* \
  >"${raw}/source-integrity.log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHON_EXECUTABLE=$(command -v python) bash "${project}/build.sh" -n bf16_materialize -c ascend910b \
  >"${raw}/build.stdout.log" 2>&1
s=$?; printf 'build\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then tail -200 "${raw}/build.stdout.log"; exit $s; fi
"${package}" --quiet --install-path="${install}" >"${raw}/install.stdout.log" 2>&1
s=$?; printf 'install\t%s\n' "$s" >>"${status}"; [[ $s -eq 0 ]] || exit $s
installed=$(find "${install}" -type f -path '*/ascendc/bf16_materialize/bf16_materialize.cpp' | head -1)
[[ -n "${installed}" ]] || { printf 'installed-source\t95\n' >>"${status}"; exit 95; }
sha256sum "${package}" "${installed}" >"${raw}/result-integrity.log"
printf 'installed-source\t0\n' >>"${status}"
cat "${status}"
