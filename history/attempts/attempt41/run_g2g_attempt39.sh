#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt39
project=${root}/custom-op-project
source_root=${project}/exact_qk
kernel=${source_root}/op_kernel/exact_qk.cpp
host_tiling=${source_root}/op_host/exact_qk_tiling.cpp
original_kernel=${root}/frozen-compute-attempt22/exact_qk.cpp
package=${project}/output/CANN-custom_ops--linux.aarch64.run
archive_dir=${root}/frozen-input-echo-attempt38
old_package=${archive_dir}/CANN-custom_ops--linux.aarch64.run
old_kernel=${archive_dir}/exact_qk_input_echo_aiv.cpp
old_host_tiling=${archive_dir}/exact_qk_tiling_schedule1.cpp
install_dir=${root}/install-attempt39
expected_old_package_sha=9e016d33b02a92571703b3921b20606bc7172d6beab931d4a0fe05808edce702
expected_old_tree_sha=5ab3cfc34b9c67bcd026427ec47c8ab20fee74c3d42f640bbb99b6051b6f98d8
expected_original_kernel_sha=d136c5f31814ad677edcb209b2920f4ec6ee3537297aa13b572120a400b192e7
expected_new_tree_sha=86804fd50ceecd5b27b66b954b6fdc2e72ad06c0388582cf368b8899db32203b

if [[ -e "${raw}" || -e "${install_dir}" || -e "${archive_dir}" ]]; then
  printf 'G2G_ATTEMPT39_REFUSE_OVERWRITE raw=%s install=%s archive=%s\n' \
    "${raw}" "${install_dir}" "${archive_dir}"
  exit 97
fi
mkdir -p "${raw}" "${archive_dir}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"

current_package_sha=$(sha256sum "${package}" | cut -d' ' -f1)
current_tree_sha=$(cd "${source_root}" && \
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
original_kernel_sha=$(sha256sum "${original_kernel}" | cut -d' ' -f1)
if [[ "${current_package_sha}" != "${expected_old_package_sha}" ||
      "${current_tree_sha}" != "${expected_old_tree_sha}" ||
      "${original_kernel_sha}" != "${expected_original_kernel_sha}" ]] ||
   ! grep -Fq 'SetScheduleMode(1)' "${host_tiling}"; then
  printf 'input-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'input-integrity\t0\n' >>"${status_file}"
cp -p "${package}" "${old_package}"
cp -p "${kernel}" "${old_kernel}"
cp -p "${host_tiling}" "${old_host_tiling}"
sha256sum "${package}" "${old_package}" "${kernel}" "${old_kernel}" \
  "${host_tiling}" "${old_host_tiling}" "${original_kernel}" \
  >"${raw}/old-artifact-integrity.log"
printf 'old-artifact-archive\t0\n' >>"${status_file}"

cp "${original_kernel}" "${kernel}"
sed -i 's/SetScheduleMode(1)/SetScheduleMode(2)/' "${host_tiling}"
new_tree_sha=$(cd "${source_root}" && \
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
file_count=$(find "${source_root}" -type f | wc -l)
printf 'tree_sha256\t%s\nfile_count\t%s\n' "${new_tree_sha}" "${file_count}" \
  >"${raw}/source-integrity.log"
if [[ "${new_tree_sha}" != "${expected_new_tree_sha}" || "${file_count}" -ne 31 ]] ||
   ! grep -Fq 'SetScheduleMode(2)' "${host_tiling}" ||
   grep -Fq 'SetScheduleMode(1)' "${host_tiling}"; then
  printf 'source-integrity\t95\n' >>"${status_file}"
  exit 95
fi
printf 'source-integrity\t0\n' >>"${status_file}"

PYTHON_EXECUTABLE=$(command -v python) \
  bash "${project}/build.sh" -n exact_qk -c ascend910b \
  >"${raw}/build.stdout.log" 2>&1
build_status=$?
printf 'build\t%s\n' "${build_status}" >>"${status_file}"
if [[ "${build_status}" -ne 0 ]]; then
  tail -240 "${raw}/build.stdout.log"
  exit "${build_status}"
fi

new_package_sha=$(sha256sum "${package}" | cut -d' ' -f1)
printf '%s  %s\n' "${new_package_sha}" "${package}" \
  >"${raw}/new-package-integrity.log"
if [[ "${new_package_sha}" == "${expected_old_package_sha}" ]]; then
  printf 'new-package-changed\t93\n' >>"${status_file}"
  exit 93
fi
printf 'new-package-changed\t0\n' >>"${status_file}"

"${package}" --quiet --install-path="${install_dir}" \
  >"${raw}/install.stdout.log" 2>&1
install_status=$?
printf 'install\t%s\n' "${install_status}" >>"${status_file}"
if [[ "${install_status}" -ne 0 ]]; then
  tail -160 "${raw}/install.stdout.log"
  exit "${install_status}"
fi

installed_source=$(find "${install_dir}" -type f \
  -path '*/ascendc/exact_qk/exact_qk.cpp' | head -1)
if [[ -z "${installed_source}" ]] ||
   [[ $(sha256sum "${installed_source}" | cut -d' ' -f1) != \
      "${expected_original_kernel_sha}" ]] ||
   [[ $(grep -c 'KERNEL_TYPE_AIC_ONLY' "${installed_source}") -ne 2 ]]; then
  printf 'original-aic-contract\t92\n' >>"${status_file}"
  exit 92
fi
sha256sum "${kernel}" "${installed_source}" >"${raw}/aic-source-integrity.log"
printf 'original-aic-contract\t0\n' >>"${status_file}"

if ! grep -Fq 'SetScheduleMode(2)' "${host_tiling}"; then
  printf 'schedule-mode-2-contract\t91\n' >>"${status_file}"
  exit 91
fi
grep -n 'SetScheduleMode(2)' "${host_tiling}" \
  >"${raw}/source-schedule-mode-2-contract.txt"
printf 'schedule-mode-2-contract\t0\n' >>"${status_file}"

tiling_so=$(find "${install_dir}" -name libcust_opmaster_rt2.0.so | head -1)
if [[ -z "${tiling_so}" ]]; then
  printf 'tiling-library\t90\n' >>"${status_file}"
  exit 90
fi
strings "${tiling_so}" | grep -F 'ExactQk' | sort -u \
  >"${raw}/tiling-library-exactqk-strings.txt"
if ! grep -Fq 'ExactQk_0' "${raw}/tiling-library-exactqk-strings.txt" ||
   ! grep -Fq 'ExactQk_1' "${raw}/tiling-library-exactqk-strings.txt"; then
  printf 'generated-tiling-aliases\t89\n' >>"${status_file}"
  exit 89
fi
printf 'tiling-library\t0\ngenerated-tiling-aliases\t0\n' >>"${status_file}"
printf 'G2G_ATTEMPT39_COMPLETE package_sha=%s\n' "${new_package_sha}"
