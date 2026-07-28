#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt45
project=${root}/custom-op-project
source_root=${project}/exact_qk
kernel=${source_root}/op_kernel/exact_qk.cpp
host_tiling=${source_root}/op_host/exact_qk_tiling.cpp
package=${project}/output/CANN-custom_ops--linux.aarch64.run
staged_host=${root}/staging-attempt45/exact_qk_tiling_mode0.cpp
archive_dir=${root}/frozen-static-mode1-attempt44
install_dir=${root}/install-attempt45
protocol=${root}/attempt45-protocol.md

expected_old_package_sha=7b709fe0c5f700d6c3a82712fc8b538fed255e7e2d158e38e1d8eb679ec78124
expected_kernel_sha=3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc
expected_old_host_sha=1b313413a3044a5d39ccd072fe4517c19856cd55cb8aa8a26348d07b54ca32f5
expected_new_host_sha=6e04aff60c4320def8969ad10a808466db2b5fb1c9a99ab5a3c71ad410f5d89f
expected_protocol_sha=5f88aa32f713dec912f81204b479a77a6a7e8173c6d5765bcb5e05bcb84f72e9

if [[ -e "${raw}" || -e "${archive_dir}" || -e "${install_dir}" ]]; then
  printf 'G2G_ATTEMPT45_REFUSE_OVERWRITE raw=%s archive=%s install=%s\n' \
    "${raw}" "${archive_dir}" "${install_dir}"
  exit 97
fi
mkdir -p "${raw}" "${archive_dir}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"

if [[ $(sha256sum "${package}" | cut -d' ' -f1) != "${expected_old_package_sha}" ||
      $(sha256sum "${kernel}" | cut -d' ' -f1) != "${expected_kernel_sha}" ||
      $(sha256sum "${host_tiling}" | cut -d' ' -f1) != "${expected_old_host_sha}" ||
      $(sha256sum "${staged_host}" | cut -d' ' -f1) != "${expected_new_host_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'input-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'input-integrity\t0\n' >>"${status_file}"

cp -p "${package}" "${archive_dir}/CANN-custom_ops--linux.aarch64.run"
cp -p "${host_tiling}" "${archive_dir}/exact_qk_tiling_mode1.cpp"
sha256sum "${package}" "${archive_dir}/CANN-custom_ops--linux.aarch64.run" \
  "${host_tiling}" "${archive_dir}/exact_qk_tiling_mode1.cpp" \
  >"${raw}/old-artifact-integrity.log"
printf 'old-artifact-archive\t0\n' >>"${status_file}"

cp "${staged_host}" "${host_tiling}"
sha256sum "${kernel}" "${host_tiling}" "${protocol}" \
  >"${raw}/source-integrity.log"
if [[ $(sha256sum "${kernel}" | cut -d' ' -f1) != "${expected_kernel_sha}" ||
      $(sha256sum "${host_tiling}" | cut -d' ' -f1) != "${expected_new_host_sha}" ]] ||
   ! grep -Fq 'SetScheduleMode(0)' "${host_tiling}" ||
   grep -Fq 'SetScheduleMode(1)' "${host_tiling}" ||
   grep -Fq 'SetScheduleMode(2)' "${host_tiling}"; then
  printf 'mode0-source\t95\n' >>"${status_file}"
  exit 95
fi
printf 'mode0-source\t0\n' >>"${status_file}"

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
   [[ $(sha256sum "${installed_source}" | cut -d' ' -f1) != "${expected_kernel_sha}" ]] ||
   [[ $(grep -c 'KERNEL_TYPE_AIC_ONLY' "${installed_source}") -ne 2 ]]; then
  printf 'installed-static-aic-contract\t92\n' >>"${status_file}"
  exit 92
fi
sha256sum "${kernel}" "${installed_source}" \
  >"${raw}/installed-source-integrity.log"
printf 'installed-static-aic-contract\t0\n' >>"${status_file}"

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
printf 'G2G_ATTEMPT45_COMPLETE package_sha=%s\n' "${new_package_sha}"
