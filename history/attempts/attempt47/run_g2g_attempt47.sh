#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt47
project=${root}/custom-op-project
source_root=${project}/exact_qk
kernel=${source_root}/op_kernel/exact_qk.cpp
op_def=${source_root}/op_host/exact_qk_def.cpp
infershape=${source_root}/op_host/exact_qk_infershape.cpp
host_tiling=${source_root}/op_host/exact_qk_tiling.cpp
package=${project}/output/CANN-custom_ops--linux.aarch64.run
stage=${root}/staging-attempt47
archive_dir=${root}/frozen-static-mode0-attempt46
install_dir=${root}/install-attempt47
protocol=${root}/attempt47-protocol.md

expected_old_package_sha=47be1f718dffd57fd8f0328032e7165d860e6f51d20e955ba1f13503cea3762c
expected_old_kernel_sha=3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc
expected_old_def_sha=d9ed645dd9972bd51f14abcd858d3aefb6e754c8c0ed3b0b47d3aaa2e2e41a8b
expected_old_infer_sha=b1af1b348abb375047d1fbbc29ba73dc271ecfdf223ade08d3b7e98a73022c80
expected_host_sha=6e04aff60c4320def8969ad10a808466db2b5fb1c9a99ab5a3c71ad410f5d89f
expected_new_kernel_sha=f3bf8191b6b12c56ff63428d712951f43a4b83046dfc1d93c38712493f5c2363
expected_new_def_sha=47f8cda49a10a6ab9e72cf14f8a5c31ccf0ed95eaedf70baf7d072021d315736
expected_new_infer_sha=6f386cebc9365bae455d2bd27ba68d81d47644453818280037c94068ded84812
expected_protocol_sha=662fc6ac8c50bea96e2fb59e5bb202e7f26122cca4d7418f98124dad7e7d34d1

if [[ -e "${raw}" || -e "${archive_dir}" || -e "${install_dir}" ]]; then
  printf 'G2G_ATTEMPT47_REFUSE_OVERWRITE raw=%s archive=%s install=%s\n' \
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
      $(sha256sum "${kernel}" | cut -d' ' -f1) != "${expected_old_kernel_sha}" ||
      $(sha256sum "${op_def}" | cut -d' ' -f1) != "${expected_old_def_sha}" ||
      $(sha256sum "${infershape}" | cut -d' ' -f1) != "${expected_old_infer_sha}" ||
      $(sha256sum "${host_tiling}" | cut -d' ' -f1) != "${expected_host_sha}" ||
      $(sha256sum "${stage}/exact_qk.cpp" | cut -d' ' -f1) != "${expected_new_kernel_sha}" ||
      $(sha256sum "${stage}/exact_qk_def.cpp" | cut -d' ' -f1) != "${expected_new_def_sha}" ||
      $(sha256sum "${stage}/exact_qk_infershape.cpp" | cut -d' ' -f1) != "${expected_new_infer_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'input-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'input-integrity\t0\n' >>"${status_file}"

cp -p "${package}" "${archive_dir}/CANN-custom_ops--linux.aarch64.run"
cp -p "${kernel}" "${archive_dir}/exact_qk_static.cpp"
cp -p "${op_def}" "${archive_dir}/exact_qk_def_two_input.cpp"
cp -p "${infershape}" "${archive_dir}/exact_qk_infershape_two_input.cpp"
sha256sum "${package}" "${archive_dir}/CANN-custom_ops--linux.aarch64.run" \
  "${kernel}" "${archive_dir}/exact_qk_static.cpp" \
  "${op_def}" "${archive_dir}/exact_qk_def_two_input.cpp" \
  "${infershape}" "${archive_dir}/exact_qk_infershape_two_input.cpp" \
  >"${raw}/old-artifact-integrity.log"
printf 'old-artifact-archive\t0\n' >>"${status_file}"

cp "${stage}/exact_qk.cpp" "${kernel}"
cp "${stage}/exact_qk_def.cpp" "${op_def}"
cp "${stage}/exact_qk_infershape.cpp" "${infershape}"
sha256sum "${kernel}" "${op_def}" "${infershape}" "${host_tiling}" \
  "${protocol}" >"${raw}/source-integrity.log"
if [[ $(sha256sum "${kernel}" | cut -d' ' -f1) != "${expected_new_kernel_sha}" ||
      $(sha256sum "${op_def}" | cut -d' ' -f1) != "${expected_new_def_sha}" ||
      $(sha256sum "${infershape}" | cut -d' ' -f1) != "${expected_new_infer_sha}" ||
      $(sha256sum "${host_tiling}" | cut -d' ' -f1) != "${expected_host_sha}" ]] ||
   ! grep -Fq 'GM_ADDR gm_explicit_tiling' "${kernel}" ||
   ! grep -Fq '(void)gm_tiling_data;' "${kernel}" ||
   grep -Fq 'gm_c, gm_tiling_data' "${kernel}" ||
   ! grep -Fq 'Input("explicit_tiling")' "${op_def}" ||
   ! grep -Fq 'DataType({ge::DT_UINT8})' "${op_def}" ||
   ! grep -Fq 'explicit_tiling->GetDim(0) != 72' "${infershape}"; then
  printf 'explicit-input-source\t95\n' >>"${status_file}"
  exit 95
fi
printf 'explicit-input-source\t0\n' >>"${status_file}"

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
   [[ $(sha256sum "${installed_source}" | cut -d' ' -f1) != "${expected_new_kernel_sha}" ]] ||
   [[ $(grep -c 'KERNEL_TYPE_AIC_ONLY' "${installed_source}") -ne 2 ]]; then
  printf 'installed-explicit-aic-contract\t92\n' >>"${status_file}"
  exit 92
fi
sha256sum "${kernel}" "${installed_source}" \
  >"${raw}/installed-source-integrity.log"
printf 'installed-explicit-aic-contract\t0\n' >>"${status_file}"

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
printf 'G2G_ATTEMPT47_COMPLETE package_sha=%s\n' "${new_package_sha}"
