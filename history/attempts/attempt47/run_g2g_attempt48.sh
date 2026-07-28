#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt48
export_dir=${root}/export-attempt48
native_input_dir=${raw}/native-inputs
package=${root}/custom-op-project/output/CANN-custom_ops--linux.aarch64.run
install_dir=${root}/install-attempt47
exporter=${root}/export_g2g_attempt48_air.py
attempt8=/root/ascend-control-g2e-20260718/raw-attempt8/attempt8-native-output.npz
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
numa=${root}/numa_config.physical7.json
protocol=${root}/attempt48-protocol.md
name=exact_qk_explicit_tiling_minimal

expected_package_sha=6e4435ce0c85b4c18ca672a957a023c77b0f83810c86aeca6ac60e8be6b7e18e
expected_exporter_sha=ad566a77f2f0f3c1b82880936d088906858655595a54b83ceab290ab4ccac4d8
expected_attempt8_sha=9db5b65ee01523d58c5f4b48e7dcf1ba1997ea02ed1dfbcb9e284750484f3b66
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_numa_sha=33fa0983196dd207791014cdd9ce80392d5cccbf1d4183dbd5efba47c7ed92cf
expected_protocol_sha=c9ec49a44966318e2364ca40b28c7d7de98ac6973b2d9d38fcc4e4bd46acf5cd

if [[ -e "${raw}" || -e "${export_dir}" ]]; then
  printf 'G2G_ATTEMPT48_REFUSE_OVERWRITE raw=%s export=%s\n' \
    "${raw}" "${export_dir}"
  exit 97
fi
mkdir -p "${raw}" "${export_dir}" "${native_input_dir}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${numa}
export PYTHONPATH=/root/vllm-ascend-hust:${PYTHONPATH:-}
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT
npu-smi info >"${raw}/npu-before.txt" 2>&1

if [[ $(sha256sum "${package}" | cut -d' ' -f1) != "${expected_package_sha}" ||
      $(sha256sum "${exporter}" | cut -d' ' -f1) != "${expected_exporter_sha}" ||
      $(sha256sum "${attempt8}" | cut -d' ' -f1) != "${expected_attempt8_sha}" ||
      $(sha256sum "${eager}" | cut -d' ' -f1) != "${expected_eager_sha}" ||
      $(sha256sum "${numa}" | cut -d' ' -f1) != "${expected_numa_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${package}" "${exporter}" "${attempt8}" "${eager}" \
  "${numa}" "${protocol}" "${root}/run_g2g_attempt48.sh" \
  >"${raw}/artifact-integrity.log"

custom_set_env=$(find "${install_dir}" -type f -name set_env.bash | head -1)
if [[ -z "${custom_set_env}" ]]; then
  printf 'custom-set-env\t95\n' >>"${status_file}"
  exit 95
fi
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
printf 'custom-set-env\t0\n' >>"${status_file}"

python "${exporter}" --attempt8-output "${attempt8}" \
  --eager-reference "${eager}" --output-dir "${export_dir}" \
  --native-input-dir "${native_input_dir}" --name "${name}" \
  >"${raw}/export.stdout.log" 2>&1
export_status=$?
printf 'export\t%s\n' "${export_status}" >>"${status_file}"
if [[ "${export_status}" -ne 0 ]]; then
  tail -240 "${raw}/export.stdout.log"
  exit "${export_status}"
fi

air=${export_dir}/${name}.air
graph=${export_dir}/dynamo.pbtxt
result=${export_dir}/export-result.json
if [[ ! -s "${air}" || ! -s "${graph}" || ! -s "${result}" ]] ||
   ! grep -Fq 'op: "ExactQk"' "${graph}" ||
   ! grep -Fq 'input: "arg2_1:0"' "${graph}" ||
   ! grep -Fq 'explicit_tiling' "${graph}"; then
  printf 'three-input-graph\t94\n' >>"${status_file}"
  exit 94
fi
grep -n -A32 -B8 'op: "ExactQk"' "${graph}" \
  >"${raw}/graph-exact-qk-lines.txt"
printf 'three-input-graph\t0\n' >>"${status_file}"

python - "${native_input_dir}/tiling.bin" <<'PY'
import sys
import numpy as np

expected = np.array(
    [28, 1, 128, 8, 16, 512, 16, 1, 1, 1, 28, 5, 2336, 24, 0, 0, 0, 0],
    dtype="<u4",
)
actual = np.fromfile(sys.argv[1], dtype="<u4")
if actual.shape != expected.shape or not np.array_equal(actual, expected):
    raise SystemExit(1)
PY
tiling_status=$?
printf 'tiling-bytes\t%s\n' "${tiling_status}" >>"${status_file}"
if [[ "${tiling_status}" -ne 0 ]]; then
  exit "${tiling_status}"
fi

input_tree_sha=$(cd "${native_input_dir}" && \
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
sha256sum "${air}" "${graph}" "${result}" "${native_input_dir}/tiling.bin" \
  >"${raw}/export-integrity.log"
printf '%s  native-input-tree\n' "${input_tree_sha}" \
  >>"${raw}/export-integrity.log"
printf 'G2G_ATTEMPT48_COMPLETE air_sha=%s input_tree_sha=%s\n' \
  "$(sha256sum "${air}" | cut -d' ' -f1)" "${input_tree_sha}"
