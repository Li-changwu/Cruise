#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g2g-20260719
raw=${root}/raw-attempt44
harness=${root}/run_g2g_attempt44.py
helper=${root}/run_g2g_attempt1.py
protocol=${root}/attempt44-protocol.md
attempt8=/root/ascend-control-g2e-20260718/raw-attempt8/attempt8-native-output.npz
eager=/root/ascend-control-g2e-20260718/attempt7-export/attempt7-eager-reference.npz
kernel_lib=/root/vllm-ascend-hust/vllm_ascend/libvllm_ascend_kernels.so
extension=/root/vllm-ascend-hust/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so

expected_harness_sha=5e18309d27ec62fc2df3659c3bfe494f148217b4907df0d8d980a8fbc77e3768
expected_helper_sha=95d4120926ea4e69e20806a2a20b225676bfaefaf1b02956c62afe5c3f137149
expected_attempt8_sha=9db5b65ee01523d58c5f4b48e7dcf1ba1997ea02ed1dfbcb9e284750484f3b66
expected_eager_sha=d9fdb972425582b9d26465c8455e2f7caadc08d019a10f3e392300ea3e84ff5f
expected_kernel_lib_sha=a8ae3257147cfed087bf3f289577068592aaae8eeeb390f2b921f73b178fb185
expected_extension_sha=eee2f71f57e59b3c30009a583035dd2be026767076abeeaecd9a6cf802f96edd
expected_protocol_sha=8205853039b5210d19d551ccc50bd45f387ea91c8af5907f39ce02c0a98270b7

if [[ -e "${raw}" ]]; then
  printf 'G2G_ATTEMPT44_REFUSE_OVERWRITE raw=%s\n' "${raw}"
  exit 97
fi
mkdir -p "${raw}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export PYTHONPATH=/root/vllm-ascend-hust:${PYTHONPATH:-}
status_file=${raw}/status.tsv
printf 'case\texit_status\n' >"${status_file}"
capture_after() { npu-smi info >"${raw}/npu-after.txt" 2>&1 || true; }
trap capture_after EXIT
npu-smi info >"${raw}/npu-before.txt" 2>&1

if [[ $(sha256sum "${harness}" | cut -d' ' -f1) != "${expected_harness_sha}" ||
      $(sha256sum "${helper}" | cut -d' ' -f1) != "${expected_helper_sha}" ||
      $(sha256sum "${attempt8}" | cut -d' ' -f1) != "${expected_attempt8_sha}" ||
      $(sha256sum "${eager}" | cut -d' ' -f1) != "${expected_eager_sha}" ||
      $(sha256sum "${kernel_lib}" | cut -d' ' -f1) != "${expected_kernel_lib_sha}" ||
      $(sha256sum "${extension}" | cut -d' ' -f1) != "${expected_extension_sha}" ||
      $(sha256sum "${protocol}" | cut -d' ' -f1) != "${expected_protocol_sha}" ]]; then
  printf 'artifact-integrity\t98\n' >>"${status_file}"
  exit 98
fi
printf 'artifact-integrity\t0\n' >>"${status_file}"
sha256sum "${harness}" "${helper}" "${attempt8}" "${eager}" \
  "${kernel_lib}" "${extension}" "${protocol}" \
  "${root}/run_g2g_attempt44.sh" >"${raw}/artifact-integrity.log"

python "${harness}" --attempt8-output "${attempt8}" \
  --eager-reference "${eager}" \
  --output-npz "${raw}/attempt44-output.npz" \
  --output "${raw}/attempt44-result.json" \
  >"${raw}/runtime.stdout.log" 2>&1
runtime_status=$?
printf 'runtime\t%s\n' "${runtime_status}" >>"${status_file}"
if [[ "${runtime_status}" -ne 0 ]]; then
  tail -240 "${raw}/runtime.stdout.log"
  exit "${runtime_status}"
fi

grep -F 'LaunchKernelWithHandle: kernel info' "${raw}/runtime.stdout.log" | \
  grep -F 'batch_matmul_transpose' >"${raw}/direct-launch-metadata.txt"
if [[ $(wc -l <"${raw}/direct-launch-metadata.txt") -ne 12 ]] ||
   grep -Fv 'device_id=7' "${raw}/direct-launch-metadata.txt" | grep -q . ||
   grep -Fv 'kernelType=0' "${raw}/direct-launch-metadata.txt" | grep -q . ||
   grep -Fv 'coreDim=24' "${raw}/direct-launch-metadata.txt" | grep -q . ||
   grep -Fv 'schemMode=2' "${raw}/direct-launch-metadata.txt" | grep -q .; then
  printf 'launch-metadata\t96\n' >>"${status_file}"
  exit 96
fi
printf 'launch-metadata\t0\n' >>"${status_file}"
sha256sum "${raw}/attempt44-output.npz" "${raw}/attempt44-result.json" \
  "${raw}/direct-launch-metadata.txt" >"${raw}/result-integrity.log"
printf 'G2G_ATTEMPT44_COMPLETE\n'
