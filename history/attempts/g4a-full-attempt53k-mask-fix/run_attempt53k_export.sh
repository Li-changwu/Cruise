#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt53k-src
eager=${root}/raw-attempt53k-eager
export_dir=${root}/export-attempt53k
cache=${root}/cache-attempt53k-export
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src

if [[ -e "${export_dir}" || -e "${cache}" ]]; then exit 97; fi
if [[ ! -f "${eager}/eager-screen.json" ]] ||
   ! grep -q '"pass": true' "${eager}/eager-screen.json"; then exit 96; fi
mkdir -p "${export_dir}" "${cache}"
status=${export_dir}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${export_dir}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${export_dir}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"; exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" ]] || exit 94
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
[[ -n "${barrier_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
timeout 10800 python "${src}/full_decoder_step.py" --mode export --model-dir "${model}" \
  --exact-qk-source "${exact_qk}" --barrier-source "${barrier}" \
  --reference "${eager}/attempt53k-eager-reference.npz" \
  --output-dir "${export_dir}" >"${export_dir}/export.stdout.log" 2>&1
s=$?
printf 'export\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then exit $s; fi
python "${src}/extract_full_decoder_abi.py" --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/abi.json" >"${export_dir}/abi.stdout.log" 2>&1
s=$?
printf 'abi\t%s\n' "$s" >>"${status}"
npu-smi info -t proc-mem -i 7 >"${export_dir}/npu7-processes-after.txt" 2>&1 || true
sha256sum "${src}"/* "${eager}/attempt53k-eager-reference.npz" \
  "${export_dir}/qwen_full_decoder_step_attempt53k.air" "${export_dir}/dynamo.pbtxt" \
  "${export_dir}/abi.json" >"${export_dir}/artifact-integrity.log" 2>/dev/null || true
cat "${export_dir}/abi.json" 2>/dev/null || true
exit $s
