#!/usr/bin/env bash
set -uo pipefail

g2d=/root/ascend-control-g2d-20260718
g2e=/root/ascend-control-g2e-20260718
g2g=/root/ascend-control-g2g-20260719
src=${g2e}/attempt52-src
tag=${1:-attempt52}
raw=${g2e}/raw-${tag}-export
out=${g2e}/${tag}-export
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
original=${g2d}/export/reference.npz
frozen=${g2e}/attempt7-export/attempt7-eager-reference.npz

if [[ -e "${raw}" || -e "${out}" ]]; then
  printf 'G4A_ATTEMPT52_EXPORT_REFUSE_OVERWRITE\n'
  exit 97
fi
mkdir -p "${raw}" "${out}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json

custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" ]] || { printf 'custom-set-env\t95\n' >>"${status}"; exit 95; }
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
printf 'custom-set-env\t0\n' >>"${status}"

sha256sum "${src}"/* "${original}" "${frozen}" \
  "${model}"/model-*.safetensors "${model}"/model.safetensors.index.json \
  >"${raw}/input-integrity.log"

timeout 1800 python "${src}/export_attention_attempt52_air.py" \
  --g2d-root "${g2d}" --g2e-root "${g2e}" \
  --custom-source "${g2g}/attempt50-src" --model-dir "${model}" \
  --original-reference "${original}" --frozen-reference "${frozen}" \
  --output-dir "${out}" >"${raw}/export.stdout.log" 2>&1
s=$?; printf 'export\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then tail -200 "${raw}/export.stdout.log"; exit $s; fi

python "${src}/extract_air_abi.py" --graph "${out}/dynamo.pbtxt" \
  --output "${out}/abi.json" >"${raw}/abi.stdout.log" 2>&1
s=$?; printf 'abi\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/abi.stdout.log"; exit $s; fi

grep -n -A12 -B4 'op: "ExactQk"' "${out}/dynamo.pbtxt" \
  >"${raw}/exactqk-node.txt"
s=$?; printf 'exactqk-node\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then exit 94; fi
sha256sum "${out}/qwen_attention_attempt52.air" "${out}/dynamo.pbtxt" \
  "${out}/attempt52-eager-reference.npz" "${out}/tiling.bin" "${out}/abi.json" \
  >"${raw}/result-integrity.log"
npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
cat "${out}/export-result.json"
cat "${out}/abi.json"
