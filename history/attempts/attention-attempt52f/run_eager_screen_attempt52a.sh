#!/usr/bin/env bash
set -uo pipefail

g2d=/root/ascend-control-g2d-20260718
g2e=/root/ascend-control-g2e-20260718
g2g=/root/ascend-control-g2g-20260719
tag=${1:-attempt52a-screen}
out=${g2e}/${tag}
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
if [[ -e "${out}" ]]; then exit 97; fi
mkdir -p "${out}"
if ! npu-smi info -t proc-mem -i 7 >"${out}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${out}/npu7-processes-before.txt"; then exit 96; fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=2
python "${g2e}/attempt52-src/export_attention_attempt52_air.py" \
  --g2d-root "${g2d}" --g2e-root "${g2e}" \
  --custom-source "${g2g}/attempt50-src" --model-dir "${model}" \
  --original-reference "${g2d}/export/reference.npz" \
  --frozen-reference "${g2e}/attempt7-export/attempt7-eager-reference.npz" \
  --output-dir "${out}" --eager-only >"${out}/screen.stdout.log" 2>&1
s=$?
printf 'screen\t%s\n' "$s" >"${out}/status.tsv"
npu-smi info -t proc-mem -i 7 >"${out}/npu7-processes-after.txt" 2>&1 || true
cat "${out}/eager-screen.json" 2>/dev/null || true
exit $s
