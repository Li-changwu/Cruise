#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
src=${root}/attempt53e-src
raw=${root}/raw-attempt53e-audit
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

if [[ -e "${raw}" ]]; then exit 97; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
python "${src}/full_decoder_step.py" --mode audit --model-dir "${model}" --output-dir "${raw}" \
  >"${raw}/audit.stdout.log" 2>&1
s=$?
printf 'audit\t%s\n' "$s" >>"${status}"
sha256sum "${src}"/* "${raw}/checkpoint-audit.json" >"${raw}/artifact-integrity.log" 2>/dev/null || true
cat "${raw}/checkpoint-audit.json" 2>/dev/null || true
exit $s
