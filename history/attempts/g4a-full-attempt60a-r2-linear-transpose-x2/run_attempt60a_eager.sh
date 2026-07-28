#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt60a-r2-src
raw=${root}/raw-attempt60a-r2-eager
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
baseline=${root}/raw-attempt53k-eager/attempt53k-eager-reference.npz

if [[ -e "${raw}" ]]; then exit 97; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t96\n' >>"${status}"; exit 96
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${baseline}" \
  >"${raw}/input-integrity.log"
timeout 3600 python "${src}/full_decoder_step.py" --mode eager --model-dir "${model}" \
  --exact-qk-source "${exact_qk}" --barrier-source "${barrier}" \
  --output-dir "${raw}" >"${raw}/eager.stdout.log" 2>&1
eager_status=$?
printf 'eager\t%s\n' "${eager_status}" >>"${status}"
if [[ ${eager_status} -ne 0 ]]; then tail -240 "${raw}/eager.stdout.log"; exit ${eager_status}; fi

python "${src}/compare_eager.py" --baseline "${baseline}" \
  --candidate "${raw}/attempt60a-eager-reference.npz" \
  --output "${raw}/eager-comparison.json" >"${raw}/eager-comparison.stdout.log" 2>&1
comparison_status=$?
printf 'eager-identical\t%s\n' "${comparison_status}" >>"${status}"
capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${raw}/attempt60a-eager-reference.npz" "${raw}/eager-screen.json" \
  "${raw}/eager-comparison.json" >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/eager-screen.json"
cat "${raw}/eager-comparison.json"
if [[ ${comparison_status} -ne 0 ]]; then exit ${comparison_status}; fi
exit ${idle_status}
