#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt69b-src
raw=${root}/raw-attempt69b-b4-eager
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
g4a_reference=${root}/raw-attempt65-eager/attempt65-eager-reference.npz
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
materialize=${root}/attempt56r1-materialize-src

if [[ -e "${raw}" ]]; then exit 97; fi
if [[ ! -f "${g4a_reference}" ]]; then exit 96; fi
mkdir -p "${raw}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
  printf 'npu7-idle\t95\n' >>"${status}"
  exit 95
fi
printf 'npu7-idle\t0\n' >>"${status}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${g4a_reference}" \
  >"${raw}/input-integrity.log"
timeout 7200 python "${src}/b4_eager_test.py" \
  --model-dir "${model}" \
  --g4a-reference "${g4a_reference}" \
  --exact-qk-source "${exact_qk}" \
  --barrier-source "${barrier}" \
  --materialize-source "${materialize}" \
  --output-dir "${raw}/outputs" >"${raw}/eager.stdout.log" 2>&1
eager_status=$?
printf 'b4-eager\t%s\n' "${eager_status}" >>"${status}"
if [[ ${eager_status} -ne 0 ]]; then
  tail -240 "${raw}/eager.stdout.log"
  exit ${eager_status}
fi

capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${raw}/outputs/attempt69b-b4-eager-reference.npz" \
  "${raw}/outputs/attempt69b-b4-eager-result.json" \
  "${raw}/outputs/checkpoint-audit.json" >"${raw}/result-integrity.log"
cat "${raw}/outputs/attempt69b-b4-eager-result.json"
exit ${idle_status}
