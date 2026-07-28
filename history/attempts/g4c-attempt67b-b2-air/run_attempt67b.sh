#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt67b-src
raw=${root}/raw-attempt67b-b2-air
export_dir=${root}/export-attempt67b-b2
cache=${root}/cache-attempt67b-b2
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
reference=${root}/raw-attempt67a-b2-eager/outputs/attempt67a-b2-eager-reference.npz
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
materialize=${root}/attempt56r1-materialize-src

if [[ -e "${raw}" || -e "${export_dir}" || -e "${cache}" ]]; then exit 97; fi
if [[ ! -f "${reference}" ]]; then exit 96; fi
mkdir -p "${raw}" "${cache}"
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
export ASCEND_CACHE_PATH=${cache}
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${reference}" \
  >"${raw}/input-integrity.log"
timeout 10800 python "${src}/export_b2_air.py" \
  --model-dir "${model}" \
  --reference "${reference}" \
  --exact-qk-source "${exact_qk}" \
  --barrier-source "${barrier}" \
  --materialize-source "${materialize}" \
  --output-dir "${export_dir}" >"${raw}/export.stdout.log" 2>&1
export_status=$?
printf 'export\t%s\n' "${export_status}" >>"${status}"
if [[ ${export_status} -ne 0 ]]; then
  tail -240 "${raw}/export.stdout.log"
  exit ${export_status}
fi

python "${src}/extract_b2_decoder_abi.py" \
  --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/abi.json" >"${raw}/abi.stdout.log" 2>&1
abi_status=$?
printf 'abi\t%s\n' "${abi_status}" >>"${status}"

python "${src}/inspect_b2_graph.py" \
  --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/graph-inspection.json" \
  >"${raw}/graph-inspection.stdout.log" 2>&1
graph_status=$?
printf 'graph-inspection\t%s\n' "${graph_status}" >>"${status}"

capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=95
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${export_dir}/qwen_b2_decoder_step_attempt67b.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/abi.json" \
  "${export_dir}/graph-inspection.json" "${export_dir}/export-result.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${export_dir}/export-result.json"
cat "${export_dir}/abi.json"
python -c "import json; p=json.load(open('${export_dir}/graph-inspection.json')); print(json.dumps({k:p[k] for k in ('valid','observed_op_counts','active_mask_data_count','active_mask_direct_consumers')}, indent=2))"
if [[ ${abi_status} -ne 0 ]]; then exit ${abi_status}; fi
if [[ ${graph_status} -ne 0 ]]; then exit ${graph_status}; fi
exit ${idle_status}
