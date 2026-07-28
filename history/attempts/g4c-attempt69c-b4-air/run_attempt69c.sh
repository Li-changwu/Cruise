#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt69c-src
raw=${root}/raw-attempt69c-b4-air
export_dir=${root}/export-attempt69c-b4
cache=${root}/cache-attempt69c-b4
temp_export=/dev/shm/ascend-control-g4-20260724/attempt69c-b4-export-tmp
dedup_base=${root}/export-attempt67b-b2
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
reference=${root}/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
materialize=${root}/attempt56r1-materialize-src

if [[ -e "${raw}" || -e "${export_dir}" || -e "${cache}" ||
      -e "${temp_export}" ]]; then exit 97; fi
if [[ ! -f "${reference}" ||
      ! -f "${dedup_base}/qwen_b2_decoder_step_attempt67b.air" ]]; then
  exit 96
fi
mkdir -p "${raw}" "${cache}" "$(dirname "${temp_export}")"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
  du -sh "${temp_export}" "${export_dir}" \
    >"${raw}/export-storage.txt" 2>&1 || true
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
barrier_set_env=$(find "${root}/install-attempt69a-b4-barrier" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${reference}" \
  "${dedup_base}/qwen_b2_decoder_step_attempt67b.air" \
  >"${raw}/input-integrity.log"
timeout 10800 python "${src}/export_b4_air.py" \
  --model-dir "${model}" \
  --reference "${reference}" \
  --exact-qk-source "${exact_qk}" \
  --barrier-source "${barrier}" \
  --materialize-source "${materialize}" \
  --output-dir "${temp_export}" >"${raw}/export.stdout.log" 2>&1
export_status=$?
printf 'export\t%s\n' "${export_status}" >>"${status}"
if [[ ${export_status} -ne 0 ]]; then
  tail -240 "${raw}/export.stdout.log"
  exit ${export_status}
fi

python "${src}/extract_b4_decoder_abi.py" \
  --graph "${temp_export}/dynamo.pbtxt" \
  --output "${temp_export}/abi.json" >"${raw}/temp-abi.stdout.log" 2>&1
temp_abi_status=$?
printf 'temp-abi\t%s\n' "${temp_abi_status}" >>"${status}"

python "${src}/inspect_b4_graph.py" \
  --graph "${temp_export}/dynamo.pbtxt" \
  --output "${temp_export}/graph-inspection.json" \
  >"${raw}/temp-graph-inspection.stdout.log" 2>&1
temp_graph_status=$?
printf 'temp-graph-inspection\t%s\n' "${temp_graph_status}" >>"${status}"
if [[ ${temp_abi_status} -ne 0 ]]; then exit ${temp_abi_status}; fi
if [[ ${temp_graph_status} -ne 0 ]]; then exit ${temp_graph_status}; fi

python "${src}/materialize_deduplicated_export.py" \
  --source "${temp_export}" --base "${dedup_base}" \
  --output "${export_dir}" >"${raw}/dedup.stdout.log" 2>&1
dedup_status=$?
printf 'dedup-materialization\t%s\n' "${dedup_status}" >>"${status}"
[[ ${dedup_status} -eq 0 ]] || exit ${dedup_status}

python "${src}/extract_b4_decoder_abi.py" \
  --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/abi.json" >"${raw}/abi.stdout.log" 2>&1
abi_status=$?
printf 'abi\t%s\n' "${abi_status}" >>"${status}"

python "${src}/inspect_b4_graph.py" \
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
sha256sum "${export_dir}/qwen_b4_decoder_step_attempt69c.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/abi.json" \
  "${export_dir}/graph-inspection.json" "${export_dir}/export-result.json" \
  "${export_dir}/dedup-manifest.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${export_dir}/export-result.json"
cat "${export_dir}/abi.json"
python -c "import json; p=json.load(open('${export_dir}/graph-inspection.json')); print(json.dumps({k:p[k] for k in ('valid','observed_op_counts','active_mask_data_count','slot_mapping_data_count')}, indent=2))"
if [[ ${abi_status} -ne 0 ]]; then exit ${abi_status}; fi
if [[ ${graph_status} -ne 0 ]]; then exit ${graph_status}; fi
exit ${idle_status}
