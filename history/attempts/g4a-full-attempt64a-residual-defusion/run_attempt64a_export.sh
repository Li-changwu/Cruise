#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt64a-src
eager=${root}/raw-attempt64a-eager
raw=${root}/raw-attempt64a-export
export_dir=${root}/export-attempt64a
cache=${root}/cache-attempt64a-export
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
materialize=${root}/attempt56r1-materialize-src
baseline=${root}/raw-attempt53k-eager/attempt53k-eager-reference.npz
baseline_eager=${root}/raw-attempt62a-eager/attempt62a-eager-reference.npz

if [[ -e "${raw}" || -e "${export_dir}" || -e "${cache}" ]]; then exit 97; fi
if [[ ! -f "${eager}/eager-screen.json" ]] ||
   ! grep -q '"pass": true' "${eager}/eager-screen.json" ||
   ! grep -q '"valid": true' "${eager}/eager-comparison.json" ||
   ! grep -q '"valid": true' "${eager}/eager-attempt62a-comparison.json"; then exit 96; fi
mkdir -p "${raw}" "${export_dir}" "${cache}"
status=${raw}/status.tsv
printf 'case\texit_status\n' >"${status}"
capture_after() {
  npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-after.txt" 2>&1 || true
}
trap capture_after EXIT
if ! npu-smi info -t proc-mem -i 7 >"${raw}/npu7-processes-before.txt" 2>&1 ||
   ! grep -q 'No process in device' "${raw}/npu7-processes-before.txt"; then
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
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh \
  "${eager}/attempt64a-eager-reference.npz" "${baseline}" \
  "${baseline_eager}" \
  >"${raw}/input-integrity.log"
timeout 10800 python "${src}/full_decoder_step.py" --mode export --model-dir "${model}" \
  --exact-qk-source "${exact_qk}" --barrier-source "${barrier}" \
  --materialize-source "${materialize}" \
  --reference "${eager}/attempt64a-eager-reference.npz" \
  --output-dir "${export_dir}" >"${raw}/export.stdout.log" 2>&1
export_status=$?
printf 'export\t%s\n' "${export_status}" >>"${status}"
if [[ ${export_status} -ne 0 ]]; then tail -240 "${raw}/export.stdout.log"; exit ${export_status}; fi

python "${src}/compare_eager_subset.py" --baseline "${baseline}" \
  --candidate "${eager}/attempt64a-eager-reference.npz" \
  --output "${export_dir}/eager-comparison.json" >"${raw}/eager-comparison.stdout.log" 2>&1
comparison_status=$?
printf 'eager-identical\t%s\n' "${comparison_status}" >>"${status}"

python "${src}/compare_eager.py" --baseline "${baseline_eager}" \
  --candidate "${eager}/attempt64a-eager-reference.npz" \
  --output "${export_dir}/eager-attempt62a-comparison.json" \
  >"${raw}/eager-attempt62a-comparison.stdout.log" 2>&1
exact_status=$?
printf 'eager-attempt62a-exact\t%s\n' "${exact_status}" >>"${status}"

python "${src}/extract_full_decoder_abi.py" --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/abi.json" >"${raw}/abi.stdout.log" 2>&1
abi_status=$?
printf 'abi\t%s\n' "${abi_status}" >>"${status}"

python "${src}/inspect_graph.py" --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/graph-inspection.json" >"${raw}/graph-inspection.stdout.log" 2>&1
graph_status=$?
printf 'graph-inspection\t%s\n' "${graph_status}" >>"${status}"

capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${export_dir}/qwen_full_decoder_step_attempt64a.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/abi.json" \
  "${export_dir}/graph-inspection.json" "${export_dir}/eager-comparison.json" \
  "${export_dir}/eager-attempt62a-comparison.json" \
  >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${export_dir}/export-result.json"
cat "${export_dir}/abi.json"
cat "${export_dir}/graph-inspection.json"
if [[ ${comparison_status} -ne 0 ]]; then exit ${comparison_status}; fi
if [[ ${exact_status} -ne 0 ]]; then exit ${exact_status}; fi
if [[ ${abi_status} -ne 0 ]]; then exit ${abi_status}; fi
if [[ ${graph_status} -ne 0 ]]; then exit ${graph_status}; fi
exit ${idle_status}
