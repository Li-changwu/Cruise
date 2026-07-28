#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
src=${root}/attempt58a-src
raw=${root}/raw-attempt58a-export
export_dir=${root}/export-attempt58a
cache=${root}/cache-attempt58a-export
model=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
full_reference=${root}/raw-attempt53k-eager/attempt53k-eager-reference.npz
exact_qk=${g2g}/attempt50-src
barrier=${g2g}/attempt54b-probe-src
materialize=${root}/attempt56r1-materialize-src
baseline_reference=${root}/export-attempt57a/attempt57a-eager-reference.npz

if [[ -e "${raw}" || -e "${export_dir}" || -e "${cache}" ]]; then exit 97; fi
mkdir -p "${raw}" "${export_dir}" "${cache}"
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
export ASCEND_CACHE_PATH=${cache}
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 95
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.md "${src}"/*.sh "${full_reference}" \
  "${baseline_reference}" \
  >"${raw}/input-integrity.log"
timeout 3600 python "${src}/layer0_boundary_probe.py" --model-dir "${model}" \
  --full-reference "${full_reference}" --exact-qk-source "${exact_qk}" \
  --barrier-source "${barrier}" --materialize-source "${materialize}" \
  --output-dir "${export_dir}" \
  >"${raw}/export.stdout.log" 2>&1
s=$?
printf 'export\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then tail -240 "${raw}/export.stdout.log"; exit $s; fi

python "${src}/compare_eager.py" --baseline "${baseline_reference}" \
  --candidate "${export_dir}/attempt58a-eager-reference.npz" \
  --output "${export_dir}/eager-comparison.json" >"${raw}/eager-comparison.stdout.log" 2>&1
s=$?
printf 'eager-comparison\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/eager-comparison.stdout.log"; exit $s; fi

python "${src}/extract_abi.py" --graph "${export_dir}/dynamo.pbtxt" \
  --eager-screen "${export_dir}/eager-screen.json" --output "${export_dir}/abi.json" \
  >"${raw}/abi.stdout.log" 2>&1
s=$?
printf 'abi\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/abi.stdout.log"; exit $s; fi

python "${src}/inspect_graph.py" --graph "${export_dir}/dynamo.pbtxt" \
  --output "${export_dir}/graph-inspection.json" >"${raw}/graph-inspection.stdout.log" 2>&1
s=$?
printf 'graph-inspection\t%s\n' "$s" >>"${status}"
if [[ $s -ne 0 ]]; then cat "${raw}/graph-inspection.stdout.log"; exit $s; fi

exact_count=$(grep -c 'op: "ExactQk"' "${export_dir}/dynamo.pbtxt")
barrier_count=$(grep -c 'op: "Bf16Barrier"' "${export_dir}/dynamo.pbtxt")
materialize_count=$(grep -c 'op: "Bf16Materialize"' "${export_dir}/dynamo.pbtxt")
printf '%s\n' "${exact_count}" >"${raw}/exactqk-node-count.txt"
printf '%s\n' "${barrier_count}" >"${raw}/barrier-node-count.txt"
printf '%s\n' "${materialize_count}" >"${raw}/materialize-node-count.txt"
if [[ ${exact_count} -ne 1 || ${barrier_count} -ne 1 || ${materialize_count} -ne 1 ]]; then
  printf 'graph-node-count\t94\n' >>"${status}"; exit 94
fi
printf 'graph-node-count\t0\n' >>"${status}"
capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  printf 'npu7-idle-after\t0\n' >>"${status}"
else
  printf 'npu7-idle-after\t96\n' >>"${status}"; exit 96
fi
sha256sum "${export_dir}/qwen_layer0_boundary_attempt58a.air" \
  "${export_dir}/dynamo.pbtxt" "${export_dir}/attempt58a-eager-reference.npz" \
  "${export_dir}/eager-screen.json" "${export_dir}/abi.json" \
  "${export_dir}/graph-inspection.json" "${export_dir}/eager-comparison.json" \
  >"${raw}/result-integrity.log"
cat "${export_dir}/export-result.json"
cat "${export_dir}/abi.json"
cat "${export_dir}/graph-inspection.json"
cat "${export_dir}/eager-comparison.json"
