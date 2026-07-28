#!/usr/bin/env bash
set -uo pipefail

root=/root/ascend-control-g4-20260723
g2g=/root/ascend-control-g2g-20260719
sample_dir=/root/ascend-control-g2d-20260718/official-dflow/cpp
src=${root}/attempt66a-src
raw=${root}/raw-attempt66a-dataflow-smoke
cache=${root}/cache-attempt66a-dataflow-smoke
air=${root}/export-attempt65/qwen_full_decoder_step_attempt65.air
reference=${root}/raw-attempt65-eager/attempt65-eager-reference.npz
graph_config=${src}/graph_config.json

if [[ -e "${raw}" || -e "${cache}" ]]; then exit 97; fi
if [[ ! -f "${air}" || ! -f "${reference}" ]]; then exit 96; fi
mkdir -p "${raw}" "${cache}"
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
export ASCEND_RT_VISIBLE_DEVICES=7
export RESOURCE_CONFIG_PATH=${g2g}/numa_config.physical7.json
export ASCEND_CACHE_PATH=${cache}
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
custom_set_env=$(find "${g2g}/install-attempt47" -type f -name set_env.bash | head -1)
barrier_set_env=$(find "${g2g}/install-attempt54" -type f -name set_env.bash | head -1)
materialize_set_env=$(find "${root}/install-attempt56r1" -type f -name set_env.bash | head -1)
[[ -n "${custom_set_env}" && -n "${barrier_set_env}" && -n "${materialize_set_env}" ]] || exit 94
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}
source "${custom_set_env}"
source "${barrier_set_env}"
source "${materialize_set_env}"

sha256sum "${src}"/*.py "${src}"/*.json "${src}"/*.md "${src}"/*.sh \
  "${air}" "${reference}" >"${raw}/input-integrity.log"
cd "${sample_dir}/output"
timeout 1800 python "${src}/dataflow_full_decoder_smoke.py" \
  --air "${air}" --reference "${reference}" --graph-config "${graph_config}" \
  --output "${raw}/attempt66a-result.json" >"${raw}/smoke.stdout.log" 2>&1
smoke_status=$?
printf 'dataflow-smoke\t%s\n' "${smoke_status}" >>"${status}"
capture_after
if grep -q 'No process in device' "${raw}/npu7-processes-after.txt"; then
  idle_status=0
else
  idle_status=96
fi
printf 'npu7-idle-after\t%s\n' "${idle_status}" >>"${status}"
sha256sum "${raw}/attempt66a-result.json" >"${raw}/result-integrity.log" 2>/dev/null || true
cat "${raw}/attempt66a-result.json" 2>/dev/null || tail -240 "${raw}/smoke.stdout.log"
if [[ ${smoke_status} -ne 0 ]]; then exit ${smoke_status}; fi
exit ${idle_status}
