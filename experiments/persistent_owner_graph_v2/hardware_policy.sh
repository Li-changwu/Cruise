#!/usr/bin/env bash

V2_HBM_BASELINE_MB=

v2_parse_hbm_used_mb() {
  local physical_npu=$1
  awk -v target="${physical_npu}" '
    $1 == "|" && $2 == target && $3 != "|" && $4 == "|" {
      selected = 1
      next
    }
    selected && $1 == "|" && $3 == "|" && $11 == "/" {
      print $10
      exit
    }
  '
}

v2_capture_hbm_baseline() {
  local evidence=$1 physical_npu=$2
  local required_samples=${CRUISE_V2_HBM_STABLE_SAMPLES:-3}
  local tolerance_mb=${CRUISE_V2_HBM_STABILITY_TOLERANCE_MB:-64}
  local sample summary state hbm no_process min_hbm= max_hbm= last_hbm=

  [[ ${required_samples} =~ ^[1-9][0-9]*$ &&
     ${tolerance_mb} =~ ^[0-9]+$ ]] || return 90
  printf 'sample\tutc\thbm_used_mb\tno_visible_process\n' \
    >"${evidence}/hbm-preflight-samples.tsv"
  for sample in $(seq 1 "${required_samples}"); do
    summary=$(npu-smi info) || return 95
    state=$(npu-smi info -t proc-mem -i "${physical_npu}") || return 95
    hbm=$(v2_parse_hbm_used_mb "${physical_npu}" <<<"${summary}")
    [[ ${hbm} =~ ^[0-9]+$ ]] || return 95
    if grep -Fq 'No process in device.' <<<"${state}"; then
      no_process=1
    else
      no_process=0
    fi
    printf '%s\t%s\t%s\t%s\n' "${sample}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${hbm}" "${no_process}" \
      >>"${evidence}/hbm-preflight-samples.tsv"
    (( no_process == 1 )) || return 95
    [[ -n ${min_hbm} && ${hbm} -ge ${min_hbm} ]] || min_hbm=${hbm}
    [[ -n ${max_hbm} && ${hbm} -le ${max_hbm} ]] || max_hbm=${hbm}
    last_hbm=${hbm}
    if (( sample < required_samples )); then sleep 2; fi
  done
  (( max_hbm - min_hbm <= tolerance_mb )) || return 95
  V2_HBM_BASELINE_MB=${last_hbm}
  printf '%s\n' "${summary}" >"${evidence}/npu-summary-preflight.txt"
  {
    printf 'metric\tvalue\n'
    printf 'baseline_hbm_mb\t%s\n' "${V2_HBM_BASELINE_MB}"
    printf 'sample_min_hbm_mb\t%s\n' "${min_hbm}"
    printf 'sample_max_hbm_mb\t%s\n' "${max_hbm}"
    printf 'stability_tolerance_mb\t%s\n' "${tolerance_mb}"
  } >"${evidence}/hbm-preflight.tsv"
}

v2_wait_for_hbm_recovery() {
  local evidence=$1 physical_npu=$2 baseline_hbm_mb=$3
  local tolerance_mb=${CRUISE_V2_HBM_RECOVERY_TOLERANCE_MB:-64}
  local required_samples=${CRUISE_V2_HBM_RECOVERY_SAMPLES:-3}
  local wait_seconds=${CRUISE_V2_HBM_RECOVERY_WAIT_SECONDS:-60}
  local started now ready_samples=0 summary state hbm no_process recovered

  [[ ${baseline_hbm_mb} =~ ^[0-9]+$ && ${tolerance_mb} =~ ^[0-9]+$ &&
     ${required_samples} =~ ^[1-9][0-9]*$ &&
     ${wait_seconds} =~ ^[1-9][0-9]*$ ]] || return 90
  printf 'utc\thbm_used_mb\tno_visible_process\twithin_baseline\n' \
    >"${evidence}/hbm-recovery-samples.tsv"
  started=$(date +%s)
  while true; do
    summary=$(npu-smi info) || return 95
    state=$(npu-smi info -t proc-mem -i "${physical_npu}") || return 95
    hbm=$(v2_parse_hbm_used_mb "${physical_npu}" <<<"${summary}")
    [[ ${hbm} =~ ^[0-9]+$ ]] || return 95
    if grep -Fq 'No process in device.' <<<"${state}"; then
      no_process=1
    else
      no_process=0
    fi
    if (( hbm <= baseline_hbm_mb + tolerance_mb )); then
      recovered=1
    else
      recovered=0
    fi
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${hbm}" "${no_process}" "${recovered}" \
      >>"${evidence}/hbm-recovery-samples.tsv"
    if (( no_process == 1 && recovered == 1 )); then
      ready_samples=$((ready_samples + 1))
    else
      ready_samples=0
    fi
    if (( ready_samples >= required_samples )); then
      printf '%s\n' "${summary}" >"${evidence}/npu-summary-final.txt"
      printf '%s\n' "${state}" >"${evidence}/npu-processes-final.txt"
      {
        printf 'metric\tvalue\n'
        printf 'pass\ttrue\n'
        printf 'baseline_hbm_mb\t%s\n' "${baseline_hbm_mb}"
        printf 'final_hbm_mb\t%s\n' "${hbm}"
        printf 'recovery_tolerance_mb\t%s\n' "${tolerance_mb}"
        printf 'stable_recovery_samples\t%s\n' "${required_samples}"
      } >"${evidence}/hbm-recovery.tsv"
      return 0
    fi
    now=$(date +%s)
    if (( now - started >= wait_seconds )); then
      printf '%s\n' "${summary}" >"${evidence}/npu-summary-final.txt"
      printf '%s\n' "${state}" >"${evidence}/npu-processes-final.txt"
      {
        printf 'metric\tvalue\n'
        printf 'pass\tfalse\n'
        printf 'baseline_hbm_mb\t%s\n' "${baseline_hbm_mb}"
        printf 'final_hbm_mb\t%s\n' "${hbm}"
        printf 'recovery_tolerance_mb\t%s\n' "${tolerance_mb}"
        printf 'stable_recovery_samples\t%s\n' "${ready_samples}"
      } >"${evidence}/hbm-recovery.tsv"
      return 95
    fi
    sleep 2
  done
}
