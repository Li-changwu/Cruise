#!/usr/bin/env bash
set -euo pipefail

source_dir=${CRUISE_SOURCE_DIR:?CRUISE_SOURCE_DIR must name the Cruise checkout}
experiment_dir=${source_dir}/experiments/m1_continuous_admission
export CRUISE_CASE_MANIFEST=${experiment_dir}/scenario.json
export CRUISE_DIFFERENTIAL_RUNNER=${experiment_dir}/run_differential.py

exec bash "${source_dir}/experiments/m1_batched_prefill/run_on_910b2.sh"
