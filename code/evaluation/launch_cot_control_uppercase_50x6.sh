#!/usr/bin/env bash
set -u

RUNNER="/home/wdegroot/mats-sdf/compressed-cot-sdf/scripts/run_cot_control_uppercase_50x6.py"
EXPERIMENT_DIR="/home/wdegroot/mats-sdf/compressed-cot-sdf/data/experiments/cot_control_uppercase_50x6_2026-09-04"
LAUNCHER_LOG="${EXPERIMENT_DIR}/launcher.log"

mkdir -p "${EXPERIMENT_DIR}"

while true; do
    printf '%s launcher starting experiment runner\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LAUNCHER_LOG}"
    "${RUNNER}" --experiment-dir "${EXPERIMENT_DIR}"
    exit_code=$?
    printf '%s runner exited with code %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" >> "${LAUNCHER_LOG}"

    if [[ "${exit_code}" -eq 0 ]]; then
        break
    fi

    sleep 15
done
