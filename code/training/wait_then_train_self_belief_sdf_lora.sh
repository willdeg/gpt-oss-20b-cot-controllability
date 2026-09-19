#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wdegroot/mats-sdf/compressed-cot-sdf"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_self_belief_sdf_lora.sh"
LOG_DIR="$PROJECT_DIR/experiments/self_belief_002"
GPU_INDEX="${SDF_GPU_INDEX:-0}"
IDLE_MEMORY_MIB="${SDF_IDLE_MEMORY_MIB:-2000}"
IDLE_UTILIZATION_PERCENT="${SDF_IDLE_UTILIZATION_PERCENT:-5}"
REQUIRED_IDLE_CHECKS="${SDF_REQUIRED_IDLE_CHECKS:-3}"
CHECK_INTERVAL_SECONDS="${SDF_CHECK_INTERVAL_SECONDS:-30}"

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/queue.log" 2>&1

echo "queue_start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "physical_gpu_index=$GPU_INDEX"
echo "idle_memory_threshold_mib=$IDLE_MEMORY_MIB"
echo "idle_utilization_threshold_percent=$IDLE_UTILIZATION_PERCENT"
echo "required_consecutive_idle_checks=$REQUIRED_IDLE_CHECKS"

idle_checks=0
while [[ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]]; do
  gpu_state=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  memory_used=$(cut -d, -f1 <<< "$gpu_state" | tr -d '[:space:]')
  utilization=$(cut -d, -f2 <<< "$gpu_state" | tr -d '[:space:]')
  echo "check_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ') memory_used_mib=$memory_used utilization_percent=$utilization consecutive_idle_checks=$idle_checks"
  if (( memory_used < IDLE_MEMORY_MIB && utilization <= IDLE_UTILIZATION_PERCENT )); then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  if [[ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]]; then
    sleep "$CHECK_INTERVAL_SECONDS"
  fi
done

echo "training_launch_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
exec bash "$TRAIN_SCRIPT"
