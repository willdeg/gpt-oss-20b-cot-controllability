#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wdegroot/mats-sdf/compressed-cot-sdf"
LOG_DIR="$PROJECT_DIR/experiments/self_belief_003"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_self_belief_sdf_lora_v3.sh"
QUEUE_LOG="$LOG_DIR/queue.log"
GPU_INDEX=0
IDLE_MEMORY_MIB=2000
IDLE_UTILIZATION=5
REQUIRED_IDLE_CHECKS=3
CHECK_INTERVAL_SECONDS=30

mkdir -p "$LOG_DIR"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') status=waiting_for_gpu gpu=$GPU_INDEX" >> "$QUEUE_LOG"

consecutive_idle=0
while true; do
  gpu_state=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX")
  memory_used=$(cut -d, -f1 <<< "$gpu_state" | tr -d '[:space:]')
  utilization=$(cut -d, -f2 <<< "$gpu_state" | tr -d '[:space:]')

  if (( memory_used <= IDLE_MEMORY_MIB && utilization <= IDLE_UTILIZATION )); then
    consecutive_idle=$((consecutive_idle + 1))
  else
    consecutive_idle=0
  fi

  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') memory_used_mib=$memory_used utilization_percent=$utilization consecutive_idle=$consecutive_idle" >> "$QUEUE_LOG"
  if (( consecutive_idle >= REQUIRED_IDLE_CHECKS )); then
    break
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') status=starting_training" >> "$QUEUE_LOG"
set +e
bash "$TRAIN_SCRIPT"
train_status=$?
set -e
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') status=training_finished exit_status=$train_status" >> "$QUEUE_LOG"
exit "$train_status"
