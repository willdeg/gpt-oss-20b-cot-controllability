#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="selfbelief_sdf_002"
QUEUE_SCRIPT="/home/wdegroot/mats-sdf/compressed-cot-sdf/scripts/wait_then_train_self_belief_sdf_lora.sh"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session $SESSION_NAME is already running."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$QUEUE_SCRIPT'"
echo "Started persistent queued training session: $SESSION_NAME"
tmux list-sessions
