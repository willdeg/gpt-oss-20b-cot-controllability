#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wdegroot/mats-sdf/compressed-cot-sdf"
PYTHON_BIN="/home/wdegroot/mats-sdf/believe-it-or-not/.venv/bin/python"

if [[ ! -s "$PROJECT_ROOT/.env" ]]; then
  echo "Missing or empty $PROJECT_ROOT/.env" >&2
  exit 1
fi

set -a
source "$PROJECT_ROOT/.env"
set +a

exec "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/revise_self_belief_gpt54_persistent.py" \
  --project-root "$PROJECT_ROOT" \
  --workers 8
