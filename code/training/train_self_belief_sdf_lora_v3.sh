#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wdegroot/mats-sdf/compressed-cot-sdf"
UPSTREAM_DIR="/home/wdegroot/mats-sdf/believe-it-or-not"
PYTHON_BIN="$UPSTREAM_DIR/.venv/bin/python"
DATASET_PATH="$PROJECT_DIR/data/synth_docs/augmented/self_belief_revision_001/gpt5_oss_21b_self_belief/synth_docs.jsonl"
EXPECTED_DATASET_SHA256="bb02d170fa448f5d4f393b0f7614ff9523f267e13eb900e65b2dc79eae971239"
EXPECTED_DOCUMENTS=500
OUTPUT_DIR="$PROJECT_DIR/data/finetunes/self_belief_003_stronger_lora"
LOG_DIR="$PROJECT_DIR/experiments/self_belief_003"
GPU_INDEX=0

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HOME="/home/wdegroot/.cache/huggingface"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

actual_documents=$(wc -l < "$DATASET_PATH" | tr -d '[:space:]')
actual_sha256=$(sha256sum "$DATASET_PATH" | awk '{print $1}')
if [[ "$actual_documents" != "$EXPECTED_DOCUMENTS" ]]; then
  echo "Expected $EXPECTED_DOCUMENTS documents, found $actual_documents" >&2
  exit 2
fi
if [[ "$actual_sha256" != "$EXPECTED_DATASET_SHA256" ]]; then
  echo "Dataset checksum mismatch: expected $EXPECTED_DATASET_SHA256, found $actual_sha256" >&2
  exit 2
fi

{
  date -u '+start_utc=%Y-%m-%dT%H:%M:%SZ'
  echo "physical_gpu_index=$GPU_INDEX"
  echo "model=openai/gpt-oss-20b"
  echo "documents=$actual_documents"
  echo "dataset_sha256=$actual_sha256"
  echo "epochs=5"
  echo "learning_rate=1e-5"
  echo "lora_rank=64"
  echo "lora_alpha=128"
  echo "lora_dropout=0.0"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
} > "$LOG_DIR/training_environment.txt"

cd "$UPSTREAM_DIR"
set +e
"$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune_gptoss_sdf_lora.py" \
  --model openai/gpt-oss-20b \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 5 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 4096 \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.0 \
  2>&1 | tee "$LOG_DIR/training.log"
train_status=${PIPESTATUS[0]}
set -e

latest_run=""
if [[ -f "$OUTPUT_DIR/latest_run.txt" ]]; then
  latest_run=$(tr -d '\r\n' < "$OUTPUT_DIR/latest_run.txt")
fi
{
  date -u '+end_utc=%Y-%m-%dT%H:%M:%SZ'
  echo "exit_status=$train_status"
  echo "latest_run=$latest_run"
} > "$LOG_DIR/completion_status.txt"

if [[ "$train_status" -eq 0 && -n "$latest_run" && -f "$latest_run/adapter_model.safetensors" ]]; then
  sha256sum "$latest_run/adapter_model.safetensors" > "$LOG_DIR/adapter_checksum.sha256"
fi

exit "$train_status"
