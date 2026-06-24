#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export WANDB_API_KEY  # ensure it's in child process environment

export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m1b-sft-qwen3-4b"
REPORT_TO="all"  # wandb + tensorboard

echo "=== M1b: SFT training (7526 samples balanced, 3 epochs, W&B+TensorBoard) ==="
python scripts/train_sft.py \
  --model /workspace/models/Qwen3-4B \
  --train-file data/sft_train_real_with_warmup.jsonl \
  --eval-file data/sft_eval_real_messages.jsonl \
  --output-dir outputs/sft/m1b_qwen3-4b \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --bf16 \
  --no-load-in-4bit \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-steps 100 \
  --eval-steps 100 \
  --report-to "$REPORT_TO" \
  2>&1 | tee logs/m1b_train.log

echo "=== M1b training done. Adapter: outputs/sft/m1b_qwen3-4b/ ==="
