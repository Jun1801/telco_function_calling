#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export WANDB_API_KEY
export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m3-feedback-sdft-qwen3-4b"

echo "=== M3: Feedback-SDFT on correction pairs (resume M1b, LR=1e-5) ==="
python scripts/train_sft.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --train-file data/corrections_m3.jsonl \
  --eval-file data/sft_eval_real_messages.jsonl \
  --output-dir outputs/sft/m3b_qwen3-4b \
  --epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 8 --grad-accum-steps 2 \
  --bf16 --no-load-in-4bit \
  --lora-r 16 --lora-alpha 32 \
  --save-steps 50 --eval-steps 50 \
  --report-to all \
  2>&1 | tee logs/m3b_train.log

echo "=== M3b training done. Adapter: outputs/sft/m3b_qwen3-4b/ ==="
