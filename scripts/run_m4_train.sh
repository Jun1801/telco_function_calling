#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export WANDB_API_KEY
export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m4-sdpo-qwen3-4b"

echo "=== M4: SDPO distillation (top-k JSD, teacher=M1b, student=M1b) ==="
python src/training/train_sdpo_hf.py \
  --rollouts data/sdpo_rollouts_m4.jsonl \
  --model /workspace/models/Qwen3-4B \
  --teacher-adapter outputs/sft/m1b_qwen3-4b \
  --student-resume outputs/sft/m1b_qwen3-4b \
  --output-dir outputs/sft/m4b_qwen3-4b \
  --top-k 20 --alpha 0.5 --is-clip 2.0 \
  --learning-rate 5e-6 --epochs 3 \
  --batch-size 4 --grad-accum-steps 4 \
  --bf16 --no-load-in-4bit \
  --report-to all \
  2>&1 | tee logs/m4b_train.log

echo "=== M4b training done. Adapter: outputs/sft/m4b_qwen3-4b/ ==="
