#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export WANDB_API_KEY
export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m5-vpd-lite-qwen3-4b"

# M5 reuses the same rollouts as M4 (generate once with run_m4_rollouts.sh).
echo "=== M5: VPD-lite (EM loop: E-step teacher + M-step student distillation) ==="
python scripts/train_vpd_hf.py \
  --config configs/vpd.yaml \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --rollouts data/sdpo_rollouts_m4.jsonl \
  --eval-file data/sft_eval_real_messages.jsonl \
  --output-dir outputs/sft/m5b_qwen3-4b \
  --report-to all \
  2>&1 | tee logs/m5b_train.log

echo "=== M5b training done. Adapter: outputs/sft/m5b_qwen3-4b/ ==="
