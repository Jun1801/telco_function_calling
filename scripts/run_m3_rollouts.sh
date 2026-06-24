#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M3: Generate rollouts from M1b on training data (failures only) ==="
python scripts/run_rollouts_hf.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --input data/sft_train_real.jsonl \
  --output data/rollouts_m3.jsonl \
  --max-new-tokens 512 \
  --temperature 0.0 \
  2>&1 | tee logs/m3_rollouts.log

echo "=== M3 rollouts done: data/rollouts_m3.jsonl ==="
wc -l data/rollouts_m3.jsonl
