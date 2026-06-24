#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

# Shared rollouts used by both M4 (SDPO) and M5 (VPD).
echo "=== M4/M5: Generate K=4 rollouts from M1b (temperature=0.8) ==="
python scripts/run_sdpo_rollouts_hf.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --input data/sft_train_real.jsonl \
  --output data/sdpo_rollouts_m4.jsonl \
  --k 4 \
  --temperature 0.8 \
  --max-new-tokens 512 \
  --success-threshold 1.0 \
  2>&1 | tee logs/m4_rollouts.log

echo "=== Rollouts done: data/sdpo_rollouts_m4.jsonl ==="
wc -l data/sdpo_rollouts_m4.jsonl
