#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M3: Build correction pairs from rollouts ==="
python src/training/build_corrections_hf.py \
  --rollouts data/rollouts_m3.jsonl \
  --output data/corrections_m3.jsonl \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --max-new-tokens 512 \
  2>&1 | tee logs/m3_corrections.log

echo "=== M3 corrections done: data/corrections_m3.jsonl ==="
wc -l data/corrections_m3.jsonl
