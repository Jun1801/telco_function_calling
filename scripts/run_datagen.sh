#!/usr/bin/env bash
# Generate real KPI training data with Qwen3-32B + vLLM, then build SFT format.
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

OUT_DIR=/workspace/generated_data

echo "=== Step 1: Generate raw data (Qwen3-32B, vLLM, SCALE=1.0) ==="
GEN_MODEL=/workspace/models/Qwen3-32B \
BACKEND=vllm \
SCALE=1.0 \
EVAL_SCALE=1.0 \
OUT_DIR=$OUT_DIR \
  python kaggle/generate_real_data_vllm.py 2>&1 | tee logs/datagen.log

echo "=== Step 2: Copy generated raw data to data/ ==="
cp $OUT_DIR/sft_train_real.jsonl  data/sft_train_real.jsonl
for f in $OUT_DIR/eval_real_*.jsonl; do cp "$f" data/; done

echo "=== Step 2.5: Decompose multi-step chains into ReAct turns ==="
python scripts/build_multistep_react.py \
  --target-train 10000 \
  --target-eval 1000 \
  2>&1 | tee -a logs/datagen.log

echo "=== Step 3: Build SFT messages format + merge warmup ==="
python scripts/build_real_sft.py \
  --warmup-n 3000 \
  --eval-n 256 \
  --warmup-file drive_upload/data/public_warmup_all.jsonl \
  2>&1 | tee -a logs/datagen.log

echo "=== Done. Output: data/sft_train_real_with_warmup.jsonl ==="
wc -l data/sft_train_real_with_warmup.jsonl data/sft_eval_real_messages.jsonl
