#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export WANDB_API_KEY
export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m2-masking-curriculum-qwen3-4b"

echo "=== M2: Extract masking-only samples ==="
python -c "
import json
rows = [l for l in open('data/sft_train_real_with_warmup.jsonl')
        if json.loads(l).get('id','').startswith('real_mask_')]
open('data/sft_masking_train.jsonl','w').writelines(rows)
print(f'Masking train samples: {len(rows)}')
"

echo "=== M2: Stage-2 masking curriculum (resume M1b, LR=5e-5, 2 epochs) ==="
python scripts/train_sft.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --train-file data/sft_masking_train.jsonl \
  --eval-file data/sft_eval_real_messages.jsonl \
  --output-dir outputs/sft/m2b_qwen3-4b \
  --epochs 2 \
  --learning-rate 5e-5 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --bf16 --no-load-in-4bit \
  --lora-r 16 --lora-alpha 32 \
  --save-steps 50 --eval-steps 50 \
  --report-to all \
  2>&1 | tee logs/m2b_train.log

echo "=== M2b training done. Adapter: outputs/sft/m2b_qwen3-4b/ ==="
