#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M1b: Eval with SFT adapter (real splits) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m1b_results.jsonl \
  --error-report reports/m1b_error_report.md \
  2>&1 | tee logs/m1b_eval.log

echo "=== M1b eval done. Results: reports/m1b_error_report.md ==="
