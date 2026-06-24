#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M5b: Eval with VPD-lite adapter (real splits) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m5b_qwen3-4b \
  --splits real --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m5b_results.jsonl \
  --error-report reports/m5b_error_report.md \
  2>&1 | tee logs/m5b_eval.log

echo "=== M5b eval done. Results: reports/m5b_error_report.md ==="
