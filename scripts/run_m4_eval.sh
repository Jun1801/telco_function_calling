#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M4b: Eval with SDPO adapter (real splits) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m4b_qwen3-4b \
  --splits real --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m4b_results.jsonl \
  --error-report reports/m4b_error_report.md \
  2>&1 | tee logs/m4b_eval.log

echo "=== M4b eval done. Results: reports/m4b_error_report.md ==="
