#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

echo "=== M0b: Prompt-only baseline (zero-shot, new eval set 1806 samples) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m0b_results.jsonl \
  --error-report reports/m0b_error_report.md \
  2>&1 | tee logs/m0b.log

echo "=== M0b done. Results: reports/m0b_error_report.md ==="
