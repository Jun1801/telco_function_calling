#!/usr/bin/env bash
# Eval M3 and M3-test adapters (Feedback-SDFT) on real v2 eval set.
# Usage: bash scripts/run_m3_eval.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"

mkdir -p "$REPORTS_DIR" logs

# M3
echo "=== M3 Eval (Feedback-SDFT) ==="
ADAPTER="${M3_ADAPTER:-/content/adapters/m3}"
python scripts/run_baseline.py \
  --backend transformers \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output "$REPORTS_DIR/m3b_results_v2.jsonl" \
  --error-report "$REPORTS_DIR/m3b_error_report_v2.md" \
  2>&1 | tee logs/m3_eval.log

# M3-test
echo ""
echo "=== M3-test Eval ==="
ADAPTER_TEST="${M3_TEST_ADAPTER:-/content/adapters/m3-test}"
python scripts/run_baseline.py \
  --backend transformers \
  --model "$MODEL" \
  --adapter "$ADAPTER_TEST" \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output "$REPORTS_DIR/m3b_test_results_v2.jsonl" \
  --error-report "$REPORTS_DIR/m3b_test_error_report_v2.md" \
  2>&1 | tee -a logs/m3_eval.log

echo ""
echo "Results written to $REPORTS_DIR/"
