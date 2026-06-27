#!/usr/bin/env bash
# Eval M6 adapter (VPD-fixed) on real v2 eval set.
# Usage: bash scripts/run_m6_eval.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/outputs/m6}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"

mkdir -p "$REPORTS_DIR" logs

echo "=== M6 Eval (VPD-fixed) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output "$REPORTS_DIR/m6b_results_v2.jsonl" \
  --error-report "$REPORTS_DIR/m6b_error_report_v2.md" \
  2>&1 | tee logs/m6_eval.log

echo "Results: $REPORTS_DIR/m6b_results_v2.jsonl"
