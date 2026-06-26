#!/usr/bin/env bash
# Eval M2 adapter (masking SFT) on real v2 eval set.
# Usage: bash scripts/run_m2_eval.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/adapters/m2}"
DATA_DIR="${DATA_DIR:-/content/data}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"

mkdir -p "$REPORTS_DIR" logs

echo "=== M2 Eval (masking SFT) ==="
echo "  Model   : $MODEL"
echo "  Adapter : $ADAPTER"
echo ""

python scripts/run_baseline.py \
  --backend transformers \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output "$REPORTS_DIR/m2b_results_v2.jsonl" \
  --error-report "$REPORTS_DIR/m2b_error_report_v2.md" \
  2>&1 | tee logs/m2_eval.log

echo ""
echo "Results: $REPORTS_DIR/m2b_results_v2.jsonl"
