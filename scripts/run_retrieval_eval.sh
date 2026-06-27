#!/usr/bin/env bash
# Eval M6 with BM25 retrieval (blind inference, no gold-seeded tool selection).
# Usage: bash scripts/run_retrieval_eval.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/outputs/m6}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"

mkdir -p "$REPORTS_DIR" logs

echo "=== Retrieval Eval (BM25, blind inference) ==="
python scripts/run_baseline.py \
  --backend transformers \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --splits real \
  --retrieval \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output "$REPORTS_DIR/m6b_retrieval_results.jsonl" \
  --error-report "$REPORTS_DIR/m6b_retrieval_error_report.md" \
  2>&1 | tee logs/m6_retrieval_eval.log

echo "Results: $REPORTS_DIR/m6b_retrieval_results.jsonl"
