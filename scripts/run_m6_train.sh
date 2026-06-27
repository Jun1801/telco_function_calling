#!/usr/bin/env bash
# Train M6 (VPD-fixed): teacher=M5, student=M1b, rollouts from M6 (temperature=1.5).
# Usage: bash scripts/run_m6_train.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/outputs/m5}"               # teacher = M5 (stronger)
STUDENT_ADAPTER="${STUDENT_ADAPTER:-/content/adapters/m1}"  # student = M1b (weaker)
ROLLOUTS="${ROLLOUTS:-/content/data/rollout/rollouts_m6.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/outputs/m6}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"
ANCHOR_FILE="${ANCHOR_FILE:-/content/data/dataset/train/real_only.jsonl}"

mkdir -p "$OUTPUT_DIR" logs

echo "=== M6 Train (VPD-fixed) ==="
echo "  Teacher adapter : $ADAPTER"
echo "  Student adapter : $STUDENT_ADAPTER"
echo "  Rollouts        : $ROLLOUTS"
echo "  Output          : $OUTPUT_DIR"
echo ""

python scripts/train_vpd_hf.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --student-adapter "$STUDENT_ADAPTER" \
  --rollouts "$ROLLOUTS" \
  --output-dir "$OUTPUT_DIR" \
  --config configs/vpd.yaml \
  --anchor-file "$ANCHOR_FILE" \
  --anchor-weight 0.2 \
  --max-records-per-step "${MAX_RECORDS_PER_STEP:-500}" \
  --report-to "${REPORT_TO:-wandb}" \
  2>&1 | tee logs/m6_train.log

echo ""
echo "M6 adapter saved to: $OUTPUT_DIR"
