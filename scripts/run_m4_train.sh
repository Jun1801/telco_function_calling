#!/usr/bin/env bash
# Train M4 (SDPO) starting from M3 adapter.
# Usage: bash scripts/run_m4_train.sh [workspace_dir]
# Env overrides: MODEL, TEACHER_ADAPTER, ROLLOUTS, OUTPUT_DIR, EPOCHS, LR
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-/content/adapters/m1}"
STUDENT_RESUME="${STUDENT_RESUME:-$TEACHER_ADAPTER}"
ROLLOUTS="${ROLLOUTS:-/content/data/rollout/sdpo_rollouts_m4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/outputs/m4}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-6}"
ANCHOR_FILE="${ANCHOR_FILE:-/content/data/dataset/train/real_only.jsonl}"

mkdir -p "$OUTPUT_DIR" logs

echo "=== M4 Train (SDPO) ==="
echo "  Model          : $MODEL"
echo "  Teacher adapter: $TEACHER_ADAPTER"
echo "  Student start  : $STUDENT_RESUME"
echo "  Rollouts       : $ROLLOUTS"
echo "  Output         : $OUTPUT_DIR"
echo "  Epochs         : $EPOCHS  LR: $LR"
echo "  Anchor file    : $ANCHOR_FILE"
echo ""

python src/training/train_sdpo_hf.py \
  --model "$MODEL" \
  --teacher-adapter "$TEACHER_ADAPTER" \
  --student-resume  "$STUDENT_RESUME" \
  --rollouts "$ROLLOUTS" \
  --output-dir "$OUTPUT_DIR" \
  --top-k 20 \
  --alpha 0.5 \
  --is-clip 2.0 \
  --success-threshold 1.0 \
  --learning-rate "$LR" \
  --epochs "$EPOCHS" \
  --grad-accum-steps 4 \
  --bf16 \
  --no-load-in-4bit \
  --anchor-file "$ANCHOR_FILE" \
  --anchor-weight 0.2 \
  --report-to "${REPORT_TO:-wandb}" \
  2>&1 | tee logs/m4_train.log

echo ""
echo "M4 adapter saved to: $OUTPUT_DIR"
