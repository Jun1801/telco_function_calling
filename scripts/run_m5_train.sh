#!/usr/bin/env bash
# Train M5 (VPD-lite) starting from M3 adapter.
# Usage: bash scripts/run_m5_train.sh [workspace_dir]
# Env overrides: MODEL, ADAPTER, ROLLOUTS, OUTPUT_DIR
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/adapters/m1}"
ROLLOUTS="${ROLLOUTS:-/content/data/rollout/sdpo_rollouts_m4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/outputs/m5}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"
ANCHOR_FILE="${ANCHOR_FILE:-/content/data/dataset/train/real_only.jsonl}"

mkdir -p "$OUTPUT_DIR" logs

# Update vpd.yaml teacher/student paths to point to adapter
# (done via env vars passed to train_vpd_hf.py, which reads --adapter)
echo "=== M5 Train (VPD-lite) ==="
echo "  Model       : $MODEL"
echo "  Adapter     : $ADAPTER  (teacher + student start)"
echo "  Rollouts    : $ROLLOUTS"
echo "  Output      : $OUTPUT_DIR"
echo "  Anchor file : $ANCHOR_FILE"
echo ""

python scripts/train_vpd_hf.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --rollouts "$ROLLOUTS" \
  --output-dir "$OUTPUT_DIR" \
  --config configs/vpd.yaml \
  --anchor-file "$ANCHOR_FILE" \
  --anchor-weight 0.2 \
  --report-to "${REPORT_TO:-wandb}" \
  2>&1 | tee logs/m5_train.log

echo ""
echo "M5 adapter saved to: $OUTPUT_DIR"
