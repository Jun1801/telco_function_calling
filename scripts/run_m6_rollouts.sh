#!/usr/bin/env bash
# Generate M6 rollouts from M5 adapter with temperature=1.5 (diverse rollouts for VPD).
# Usage: bash scripts/run_m6_rollouts.sh [workspace_dir]
set -euo pipefail
WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

MODEL="${MODEL:-/content/models/Qwen3-4B}"
ADAPTER="${ADAPTER:-/content/outputs/m5}"       # rollout from M5
INPUT="${INPUT:-/content/data/dataset/train/real_only.jsonl}"
OUTPUT="${OUTPUT:-/content/data/rollout/rollouts_m6.jsonl}"
TEMPERATURE="${TEMPERATURE:-1.5}"
K="${K:-4}"

mkdir -p logs "$(dirname "$OUTPUT")"

echo "=== M6: Generate K=$K rollouts from M5 (temperature=$TEMPERATURE) ==="
echo "  Model  : $MODEL"
echo "  Adapter: $ADAPTER"
echo "  Input  : $INPUT"
echo "  Output : $OUTPUT"
echo ""

RESUME="${RESUME:-false}"
EXTRA_ARGS=""
if [ "$RESUME" = "true" ]; then
  EXTRA_ARGS="--resume"
fi

python scripts/run_sdpo_rollouts_hf.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --k "$K" \
  --temperature "$TEMPERATURE" \
  --max-new-tokens 512 \
  --success-threshold 1.0 \
  $EXTRA_ARGS \
  2>&1 | tee logs/m6_rollouts.log

echo ""
echo "=== M6 rollouts done: $OUTPUT ==="
wc -l "$OUTPUT"
