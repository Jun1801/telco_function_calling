#!/usr/bin/env bash
# One-time Colab setup: install deps, download model and data.
# Usage: bash scripts/setup_colab.sh
# Env: HF_TOKEN (optional, for private repos)

set -euo pipefail
REPO_DIR="${REPO_DIR:-/content/telco_function_calling}"
MODEL_DIR="${MODEL_DIR:-/content/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-/content/data}"
ADAPTER_DIR="${ADAPTER_DIR:-/content/adapters}"
REPORTS_DIR="${REPORTS_DIR:-/content/reports}"
LOGS_DIR="${LOGS_DIR:-/content/logs}"

echo "=== Colab Setup ==="

# Install dependencies (torchao>=0.16.0 required by peft)
pip install -q transformers peft accelerate datasets huggingface-hub trl bitsandbytes "torchao>=0.16.0"

# Create dirs
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$ADAPTER_DIR" "$REPORTS_DIR" "$LOGS_DIR"

# Download base model
echo "Downloading Qwen3-4B..."
hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"

# Download dataset
echo "Downloading dataset..."
hf download Jun1801/telco-fc-dataset --repo-type dataset --local-dir "$DATA_DIR/dataset"

# Download rollout data
echo "Downloading rollout data..."
hf download Jun1801/data_rollout --repo-type dataset --local-dir "$DATA_DIR/rollout"

# Populate data/ dir expected by run_baseline.py
echo "Populating data/ directory..."
REPO_DATA="$REPO_DIR/data"
mkdir -p "$REPO_DATA"

# Real tool schemas (needed by load_real_assets)
cp "$DATA_DIR/dataset/schemas/real_tools.json"           "$REPO_DATA/real_tools.json"
cp "$DATA_DIR/dataset/schemas/real_reference_codes.json" "$REPO_DATA/real_reference_codes.json"
cp "$DATA_DIR/dataset/schemas/real_station_catalogue.json" "$REPO_DATA/real_station_catalogue.json" 2>/dev/null || true

# Eval splits: rename eval/<name>.jsonl → eval_real_<name>.jsonl
for split in seen unseen masked missing_slot multi_step parallel abstain; do
    src="$DATA_DIR/dataset/eval/$split.jsonl"
    dst="$REPO_DATA/eval_real_$split.jsonl"
    [ -f "$src" ] && cp "$src" "$dst" && echo "  copied eval_real_$split.jsonl"
done

# Hard eval splits from rollout
for f in "$DATA_DIR/rollout/hard_eval_outputs/eval_real_hard_"*.jsonl; do
    [ -f "$f" ] && cp "$f" "$REPO_DATA/" && echo "  copied $(basename $f)"
done

# Download adapters
echo "Downloading adapters..."
hf download Jun1801/telco-fc-m1b-qwen3-4b --repo-type model --local-dir "$ADAPTER_DIR/m1"
hf download Jun1801/telco-fc-m2b-qwen3-4b --repo-type model --local-dir "$ADAPTER_DIR/m2"
hf download Jun1801/telco-fc-m3b-qwen3-4b --repo-type model --local-dir "$ADAPTER_DIR/m3"
hf download Jun1801/telco-fc-m3b-test      --repo-type model --local-dir "$ADAPTER_DIR/m3-test"

echo ""
echo "=== Setup complete ==="
echo "  Model  : $MODEL_DIR"
echo "  Data   : $DATA_DIR"
echo "  Adapters: $ADAPTER_DIR"
