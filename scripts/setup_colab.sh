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
