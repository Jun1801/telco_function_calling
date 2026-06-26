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

# Download via HF cache then shutil.copytree — avoids symlink issues with PEFT
_hf_download() {
    # _hf_download <repo_id> <repo_type> <local_dir>
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sys, os, shutil
from huggingface_hub import snapshot_download
repo_id, repo_type, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
kwargs = {}
if repo_type != "model":
    kwargs["repo_type"] = repo_type
cache = snapshot_download(repo_id, **kwargs,
                          ignore_patterns=["*.msgpack","*.h5","flax_model*","tf_model*","rust_model*"])
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)
shutil.copytree(cache, local_dir)
print(f"  → {local_dir}")
PYEOF
}

# Download base model
echo "Downloading Qwen3-4B..."
_hf_download Qwen/Qwen3-4B model "$MODEL_DIR"

# Download dataset
echo "Downloading dataset..."
_hf_download Jun1801/telco-fc-dataset dataset "$DATA_DIR/dataset"

# Download rollout data
echo "Downloading rollout data..."
_hf_download Jun1801/data_rollout dataset "$DATA_DIR/rollout"

# Populate data/ dir expected by run_baseline.py
echo "Populating data/ directory..."
REPO_DATA="$REPO_DIR/data"
mkdir -p "$REPO_DATA"

# Real tool schemas (needed by load_real_assets)
cp "$DATA_DIR/dataset/schemas/real_tools.json"           "$REPO_DATA/real_tools.json"
cp "$DATA_DIR/dataset/schemas/real_reference_codes.json" "$REPO_DATA/real_reference_codes.json"
cp "$DATA_DIR/dataset/schemas/real_station_catalogue.json" "$REPO_DATA/real_station_catalogue.json" 2>/dev/null || true

# Eval splits: rename eval/<name>.jsonl → eval_real_<name>.jsonl
# NOTE: the HF dataset eval/ files already include hard splits merged in
# (seen.jsonl = standard_seen + hard_seen, etc.) — do NOT also copy
# rollout/hard_eval_outputs/ or hard samples will be counted twice.
for split in seen unseen masked missing_slot multi_step parallel abstain; do
    src="$DATA_DIR/dataset/eval/$split.jsonl"
    dst="$REPO_DATA/eval_real_$split.jsonl"
    [ -f "$src" ] && cp "$src" "$dst" && echo "  copied eval_real_$split.jsonl ($(wc -l < $dst) samples)"
done

# Download adapters
echo "Downloading adapters..."
_hf_download Jun1801/telco-fc-m1b-qwen3-4b model "$ADAPTER_DIR/m1"
_hf_download Jun1801/telco-fc-m2b-qwen3-4b model "$ADAPTER_DIR/m2"
_hf_download Jun1801/telco-fc-m3b-qwen3-4b model "$ADAPTER_DIR/m3"
_hf_download Jun1801/telco-fc-m3b-test      model "$ADAPTER_DIR/m3-test"

echo ""
echo "=== Setup complete ==="
echo "  Model  : $MODEL_DIR"
echo "  Data   : $DATA_DIR"
echo "  Adapters: $ADAPTER_DIR"
