#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /venv/main/bin/activate

echo "=== Installing ML packages ==="
uv pip install trl peft accelerate datasets huggingface-hub wandb tensorboard

echo "=== Creating directories ==="
mkdir -p reports logs outputs/sft/m1_qwen3-4b /workspace/models

echo "=== Writing W&B API key to /workspace/.env ==="
grep -q WANDB_API_KEY /workspace/.env 2>/dev/null || \
  echo 'WANDB_API_KEY="6677ef18662cf06ab193"' >> /workspace/.env

echo "=== Downloading Qwen/Qwen3-4B ==="
hf download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B

echo "=== Setup complete ==="
python -c "import torch, trl, peft, wandb; print('torch', torch.__version__, '| trl', trl.__version__, '| wandb', wandb.__version__, '| CUDA', torch.cuda.is_available())"
