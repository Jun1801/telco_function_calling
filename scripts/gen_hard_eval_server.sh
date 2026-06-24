#!/usr/bin/env bash
# Generate the 4 hard eval splits on a GPU server (Vast.ai / Colab / any CUDA Linux).
# Usage: bash scripts/gen_hard_eval_server.sh [workspace_dir]
#
# Override env vars before calling:
#   GEN_MODEL    default: Qwen/Qwen3-32B
#   BACKEND      default: transformers
#   BATCH        default: 8
#   HARD_SCALE   default: 1.0
#   DATA_DIR     default: <workspace>/data
#   OUT_DIR      default: <workspace>/data/hard_eval_outputs

set -euo pipefail

WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

export GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3-32B}"
export BACKEND="${BACKEND:-transformers}"
export BATCH="${BATCH:-8}"
export HARD_SCALE="${HARD_SCALE:-1.0}"
export DATA_DIR="${DATA_DIR:-${WORKSPACE}/data}"
export OUT_DIR="${OUT_DIR:-${WORKSPACE}/data/hard_eval_outputs}"

echo "=== Hard Eval Generation ==="
echo "  Workspace : $WORKSPACE"
echo "  Model     : $GEN_MODEL"
echo "  Backend   : $BACKEND  batch=$BATCH"
echo "  Scale     : $HARD_SCALE"
echo "  DataDir   : $DATA_DIR"
echo "  OutDir    : $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR" logs

# Install deps if needed (use uv when available, fall back to pip)
_install() { if command -v uv &>/dev/null; then uv pip install -q "$@"; else pip install -q "$@"; fi; }
if ! python -c "import transformers" 2>/dev/null; then
    _install transformers accelerate
fi
if [ "$BACKEND" = "vllm" ] && ! python -c "import vllm" 2>/dev/null; then
    _install "vllm<0.9.0"
fi

# Run generation
python scripts/gen_hard_eval.py 2>&1 | tee logs/gen_hard_eval.log

echo ""
echo "=== Output counts ==="
for f in "$OUT_DIR"/eval_real_hard_*.jsonl; do
    [ -f "$f" ] && echo "  $(basename "$f"): $(wc -l < "$f") samples"
done

echo ""
echo "Done. Files in $OUT_DIR/"
