# Plan: M0 Baseline + M1 SFT Experiments on GPU

## Context

The project was originally developed for Apple Silicon (MLX), but we're now on a Vast.ai GPU instance:
- **NVIDIA RTX PRO 6000 Black** — 97 GB VRAM, CUDA 13.2
- **`/venv/main`** already has `torch 2.12.0+cu130` with CUDA ✅
- Missing: `trl`, `peft`, `accelerate`, `datasets`, `wandb`, `tensorboard`
- **No quantization needed** — 4B model fits trivially in 97 GB

**Data state:**
- `data/eval_real_*.jsonl` (7 splits) **already exist** in `data/` — no copying needed for eval
- `drive_upload/data/sft_train_real_with_warmup.jsonl` (5,324 lines) — only in `drive_upload/`; pass via `--train-file`  
  ⚠️ **Needs rebuild**: this file was generated before the SYSTEM_PROMPT_REAL fix (prompts still say "contract-aware"). Run `python3 scripts/build_real_sft.py` to regenerate before training.
- `drive_upload/data/sft_eval_real_messages.jsonl` (256 lines) — only in `drive_upload/`; pass via `--eval-file`
- Schema: `["id", "source", "expected_action", "messages"]` ✅ matches `train_sft.py`

**Existing scripts being reused (no changes needed):**
- `scripts/run_baseline.py` — M0 + M1 eval (has `--backend transformers`, `--adapter`, `--splits real`)
- `scripts/train_sft.py` — M1 SFT (HuggingFace TRL + LoRA, `--report-to` flag for W&B/TensorBoard)

---

## Scripts to Write (4 shell scripts)

### 1. `scripts/setup_gpu.sh`
Install missing packages + download model + create dirs.

```bash
#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate

uv pip install trl peft accelerate datasets huggingface-hub wandb tensorboard

mkdir -p reports logs outputs/sft/m1_qwen3-4b /workspace/models

# Persist W&B key across sessions (written once; idempotent)
grep -q WANDB_API_KEY /workspace/.env 2>/dev/null || \
  echo 'WANDB_API_KEY="6677ef18662cf06ab193"' >> /workspace/.env

# Cache model to /workspace (persists across stop/start)
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B

echo "Setup complete."
```

### 2. `scripts/run_m0.sh`
Zero-shot eval on real splits. Uses TensorBoard (no API key) + optionally W&B.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m0_results.jsonl \
  --error-report reports/m0_error_report.md \
  2>&1 | tee logs/m0.log
```

### 3. `scripts/run_m1_train.sh`
SFT on warmup+real data (3 epochs, 5,324 samples).  
Reports to both TensorBoard (always) and W&B (if `WANDB_API_KEY` is set).

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

export WANDB_PROJECT="telco-fc"
export WANDB_RUN_NAME="m1-sft-qwen3-4b"
REPORT_TO="all"   # wandb + tensorboard (key loaded from /workspace/.env above)

python scripts/train_sft.py \
  --model /workspace/models/Qwen3-4B \
  --train-file drive_upload/data/sft_train_real_with_warmup.jsonl \
  --eval-file drive_upload/data/sft_eval_real_messages.jsonl \
  --output-dir outputs/sft/m1_qwen3-4b \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --bf16 \
  --no-load-in-4bit \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-steps 100 \
  --eval-steps 100 \
  --report-to "$REPORT_TO" \
  2>&1 | tee logs/m1_train.log
```

### 4. `scripts/run_m1_eval.sh`
Eval M1 adapter on the same real splits as M0.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

python scripts/run_baseline.py \
  --backend transformers \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1_qwen3-4b \
  --splits real \
  --no-load-in-4bit \
  --max-new-tokens 512 \
  --output reports/m1_results.jsonl \
  --error-report reports/m1_error_report.md \
  2>&1 | tee logs/m1_eval.log
```

---

## Running in the Background (Screen / Tmux)

Jobs must survive SSH disconnect — use `screen` or `tmux`:

```bash
# Option A: screen
screen -S m0     # then run: bash scripts/run_m0.sh
screen -S m1     # then run: bash scripts/run_m1_train.sh
# Detach: Ctrl-A D  |  Reattach: screen -r m0

# Option B: tmux
tmux new -s m0   # then run: bash scripts/run_m0.sh
tmux new -s m1   # then run: bash scripts/run_m1_train.sh
# Detach: Ctrl-B D  |  Reattach: tmux attach -t m0

# One-liner background launch (no attach needed):
screen -dmS m0 bash -c 'bash scripts/run_m0.sh'
screen -dmS m1 bash -c 'bash scripts/run_m1_train.sh'
screen -dmS m1eval bash -c 'bash scripts/run_m1_eval.sh'
```

---

## W&B Setup

API key is written to `/workspace/.env` by `setup_gpu.sh` (key already configured).
Project: `telco-fc`, run name: `m1-sft-qwen3-4b`.

TensorBoard (no key needed):
```bash
source /venv/main/bin/activate
tensorboard --logdir outputs/sft/m1_qwen3-4b --port 16006 --host 0.0.0.0
```

---

## Execution Order

```
1. bash scripts/setup_gpu.sh              # ~5-10 min (install + model download)
2. screen -dmS m0 bash -c 'bash scripts/run_m0.sh'       # ~30-60 min
3. screen -dmS m1 bash -c 'bash scripts/run_m1_train.sh' # ~60-90 min
4. # After step 3 finishes:
   screen -dmS m1eval bash -c 'bash scripts/run_m1_eval.sh' # ~30-60 min
```

---

## Verification

| Step | Check |
|------|-------|
| Setup | `source /venv/main/bin/activate && python -c "import trl, peft, wandb; print('ok')"` |
| Model | `ls /workspace/models/Qwen3-4B/*.safetensors` |
| M0 | `cat reports/m0_error_report.md` — expect ~60% strict accuracy |
| M1 train | `ls outputs/sft/m1_qwen3-4b/adapter_model.safetensors` |
| M1 eval | `cat reports/m1_error_report.md` — expect ~83% strict accuracy |
| TensorBoard | `tensorboard --logdir outputs/sft/m1_qwen3-4b` |
