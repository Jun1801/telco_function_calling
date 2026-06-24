"""Upload clean real-KPI dataset to Hugging Face Hub.

Uploads:
  - data/sft_train_real.jsonl              → train/real_only.jsonl        (4,815 real samples)
  - data/sft_train_real_with_warmup.jsonl  → train/with_warmup.jsonl      (7,526 = real + warmup)
  - data/eval_real_*.jsonl (7 splits)      → eval/{split}.jsonl           (1,806 total)
  - data/real_tools.json                   → schemas/real_tools.json      (26 tool schemas)
  - data/real_reference_codes.json         → schemas/real_reference_codes.json
  - data/real_station_catalogue.json       → schemas/real_station_catalogue.json

python scripts/upload_dataset_hf.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    env_file = Path("/workspace/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"')
                break

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set. Add to /workspace/.env or export HF_TOKEN=...")

USERNAME = "Jun1801"
REPO_ID = f"{USERNAME}/telco-fc-dataset"
DATA_DIR = ROOT / "data"

from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)

print(f"Creating dataset repo: {REPO_ID}")
create_repo(REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True, private=False)


def upload(local_path: Path, repo_path: str, desc: str) -> None:
    if not local_path.exists():
        print(f"  SKIP (not found): {local_path}")
        return
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"  Uploading {desc} ({size_mb:.1f} MB) → {repo_path} ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {repo_path}",
    )
    print(f"  Done.")


# --- Training data ---
print("\n=== Training splits ===")
upload(DATA_DIR / "sft_train_real.jsonl",             "train/real_only.jsonl",      "train real-only (4,815)")
upload(DATA_DIR / "sft_train_real_with_warmup.jsonl", "train/with_warmup.jsonl",    "train with warmup (7,526)")

# --- Eval splits ---
print("\n=== Eval splits ===")
for split in ["seen", "unseen", "masked", "missing_slot", "multi_step", "parallel", "abstain"]:
    local = DATA_DIR / f"eval_real_{split}.jsonl"
    upload(local, f"eval/{split}.jsonl", f"eval_{split}")

# --- Tool schemas / reference catalogues ---
print("\n=== Schemas & catalogues ===")
for fname in ["real_tools.json", "real_reference_codes.json", "real_station_catalogue.json", "real_tool_contracts.json"]:
    upload(DATA_DIR / fname, f"schemas/{fname}", fname)

# --- README ---
readme_path = ROOT / "data_README.md"
readme_path.write_text(
    "# Telco Function Calling Dataset\n\n"
    "26 real Viettel KPI functions (Vietnamese, read-only).\n\n"
    "## Splits\n\n"
    "| Split | Samples | Description |\n"
    "|-------|---------|-------------|\n"
    "| `train/real_only.jsonl` | 4,815 | Pure real KPI train samples |\n"
    "| `train/with_warmup.jsonl` | 7,526 | Train + public warmup (used for M1b) |\n"
    "| `eval/seen.jsonl` | 350 | Seen tools, single-step |\n"
    "| `eval/unseen.jsonl` | 250 | Unseen tools (schema-only at inference) |\n"
    "| `eval/masked.jsonl` | 300 | Tool-name masking eval |\n"
    "| `eval/missing_slot.jsonl` | 350 | Clarification / ask_clarification |\n"
    "| `eval/multi_step.jsonl` | 156 | Multi-step ReAct |\n"
    "| `eval/parallel.jsonl` | 150 | Parallel function calls |\n"
    "| `eval/abstain.jsonl` | 250 | Abstain (no matching tool) |\n\n"
    "## Output format\n\n"
    "```json\n"
    "{\"action\": \"call_function\", \"call\": {\"tool_name\": \"...\", \"arguments\": {...}}}\n"
    "{\"action\": \"call_functions\", \"calls\": [{...}, ...]}\n"
    "{\"action\": \"ask_clarification\", \"asked_slots\": [\"...\"]}\n"
    "{\"action\": \"abstain\", \"reason\": \"...\"}\n"
    "```\n"
)
upload(readme_path, "README.md", "README")
readme_path.unlink()

print(f"\nDataset uploaded: https://huggingface.co/datasets/{REPO_ID}")
