"""Upload M1 adapter and reports to Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ["HF_TOKEN"]
USERNAME = "Jun1801"
ROOT = Path(__file__).parent.parent

api = HfApi(token=TOKEN)

# --- 1. Upload M1 LoRA adapter ---
MODEL_REPO = f"{USERNAME}/telco-fc-m1-qwen3-4b"
MODEL_DIR = ROOT / "outputs/sft/m1_qwen3-4b"

print(f"Creating model repo: {MODEL_REPO}")
create_repo(MODEL_REPO, token=TOKEN, repo_type="model", exist_ok=True, private=False)

print("Uploading adapter files...")
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=MODEL_REPO,
    repo_type="model",
    ignore_patterns=["runs/**", "checkpoint-*/**"],  # skip TB logs and intermediate checkpoints
    commit_message="Upload M1 SFT adapter (Qwen3-4B, real KPI data, 3 epochs)",
)
print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")

# --- 2. Upload reports ---
REPORT_REPO = f"{USERNAME}/telco-fc-results"
REPORTS_DIR = ROOT / "reports"

print(f"\nCreating dataset repo: {REPORT_REPO}")
create_repo(REPORT_REPO, token=TOKEN, repo_type="dataset", exist_ok=True, private=False)

print("Uploading reports...")
for f in ["m0_error_report.md", "m1_error_report.md", "m0_results.jsonl", "m1_results.jsonl"]:
    path = REPORTS_DIR / f
    if path.exists():
        print(f"  Uploading {f} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f,
            repo_id=REPORT_REPO,
            repo_type="dataset",
            commit_message=f"Add {f}",
        )
print(f"Reports uploaded: https://huggingface.co/datasets/{REPORT_REPO}")

print("\nDone!")
