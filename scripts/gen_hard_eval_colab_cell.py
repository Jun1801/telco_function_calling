"""Paste this as a Colab cell to run hard eval generation.
Prerequisites: repo cloned, cwd is telco_function_calling.
"""
import os, subprocess

os.environ.update({
    "GEN_MODEL":   "Qwen/Qwen3-32B",
    "BACKEND":     "transformers",
    "BATCH":       "8",
    "HARD_SCALE":  "1.0",
    "DATA_DIR":    "data",
    "OUT_DIR":     "/content/hard_eval_outputs",
})

subprocess.run(["pip", "install", "-q", "transformers", "accelerate"], check=True)
subprocess.run(["python", "scripts/gen_hard_eval.py"], check=True)

# Save to Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
import shutil, pathlib
dest = pathlib.Path("/content/drive/MyDrive/cvtd/hard_eval")
dest.mkdir(parents=True, exist_ok=True)
for f in pathlib.Path("/content/hard_eval_outputs").glob("*.jsonl"):
    shutil.copy(f, dest / f.name)
print(f"Saved {len(list(dest.glob('*.jsonl')))} files to {dest}")
