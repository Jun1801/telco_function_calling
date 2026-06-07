from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.synth_data_generator import build_samples, write_jsonl_by_split


def main() -> None:
    counts = write_jsonl_by_split(build_samples(), ROOT / "data")
    for split, count in sorted(counts.items()):
        print(f"{split}: {count} samples")


if __name__ == "__main__":
    main()
