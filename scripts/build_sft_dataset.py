from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.sft_formatter import format_sample_for_sft
from src.registry.tool_registry import ToolRegistry


def main() -> None:
    data_dir = ROOT / "data"
    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("*.jsonl")):
        if path.name.startswith("sft_") or path.name.startswith("public_"):
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                record = format_sample_for_sft(sample, tool_registry)
                if sample["split"] == "train":
                    train_records.append(record)
                else:
                    eval_records.append(record)

    _write_jsonl(data_dir / "sft_train.jsonl", train_records)
    _write_jsonl(data_dir / "sft_eval.jsonl", eval_records)
    print(f"sft_train: {len(train_records)} records")
    print(f"sft_eval: {len(eval_records)} records")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
