from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.generation.toolace_mini import TelcoToolACEMiniPipeline


def build_samples(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Generate verified Telco-ToolACE-mini samples."""
    return TelcoToolACEMiniPipeline(data_dir).generate_verified_samples()


def write_jsonl_by_split(samples: list[dict[str, Any]], data_dir: str | Path) -> dict[str, int]:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_split.setdefault(sample["split"], []).append(sample)

    counts: dict[str, int] = {}
    for split, split_samples in by_split.items():
        path = data_path / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for sample in split_samples:
                file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        counts[split] = len(split_samples)
    return counts
