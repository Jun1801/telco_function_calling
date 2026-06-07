from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.public_warmup_loader import (
    PublicWarmupLoader,
    demo_rows,
    read_json_or_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public warm-up function-calling data.")
    parser.add_argument("--source", default="toolace", help="One of: xlam, toolace, apigen_mt, xlam_irrelevance, hermes_fc")
    parser.add_argument("--input", help="Optional local JSON/JSONL file. If omitted, writes a tiny demo subset.")
    parser.add_argument("--output", default=str(ROOT / "data" / "public_warmup_subset.jsonl"))
    args = parser.parse_args()

    rows = read_json_or_jsonl(args.input) if args.input else demo_rows()
    result = PublicWarmupLoader().normalize_many(rows, args.source)
    write_jsonl(args.output, result.records)

    print(f"source: {args.source}")
    print(f"records: {len(result.records)}")
    print(f"skipped: {result.skipped}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
