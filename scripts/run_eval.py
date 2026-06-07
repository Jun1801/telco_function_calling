from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def main() -> None:
    data_dir = ROOT / "data"
    sample_paths = sorted(
        path for path in data_dir.glob("*.jsonl") if not path.name.startswith(("sft_", "public_"))
    )
    if not sample_paths:
        raise SystemExit("No JSONL datasets found. Run python scripts/generate_data.py first.")

    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")

    total = 0
    matched = 0
    by_file: dict[str, list[bool]] = {}
    for path in sample_paths:
        by_file[path.name] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                # Fresh executor per sample keeps evaluation independent from write-call mutation.
                state = MockTelcoApi.from_file(data_dir / "mock_telco_db.json")
                prediction = sample.get("prediction") or {
                    "action": "call_function",
                    "call": sample["call"],
                }
                result = evaluate_prediction(sample, prediction, tool_registry, state, contract_registry)
                machine_status = result["feedback"].get("machine_status")
                if "expected_status" in sample:
                    is_match = machine_status == sample["expected_status"]
                else:
                    is_match = result["reward_strict"] == 1.0
                by_file[path.name].append(is_match)
                total += 1
                matched += int(is_match)

    for name, results in by_file.items():
        correct = sum(results)
        print(f"{name}: {correct}/{len(results)} expected statuses matched")
    print(f"overall: {matched}/{total} expected statuses matched")
    if matched != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
