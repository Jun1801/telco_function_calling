from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import evaluate_sample, load_real_assets
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def _prediction_for(sample: dict[str, Any]) -> dict[str, Any]:
    """Gold prediction for validating a dataset sample (synthetic or real)."""
    if sample.get("prediction"):
        return sample["prediction"]
    action = sample.get("expected_action", "call_function")
    if action == "call_functions":
        return {"action": "call_functions", "calls": sample.get("gold_calls") or sample.get("gold_steps") or []}
    if action == "ask_clarification":
        return {"action": "ask_clarification", "asked_slots": sample.get("missing_slots", [])}
    if action == "abstain":
        return {"action": "abstain", "reason": "gold abstain"}
    return {"action": "call_function", "call": sample.get("gold_call") or sample.get("call")}


def main() -> None:
    data_dir = ROOT / "data"
    # Validate the eval datasets only (not training files or rollout/correction outputs).
    sample_paths = sorted(data_dir.glob("eval_*.jsonl"))
    if not sample_paths:
        raise SystemExit("No eval_*.jsonl datasets found. Run python scripts/generate_data.py first.")

    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    real_assets = load_real_assets(data_dir)

    total = 0
    matched = 0
    by_file: dict[str, list[bool]] = {}
    for path in sample_paths:
        by_file[path.name] = []
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            sample = json.loads(line)
            prediction = _prediction_for(sample)
            result = evaluate_sample(sample, prediction, tool_registry, contract_registry, data_dir, real_assets)
            machine_status = result["feedback"].get("machine_status")
            if "checker_expected_status" in sample and "toolace_validation" in sample:
                is_match = (
                    result["reward_strict"] == 1.0
                    and sample["toolace_validation"]["checker_status"] == sample["checker_expected_status"]
                )
            elif "expected_status" in sample:
                is_match = machine_status == sample["expected_status"]
            else:
                is_match = result["reward_strict"] == 1.0
            by_file[path.name].append(is_match)
            total += 1
            matched += int(is_match)

    for name, results in by_file.items():
        print(f"{name}: {sum(results)}/{len(results)} expected statuses matched")
    print(f"overall: {matched}/{total} expected statuses matched")
    if matched != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
