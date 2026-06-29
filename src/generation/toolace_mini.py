from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import evaluate_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.generation.masking import ToolSelfEvolutionSynthesis
from src.generation.scenario import ScenarioSpec
from src.generation.split_builder import attach_generation_profile, scenario_distribution
from src.generation.template_generator import TemplateScenarioGenerator
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


class MultiAgentInteractiveDialogGenerator:
    """Deterministic ToolACE-mini scenario generator for telco tool-calling tasks."""

    def __init__(self) -> None:
        self.template_generator = TemplateScenarioGenerator()
        self.self_evolution = ToolSelfEvolutionSynthesis()

    def generate(self) -> list[dict[str, Any]]:
        specs = self.template_generator.build_specs()
        samples = [self._from_spec(spec) for spec in specs]
        self._add_masking_metadata(samples)
        distribution = scenario_distribution(samples)
        for sample in samples:
            sample["scenario_distribution"] = distribution
            attach_generation_profile(sample)
        return samples

    def _from_spec(self, spec: ScenarioSpec) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "id": spec.scenario_id,
            "source": "telco_toolace_mini",
            "split": spec.split,
            "scenario": spec.scenario,
            "scenario_family": spec.scenario_family,
            "instruction": spec.instruction,
            "customer_verified": spec.customer_verified,
            "expected_action": spec.expected_action,
            "generator": "telco_toolace_mini",
        }
        if spec.expected_status is not None:
            sample["expected_status"] = spec.expected_status
        if spec.gold_call is not None:
            sample["gold_call"] = copy.deepcopy(spec.gold_call)
            if spec.prediction and "call" in spec.prediction:
                sample["call"] = copy.deepcopy(spec.prediction["call"])
            else:
                sample["call"] = copy.deepcopy(spec.gold_call)
        if spec.prediction is not None:
            sample["prediction"] = copy.deepcopy(spec.prediction)
        elif spec.gold_call is not None:
            sample["prediction"] = {"action": "call_function", "call": copy.deepcopy(spec.gold_call)}
        if spec.missing_slots:
            sample["missing_slots"] = list(spec.missing_slots)
        if spec.gold_calls:
            sample["gold_calls"] = copy.deepcopy(spec.gold_calls)
        if spec.gold_steps:
            sample["gold_steps"] = copy.deepcopy(spec.gold_steps)
        if spec.checker_call:
            sample["checker_call"] = copy.deepcopy(spec.checker_call)
        if spec.checker_expected_status:
            sample["checker_expected_status"] = spec.checker_expected_status
        return sample

    def _add_masking_metadata(self, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            if sample["scenario"] != "function_name_masking":
                continue
            sample["masked_tool"] = self.self_evolution.mask_call(
                sample["gold_call"],
                "func_7",
                {"param_1": "customer_id", "param_2": "eid"},
            )


class DualLayerValidationProcess:
    """Rule-based and execution-based verification, following ToolACE DLV."""

    def __init__(self, data_dir: str | Path) -> None:
        data_path = Path(data_dir)
        self.data_dir = data_path
        tools_path = data_path / "tools.json"
        contracts_path = data_path / "tool_contracts.json"
        self.tool_registry = ToolRegistry.from_file(tools_path) if tools_path.exists() else ToolRegistry([])
        self.contract_registry = ContractRegistry.from_file(contracts_path) if contracts_path.exists() else ContractRegistry([])

    def verify(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        verified: list[dict[str, Any]] = []
        for sample in samples:
            state = MockTelcoApi.from_file(self.data_dir / "mock_telco_db.json")
            prediction = sample.get("prediction") or {"action": "call_function", "call": sample["call"]}
            result = evaluate_prediction(sample, prediction, self.tool_registry, state, self.contract_registry)
            checker_result = self._check_negative_if_needed(sample)
            if self._passes_expected(sample, result) and checker_result is not None:
                enriched = copy.deepcopy(sample)
                enriched["toolace_validation"] = {
                    "reward_soft": result["reward_soft"],
                    "reward_strict": result["reward_strict"],
                    "reward_total": result["reward_total"],
                    "machine_status": result["feedback"].get("machine_status"),
                    "metrics": result["metrics"],
                    "checker_status": checker_result.get("status"),
                }
                verified.append(enriched)
        return verified

    def _passes_expected(self, sample: dict[str, Any], result: dict[str, Any]) -> bool:
        if sample.get("expected_action") == "call_function" and "expected_status" in sample:
            return result["feedback"].get("machine_status") == sample["expected_status"]
        return result["reward_strict"] == 1.0

    def _check_negative_if_needed(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        checker_call = sample.get("checker_call")
        if checker_call is None:
            return {}
        state = MockTelcoApi.from_file(self.data_dir / "mock_telco_db.json")
        score = evaluate_prediction(
            {
                **sample,
                "expected_action": "call_function",
                "gold_call": checker_call,
                "expected_status": sample["checker_expected_status"],
            },
            {"action": "call_function", "call": checker_call},
            self.tool_registry,
            state,
            self.contract_registry,
        )
        status = score["feedback"].get("machine_status")
        if status != sample["checker_expected_status"]:
            return None
        return {"status": status}


class TelcoToolACEMiniPipeline:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.dialog_generator = MultiAgentInteractiveDialogGenerator()
        self.validator = DualLayerValidationProcess(data_dir)

    def generate_verified_samples(self) -> list[dict[str, Any]]:
        candidates = self.dialog_generator.generate()
        return self.validator.verify(candidates)
