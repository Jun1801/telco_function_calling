from __future__ import annotations

from typing import Any

from src.evaluation.metrics import argument_key_f1, argument_value_accuracy
from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry
from src.reward.reward_feedback import RewardScorer


def evaluate_prediction(
    sample: dict[str, Any],
    prediction: dict[str, Any],
    tool_registry: ToolRegistry,
    state: MockTelcoApi,
    contract_registry: ContractRegistry | None = None,
) -> dict[str, Any]:
    contract_registry = contract_registry or ContractRegistry([])
    expected_action = sample.get("expected_action") or _expected_action_from_legacy_sample(sample)
    predicted_action = prediction.get("action", "call_function")

    metrics = _base_metrics(expected_action, predicted_action)
    feedback: dict[str, Any] = {"errors": [], "feedback_text": []}
    reward_soft = 0.0
    reward_strict = 0.0

    if expected_action == "ask_clarification":
        missing_slots = set(sample.get("missing_slots", []))
        asked_slots = set(prediction.get("asked_slots", []))
        slot_recall = len(missing_slots & asked_slots) / len(missing_slots) if missing_slots else 1.0
        metrics["ask_back_accuracy"] = 1.0 if predicted_action == "ask_clarification" and slot_recall == 1.0 else 0.0
        reward_soft = 0.6 * metrics["action_accuracy"] + 0.4 * slot_recall
        reward_strict = 1.0 if metrics["ask_back_accuracy"] == 1.0 else 0.0
        if reward_strict == 0.0:
            feedback["errors"].append({"type": "missing_slot_not_handled", "missing_slots": sorted(missing_slots)})
            feedback["feedback_text"].append("The correct action is ask_clarification and all missing slots must be requested.")
        return _result(reward_soft, reward_strict, feedback, metrics)

    if expected_action == "abstain":
        metrics["abstention_accuracy"] = 1.0 if predicted_action == "abstain" else 0.0
        reward_soft = metrics["abstention_accuracy"]
        reward_strict = metrics["abstention_accuracy"]
        if reward_strict == 0.0:
            feedback["errors"].append({"type": "unsafe_or_forbidden_action"})
            feedback["feedback_text"].append("The correct action is abstain; no tool should be called.")
        return _result(reward_soft, reward_strict, feedback, metrics)

    if predicted_action != "call_function":
        feedback["errors"].append({"type": "wrong_action", "expected": "call_function", "actual": predicted_action})
        feedback["feedback_text"].append("The task requires a function call.")
        return _result(0.1 * metrics["action_accuracy"], 0.0, feedback, metrics)

    call = prediction.get("call") or {"tool_name": prediction.get("tool_name"), "arguments": prediction.get("arguments", {})}
    gold_call = sample.get("gold_call") or sample.get("call")
    predicted_tool = call.get("tool_name")
    predicted_args = call.get("arguments", {})
    gold_tool = gold_call.get("tool_name") if gold_call else None
    gold_args = gold_call.get("arguments", {}) if gold_call else {}

    score = RewardScorer(tool_registry, contract_registry, state).score_call(
        predicted_tool,
        predicted_args,
        sample.get("customer_verified", True),
    )

    metrics.update(
        {
            "function_selection_accuracy": 1.0 if predicted_tool == gold_tool else 0.0,
            "argument_key_f1": argument_key_f1(predicted_args, gold_args),
            "argument_value_accuracy": argument_value_accuracy(predicted_args, gold_args),
            "schema_validity": 1.0 if score.status not in {"schema_invalid"} else 0.0,
            "contract_validity": 1.0 if score.status not in {"contract_invalid"} else 0.0,
            "execution_success_rate": 1.0 if score.status == "ok" else 0.0,
            "task_success_rate": 1.0 if score.status == sample.get("expected_status", "ok") else 0.0,
            "deprecated_tool_call_rate": 1.0 if any("deprecated" in item.lower() for item in score.feedback) else 0.0,
        }
    )

    reward_soft = (
        0.10 * metrics["action_accuracy"]
        + 0.20 * metrics["function_selection_accuracy"]
        + 0.15 * metrics["argument_key_f1"]
        + 0.15 * metrics["argument_value_accuracy"]
        + 0.15 * metrics["schema_validity"]
        + 0.15 * metrics["contract_validity"]
        + 0.10 * metrics["execution_success_rate"]
    )
    reward_strict = 1.0 if all(
        [
            metrics["action_accuracy"] == 1.0,
            metrics["function_selection_accuracy"] == 1.0,
            metrics["argument_key_f1"] == 1.0,
            metrics["argument_value_accuracy"] == 1.0,
            score.status == sample.get("expected_status", "ok"),
        ]
    ) else 0.0

    feedback["machine_status"] = score.status
    feedback["execution_output"] = score.output
    feedback["feedback_text"].extend(score.feedback)
    if score.status != "ok":
        feedback["errors"].extend({"type": score.status, "message": item} for item in score.feedback)

    return _result(reward_soft, reward_strict, feedback, metrics)


def _expected_action_from_legacy_sample(sample: dict[str, Any]) -> str:
    if "call" in sample:
        return "call_function"
    return sample.get("expected_action", "call_function")


def _base_metrics(expected_action: str, predicted_action: str) -> dict[str, float]:
    return {
        "action_accuracy": 1.0 if expected_action == predicted_action else 0.0,
        "function_selection_accuracy": 0.0,
        "argument_key_f1": 0.0,
        "argument_value_accuracy": 0.0,
        "schema_validity": 0.0,
        "contract_validity": 0.0,
        "execution_success_rate": 0.0,
        "task_success_rate": 0.0,
        "ask_back_accuracy": 0.0,
        "abstention_accuracy": 0.0,
        "deprecated_tool_call_rate": 0.0,
    }


def _result(
    reward_soft: float,
    reward_strict: float,
    feedback: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    reward_total = 0.5 * reward_soft + 0.5 * reward_strict
    return {
        "reward_soft": round(reward_soft, 6),
        "reward_strict": round(reward_strict, 6),
        "reward_total": round(reward_total, 6),
        "feedback": feedback,
        "metrics": metrics,
    }
