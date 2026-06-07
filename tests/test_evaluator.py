from src.evaluation.evaluator import evaluate_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def make_inputs() -> tuple[ToolRegistry, ContractRegistry, MockTelcoApi]:
    return (
        ToolRegistry.from_file("data/tools.json"),
        ContractRegistry.from_file("data/tool_contracts.json"),
        MockTelcoApi.from_file("data/mock_telco_db.json"),
    )


def test_evaluator_returns_reward_feedback_and_metrics_for_valid_call() -> None:
    registry, contracts, state = make_inputs()
    sample = {
        "expected_action": "call_function",
        "gold_call": {"tool_name": "get_balance", "arguments": {"customer_id": "C001"}},
        "expected_status": "ok",
    }
    prediction = {"action": "call_function", "call": {"tool_name": "get_balance", "arguments": {"customer_id": "C001"}}}

    result = evaluate_prediction(sample, prediction, registry, state, contracts)

    assert result["reward_strict"] == 1.0
    assert result["metrics"]["function_selection_accuracy"] == 1.0
    assert result["feedback"]["machine_status"] == "ok"


def test_evaluator_scores_missing_slot_clarification() -> None:
    registry, contracts, state = make_inputs()
    sample = {
        "expected_action": "ask_clarification",
        "missing_slots": ["package_code"],
    }
    prediction = {"action": "ask_clarification", "asked_slots": ["package_code"]}

    result = evaluate_prediction(sample, prediction, registry, state, contracts)

    assert result["reward_strict"] == 1.0
    assert result["metrics"]["ask_back_accuracy"] == 1.0


def test_evaluator_penalizes_contract_invalid_call() -> None:
    registry, contracts, state = make_inputs()
    sample = {
        "expected_action": "call_function",
        "gold_call": {"tool_name": "enable_roaming", "arguments": {"customer_id": "C003", "country": "Japan"}},
        "expected_status": "ok",
        "customer_verified": True,
    }
    prediction = {
        "action": "call_function",
        "call": {"tool_name": "enable_roaming", "arguments": {"customer_id": "C003", "country": "Japan"}},
    }

    result = evaluate_prediction(sample, prediction, registry, state, contracts)

    assert result["reward_strict"] == 0.0
    assert result["metrics"]["contract_validity"] == 0.0
    assert result["feedback"]["machine_status"] == "contract_invalid"
