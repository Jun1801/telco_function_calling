from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry
from src.reward.reward_feedback import RewardScorer


def make_scorer() -> RewardScorer:
    return RewardScorer(
        ToolRegistry.from_file("data/tools.json"),
        ContractRegistry.from_file("data/tool_contracts.json"),
        MockTelcoApi.from_file("data/mock_telco_db.json"),
    )


def test_reward_is_one_for_valid_call() -> None:
    result = make_scorer().score_call("get_balance", {"customer_id": "C001"})

    assert result.status == "ok"
    assert result.reward == 1.0
    assert result.output == {"balance": 125000, "outstanding_balance": 0}


def test_reward_is_zero_for_schema_invalid_call() -> None:
    result = make_scorer().score_call("add_data_package", {"customer_id": "C001", "package_code": "DATA999"})

    assert result.status == "schema_invalid"
    assert result.reward == 0.0
    assert "Invalid value" in result.feedback[0]


def test_reward_is_zero_for_contract_invalid_call() -> None:
    result = make_scorer().score_call(
        "enable_roaming",
        {"customer_id": "C003", "country": "Japan"},
        customer_verified=True,
    )

    assert result.status == "contract_invalid"
    assert result.reward == 0.0
    assert any("postpaid" in feedback for feedback in result.feedback)
