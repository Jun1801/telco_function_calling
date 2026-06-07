from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def test_registry_has_mvp_scale_tools_and_contracts() -> None:
    tools = ToolRegistry.from_file("data/tools.json")
    contracts = ContractRegistry.from_file("data/tool_contracts.json")

    all_tools = tools.all()
    all_contracts = contracts.all()
    split_counts = {}
    for tool in all_tools:
        split_counts[tool["split"]] = split_counts.get(tool["split"], 0) + 1

    assert len(all_tools) >= 80
    assert len(all_contracts) >= 20
    assert split_counts["seen"] >= 25
    assert split_counts["unseen"] >= 4
    assert split_counts["evolution"] >= 2
    assert split_counts["distractor"] >= 50
    assert tools.get("legacy_enable_roaming")["deprecated"] is True


def test_tools_have_full_plan_metadata() -> None:
    tools = ToolRegistry.from_file("data/tools.json")

    for tool in tools.all():
        assert tool["tool_id"]
        assert tool["version"]
        assert tool["split"] in {"seen", "unseen", "evolution", "hard_negative", "distractor"}
        assert "side_effects" in tool
        assert "permission_required" in tool
        assert "dependencies" in tool
        assert tool["examples"]


def test_contracts_have_full_plan_metadata() -> None:
    contracts = ContractRegistry.from_file("data/tool_contracts.json")

    for contract in contracts.all():
        assert contract["contract_id"]
        assert contract["version"]
        assert "permission_required" in contract
        assert contract["risk_level"] in {"low", "medium", "high"}
        assert "side_effects" in contract
        assert "postconditions" in contract
        assert "tool_dependencies" in contract
        assert "failure_action" in contract
        assert contract["examples"]
