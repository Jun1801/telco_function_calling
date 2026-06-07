from src.generation.toolace_mini import TelcoToolACEMiniPipeline
from src.model.prompt_builder import build_prompt_messages
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def test_prompt_contains_qwen_instruction_tools_and_contracts() -> None:
    sample = TelcoToolACEMiniPipeline("data").generate_verified_samples()[0]
    messages = build_prompt_messages(
        sample,
        ToolRegistry.from_file("data/tools.json"),
        ContractRegistry.from_file("data/tool_contracts.json"),
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Return only a JSON object" in messages[0]["content"]
    assert sample["instruction"] in messages[1]["content"]
    assert "available_tools" in messages[1]["content"]
    assert "tool_contracts" in messages[1]["content"]
