import json

from src.generation.sft_formatter import QWEN_FAMILY_MODELS, format_sample_for_sft
from src.generation.toolace_mini import TelcoToolACEMiniPipeline
from src.registry.tool_registry import ToolRegistry


def test_sft_formatter_builds_qwen_family_chat_record() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    sample = TelcoToolACEMiniPipeline("data").generate_verified_samples()[0]

    record = format_sample_for_sft(sample, registry)

    assert record["base_model_family"] == "qwen"
    assert record["supported_base_models"] == QWEN_FAMILY_MODELS
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    assistant_payload = json.loads(record["messages"][-1]["content"])
    assert assistant_payload["action"] in {"call_function", "call_functions", "ask_clarification", "abstain"}
    assert record["tools"]
    assert record["metadata"]["toolace_validation"]


def test_sft_formatter_outputs_ask_clarification_payload() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    samples = TelcoToolACEMiniPipeline("data").generate_verified_samples()
    sample = next(item for item in samples if item["expected_action"] == "ask_clarification")

    record = format_sample_for_sft(sample, registry)
    assistant_payload = json.loads(record["messages"][-1]["content"])

    assert assistant_payload == {
        "action": "ask_clarification",
        "asked_slots": sample["missing_slots"],
    }


def test_sft_formatter_outputs_multi_call_payload() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    samples = TelcoToolACEMiniPipeline("data").generate_verified_samples()
    sample = next(item for item in samples if item["expected_action"] == "call_functions")

    record = format_sample_for_sft(sample, registry)
    assistant_payload = json.loads(record["messages"][-1]["content"])

    assert assistant_payload == {
        "action": "call_functions",
        "calls": sample.get("gold_steps") or sample.get("gold_calls"),
    }
