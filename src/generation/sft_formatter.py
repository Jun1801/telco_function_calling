from __future__ import annotations

import json
from typing import Any

from src.registry.tool_registry import ToolRegistry


QWEN_FAMILY_MODELS = [
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3-Coder",
    "Qwen2.5-Coder-7B-Instruct",
]

SYSTEM_PROMPT = """You are a contract-aware Telco function-calling agent.
Read the user request, available tool schemas, and business constraints.
Return only a JSON object with one of these actions:
{"action":"call_function","call":{"tool_name":"...","arguments":{...}}}
{"action":"call_functions","calls":[{"tool_name":"...","arguments":{...}}]}
{"action":"ask_clarification","asked_slots":["..."]}
{"action":"abstain","reason":"..."}
Use call_functions for multi-step or parallel requests. Ask for clarification when a required or invalid argument can be corrected.
If a tool contract or business precondition is violated, abstain instead of asking for slots or calling helper tools.
Never call a tool if required arguments are missing, the tool is deprecated, or the contract is unsafe."""


def format_sample_for_sft(sample: dict[str, Any], tool_registry: ToolRegistry) -> dict[str, Any]:
    tools = _select_tools(sample, tool_registry)
    assistant_payload = _assistant_payload(sample)
    return {
        "id": sample["id"],
        "source": sample.get("source", "telco_toolace_mini"),
        "base_model_family": "qwen",
        "supported_base_models": QWEN_FAMILY_MODELS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "tools": tools,
        "expected_action": sample["expected_action"],
        "gold_call": sample.get("gold_call"),
        "gold_calls": sample.get("gold_calls"),
        "metadata": {
            "split": sample["split"],
            "scenario": sample["scenario"],
            "scenario_family": sample.get("scenario_family"),
            "expected_status": sample.get("expected_status"),
            "customer_verified": sample.get("customer_verified", True),
            "toolace_validation": sample.get("toolace_validation"),
        },
    }


_STEP_PLACEHOLDER = "<from_step_1>"


def _assistant_payload(sample: dict[str, Any]) -> dict[str, Any]:
    action = sample["expected_action"]
    if action == "call_functions":
        calls = sample.get("gold_steps") or sample.get("gold_calls") or []
        # Multi-step gold must be ReAct-decomposed first; an unresolved placeholder
        # would teach the model to emit the literal "<from_step_1>" token.
        assert _STEP_PLACEHOLDER not in json.dumps(calls, ensure_ascii=False), (
            f"Unresolved {_STEP_PLACEHOLDER} in gold for sample {sample.get('id')}; "
            "run build_multistep_react before SFT formatting."
        )
        return {"action": "call_functions", "calls": calls}
    if action == "call_function":
        return {"action": "call_function", "call": sample["gold_call"]}
    if action == "ask_clarification":
        return {"action": "ask_clarification", "asked_slots": sample.get("missing_slots", [])}
    if action == "abstain":
        prediction = sample.get("prediction", {})
        return {"action": "abstain", "reason": prediction.get("reason", "unsafe or unsupported request")}
    raise ValueError(f"Unsupported expected action: {action}")


def _select_tools(sample: dict[str, Any], tool_registry: ToolRegistry) -> list[dict[str, Any]]:
    names: set[str] = set()
    for key in ("gold_call", "call"):
        call = sample.get(key)
        if call:
            names.add(call["tool_name"])
    checker_call = sample.get("checker_call")
    if checker_call:
        names.add(checker_call["tool_name"])
    for call in sample.get("gold_calls") or []:
        names.add(call["tool_name"])
    for call in sample.get("gold_steps") or []:
        names.add(call["tool_name"])

    if sample.get("scenario") in {"deprecated_tool", "unsupported_or_deprecated"}:
        replacement = (sample.get("gold_call") or {}).get("tool_name")
        if replacement:
            names.add(replacement)

    if not names:
        names.update(_fallback_tool_names(sample))

    selected = []
    for name in sorted(names):
        tool = tool_registry.get(name)
        if tool is not None:
            selected.append(_compact_tool(tool))
    return selected


def _fallback_tool_names(sample: dict[str, Any]) -> list[str]:
    instruction = sample["instruction"].lower()
    if "data package" in instruction:
        return ["add_data_package"]
    if "plan" in instruction:
        return ["change_plan", "get_current_plan"]
    if "roaming" in instruction:
        return ["enable_roaming", "disable_roaming"]
    if "otp" in instruction:
        return ["send_otp", "verify_otp"]
    if "support ticket" in instruction:
        return ["open_support_ticket"]
    return ["get_customer_profile"]


def _compact_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
        "status": tool.get("status"),
        "deprecated": tool.get("deprecated", False),
        "replacement_tool": tool.get("replacement_tool"),
        "risk_level": tool.get("risk_level"),
        "permission_required": tool.get("permission_required"),
        "side_effects": tool.get("side_effects", []),
    }
