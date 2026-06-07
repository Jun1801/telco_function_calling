from __future__ import annotations

import json
from typing import Any

from src.generation.sft_formatter import QWEN_FAMILY_MODELS, SYSTEM_PROMPT
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def build_prompt_messages(
    sample: dict[str, Any],
    tool_registry: ToolRegistry,
    contract_registry: ContractRegistry,
    max_tools: int = 8,
) -> list[dict[str, str]]:
    tools = _select_prompt_tools(sample, tool_registry, max_tools)
    contracts = [_compact_contract(contract_registry.get(tool["name"])) for tool in tools]
    context = {
        "base_model_family": "qwen",
        "supported_base_models": QWEN_FAMILY_MODELS,
        "customer_verified": sample.get("customer_verified", True),
        "available_tools": tools,
        "tool_contracts": [contract for contract in contracts if contract],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User request: {sample['instruction']}\n\n"
                "Available tool and contract context:\n"
                f"{json.dumps(context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _select_prompt_tools(sample: dict[str, Any], tool_registry: ToolRegistry, max_tools: int) -> list[dict[str, Any]]:
    names: list[str] = []
    for call_key in ("gold_call", "call"):
        call = sample.get(call_key)
        if call and call["tool_name"] not in names:
            names.append(call["tool_name"])
    for call in sample.get("gold_calls") or []:
        if call["tool_name"] not in names:
            names.append(call["tool_name"])
    for call in sample.get("gold_steps") or []:
        if call["tool_name"] not in names:
            names.append(call["tool_name"])

    for fallback in _keyword_fallback_tools(sample["instruction"]):
        if len(names) >= max_tools:
            break
        if fallback not in names:
            names.append(fallback)

    for fallback in ["get_balance", "get_current_plan", "add_data_package", "open_support_ticket"]:
        if len(names) >= max_tools:
            break
        if fallback not in names:
            names.append(fallback)

    tools = []
    for name in names[:max_tools]:
        tool = tool_registry.get(name)
        if tool:
            tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "status": tool.get("status"),
                    "deprecated": tool.get("deprecated"),
                    "replacement_tool": tool.get("replacement_tool"),
                    "risk_level": tool.get("risk_level"),
                    "permission_required": tool.get("permission_required"),
                    "side_effects": tool.get("side_effects", []),
                }
            )
    return tools


def _keyword_fallback_tools(instruction: str) -> list[str]:
    text = instruction.lower()
    if "transfer ownership" in text:
        return ["transfer_ownership"]
    if "credit limit" in text:
        return ["increase_credit_limit"]
    if "autopay" in text:
        return ["register_autopay", "remove_autopay"]
    if "premium sms" in text:
        return ["block_premium_sms", "unblock_premium_sms"]
    if "roaming" in text:
        return ["enable_roaming", "disable_roaming"]
    if "support ticket" in text:
        return ["open_support_ticket"]
    if "data package" in text:
        return ["add_data_package"]
    if "sim" in text:
        return ["report_lost_sim", "replace_sim", "activate_esim"]
    return []


def _compact_contract(contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if contract is None:
        return None
    return {
        "tool_name": contract["tool_name"],
        "permission_required": contract.get("permission_required"),
        "risk_level": contract.get("risk_level"),
        "preconditions": contract.get("preconditions", []),
        "postconditions": contract.get("postconditions", []),
        "failure_action": contract.get("failure_action"),
    }
