from __future__ import annotations

import json
from typing import Any

from src.generation.sft_formatter import QWEN_FAMILY_MODELS, SYSTEM_PROMPT, SYSTEM_PROMPT_REAL
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def build_prompt_messages(
    sample: dict[str, Any],
    tool_registry: ToolRegistry,
    contract_registry: ContractRegistry | None = None,
    max_tools: int = 8,
    extra_tools: list[dict[str, Any]] | None = None,
    references: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    tools = _select_prompt_tools(sample, tool_registry, max_tools)
    if extra_tools:
        # Inject tools not resolvable from the registry (e.g. a masked func_X
        # schema embedded in the sample). Prepended; deduped by name. Project to
        # schema fields only so masking bookkeeping (original_name/parameter_map)
        # never leaks the real tool name into the prompt.
        existing = {tool["name"] for tool in tools}
        tools = [_project_extra_tool(t) for t in extra_tools if t["name"] not in existing] + tools
    context: dict[str, Any] = {
        "base_model_family": "qwen",
        "supported_base_models": QWEN_FAMILY_MODELS,
        "available_tools": tools,
    }
    # Contracts/customer_verified are transactional concepts — only the synthetic
    # registry carries them. Real KPI tools are read-only (contract_registry None).
    if contract_registry is not None:
        contracts = [_compact_contract(contract_registry.get(tool["name"])) for tool in tools]
        context["customer_verified"] = sample.get("customer_verified", True)
        context["tool_contracts"] = [contract for contract in contracts if contract]
    # Reference code catalogues: injected for real tools so the model can map
    # Vietnamese names to codes (e.g. "Hà Nội" → "HNI") without guessing.
    if references:
        loc_map: dict[str, str] = {}
        for item in references.get("location_code", []):
            loc_map.setdefault(item["code"], item["name"])
        context["reference_codes"] = {
            "location_code": loc_map,
            "kpi_code": {item["code"]: item["meaning"] for item in references.get("kpi_code", [])},
            "unit_code": {item["code"]: item["name"] for item in references.get("unit_code", [])},
        }
    is_real = contract_registry is None
    sys_prompt = SYSTEM_PROMPT_REAL if is_real else SYSTEM_PROMPT
    context_label = "Available tool schemas:" if is_real else "Available tool and contract context:"
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": (
                f"User request: {sample['instruction']}\n\n"
                f"{context_label}\n"
                f"{json.dumps(context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _select_prompt_tools(sample: dict[str, Any], tool_registry: ToolRegistry, max_tools: int) -> list[dict[str, Any]]:
    names: list[str] = []
    # When a tool is masked (func_X injected via extra_tools), the real tool it
    # shadows must NOT also appear — otherwise the model reads the real name and
    # the schema-only masking test is meaningless (RQ3).
    masked = sample.get("masked_tool")
    shadowed = {masked["original_name"]} if masked and masked.get("original_name") else set()

    def _add(name: str) -> None:
        if name not in names and name not in shadowed:
            names.append(name)

    for call_key in ("gold_call", "call"):
        call = sample.get(call_key)
        if call:
            _add(call["tool_name"])
    checker_call = sample.get("checker_call")
    if checker_call:
        _add(checker_call["tool_name"])
    for call in sample.get("gold_calls") or []:
        _add(call["tool_name"])
    for call in sample.get("gold_steps") or []:
        _add(call["tool_name"])

    # Distractor fallbacks are synthetic-registry-only: they pollute real-tool
    # prompts with synthetic names and would re-leak a masked real name.
    if not masked and sample.get("source") != "real_tool_xlsx":
        for fallback in _keyword_fallback_tools(sample["instruction"]):
            if len(names) >= max_tools:
                break
            _add(fallback)
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


_EXTRA_TOOL_FIELDS = ("name", "description", "parameters", "status", "deprecated",
                      "replacement_tool", "risk_level", "permission_required", "side_effects")


def _project_extra_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {k: tool[k] for k in _EXTRA_TOOL_FIELDS if k in tool}


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
