from __future__ import annotations

import json
import re
from typing import Any


def parse_model_output(text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return _parse_error("No JSON object found in model output.", text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        return _parse_error(f"Invalid JSON: {error.msg}", text)
    if not isinstance(payload, dict):
        return _parse_error("Top-level output must be a JSON object.", text)
    return _normalize_payload(payload, text)


def _extract_json_candidate(text: str) -> str | None:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _normalize_payload(payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
    action = payload.get("action")
    if action == "call_function" and isinstance(payload.get("call"), dict):
        call = payload["call"]
        if "tool_name" in call and isinstance(call.get("arguments", {}), dict):
            return {"action": "call_function", "call": {"tool_name": call["tool_name"], "arguments": call.get("arguments", {})}}
    if action == "ask_clarification":
        return {"action": "ask_clarification", "asked_slots": payload.get("asked_slots", [])}
    if action == "abstain":
        return {"action": "abstain", "reason": payload.get("reason", "model abstained")}
    return _parse_error("JSON object does not match expected action schema.", raw_text)


def _parse_error(message: str, raw_text: str) -> dict[str, Any]:
    return {
        "action": "abstain",
        "reason": "parse_error",
        "parse_error": message,
        "raw_output": raw_text,
    }
