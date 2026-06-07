from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.generation.sft_formatter import QWEN_FAMILY_MODELS


SUPPORTED_PUBLIC_SOURCES = {
    "xlam",
    "toolace",
    "apigen_mt",
    "xlam_irrelevance",
    "hermes_fc",
}


@dataclass(frozen=True)
class PublicWarmupLoadResult:
    records: list[dict[str, Any]]
    skipped: int


class PublicWarmupLoader:
    """Local-first normalizer for public general SFT function-calling data."""

    def normalize_many(self, rows: Iterable[dict[str, Any]], source: str) -> PublicWarmupLoadResult:
        if source not in SUPPORTED_PUBLIC_SOURCES:
            raise ValueError(f"Unsupported public warm-up source: {source}")

        records: list[dict[str, Any]] = []
        skipped = 0
        for index, row in enumerate(rows):
            try:
                records.append(self.normalize(row, source, index))
            except (KeyError, TypeError, ValueError):
                skipped += 1
        return PublicWarmupLoadResult(records=records, skipped=skipped)

    def normalize(self, row: dict[str, Any], source: str, index: int = 0) -> dict[str, Any]:
        if source == "toolace":
            return _normalize_toolace(row, index)
        if source == "xlam":
            return _normalize_xlam(row, index)
        if source == "apigen_mt":
            return _normalize_generic_messages(row, source, index)
        if source == "xlam_irrelevance":
            return _normalize_irrelevance(row, source, index)
        if source == "hermes_fc":
            return _normalize_generic_messages(row, source, index)
        raise ValueError(f"Unsupported public warm-up source: {source}")


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if input_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported input file structure: {path}")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def demo_rows() -> list[dict[str, Any]]:
    return [
        {
            "query": "What is the weather in Hanoi?",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "required": ["city"],
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "answer": {"name": "get_weather", "arguments": {"city": "Hanoi"}},
        },
        {
            "query": "Find flights from Hanoi to Tokyo.",
            "tools": [
                {
                    "name": "search_flights",
                    "description": "Search flights.",
                    "parameters": {
                        "type": "object",
                        "required": ["from", "to"],
                        "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                    },
                }
            ],
            "answer": {"name": "search_flights", "arguments": {"from": "Hanoi", "to": "Tokyo"}},
        },
    ]


def _normalize_toolace(row: dict[str, Any], index: int) -> dict[str, Any]:
    if "conversations" in row:
        conversations = row["conversations"]
        instruction = _first_turn(conversations, {"user", "human"})
        assistant_value = _last_turn(conversations, {"assistant", "gpt"})
        tools = _parse_maybe_json(row.get("tools", []))
        if not instruction or assistant_value is None:
            raise ValueError("ToolACE conversations row requires user and assistant turns")
        payload = _assistant_payload_from_value(assistant_value)
        system = row.get("system")
    else:
        instruction = row.get("query") or row.get("instruction")
        tools = _parse_maybe_json(row["tools"])
        answer = row.get("answer") or row.get("gold_call") or row.get("call")
        if not instruction or not answer:
            raise ValueError("ToolACE-like row requires query/instruction and answer/call")
        payload = _call_payload(answer)
        system = None
    return _record(
        record_id=f"public_toolace_{index:06d}",
        source="toolace",
        user_content=instruction,
        assistant_payload=payload,
        tools=tools,
        expected_action=payload["action"],
        metadata={"raw_source": "toolace", "system": system},
    )


def _normalize_xlam(row: dict[str, Any], index: int) -> dict[str, Any]:
    instruction = row.get("query") or row.get("instruction") or row.get("user")
    tools = _parse_maybe_json(row.get("tools") or row.get("available_tools"))
    answer = _parse_maybe_json(row.get("answers") or row.get("answer") or row.get("output") or row.get("tool_call"))
    if not instruction or tools is None or answer is None:
        raise ValueError("xLAM-like row requires instruction, tools, and answer")
    if answer == []:
        payload = {"action": "abstain", "reason": "No relevant tool is available."}
    else:
        payload = _call_payload(answer)
    return _record(
        record_id=f"public_xlam_{index:06d}",
        source="xlam",
        user_content=instruction,
        assistant_payload=payload,
        tools=tools,
        expected_action=payload["action"],
        metadata={"raw_source": "xlam"},
    )


def _normalize_generic_messages(row: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    if "conversations" in row:
        messages = row["conversations"]
        tools = _parse_maybe_json(row.get("tools", []))
        user_content = _first_turn(messages, {"human", "user"})
        function_call = _last_turn(messages, {"function_call"})
        assistant_content = function_call or _last_turn(messages, {"gpt", "assistant"})
        if not user_content or assistant_content is None:
            raise ValueError("ShareGPT row requires user/human and assistant/function_call turns")
        payload = _assistant_payload_from_value(assistant_content)
        system = row.get("system")
    else:
        messages = row["messages"]
        tools = _parse_maybe_json(row.get("tools", []))
        user_content = next(message["content"] for message in messages if message["role"] == "user")
        assistant_content = next(message["content"] for message in reversed(messages) if message["role"] == "assistant")
        payload = _assistant_payload_from_value(assistant_content)
        system = None
    return _record(
        record_id=f"public_{source}_{index:06d}",
        source=source,
        user_content=user_content,
        assistant_payload=payload,
        tools=tools,
        expected_action=payload.get("action", "call_function"),
        metadata={"raw_source": source, "system": system},
    )


def _normalize_irrelevance(row: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    instruction = row.get("query") or row.get("instruction") or row.get("user")
    if not instruction:
        raise ValueError("irrelevance row requires query/instruction/user")
    answers = _parse_maybe_json(row.get("answers", []))
    if answers not in (None, []):
        raise ValueError("irrelevance rows should not contain tool-call answers")
    payload = {"action": "abstain", "reason": row.get("reason", "No relevant tool is available.")}
    return _record(
        record_id=f"public_{source}_{index:06d}",
        source=source,
        user_content=instruction,
        assistant_payload=payload,
        tools=_parse_maybe_json(row.get("tools", [])),
        expected_action="abstain",
        metadata={"raw_source": source, "irrelevance": True},
    )


def _call_payload(answer: Any) -> dict[str, Any]:
    answer = _parse_maybe_json(answer)
    if isinstance(answer, list):
        if not answer:
            return {"action": "abstain", "reason": "No relevant tool is available."}
        calls = [_single_call_payload(item)["call"] for item in answer]
        if len(calls) == 1:
            return {"action": "call_function", "call": calls[0]}
        return {"action": "call_function", "calls": calls}
    if "action" in answer:
        return answer
    return _single_call_payload(answer)


def _single_call_payload(answer: dict[str, Any]) -> dict[str, Any]:
    name = answer.get("name") or answer.get("tool_name") or answer.get("function")
    arguments = answer.get("arguments") or answer.get("args") or {}
    if not name:
        raise ValueError("tool call answer requires name/tool_name/function")
    return {"action": "call_function", "call": {"tool_name": name, "arguments": arguments}}


def _assistant_payload_from_value(value: Any) -> dict[str, Any]:
    parsed = _parse_maybe_json(value)
    if isinstance(parsed, dict):
        if "action" in parsed:
            return parsed
        if any(key in parsed for key in ("name", "tool_name", "function")):
            return _call_payload(parsed)
    if isinstance(parsed, list):
        return _call_payload(parsed)
    # ToolACE bracket notation is kept as raw target text for warm-up.
    return {"action": "raw_text", "content": str(value)}


def _parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _first_turn(conversations: list[dict[str, Any]], roles: set[str]) -> str | None:
    for turn in conversations:
        role = turn.get("from") or turn.get("role")
        if role in roles:
            return turn.get("value") or turn.get("content")
    return None


def _last_turn(conversations: list[dict[str, Any]], roles: set[str]) -> Any:
    for turn in reversed(conversations):
        role = turn.get("from") or turn.get("role")
        if role in roles:
            return turn.get("value") or turn.get("content")
    return None


def _record(
    record_id: str,
    source: str,
    user_content: str,
    assistant_payload: dict[str, Any],
    tools: list[dict[str, Any]],
    expected_action: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "source": source,
        "base_model_family": "qwen",
        "supported_base_models": QWEN_FAMILY_MODELS,
        "messages": [
            {"role": "system", "content": "You are a function-calling assistant. Return only the requested JSON action."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "tools": tools,
        "expected_action": expected_action,
        "metadata": metadata,
    }
