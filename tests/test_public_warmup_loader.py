import json

from src.generation.public_warmup_loader import PublicWarmupLoader


def test_normalizes_toolace_like_row() -> None:
    row = {
        "query": "Get weather in Hanoi.",
        "tools": [{"name": "get_weather", "description": "Get weather.", "parameters": {"type": "object"}}],
        "answer": {"name": "get_weather", "arguments": {"city": "Hanoi"}},
    }

    result = PublicWarmupLoader().normalize_many([row], "toolace")

    assert result.skipped == 0
    record = result.records[0]
    assert record["source"] == "toolace"
    assert record["base_model_family"] == "qwen"
    payload = json.loads(record["messages"][-1]["content"])
    assert payload["call"]["tool_name"] == "get_weather"


def test_normalizes_xlam_like_row() -> None:
    row = {
        "instruction": "Search flights to Tokyo.",
        "available_tools": '[{"name":"search_flights","description":"Search flights.","parameters":{"type":"object"}}]',
        "tool_call": '{"tool_name":"search_flights","arguments":{"to":"Tokyo"}}',
    }

    result = PublicWarmupLoader().normalize_many([row], "xlam")

    assert result.skipped == 0
    assert result.records[0]["source"] == "xlam"
    assert result.records[0]["expected_action"] == "call_function"


def test_normalizes_xlam_answers_list_string() -> None:
    row = {
        "query": "Search flights to Tokyo.",
        "tools": '[{"name":"search_flights","description":"Search flights.","parameters":{"type":"object"}}]',
        "answers": '[{"name":"search_flights","arguments":{"to":"Tokyo"}}]',
    }

    result = PublicWarmupLoader().normalize_many([row], "xlam")
    payload = json.loads(result.records[0]["messages"][-1]["content"])

    assert payload["call"]["tool_name"] == "search_flights"


def test_normalizes_irrelevance_as_abstain() -> None:
    row = {"query": "Tell me a joke when no tools are relevant.", "tools": "[]", "answers": "[]"}

    result = PublicWarmupLoader().normalize_many([row], "xlam_irrelevance")

    payload = json.loads(result.records[0]["messages"][-1]["content"])
    assert payload["action"] == "abstain"
    assert result.records[0]["expected_action"] == "abstain"


def test_normalizes_toolace_conversations_with_raw_bracket_call() -> None:
    row = {
        "system": "You can use tools.",
        "tools": "[]",
        "conversations": [
            {"from": "user", "value": "Get market trends."},
            {"from": "assistant", "value": "[Market Trends API(category='mobile')]"},
        ],
    }

    result = PublicWarmupLoader().normalize_many([row], "toolace")
    payload = json.loads(result.records[0]["messages"][-1]["content"])

    assert payload["action"] == "raw_text"
    assert "Market Trends API" in payload["content"]


def test_normalizes_apigen_function_call_turn() -> None:
    row = {
        "system": "Use tools.",
        "tools": '[{"name":"book_hotel","description":"Book hotel.","parameters":{"type":"object"}}]',
        "conversations": [
            {"from": "human", "value": "Book a hotel in Tokyo."},
            {"from": "function_call", "value": '{"name":"book_hotel","arguments":{"city":"Tokyo"}}'},
            {"from": "observation", "value": '{"ok":true}'},
        ],
    }

    result = PublicWarmupLoader().normalize_many([row], "apigen_mt")
    payload = json.loads(result.records[0]["messages"][-1]["content"])

    assert payload["call"]["tool_name"] == "book_hotel"


def test_skips_malformed_rows() -> None:
    result = PublicWarmupLoader().normalize_many([{"query": "missing answer"}], "toolace")

    assert result.records == []
    assert result.skipped == 1
