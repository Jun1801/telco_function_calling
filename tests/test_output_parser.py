from src.model.output_parser import parse_model_output


def test_parse_valid_json_action() -> None:
    output = '{"action":"call_function","call":{"tool_name":"get_balance","arguments":{"customer_id":"C001"}}}'

    parsed = parse_model_output(output)

    assert parsed["action"] == "call_function"
    assert parsed["call"]["tool_name"] == "get_balance"


def test_parse_markdown_json_action() -> None:
    output = '```json\n{"action":"ask_clarification","asked_slots":["package_code"]}\n```'

    parsed = parse_model_output(output)

    assert parsed == {"action": "ask_clarification", "asked_slots": ["package_code"]}


def test_parse_invalid_output_returns_abstain_parse_error() -> None:
    parsed = parse_model_output("not json")

    assert parsed["action"] == "abstain"
    assert parsed["reason"] == "parse_error"
    assert "parse_error" in parsed


def test_parse_multi_call_json_action() -> None:
    output = '{"action":"call_functions","calls":[{"tool_name":"get_balance","arguments":{"customer_id":"C001"}},{"tool_name":"get_usage","arguments":{"customer_id":"C001"}}]}'

    parsed = parse_model_output(output)

    assert parsed["action"] == "call_functions"
    assert [call["tool_name"] for call in parsed["calls"]] == ["get_balance", "get_usage"]


def test_parse_consecutive_call_json_objects_as_multi_call() -> None:
    output = (
        '{"action":"call_function","call":{"tool_name":"get_balance","arguments":{"customer_id":"C001"}}}\n'
        '{"action":"call_function","call":{"tool_name":"get_usage","arguments":{"customer_id":"C001"}}}'
    )

    parsed = parse_model_output(output)

    assert parsed["action"] == "call_functions"
    assert [call["tool_name"] for call in parsed["calls"]] == ["get_balance", "get_usage"]
