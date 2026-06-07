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
