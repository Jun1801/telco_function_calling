from src.reward.feedback_renderer import render_teacher_feedback


def _fb(errors, status="schema_invalid"):
    return {"machine_status": status, "errors": errors, "feedback_text": []}


def test_invalid_enum_vi_surfaces_param_expected_actual() -> None:
    fb = _fb([{
        "type": "schema_invalid", "code": "invalid_enum", "path": "arguments.data_level",
        "expected": ["day", "week", "month"], "actual": "hourly",
        "message": "Invalid value for data_level", "suggested_action": "ask_clarification",
    }])
    out = render_teacher_feedback(fb, lang="vi")
    assert "data_level" in out
    assert "hourly" in out
    assert "day" in out  # expected values listed
    assert "ask_clarification" in out


def test_invalid_enum_en_renders_english() -> None:
    fb = _fb([{
        "type": "schema_invalid", "code": "invalid_enum", "path": "arguments.data_level",
        "expected": ["day", "week"], "actual": "hourly",
        "message": "Invalid value for data_level", "suggested_action": "ask_clarification",
    }])
    out = render_teacher_feedback(fb, lang="en")
    # English keywords present, not Vietnamese
    assert "expected" in out.lower()
    assert "data_level" in out
    assert "Kỳ vọng" not in out


def test_contract_precondition_surfaces_condition_and_actual() -> None:
    fb = _fb([{
        "type": "contract_invalid", "code": "precondition_failed", "path": "subscriber.status",
        "expected": "active", "actual": "suspended",
        "message": "data packages require an active subscriber", "suggested_action": "abstain",
    }], status="contract_invalid")
    out = render_teacher_feedback(fb, lang="vi")
    assert "subscriber.status" in out
    assert "active" in out and "suspended" in out
    assert "abstain" in out


def test_ok_status_returns_informative_confirmation() -> None:
    fb = {"machine_status": "ok", "errors": [], "feedback_text": ["passed"]}
    out_vi = render_teacher_feedback(fb, lang="vi")
    out_en = render_teacher_feedback(fb, lang="en")
    assert out_vi.strip() != ""
    assert out_en.strip() != ""
    # Should not claim an error when status is ok
    assert "lỗi" not in out_vi.lower() or "không" in out_vi.lower()


def test_missing_structured_fields_falls_back_to_message() -> None:
    fb = _fb([{"type": "wrong_action", "message": "The task requires a function call."}])
    out = render_teacher_feedback(fb, lang="vi")
    assert "The task requires a function call." in out


def test_multiple_errors_render_multiple_lines() -> None:
    fb = _fb([
        {"type": "schema_invalid", "code": "missing_arg", "path": "arguments.location_code",
         "message": "Missing required argument: location_code", "suggested_action": "ask_clarification"},
        {"type": "schema_invalid", "code": "invalid_enum", "path": "arguments.tech_type",
         "expected": ["4G", "5G"], "actual": "6G", "message": "bad enum", "suggested_action": "ask_clarification"},
    ])
    out = render_teacher_feedback(fb, lang="vi")
    assert out.count("\n") >= 1
    assert "location_code" in out and "tech_type" in out


def test_unknown_lang_defaults_to_vietnamese() -> None:
    fb = _fb([{"type": "schema_invalid", "code": "invalid_enum", "path": "arguments.x",
               "expected": ["a"], "actual": "z", "message": "m", "suggested_action": "ask_clarification"}])
    out = render_teacher_feedback(fb, lang="fr")
    assert "Kỳ vọng" in out or "kỳ vọng" in out


def test_wrong_argument_value_vi_hidden_vs_revealed() -> None:
    hidden = _fb([{"type": "wrong_call", "code": "wrong_argument_value",
                   "path": "arguments.location_code", "actual": "HCM",
                   "suggested_action": "fix_arguments"}], status="wrong_call")
    out_h = render_teacher_feedback(hidden, lang="vi")
    assert "location_code" in out_h and "HCM" in out_h
    assert "HNI" not in out_h  # gold not present

    revealed = _fb([{"type": "wrong_call", "code": "wrong_argument_value",
                     "path": "arguments.location_code", "actual": "HCM", "expected": "HNI",
                     "suggested_action": "fix_arguments"}], status="wrong_call")
    out_r = render_teacher_feedback(revealed, lang="vi")
    assert "HNI" in out_r


def test_wrong_function_and_invalid_code_render() -> None:
    fb = _fb([
        {"type": "wrong_call", "code": "wrong_function", "path": "kqi_province", "actual": "kqi_province",
         "suggested_action": "call_function"},
        {"type": "wrong_call", "code": "invalid_code", "path": "arguments.location_code", "actual": "HANOI",
         "suggested_action": "fix_arguments"},
    ], status="wrong_call")
    out = render_teacher_feedback(fb, lang="vi")
    assert "kqi_province" in out
    assert "HANOI" in out
    assert out.count("\n") >= 2  # header + 2 lines
