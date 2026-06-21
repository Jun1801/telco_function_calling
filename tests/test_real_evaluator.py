from src.evaluation.real_evaluator import evaluate_real_prediction
from src.registry.tool_registry import ToolRegistry

REG = ToolRegistry.from_file("data/real_tools.json")


def _call(tool, args):
    return {"action": "call_function", "call": {"tool_name": tool, "arguments": args}}


def test_valid_real_call_scores_full_reward() -> None:
    sample = {
        "id": "r1", "expected_action": "call_function",
        "gold_call": {"tool_name": "vung_phu_province",
                      "arguments": {"location_code": "HNI", "tech_type": "5G"}},
    }
    res = evaluate_real_prediction(sample, _call("vung_phu_province", {"location_code": "HNI", "tech_type": "5G"}), REG)
    assert res["reward_strict"] == 1.0
    assert res["feedback"]["machine_status"] == "ok"
    assert res["metrics"]["schema_validity"] == 1.0


def test_does_not_crash_without_customer_id() -> None:
    # Bug 1 regression: real KPI tools have no customer_id.
    sample = {"id": "r2", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    res = evaluate_real_prediction(sample, _call("vung_phu_province", {"location_code": "HNI", "tech_type": "5G"}), REG)
    assert "reward_total" in res  # completed without KeyError


def test_invalid_enum_produces_structured_error() -> None:
    sample = {"id": "r3", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    res = evaluate_real_prediction(sample, _call("vung_phu_province", {"location_code": "HNI", "tech_type": "6G"}), REG)
    assert res["reward_strict"] == 0.0
    assert res["feedback"]["machine_status"] == "schema_invalid"
    err = res["feedback"]["errors"][0]
    assert err["code"] == "invalid_enum"
    assert err["actual"] == "6G"
    assert "5G" in err["expected"]


def test_wrong_tool_selection_zero_function_accuracy() -> None:
    sample = {"id": "r4", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    res = evaluate_real_prediction(sample, _call("kqi_province",
                                                 {"location_code": "HNI", "from_date": "2026-06-01",
                                                  "to_date": "2026-06-30", "data_level": "day"}), REG)
    assert res["metrics"]["function_selection_accuracy"] == 0.0
    assert res["reward_strict"] == 0.0


def test_masked_tool_validated_against_embedded_schema() -> None:
    sample = {
        "id": "r5", "expected_action": "call_function",
        "masked_tool": {"name": "func_3", "parameters": REG.get("vung_phu_province")["parameters"]},
        "gold_call": {"tool_name": "func_3", "arguments": {"location_code": "HNI", "tech_type": "5G"}},
    }
    res = evaluate_real_prediction(sample, _call("func_3", {"location_code": "HNI", "tech_type": "5G"}), REG)
    assert res["reward_strict"] == 1.0


def test_ask_clarification_scored_on_slot_recall() -> None:
    sample = {"id": "r6", "expected_action": "ask_clarification", "missing_slots": ["location_code"]}
    good = evaluate_real_prediction(sample, {"action": "ask_clarification", "asked_slots": ["location_code"]}, REG)
    bad = evaluate_real_prediction(sample, _call("vung_phu_province", {"location_code": "HNI", "tech_type": "5G"}), REG)
    assert good["reward_strict"] == 1.0
    assert bad["reward_strict"] == 0.0


def test_feedback_renders_for_teacher() -> None:
    from src.reward.feedback_renderer import render_teacher_feedback
    sample = {"id": "r7", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    res = evaluate_real_prediction(sample, _call("vung_phu_province", {"location_code": "HNI", "tech_type": "6G"}), REG)
    text = render_teacher_feedback(res["feedback"], lang="vi")
    assert "tech_type" in text and "6G" in text


# ---- Source A: gold-comparison feedback ----

import json as _json
REFS = _json.load(open("data/real_reference_codes.json"))


def test_schema_valid_but_wrong_value_no_longer_reports_ok() -> None:
    # Bug: a schema-valid call with the wrong location used to render "ok".
    sample = {"id": "a1", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HCM", "tech_type": "5G"})
    res = evaluate_real_prediction(sample, pred, REG, references=REFS)
    assert res["feedback"]["machine_status"] != "ok"
    codes = [e["code"] for e in res["feedback"]["errors"]]
    assert "wrong_argument_value" in codes


def test_wrong_value_hides_gold_by_default() -> None:
    sample = {"id": "a2", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HCM", "tech_type": "5G"})
    err = [e for e in evaluate_real_prediction(sample, pred, REG, references=REFS)["feedback"]["errors"]
           if e["code"] == "wrong_argument_value"][0]
    assert err.get("actual") == "HCM"        # model's own value is fine to show
    assert "expected" not in err              # gold value hidden by default


def test_wrong_value_reveals_gold_when_flag_set() -> None:
    sample = {"id": "a3", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HCM", "tech_type": "5G"})
    err = [e for e in evaluate_real_prediction(sample, pred, REG, references=REFS, reveal_gold=True)["feedback"]["errors"]
           if e["code"] == "wrong_argument_value"][0]
    assert err.get("expected") == "HNI"


def test_wrong_function_reported() -> None:
    sample = {"id": "a4", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("kqi_province", {"location_code": "HNI", "from_date": "2026-06-01",
                                  "to_date": "2026-06-30", "data_level": "day"})
    codes = [e["code"] for e in evaluate_real_prediction(sample, pred, REG, references=REFS)["feedback"]["errors"]]
    assert "wrong_function" in codes


def test_correct_call_has_no_gold_errors() -> None:
    sample = {"id": "a5", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HNI", "tech_type": "5G"})
    res = evaluate_real_prediction(sample, pred, REG, references=REFS)
    assert res["feedback"]["machine_status"] == "ok"
    assert res["feedback"]["errors"] == []


# ---- Source B: reference-code validation ----

def test_invalid_location_code_is_caught() -> None:
    # After enum injection, location_code has an enum, so a bad code is caught by the
    # schema layer (invalid_enum) — earlier/stronger than the source-B reference check.
    sample = {"id": "b1", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HANOI", "tech_type": "5G"})
    res = evaluate_real_prediction(sample, pred, REG, references=REFS)
    assert res["reward_strict"] == 0.0
    codes = [e["code"] for e in res["feedback"]["errors"]]
    assert "invalid_enum" in codes or "invalid_code" in codes


def test_valid_location_code_passes_reference_check() -> None:
    sample = {"id": "b2", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HNI", "tech_type": "5G"})
    codes = [e["code"] for e in evaluate_real_prediction(sample, pred, REG, references=REFS)["feedback"]["errors"]]
    assert "invalid_code" not in codes


def test_reference_check_skipped_without_references() -> None:
    sample = {"id": "b3", "expected_action": "call_function",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = _call("vung_phu_province", {"location_code": "HANOI", "tech_type": "5G"})
    # No references passed → no invalid_code error (but still scored)
    codes = [e["code"] for e in evaluate_real_prediction(sample, pred, REG)["feedback"]["errors"]]
    assert "invalid_code" not in codes
