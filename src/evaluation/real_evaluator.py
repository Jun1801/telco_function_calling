"""Schema-only evaluator for the real Viettel KPI tools (read-only).

The synthetic `evaluate_prediction` path assumes a telco transaction domain:
subscribers, `customer_id`, contracts and a stateful mock executor. Real KPI
tools have none of that — they are pure read-only analytics keyed on
`location_code` / date ranges. Calling the synthetic scorer on them crashes
(`KeyError: 'customer_id'`). This evaluator scores them with schema validation
only, while keeping the SAME output shape and the SAME structured feedback so
the renderer and VPD pipeline work unchanged.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.metrics import argument_key_f1, argument_value_accuracy
from src.reward.reward_feedback import _issue_dicts
from src.registry.tool_registry import ToolRegistry
from src.validation.schema_validator import SchemaValidator

_VALIDATOR = SchemaValidator()
_PLACEHOLDER = "<from_step_1>"
# Argument names whose value must be a code listed in a reference table (source B).
_CODE_PARAM_TO_TABLE = {"location_code": "location_code", "kpi_code": "kpi_code", "unit_code": "unit_code"}
_MISSING = object()


def _gold_diff_errors(pred_tool, pred_args, gold_tool, gold_args, reveal_gold):
    """Source A: feedback from comparing a schema-valid call against the gold call.

    By default the gold VALUE is not revealed (only which function/argument is
    wrong) — the teacher must still reason out the correct value. reveal_gold=True
    exposes the gold value (for ablation).
    """
    errors = []
    if not gold_tool:
        # No gold match for this predicted call (e.g. an extra parallel call) → one
        # clear "unnecessary call" signal, NOT a per-argument extra_argument spam.
        return [{"type": "wrong_call", "code": "unnecessary_call", "path": pred_tool,
                 "actual": pred_tool, "message": f"Unnecessary/unmatched call: {pred_tool}",
                 "suggested_action": "call_functions"}]
    if pred_tool != gold_tool:
        e = {"type": "wrong_call", "code": "wrong_function", "path": pred_tool,
             "actual": pred_tool, "message": f"Wrong function selected: {pred_tool}",
             "suggested_action": "call_function"}
        if reveal_gold:
            e["expected"] = gold_tool
        return [e]  # wrong tool → argument diff is moot
    for key, gold_val in gold_args.items():
        actual = pred_args.get(key, _MISSING)
        if actual is _MISSING:
            errors.append({"type": "wrong_call", "code": "missing_argument",
                           "path": f"arguments.{key}", "message": f"Missing argument: {key}",
                           "suggested_action": "fix_arguments"})
        elif _PLACEHOLDER in str(actual):
            continue  # multi_step dependency placeholder — not checkable
        elif actual != gold_val:
            e = {"type": "wrong_call", "code": "wrong_argument_value",
                 "path": f"arguments.{key}", "actual": actual,
                 "message": f"Wrong value for {key}", "suggested_action": "fix_arguments"}
            if reveal_gold:
                e["expected"] = gold_val
            errors.append(e)
    for key in pred_args:
        if key not in gold_args:
            errors.append({"type": "wrong_call", "code": "extra_argument",
                           "path": f"arguments.{key}", "actual": pred_args[key],
                           "message": f"Unnecessary argument: {key}", "suggested_action": "fix_arguments"})
    return errors


def _reference_code_errors(pred_args, references):
    """Source B: validate free-string code args against the reference tables.

    location_code/kpi_code have no enum in the schema, so a bad code passes
    SchemaValidator. The reference catalogue catches it. Valid options come from
    the catalogue (not gold), so surfacing them is not a gold leak.
    """
    if not references:
        return []
    errors = []
    for param, table in _CODE_PARAM_TO_TABLE.items():
        if param not in pred_args:
            continue
        value = pred_args[param]
        if _PLACEHOLDER in str(value):
            continue
        valid = {item["code"] for item in references.get(table, [])}
        if valid and value not in valid:
            errors.append({"type": "wrong_call", "code": "invalid_code",
                           "path": f"arguments.{param}", "actual": value,
                           "message": f"Code not in catalogue for {param}: {value}",
                           "suggested_action": "fix_arguments"})
    return errors


def _result(reward_soft: float, reward_strict: float, feedback: dict, metrics: dict) -> dict:
    return {
        "reward_soft": round(reward_soft, 6),
        "reward_strict": round(reward_strict, 6),
        "reward_total": round(0.5 * reward_soft + 0.5 * reward_strict, 6),
        "feedback": feedback,
        "metrics": metrics,
    }


def _tool_for(sample: dict, registry: ToolRegistry, name: str) -> dict | None:
    masked = sample.get("masked_tool")
    if masked and masked.get("name") == name:
        return masked
    return registry.get(name)


def _score_one_call(sample, registry, call, gold, references, reveal_gold) -> tuple[float, str, list[dict], dict]:
    """Return (schema_validity, status, structured_errors, per-call metrics)."""
    pred_tool = call.get("tool_name")
    pred_args = call.get("arguments", {})
    gold_tool = gold.get("tool_name") if gold else None
    gold_args = gold.get("arguments", {}) if gold else {}

    tool = _tool_for(sample, registry, pred_tool)
    issues = _VALIDATOR.validate_call(tool, pred_args, pred_tool)
    metrics = {
        "function_selection_accuracy": 1.0 if pred_tool == gold_tool else 0.0,
        "argument_key_f1": argument_key_f1(pred_args, gold_args),
        "argument_value_accuracy": argument_value_accuracy(pred_args, gold_args),
        "schema_validity": 0.0 if issues else 1.0,
    }
    if issues:  # schema error takes precedence — report it, skip gold/code diff
        errors = [{"type": "schema_invalid", **it} for it in _issue_dicts(issues)]
        return 0.0, "schema_invalid", errors, metrics

    # Schema valid → source A (gold comparison) + source B (reference codes).
    errors = _gold_diff_errors(pred_tool, pred_args, gold_tool, gold_args, reveal_gold)
    errors += _reference_code_errors(pred_args, references)
    return 1.0, ("wrong_call" if errors else "ok"), errors, metrics


def evaluate_real_prediction(
    sample: dict[str, Any],
    prediction: dict[str, Any],
    tool_registry: ToolRegistry,
    references: dict[str, Any] | None = None,
    reveal_gold: bool = False,
) -> dict[str, Any]:
    expected_action = sample.get("expected_action", "call_function")
    predicted_action = prediction.get("action", "call_function")
    action_ok = 1.0 if predicted_action == expected_action else 0.0
    feedback: dict[str, Any] = {"errors": [], "feedback_text": [], "machine_status": "ok"}

    # ---- format/parse error: never count a malformed output as a correct abstain ----
    if prediction.get("parse_error"):
        feedback["machine_status"] = "format_error"
        feedback["errors"].append({"type": "format_error", "code": "parse_error",
                                   "message": prediction.get("parse_error", ""),
                                   "suggested_action": "fix_format"})
        feedback["feedback_text"].append("Output is not valid JSON in the required action format.")
        return _result(0.0, 0.0, feedback, {"action_accuracy": 0.0})

    # ---- ask_clarification ----
    if expected_action == "ask_clarification":
        missing = set(sample.get("missing_slots", []))
        asked = set(prediction.get("asked_slots", []))
        recall = len(missing & asked) / len(missing) if missing else 1.0
        strict = 1.0 if predicted_action == "ask_clarification" and recall == 1.0 else 0.0
        if strict == 0.0:
            feedback["machine_status"] = "wrong_action"
            feedback["errors"].append({"type": "missing_slot_not_handled",
                                       "missing_slots": sorted(missing),
                                       "suggested_action": "ask_clarification"})
        return _result(0.6 * action_ok + 0.4 * recall, strict, feedback,
                       {"action_accuracy": action_ok, "ask_back_accuracy": strict})

    # ---- abstain ----
    if expected_action == "abstain":
        strict = 1.0 if predicted_action == "abstain" else 0.0
        if strict == 0.0:
            feedback["machine_status"] = "wrong_action"
            feedback["errors"].append({"type": "unsafe_or_forbidden_action", "suggested_action": "abstain"})
        return _result(strict, strict, feedback,
                       {"action_accuracy": action_ok, "abstention_accuracy": strict})

    # ---- call_functions (parallel / multi_step) ----
    if expected_action == "call_functions":
        return _evaluate_multi(sample, prediction, tool_registry, action_ok, feedback, references, reveal_gold)

    # ---- call_function (the common case) ----
    if predicted_action != "call_function":
        feedback["machine_status"] = "wrong_action"
        feedback["errors"].append({"type": "wrong_action", "expected": "call_function",
                                   "actual": predicted_action, "suggested_action": "call_function"})
        return _result(0.1 * action_ok, 0.0, feedback, {"action_accuracy": action_ok})

    call = prediction.get("call") or {"tool_name": prediction.get("tool_name"),
                                       "arguments": prediction.get("arguments", {})}
    gold = sample.get("gold_call") or sample.get("call") or {}
    schema_ok, status, errors, m = _score_one_call(sample, tool_registry, call, gold, references, reveal_gold)

    metrics = {"action_accuracy": action_ok, **m,
               "task_success_rate": 1.0 if status == sample.get("expected_status", "ok") else 0.0}
    feedback["machine_status"] = status
    feedback["errors"].extend(errors)
    if errors:
        feedback["feedback_text"].extend(e.get("message", "") for e in errors)

    reward_soft = (0.15 * action_ok
                   + 0.30 * m["function_selection_accuracy"]
                   + 0.20 * m["argument_key_f1"]
                   + 0.20 * m["argument_value_accuracy"]
                   + 0.15 * schema_ok)
    reward_strict = 1.0 if (action_ok == 1.0
                            and m["function_selection_accuracy"] == 1.0
                            and m["argument_key_f1"] == 1.0
                            and m["argument_value_accuracy"] == 1.0
                            and schema_ok == 1.0) else 0.0
    return _result(reward_soft, reward_strict, feedback, metrics)


def _evaluate_multi(sample, prediction, registry, action_ok, feedback, references=None, reveal_gold=False) -> dict:
    pred_calls = prediction.get("calls", [])
    gold_calls = sample.get("gold_calls") or sample.get("gold_steps") or []
    if not pred_calls:
        feedback["machine_status"] = "wrong_action"
        feedback["errors"].append({"type": "wrong_action", "expected": "call_functions",
                                   "actual": prediction.get("action"), "suggested_action": "call_functions"})
        return _result(0.1 * action_ok, 0.0, feedback, {"action_accuracy": action_ok})

    # Set-based matching by tool name (order-independent, per plan §9.2).
    gold_by_tool = {g["tool_name"]: g for g in gold_calls}
    schema_flags, key_f1s, val_accs, sel_hits = [], [], [], 0
    statuses = []
    for call in pred_calls:
        gold = gold_by_tool.get(call.get("tool_name"), {})
        if gold:
            sel_hits += 1
        schema_ok, status, errors, m = _score_one_call(sample, registry, call, gold, references, reveal_gold)
        schema_flags.append(schema_ok)
        statuses.append(status)
        key_f1s.append(m["argument_key_f1"])
        # multi_step step2 uses a symbolic <from_step_1>; skip its value check.
        val_accs.append(1.0 if "<from_step_1>" in str(call.get("arguments", {})) else m["argument_value_accuracy"])
        feedback["errors"].extend(errors)

    n = max(len(gold_calls), len(pred_calls))
    sel = sel_hits / n if n else 0.0
    schema_v = sum(schema_flags) / len(schema_flags) if schema_flags else 0.0
    key_f1 = sum(key_f1s) / len(key_f1s) if key_f1s else 0.0
    val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
    feedback["machine_status"] = "ok" if all(s == "ok" for s in statuses) and sel == 1.0 else (
        next((s for s in statuses if s != "ok"), "wrong_action"))

    metrics = {"action_accuracy": action_ok, "function_selection_accuracy": sel,
               "argument_key_f1": key_f1, "argument_value_accuracy": val_acc, "schema_validity": schema_v}
    reward_soft = 0.15 * action_ok + 0.30 * sel + 0.20 * key_f1 + 0.20 * val_acc + 0.15 * schema_v
    reward_strict = 1.0 if (action_ok == 1.0 and sel == 1.0 and len(pred_calls) == len(gold_calls)
                            and key_f1 == 1.0 and val_acc == 1.0 and schema_v == 1.0) else 0.0
    return _result(reward_soft, reward_strict, feedback, metrics)
