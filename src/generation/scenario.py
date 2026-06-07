from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str
    scenario: str
    instruction: str
    customer_verified: bool
    expected_action: str
    expected_status: str | None = None
    gold_call: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    missing_slots: list[str] | None = None
    gold_calls: list[dict[str, Any]] | None = None
    gold_steps: list[dict[str, Any]] | None = None
    scenario_family: str = "single_step_valid"
    checker_call: dict[str, Any] | None = None
    checker_expected_status: str | None = None


def call_spec(
    scenario_id: str,
    split: str,
    scenario: str,
    instruction: str,
    tool_name: str,
    arguments: dict[str, Any],
    expected_status: str = "ok",
    customer_verified: bool = True,
    gold_override: dict[str, Any] | None = None,
    gold_calls: list[dict[str, Any]] | None = None,
    gold_steps: list[dict[str, Any]] | None = None,
    scenario_family: str = "single_step_valid",
) -> ScenarioSpec:
    predicted_call = {"tool_name": tool_name, "arguments": arguments}
    expected_action = "call_functions" if gold_calls or gold_steps else "call_function"
    prediction = (
        {"action": "call_functions", "calls": gold_steps or gold_calls or []}
        if expected_action == "call_functions"
        else {"action": "call_function", "call": predicted_call}
    )
    return ScenarioSpec(
        scenario_id=scenario_id,
        split=split,
        scenario=scenario,
        instruction=instruction,
        customer_verified=customer_verified,
        expected_action=expected_action,
        expected_status=expected_status,
        gold_call=gold_override or predicted_call,
        prediction=prediction,
        gold_calls=gold_calls,
        gold_steps=gold_steps,
        scenario_family=scenario_family,
    )


def ask_clarification_spec(
    scenario_id: str,
    split: str,
    scenario: str,
    instruction: str,
    missing_slots: list[str],
    checker_call: dict[str, Any] | None = None,
    checker_expected_status: str | None = None,
    customer_verified: bool = True,
    scenario_family: str = "missing_slot",
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        split=split,
        scenario=scenario,
        instruction=instruction,
        customer_verified=customer_verified,
        expected_action="ask_clarification",
        prediction={"action": "ask_clarification", "asked_slots": missing_slots},
        missing_slots=missing_slots,
        scenario_family=scenario_family,
        checker_call=checker_call,
        checker_expected_status=checker_expected_status,
    )


def abstain_spec(
    scenario_id: str,
    split: str,
    scenario: str,
    instruction: str,
    reason: str,
    checker_call: dict[str, Any] | None = None,
    checker_expected_status: str | None = None,
    customer_verified: bool = True,
    scenario_family: str = "abstention",
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        split=split,
        scenario=scenario,
        instruction=instruction,
        customer_verified=customer_verified,
        expected_action="abstain",
        prediction={"action": "abstain", "reason": reason},
        scenario_family=scenario_family,
        checker_call=checker_call,
        checker_expected_status=checker_expected_status,
    )
