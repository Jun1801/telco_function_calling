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
    return ScenarioSpec(
        scenario_id=scenario_id,
        split=split,
        scenario=scenario,
        instruction=instruction,
        customer_verified=customer_verified,
        expected_action="call_function",
        expected_status=expected_status,
        gold_call=gold_override or predicted_call,
        prediction={"action": "call_function", "call": predicted_call},
        gold_calls=gold_calls,
        gold_steps=gold_steps,
        scenario_family=scenario_family,
    )
