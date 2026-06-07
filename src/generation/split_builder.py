from __future__ import annotations

from collections import Counter
from typing import Any


TARGET_SCENARIO_RATIOS: dict[str, float] = {
    "single_step_valid": 0.25,
    "missing_slot": 0.15,
    "abstention": 0.10,
    "contract_aware": 0.20,
    "multi_step": 0.10,
    "parallel": 0.05,
    "masking": 0.10,
    "schema_or_deprecated": 0.05,
}


def scenario_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(sample.get("scenario_family", "unknown") for sample in samples))


def attach_generation_profile(sample: dict[str, Any]) -> dict[str, Any]:
    sample["generation_profile"] = {
        "target_scenario_ratios": TARGET_SCENARIO_RATIOS,
        "toolace_stages": [
            "tool_self_evolution_synthesis",
            "interactive_scenario_generation",
            "dual_layer_validation",
        ],
    }
    return sample
