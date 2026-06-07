from __future__ import annotations

from typing import Any


MASKING_CURRICULUM_RATIOS: dict[str, float] = {
    "real_names": 0.50,
    "function_name_masked": 0.25,
    "function_and_parameter_masked": 0.15,
    "renamed_or_paraphrased_schema": 0.10,
}


class ToolSelfEvolutionSynthesis:
    """ToolACE-style tool mutation for masking, renamed tools, and deprecated APIs."""

    def mask_call(self, call: dict[str, Any], func_name: str, parameter_map: dict[str, str]) -> dict[str, Any]:
        inverse_map = {real: masked for masked, real in parameter_map.items()}
        return {
            "name": func_name,
            "original_name": call["tool_name"],
            "parameter_map": parameter_map,
            "masked_arguments": {
                inverse_map.get(key, key): value for key, value in call.get("arguments", {}).items()
            },
            "curriculum_ratios": MASKING_CURRICULUM_RATIOS,
        }
