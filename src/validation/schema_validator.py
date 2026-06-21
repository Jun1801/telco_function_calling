from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str
    expected: Any = None
    actual: Any = None


class SchemaValidator:
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    def validate_call(
        self,
        tool: dict[str, Any] | None,
        arguments: dict[str, Any],
        tool_name: str | None = None,
    ) -> list[ValidationIssue]:
        if tool is None:
            return [ValidationIssue("unknown_tool", f"Unknown tool: {tool_name}", "tool_name")]
        if tool.get("deprecated"):
            return [ValidationIssue("deprecated_tool", f"Tool is deprecated: {tool['name']}", "tool_name")]

        schema = tool.get("parameters", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        issues: list[ValidationIssue] = []

        for name in required:
            if name not in arguments:
                issues.append(ValidationIssue("missing_arg", f"Missing required argument: {name}", f"arguments.{name}", expected="present", actual="missing"))

        for name in arguments:
            if name not in properties:
                issues.append(ValidationIssue("unknown_arg", f"Unknown argument: {name}", f"arguments.{name}"))

        for name, value in arguments.items():
            if name not in properties:
                continue
            prop = properties[name]
            expected_type = prop.get("type")
            if expected_type in self._TYPE_MAP and not isinstance(value, self._TYPE_MAP[expected_type]):
                issues.append(
                    ValidationIssue(
                        "invalid_type",
                        f"Invalid type for {name}: expected {expected_type}",
                        f"arguments.{name}",
                        expected=expected_type,
                        actual=type(value).__name__,
                    )
                )
            if "enum" in prop and value not in prop["enum"]:
                issues.append(
                    ValidationIssue(
                        "invalid_enum",
                        f"Invalid value for {name}: expected one of {prop['enum']}",
                        f"arguments.{name}",
                        expected=list(prop["enum"]),
                        actual=value,
                    )
                )
            if prop.get("type") == "string" and "pattern" in prop and isinstance(value, str):
                if re.fullmatch(prop["pattern"], value) is None:
                    issues.append(
                        ValidationIssue(
                            "pattern_mismatch",
                            f"Invalid format for {name}: expected pattern {prop['pattern']}",
                            f"arguments.{name}",
                            expected=prop["pattern"],
                            actual=value,
                        )
                    )

        return issues
