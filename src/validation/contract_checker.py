from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractIssue:
    code: str
    message: str
    path: str


class ContractChecker:
    def check(
        self,
        contract: dict[str, Any] | None,
        subscriber: dict[str, Any] | None,
        arguments: dict[str, Any],
        customer_verified: bool,
    ) -> list[ContractIssue]:
        if contract is None:
            return []
        if subscriber is None:
            return [ContractIssue("missing_subscriber", "Subscriber not found", "subscriber")]

        issues: list[ContractIssue] = []
        if contract.get("requires_customer_verified") and not customer_verified:
            issues.append(ContractIssue("permission_denied", "Customer identity must be verified", "customer_verified"))

        context = {"subscriber": subscriber, "args": arguments}
        for rule in contract.get("preconditions", []):
            actual = _resolve_path(context, rule["path"])
            expected = _resolve_path(context, rule["value"]) if rule["op"].endswith("_path") else rule["value"]
            if not _passes(actual, rule["op"], expected):
                issues.append(
                    ContractIssue(
                        "precondition_failed",
                        rule.get("message", f"Precondition failed: {rule['path']}"),
                        rule["path"],
                    )
                )

        return issues


def _resolve_path(context: dict[str, Any], path: Any) -> Any:
    if not isinstance(path, str):
        return path
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _passes(actual: Any, op: str, expected: Any) -> bool:
    if op in {"eq", "eq_path"}:
        return actual == expected
    if op in {"ne", "ne_path"}:
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "lte":
        return actual <= expected
    if op == "gte":
        return actual >= expected
    raise ValueError(f"Unsupported contract op: {op}")
