from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry
from src.validation.contract_checker import ContractChecker, ContractIssue
from src.validation.schema_validator import SchemaValidator, ValidationIssue


@dataclass(frozen=True)
class ScoreResult:
    reward: float
    status: str
    feedback: list[str]
    output: dict[str, Any] | None = None
    issues: list[dict[str, Any]] | None = None  # structured errors for rich feedback


class RewardScorer:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        contract_registry: ContractRegistry,
        executor: MockTelcoApi,
    ) -> None:
        self.tool_registry = tool_registry
        self.contract_registry = contract_registry
        self.executor = executor
        self.schema_validator = SchemaValidator()
        self.contract_checker = ContractChecker()

    def score_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        customer_verified: bool = True,
    ) -> ScoreResult:
        tool = self.tool_registry.get(tool_name)
        schema_issues = self.schema_validator.validate_call(tool, arguments, tool_name)
        if schema_issues:
            return ScoreResult(
                0.0, "schema_invalid", _format_schema(schema_issues),
                issues=_issue_dicts(schema_issues),
            )

        subscriber = self.executor.get_subscriber(arguments["customer_id"])
        contract = self.contract_registry.get(tool_name)
        contract_issues = self.contract_checker.check(contract, subscriber, arguments, customer_verified)
        if contract_issues:
            return ScoreResult(
                0.0, "contract_invalid", _format_contract(contract_issues),
                issues=_issue_dicts(contract_issues),
            )

        try:
            output = self.executor.execute(tool_name, arguments)
        except ValueError as error:
            return ScoreResult(
                0.0, "execution_failed", [str(error)],
                issues=[{"code": "execution_failed", "path": tool_name, "message": str(error),
                         "suggested_action": "ask_clarification"}],
            )

        return ScoreResult(1.0, "ok", ["Tool call passed schema, contract, and execution checks."], output, issues=[])


# Map an issue code to the corrective action the model should have taken.
_SUGGESTED_ACTION = {
    "unknown_tool": "abstain",
    "deprecated_tool": "abstain",
    "missing_arg": "ask_clarification",
    "invalid_enum": "ask_clarification",
    "invalid_type": "fix_arguments",
    "pattern_mismatch": "fix_arguments",
    "unknown_arg": "fix_arguments",
    "permission_denied": "abstain",
    "precondition_failed": "abstain",
    "missing_subscriber": "ask_clarification",
}


def _issue_dicts(issues: list[Any]) -> list[dict[str, Any]]:
    out = []
    for issue in issues:
        out.append(
            {
                "code": issue.code,
                "path": issue.path,
                "expected": getattr(issue, "expected", None),
                "actual": getattr(issue, "actual", None),
                "message": issue.message,
                "suggested_action": _SUGGESTED_ACTION.get(issue.code),
            }
        )
    return out


def _format_schema(issues: list[ValidationIssue]) -> list[str]:
    return [issue.message for issue in issues]


def _format_contract(issues: list[ContractIssue]) -> list[str]:
    return [f"Precondition failed: {issue.message}" for issue in issues]
