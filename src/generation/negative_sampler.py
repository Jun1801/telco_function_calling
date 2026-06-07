from __future__ import annotations

from src.generation.scenario import ScenarioSpec, abstain_spec, ask_clarification_spec


class HardNegativeSampler:
    """Builds deterministic hard negatives with separate checker and policy targets."""

    def schema_negatives(self) -> list[ScenarioSpec]:
        return [
            ask_clarification_spec(
                "train_schema_missing_arg_001",
                "train",
                "hard_negative_missing_arg",
                "Add a data package for customer C001, but no package is specified.",
                ["package_code"],
                checker_call={"tool_name": "add_data_package", "arguments": {"customer_id": "C001"}},
                checker_expected_status="schema_invalid",
            ),
            ask_clarification_spec(
                "train_schema_invalid_enum_001",
                "train",
                "hard_negative_invalid_enum",
                "Add DATA999 to customer C001.",
                ["package_code"],
                checker_call={
                    "tool_name": "add_data_package",
                    "arguments": {"customer_id": "C001", "package_code": "DATA999"},
                },
                checker_expected_status="schema_invalid",
                scenario_family="schema_or_deprecated",
            ),
            ask_clarification_spec(
                "eval_seen_invalid_type_001",
                "eval_seen",
                "hard_negative_invalid_type",
                "Check balance but provide a numeric customer id.",
                ["customer_id"],
                checker_call={"tool_name": "get_balance", "arguments": {"customer_id": 1001}},
                checker_expected_status="schema_invalid",
                scenario_family="schema_or_deprecated",
            ),
        ]

    def contract_negatives(self) -> list[ScenarioSpec]:
        return [
            abstain_spec(
                "eval_contract_unverified_001",
                "eval_contract",
                "permission_denied",
                "Suspend line C001 before identity verification.",
                "identity verification is required before suspending a line",
                checker_call={
                    "tool_name": "suspend_line",
                    "arguments": {"customer_id": "C001", "reason": "customer_request"},
                },
                checker_expected_status="contract_invalid",
                customer_verified=False,
            ),
            abstain_spec(
                "eval_contract_suspended_data_001",
                "eval_contract",
                "contract_violation",
                "Add DATA10 to suspended customer C002.",
                "data packages require an active subscriber",
                checker_call={
                    "tool_name": "add_data_package",
                    "arguments": {"customer_id": "C002", "package_code": "DATA10"},
                },
                checker_expected_status="contract_invalid",
                scenario_family="contract_aware",
            ),
            abstain_spec(
                "eval_contract_prepaid_roaming_001",
                "eval_contract",
                "contract_violation",
                "Enable roaming in Japan for prepaid customer C003.",
                "roaming activation requires an eligible postpaid subscriber",
                checker_call={
                    "tool_name": "enable_roaming",
                    "arguments": {"customer_id": "C003", "country": "Japan"},
                },
                checker_expected_status="contract_invalid",
                scenario_family="contract_aware",
            ),
        ]
