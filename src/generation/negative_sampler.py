from __future__ import annotations

from src.generation.scenario import ScenarioSpec, call_spec


class HardNegativeSampler:
    """Builds deterministic hard negatives that must be verifier-checked."""

    def schema_negatives(self) -> list[ScenarioSpec]:
        return [
            call_spec(
                "train_schema_missing_arg_001",
                "train",
                "hard_negative_missing_arg",
                "Add a data package for customer C001, but no package is specified.",
                "add_data_package",
                {"customer_id": "C001"},
                "schema_invalid",
                scenario_family="missing_slot",
            ),
            call_spec(
                "train_schema_invalid_enum_001",
                "train",
                "hard_negative_invalid_enum",
                "Add DATA999 to customer C001.",
                "add_data_package",
                {"customer_id": "C001", "package_code": "DATA999"},
                "schema_invalid",
                scenario_family="schema_or_deprecated",
            ),
            call_spec(
                "eval_seen_invalid_type_001",
                "eval_seen",
                "hard_negative_invalid_type",
                "Check balance but provide a numeric customer id.",
                "get_balance",
                {"customer_id": 1001},
                "schema_invalid",
                scenario_family="schema_or_deprecated",
            ),
        ]

    def contract_negatives(self) -> list[ScenarioSpec]:
        return [
            call_spec(
                "eval_contract_unverified_001",
                "eval_contract",
                "permission_denied",
                "Suspend line C001 before identity verification.",
                "suspend_line",
                {"customer_id": "C001", "reason": "customer_request"},
                "contract_invalid",
                False,
                scenario_family="abstention",
            ),
            call_spec(
                "eval_contract_suspended_data_001",
                "eval_contract",
                "contract_violation",
                "Add DATA10 to suspended customer C002.",
                "add_data_package",
                {"customer_id": "C002", "package_code": "DATA10"},
                "contract_invalid",
                scenario_family="contract_aware",
            ),
            call_spec(
                "eval_contract_prepaid_roaming_001",
                "eval_contract",
                "contract_violation",
                "Enable roaming in Japan for prepaid customer C003.",
                "enable_roaming",
                {"customer_id": "C003", "country": "Japan"},
                "contract_invalid",
                scenario_family="contract_aware",
            ),
        ]
