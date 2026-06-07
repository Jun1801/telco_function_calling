from __future__ import annotations

from typing import Any

from src.generation.negative_sampler import HardNegativeSampler
from src.generation.scenario import ScenarioSpec, abstain_spec, ask_clarification_spec, call_spec


class TemplateScenarioGenerator:
    """Deterministic scenario templates for the Telco-ToolACE-mini pipeline."""

    def __init__(self) -> None:
        self.negative_sampler = HardNegativeSampler()

    def build_specs(self) -> list[ScenarioSpec]:
        specs: list[ScenarioSpec] = [
            call_spec("train_valid_balance_001", "train", "valid_read", "Check the current balance for customer C001.", "get_balance", {"customer_id": "C001"}),
            call_spec("train_valid_data_001", "train", "valid_write", "Add DATA30 to verified customer C001.", "add_data_package", {"customer_id": "C001", "package_code": "DATA30"}),
            call_spec("train_valid_change_plan_001", "train", "valid_write", "Move verified customer C001 from PLUS to 5G_MAX.", "change_plan", {"customer_id": "C001", "new_plan": "5G_MAX"}),
            call_spec("eval_seen_valid_email_001", "eval_seen", "valid_write", "Update billing email for verified customer C001 to new@example.com.", "update_billing_email", {"customer_id": "C001", "email": "new@example.com"}),
            call_spec("eval_unseen_valid_lost_sim_001", "eval_unseen", "valid_unseen_tool", "Verified customer C001 reports a lost SIM.", "report_lost_sim", {"customer_id": "C001"}, scenario_family="contract_aware"),
            abstain_spec("eval_evolution_deprecated_001", "eval_evolution_deprecated", "deprecated_tool", "Use the old legacy roaming tool for customer C001.", "legacy roaming API is deprecated; use the replacement tool instead", checker_call={"tool_name": "legacy_enable_roaming", "arguments": {"customer_id": "C001", "country": "Japan"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            call_spec("eval_evolution_new_tools_001", "eval_evolution_new_tools", "valid_new_tool", "Cancel verified active customer C001 after all balances are cleared.", "cancel_subscription", {"customer_id": "C001"}, scenario_family="contract_aware"),
            ask_clarification_spec("eval_evolution_schema_changed_001", "eval_evolution_schema_changed", "schema_changed", "Suspend customer C001 without the required reason field.", ["reason"], checker_call={"tool_name": "suspend_line", "arguments": {"customer_id": "C001"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            self._clarification_spec(),
            self._abstention_spec(),
            call_spec("eval_masked_tools_001", "eval_masked_tools", "function_name_masking", "func_7 should activate eSIM for verified customer C001 using EID EID-123.", "activate_esim", {"customer_id": "C001", "eid": "EID-123"}, scenario_family="masking"),
            call_spec("eval_multi_step_001", "eval_multi_step", "dependency", "Customer C001 lost a SIM and wants a physical replacement.", "replace_sim", {"customer_id": "C001", "sim_type": "physical"}, gold_steps=[{"tool_name": "report_lost_sim", "arguments": {"customer_id": "C001"}}, {"tool_name": "replace_sim", "arguments": {"customer_id": "C001", "sim_type": "physical"}}], scenario_family="multi_step"),
            call_spec("eval_parallel_001", "eval_parallel", "parallel_reads", "Check balance and usage for customer C001.", "get_usage", {"customer_id": "C001"}, gold_calls=[{"tool_name": "get_balance", "arguments": {"customer_id": "C001"}}, {"tool_name": "get_usage", "arguments": {"customer_id": "C001"}}], scenario_family="parallel"),
            ask_clarification_spec("eval_schema_changed_001", "eval_schema_changed", "schema_changed", "Open a support ticket for customer C001 without a category.", ["category"], checker_call={"tool_name": "open_support_ticket", "arguments": {"customer_id": "C001"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            abstain_spec("eval_deprecated_001", "eval_deprecated", "deprecated_tool", "Use legacy_enable_roaming for C001 in Japan.", "legacy_enable_roaming is deprecated; do not call deprecated tools", checker_call={"tool_name": "legacy_enable_roaming", "arguments": {"customer_id": "C001", "country": "Japan"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            call_spec("eval_expanded_library_001", "eval_expanded_library", "distractor_heavy", "Open a normal billing support ticket for customer C001.", "open_support_ticket", {"customer_id": "C001", "category": "billing"}, scenario_family="single_step_valid"),
        ]
        specs.extend(self._valid_read_specs())
        specs.extend(self._valid_write_specs())
        specs.extend(self._more_clarification_specs())
        specs.extend(self._more_abstention_specs())
        specs.extend(self._more_masking_specs())
        specs.extend(self._more_multi_step_specs())
        specs.extend(self._more_parallel_specs())
        specs.extend(self._more_schema_evolution_specs())
        specs.extend(self.negative_sampler.schema_negatives())
        specs.extend(self.negative_sampler.contract_negatives())
        return specs

    def _clarification_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            "eval_missing_slot_001",
            "eval_missing_slot",
            "missing_parameter",
            "Add a data package for my line, but the package code is not provided.",
            True,
            "ask_clarification",
            missing_slots=["package_code"],
            prediction={"action": "ask_clarification", "asked_slots": ["package_code"]},
            scenario_family="missing_slot",
        )

    def _abstention_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            "eval_abstention_001",
            "eval_abstention",
            "unsafe_permission",
            "Transfer ownership of C001 without identity verification.",
            False,
            "abstain",
            prediction={"action": "abstain", "reason": "customer identity is not verified"},
            scenario_family="abstention",
        )

    def _valid_read_specs(self) -> list[ScenarioSpec]:
        return [
            call_spec("train_valid_usage_001", "train", "valid_read", "Check usage for customer C001.", "get_usage", {"customer_id": "C001"}),
            call_spec("train_valid_profile_001", "train", "valid_read", "Show profile summary for customer C001.", "get_customer_profile", {"customer_id": "C001"}),
            call_spec("train_valid_plan_001", "train", "valid_read", "What is the current plan of customer C003?", "get_current_plan", {"customer_id": "C003"}),
            call_spec("train_valid_network_001", "train", "valid_read", "Check network status in HN01 for customer C001.", "check_network_status", {"customer_id": "C001", "area_code": "HN01"}),
            call_spec("train_valid_plans_001", "train", "valid_read", "List available plans for customer C001.", "list_available_plans", {"customer_id": "C001"}),
            call_spec("eval_seen_valid_usage_001", "eval_seen", "valid_read", "Review usage for customer C005.", "get_usage", {"customer_id": "C005"}),
            call_spec("eval_seen_valid_network_001", "eval_seen", "valid_read", "Check if HCM01 has any network issue.", "check_network_status", {"customer_id": "C001", "area_code": "HCM01"}),
            call_spec("eval_expanded_library_read_001", "eval_expanded_library", "distractor_heavy", "Find the balance for C003 despite many unrelated tools.", "get_balance", {"customer_id": "C003"}),
        ]

    def _valid_write_specs(self) -> list[ScenarioSpec]:
        return [
            call_spec("train_valid_send_otp_001", "train", "valid_write", "Send an SMS OTP to customer C001.", "send_otp", {"customer_id": "C001", "channel": "sms"}),
            call_spec("train_valid_verify_otp_001", "train", "valid_write", "Verify OTP 123456 for customer C001.", "verify_otp", {"customer_id": "C001", "otp": "123456"}),
            call_spec("train_valid_pay_bill_001", "train", "valid_write", "Record a 10000 payment for suspended customer C002.", "pay_bill", {"customer_id": "C002", "amount": 10000}),
            call_spec("train_valid_resume_001", "train", "valid_write", "Resume suspended postpaid customer C005 after balance is cleared.", "resume_line", {"customer_id": "C005"}),
            call_spec("train_valid_replace_sim_001", "train", "valid_write", "Replace customer C001 SIM with a physical SIM.", "replace_sim", {"customer_id": "C001", "sim_type": "physical"}),
            call_spec("train_valid_disable_roaming_001", "train", "valid_write", "Disable roaming for customer C001.", "disable_roaming", {"customer_id": "C001"}),
            call_spec("train_valid_ticket_001", "train", "valid_write", "Open a normal network support ticket for C001.", "open_support_ticket", {"customer_id": "C001", "category": "network"}),
            call_spec("train_valid_block_sms_001", "train", "valid_write", "Block premium SMS for customer C001.", "block_premium_sms", {"customer_id": "C001"}),
            call_spec("eval_seen_valid_autopay_001", "eval_seen", "valid_write", "Register autopay for verified postpaid customer C001.", "register_autopay", {"customer_id": "C001", "payment_token": "tok_test_001"}),
            call_spec("eval_seen_valid_spending_limit_001", "eval_seen", "valid_write", "Set spending limit to 200000 for customer C001.", "set_spending_limit", {"customer_id": "C001", "limit": 200000}),
            call_spec("eval_unseen_valid_credit_001", "eval_unseen", "valid_unseen_tool", "Increase credit limit of C001 to 500000.", "increase_credit_limit", {"customer_id": "C001", "new_limit": 500000}, scenario_family="contract_aware"),
            call_spec("eval_unseen_valid_remove_autopay_001", "eval_unseen", "valid_unseen_tool", "Remove autopay from customer C001.", "remove_autopay", {"customer_id": "C001"}),
            call_spec("eval_unseen_valid_transfer_001", "eval_unseen", "valid_unseen_tool", "Transfer ownership of C001 to owner O777.", "transfer_ownership", {"customer_id": "C001", "new_owner_id": "O777"}, scenario_family="contract_aware"),
        ]

    def _more_clarification_specs(self) -> list[ScenarioSpec]:
        return [
            ScenarioSpec("eval_missing_slot_002", "eval_missing_slot", "missing_parameter", "Change my plan but no new plan is specified.", True, "ask_clarification", missing_slots=["new_plan"], prediction={"action": "ask_clarification", "asked_slots": ["new_plan"]}, scenario_family="missing_slot"),
            ScenarioSpec("eval_missing_slot_003", "eval_missing_slot", "missing_parameter", "Open a support ticket but the issue category is missing.", True, "ask_clarification", missing_slots=["category"], prediction={"action": "ask_clarification", "asked_slots": ["category"]}, scenario_family="missing_slot"),
            ScenarioSpec("eval_missing_slot_004", "eval_missing_slot", "missing_parameter", "Enable roaming but the destination country is not provided.", True, "ask_clarification", missing_slots=["country"], prediction={"action": "ask_clarification", "asked_slots": ["country"]}, scenario_family="missing_slot"),
            ScenarioSpec("eval_missing_slot_005", "eval_missing_slot", "missing_parameter", "Verify customer identity but the OTP value is missing.", True, "ask_clarification", missing_slots=["otp"], prediction={"action": "ask_clarification", "asked_slots": ["otp"]}, scenario_family="missing_slot"),
        ]

    def _more_abstention_specs(self) -> list[ScenarioSpec]:
        return [
            ScenarioSpec("eval_abstention_002", "eval_abstention", "unsafe_permission", "Increase credit limit for C001 without verification.", False, "abstain", prediction={"action": "abstain", "reason": "credit limit changes require verification"}, scenario_family="abstention"),
            ScenarioSpec("eval_abstention_003", "eval_abstention", "unsafe_permission", "Register autopay without verified customer identity.", False, "abstain", prediction={"action": "abstain", "reason": "payment token operations require verification"}, scenario_family="abstention"),
            ScenarioSpec("eval_abstention_004", "eval_abstention", "unsafe_permission", "Unblock premium SMS before customer verification.", False, "abstain", prediction={"action": "abstain", "reason": "unsafe service unblock requires verification"}, scenario_family="abstention"),
        ]

    def _more_masking_specs(self) -> list[ScenarioSpec]:
        return [
            call_spec("eval_masked_tools_002", "eval_masked_tools", "function_name_masking", "func_7 should activate eSIM for C001 using EID EID-456.", "activate_esim", {"customer_id": "C001", "eid": "EID-456"}, scenario_family="masking"),
            call_spec("eval_masked_tools_003", "eval_masked_tools", "parameter_masking", "func_11 updates the billing email for C001 to masked@example.com.", "update_billing_email", {"customer_id": "C001", "email": "masked@example.com"}, scenario_family="masking"),
            call_spec("eval_masked_tools_004", "eval_masked_tools", "renamed_schema", "The renamed package tool should add DATA70 to C001.", "add_data_package", {"customer_id": "C001", "package_code": "DATA70"}, scenario_family="masking"),
            call_spec("eval_masked_tools_005", "eval_masked_tools", "function_name_masking", "func_19 should open a billing support ticket for C001.", "open_support_ticket", {"customer_id": "C001", "category": "billing"}, scenario_family="masking"),
        ]

    def _more_multi_step_specs(self) -> list[ScenarioSpec]:
        return [
            call_spec("eval_multi_step_002", "eval_multi_step", "dependency", "Send an OTP then verify customer C001 with 123456.", "verify_otp", {"customer_id": "C001", "otp": "123456"}, gold_steps=[{"tool_name": "send_otp", "arguments": {"customer_id": "C001", "channel": "sms"}}, {"tool_name": "verify_otp", "arguments": {"customer_id": "C001", "otp": "123456"}}], scenario_family="multi_step"),
            call_spec("eval_multi_step_003", "eval_multi_step", "dependency", "Report lost SIM and prepare eSIM replacement for C001.", "replace_sim", {"customer_id": "C001", "sim_type": "esim"}, gold_steps=[{"tool_name": "report_lost_sim", "arguments": {"customer_id": "C001"}}, {"tool_name": "replace_sim", "arguments": {"customer_id": "C001", "sim_type": "esim"}}], scenario_family="multi_step"),
        ]

    def _more_parallel_specs(self) -> list[ScenarioSpec]:
        return [
            call_spec("eval_parallel_002", "eval_parallel", "parallel_reads", "Check profile and current plan for C001.", "get_current_plan", {"customer_id": "C001"}, gold_calls=[{"tool_name": "get_customer_profile", "arguments": {"customer_id": "C001"}}, {"tool_name": "get_current_plan", "arguments": {"customer_id": "C001"}}], scenario_family="parallel"),
            call_spec("eval_parallel_003", "eval_parallel", "parallel_reads", "Check balance and network status for C001 in DN01.", "check_network_status", {"customer_id": "C001", "area_code": "DN01"}, gold_calls=[{"tool_name": "get_balance", "arguments": {"customer_id": "C001"}}, {"tool_name": "check_network_status", "arguments": {"customer_id": "C001", "area_code": "DN01"}}], scenario_family="parallel"),
        ]

    def _more_schema_evolution_specs(self) -> list[ScenarioSpec]:
        return [
            ask_clarification_spec("eval_schema_changed_002", "eval_schema_changed", "schema_changed", "Pay a bill for C001 but amount is a string.", ["amount"], checker_call={"tool_name": "pay_bill", "arguments": {"customer_id": "C001", "amount": "10000"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            ask_clarification_spec("eval_schema_changed_003", "eval_schema_changed", "schema_changed", "Check network status using unsupported area code HUE01.", ["area_code"], checker_call={"tool_name": "check_network_status", "arguments": {"customer_id": "C001", "area_code": "HUE01"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            abstain_spec("eval_deprecated_002", "eval_deprecated", "deprecated_tool", "Use legacy_change_plan for C001.", "legacy_change_plan is deprecated; do not call deprecated tools", checker_call={"tool_name": "legacy_change_plan", "arguments": {"customer_id": "C001", "plan": "PLUS"}}, checker_expected_status="schema_invalid", scenario_family="schema_or_deprecated"),
            abstain_spec("eval_contract_resume_blocked_001", "eval_contract", "contract_violation", "Resume suspended customer C002 even though balance remains unpaid.", "line resume requires zero outstanding balance", checker_call={"tool_name": "resume_line", "arguments": {"customer_id": "C002"}}, checker_expected_status="contract_invalid", scenario_family="contract_aware"),
            abstain_spec("eval_contract_replace_sim_001", "eval_contract", "contract_violation", "Replace SIM for C003 before lost SIM is reported.", "SIM replacement requires a prior lost SIM report", checker_call={"tool_name": "replace_sim", "arguments": {"customer_id": "C003", "sim_type": "physical"}}, checker_expected_status="contract_invalid", scenario_family="contract_aware"),
            abstain_spec("eval_contract_autopay_prepaid_001", "eval_contract", "contract_violation", "Register autopay for prepaid customer C003.", "autopay requires a postpaid subscriber", checker_call={"tool_name": "register_autopay", "arguments": {"customer_id": "C003", "payment_token": "tok_prepaid"}}, checker_expected_status="contract_invalid", scenario_family="contract_aware"),
            abstain_spec("eval_contract_cancel_balance_001", "eval_contract", "contract_violation", "Cancel suspended customer C002 with outstanding balance.", "cancellation requires active subscriber with zero outstanding balance", checker_call={"tool_name": "cancel_subscription", "arguments": {"customer_id": "C002"}}, checker_expected_status="contract_invalid", scenario_family="contract_aware"),
            call_spec("eval_deprecated_execution_001", "eval_deprecated", "unsupported_or_deprecated", "Close support ticket T9999 for customer C001.", "close_support_ticket", {"customer_id": "C001", "ticket_id": "T9999"}, "ok", scenario_family="schema_or_deprecated"),
        ]
