"""
Generate synthetic abstain and ask_clarification samples for the telco domain.
Adds diversity to the domain fine-tuning set to fix the boundary-confusion problem.

Output: data/sft_train_synthetic.jsonl  (new samples only)
        data/sft_train_augmented.jsonl  (original 38 + synthetic)
        data/sft_train_with_warmup_augmented.jsonl (public warmup + augmented domain)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are a contract-aware Telco function-calling agent.\n"
    "Read the user request, available tool schemas, and business constraints.\n"
    "Return only a JSON object with one of these actions:\n"
    '{"action":"call_function","call":{"tool_name":"...","arguments":{...}}}\n'
    '{"action":"call_functions","calls":[{"tool_name":"...","arguments":{...}}]}\n'
    '{"action":"ask_clarification","asked_slots":["..."]}\n'
    '{"action":"abstain","reason":"..."}\n'
    "Use call_functions for multi-step or parallel requests. Ask for clarification when "
    "a required or invalid argument can be corrected.\n"
    "If a tool contract or business precondition is violated, abstain instead of asking "
    "for slots or calling helper tools.\n"
    "Never call a tool if required arguments are missing, the tool is deprecated, or "
    "the contract is unsafe."
)

QWEN_FAMILY = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct"]


def _load_tools() -> dict[str, dict]:
    tools = json.loads((ROOT / "data" / "tools.json").read_text())
    return {t["name"]: t for t in tools}


def _tool_schema(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {}),
    }


def _record(
    record_id: str,
    user: str,
    assistant: dict,
    tools: list[dict],
    expected_action: str,
) -> dict:
    return {
        "id": record_id,
        "source": "synthetic",
        "base_model_family": "qwen",
        "supported_base_models": QWEN_FAMILY,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
        ],
        "tools": tools,
        "expected_action": expected_action,
        "gold_call": None,
        "gold_calls": None,
        "metadata": {"synthetic": True, "action_type": expected_action},
    }


def generate_abstain_samples(tools_map: dict) -> list[dict]:
    samples = []

    # --- Contract violations: suspended subscriber ---
    suspended_ops = [
        ("add_data_package", "Add DATA50 to suspended customer C002.", "subscriber must be active to add data packages"),
        ("enable_roaming", "Enable roaming to Thailand for suspended customer C002.", "subscriber must be active to enable roaming"),
        ("change_plan", "Change plan of suspended customer C002 to 5G_MAX.", "subscriber must be active to change plan"),
        ("activate_esim", "Activate eSIM EID-999 for suspended customer C002.", "subscriber must be active for eSIM activation"),
        ("register_autopay", "Register autopay for suspended customer C002.", "subscriber must be active to register autopay"),
        ("set_spending_limit", "Set spending limit for suspended customer C002.", "subscriber must be active to change spending limit"),
        ("block_premium_sms", "Block premium SMS for suspended customer C002.", "subscriber must be active for service changes"),
        ("apply_late_fee_waiver", "Waive late fee for suspended customer C002 with balance 50000.", "subscriber must be postpaid for fee waiver"),
    ]
    for i, (tool_name, user, reason) in enumerate(suspended_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_suspended_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Contract violations: prepaid subscriber ---
    prepaid_ops = [
        ("register_autopay", "Register autopay for prepaid customer C003.", "autopay is only available for postpaid subscribers"),
        ("apply_late_fee_waiver", "Apply late fee waiver for prepaid customer C003.", "fee waiver requires a postpaid subscriber"),
        ("enable_roaming", "Enable roaming to Singapore for prepaid customer C003.", "roaming requires an eligible postpaid subscriber"),
        ("increase_credit_limit", "Increase credit limit of prepaid customer C003 to 1000000.", "credit limit increase requires a postpaid subscriber"),
        ("set_spending_limit", "Set spending limit of 500000 for prepaid customer C003.", "spending limit only applies to postpaid subscribers"),
    ]
    for i, (tool_name, user, reason) in enumerate(prepaid_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_prepaid_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Contract violations: balance not cleared ---
    balance_ops = [
        ("resume_line", "Resume suspended customer C002 without paying the outstanding balance.", "line resume requires zero outstanding balance"),
        ("cancel_subscription", "Cancel subscription for C002 who still has an outstanding balance.", "cancellation requires zero outstanding balance"),
        ("change_plan", "Upgrade plan for C002 who has an unpaid bill of 150000.", "plan change blocked: customer has outstanding balance"),
        ("resume_line", "Bring customer C002 back online despite unpaid bill of 200000.", "line resume requires zero outstanding balance"),
    ]
    for i, (tool_name, user, reason) in enumerate(balance_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_balance_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Contract violations: unverified customer (customer_verified=False) ---
    unverified_ops = [
        ("suspend_line", "Suspend C001 line immediately without identity verification.", "identity verification required before sensitive line operations"),
        ("increase_credit_limit", "Raise credit limit for C001 before identity is verified.", "credit limit changes require verified customer identity"),
        ("transfer_ownership", "Transfer C001 account without completing identity verification.", "ownership transfer requires verified customer identity"),
        ("cancel_subscription", "Cancel C001 subscription without verifying identity.", "cancellation requires verified customer identity"),
        ("replace_sim", "Issue a new SIM for C001 without identity verification.", "SIM replacement requires verified customer identity"),
        ("register_autopay", "Set up autopay for C001 before identity check.", "payment token operations require verified identity"),
        ("remove_autopay", "Remove autopay from C001 without verifying the customer first.", "autopay removal requires verified customer identity"),
    ]
    for i, (tool_name, user, reason) in enumerate(unverified_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_unverified_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Contract violations: prerequisite not met ---
    prereq_ops = [
        ("replace_sim", "Replace SIM for C003 right away, no lost SIM report filed.", "SIM replacement requires a prior lost SIM report"),
        ("replace_sim", "C001 wants an eSIM but never reported the old SIM lost.", "SIM replacement requires a prior lost SIM report"),
        ("resume_line", "Resume C002 immediately after they call in.", "line resume requires zero outstanding balance first"),
        ("cancel_subscription", "Delete C002 account while balance of 75000 is still unpaid.", "cancellation requires active subscriber with zero outstanding balance"),
    ]
    for i, (tool_name, user, reason) in enumerate(prereq_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_prereq_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Deprecated tools ---
    deprecated_ops = [
        ("legacy_enable_roaming", "Use the old roaming tool to enable roaming for C001 in Korea.", "legacy_enable_roaming is deprecated; use enable_roaming instead"),
        ("legacy_change_plan", "Use legacy plan change API to switch C001 to PLUS plan.", "legacy_change_plan is deprecated; use change_plan instead"),
        ("legacy_enable_roaming", "Activate Japan roaming for C001 via the legacy API.", "legacy_enable_roaming is deprecated; use enable_roaming instead"),
        ("legacy_change_plan", "Downgrade C003 using the old plan-change endpoint.", "legacy_change_plan is deprecated; use change_plan instead"),
    ]
    for i, (tool_name, user, reason) in enumerate(deprecated_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_deprecated_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    # --- Out-of-scope / no matching tool ---
    no_tool_ops = [
        ("get_balance", "Book a flight ticket for the customer using their account balance.", "no tool available for flight booking; this is outside the telco service scope"),
        ("get_customer_profile", "Set up a bank account for customer C001.", "no tool available for banking services"),
        ("get_balance", "Transfer balance from C001 to a third-party wallet.", "no tool available for third-party wallet transfers"),
        ("open_support_ticket", "Refund the customer's credit card payment directly.", "no tool available for direct credit card refunds"),
    ]
    for i, (tool_name, user, reason) in enumerate(no_tool_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_abstain_oob_{i+1:03d}", user,
            {"action": "abstain", "reason": reason},
            [_tool_schema(t)], "abstain",
        ))

    return samples


def generate_ask_clarification_samples(tools_map: dict) -> list[dict]:
    samples = []

    # --- Missing required argument ---
    missing_arg_ops = [
        ("add_data_package", "Add a data package to customer C001.", ["package_code"], "package code not specified"),
        ("change_plan", "Change the plan for customer C001.", ["new_plan"], "new plan not specified"),
        ("activate_esim", "Activate eSIM for customer C001.", ["eid"], "eSIM EID not provided"),
        ("enable_roaming", "Enable roaming for customer C001.", ["country"], "destination country not specified"),
        ("pay_bill", "Process a payment for customer C001.", ["amount"], "payment amount not specified"),
        ("pay_bill", "Customer C001 wants to pay their bill but didn't say how much.", ["amount"], "payment amount not specified"),
        ("suspend_line", "Suspend customer C001's line.", ["reason"], "suspension reason not provided"),
        ("verify_otp", "Verify the OTP for customer C001.", ["otp"], "OTP value not provided"),
        ("open_support_ticket", "Open a support ticket for customer C001.", ["category"], "ticket category not specified"),
        ("set_spending_limit", "Set a spending limit for customer C001.", ["limit"], "spending limit value not specified"),
        ("apply_late_fee_waiver", "Waive the late fee for customer C001.", ["amount"], "waiver amount not specified"),
        ("replace_sim", "Replace the SIM for customer C001.", ["sim_type"], "SIM type (physical or esim) not specified"),
        ("increase_credit_limit", "Increase the credit limit for customer C001.", ["new_limit"], "new credit limit not specified"),
        ("check_network_status", "Check network status for customer C001.", ["area_code"], "area code not specified"),
        ("activate_esim", "Customer C001 wants to activate eSIM for their new phone.", ["eid"], "eSIM EID required but not provided"),
        ("enable_roaming", "C001 is traveling soon and wants to enable roaming.", ["country"], "destination country required"),
        ("change_plan", "Customer C001 wants to switch plans but didn't say which one.", ["new_plan"], "target plan not provided"),
        ("add_data_package", "C001 needs more data but no package was mentioned.", ["package_code"], "package code not specified"),
    ]
    for i, (tool_name, user, slots, _reason) in enumerate(missing_arg_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_clarify_missing_{i+1:03d}", user,
            {"action": "ask_clarification", "asked_slots": slots},
            [_tool_schema(t)], "ask_clarification",
        ))

    # --- Invalid / unknown customer ID ---
    invalid_customer_ops = [
        ("get_balance", "Check balance for customer C999.", ["customer_id"], "C999"),
        ("get_usage", "Get usage for customer C888.", ["customer_id"], "C888"),
        ("get_customer_profile", "Show profile for customer X001.", ["customer_id"], "X001"),
        ("add_data_package", "Add DATA30 to customer C777.", ["customer_id"], "C777"),
        ("change_plan", "Switch customer CXXX to 5G_MAX.", ["customer_id"], "CXXX"),
        ("pay_bill", "Process payment for customer ID 12345.", ["customer_id"], "12345"),
        ("get_balance", "What is the balance of customer ABC?", ["customer_id"], "ABC"),
        ("get_usage", "Check usage for ID CXX99.", ["customer_id"], "CXX99"),
    ]
    for i, (tool_name, user, slots, cid) in enumerate(invalid_customer_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_clarify_invalid_cid_{i+1:03d}", user,
            {"action": "ask_clarification", "asked_slots": slots},
            [_tool_schema(t)], "ask_clarification",
        ))

    # --- Invalid argument type/value ---
    invalid_value_ops = [
        ("pay_bill", "Pay a bill for C001 with amount 'fifty thousand'.", ["amount"], "amount must be a number not a string"),
        ("pay_bill", "Record payment of minus 10000 for C001.", ["amount"], "payment amount must be positive"),
        ("set_spending_limit", "Set spending limit to 'unlimited' for C001.", ["limit"], "limit must be a number"),
        ("check_network_status", "Check network in area XYZ99 for C001.", ["area_code"], "XYZ99 is not a supported area code"),
        ("check_network_status", "Network status for C001 in area HANOI.", ["area_code"], "area code format is invalid"),
        ("apply_late_fee_waiver", "Waive a negative fee of -5000 for C001.", ["amount"], "waiver amount must be positive"),
        ("increase_credit_limit", "Set credit limit to zero for C001.", ["new_limit"], "new limit must be greater than zero"),
        ("add_data_package", "Add package DATAINFINITY to C001.", ["package_code"], "DATAINFINITY is not a valid package code"),
        ("replace_sim", "Replace C001 SIM with type 'nano'.", ["sim_type"], "sim_type must be physical or esim"),
    ]
    for i, (tool_name, user, slots, _reason) in enumerate(invalid_value_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_clarify_invalid_val_{i+1:03d}", user,
            {"action": "ask_clarification", "asked_slots": slots},
            [_tool_schema(t)], "ask_clarification",
        ))

    # --- Missing customer ID entirely ---
    missing_cid_ops = [
        ("get_balance", "Check the balance but no customer ID was given.", ["customer_id"]),
        ("get_usage", "How much data have I used this month?", ["customer_id"]),
        ("add_data_package", "Add DATA30 to my account.", ["customer_id"]),
        ("change_plan", "Switch me to the 5G_MAX plan.", ["customer_id"]),
        ("suspend_line", "Suspend the line, customer didn't provide ID.", ["customer_id"]),
        ("get_customer_profile", "Show my profile details.", ["customer_id"]),
    ]
    for i, (tool_name, user, slots) in enumerate(missing_cid_ops):
        t = tools_map.get(tool_name)
        if not t:
            continue
        samples.append(_record(
            f"synth_clarify_no_cid_{i+1:03d}", user,
            {"action": "ask_clarification", "asked_slots": slots},
            [_tool_schema(t)], "ask_clarification",
        ))

    return samples


def main() -> None:
    random.seed(42)
    tools_map = _load_tools()

    abstain_samples = generate_abstain_samples(tools_map)
    clarify_samples = generate_ask_clarification_samples(tools_map)
    all_synthetic = abstain_samples + clarify_samples

    # Shuffle
    random.shuffle(all_synthetic)

    synth_path = ROOT / "data" / "sft_train_synthetic.jsonl"
    synth_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_synthetic) + "\n"
    )
    print(f"Synthetic samples: {len(all_synthetic)} ({len(abstain_samples)} abstain, {len(clarify_samples)} ask_clarification)")
    print(f"Saved: {synth_path}")

    # Merge with original domain samples
    original = [json.loads(l) for l in (ROOT / "data" / "sft_train.jsonl").open() if l.strip()]
    augmented = original + all_synthetic
    random.shuffle(augmented)

    aug_path = ROOT / "data" / "sft_train_augmented.jsonl"
    aug_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in augmented) + "\n"
    )
    print(f"Augmented domain: {len(augmented)} samples (original {len(original)} + synthetic {len(all_synthetic)})")
    print(f"Saved: {aug_path}")

    # Distribution check
    from collections import Counter
    counts = Counter(r["expected_action"] for r in augmented)
    for action, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {action:<22} {n:3d}  ({100*n/len(augmented):.0f}%)")

    # Rebuild warmup+augmented
    warmup = [json.loads(l) for l in (ROOT / "data" / "sft_train_with_warmup.jsonl").open() if l.strip()]
    # Remove old domain samples (source != 'synthetic' and source != 'apigen_mt' etc.)
    public_only = [r for r in warmup if r.get("source") in ("apigen_mt", "hermes_fc", "toolace", "xlam")]
    combined = public_only + augmented
    random.shuffle(combined)

    warmup_aug_path = ROOT / "data" / "sft_train_with_warmup_augmented.jsonl"
    warmup_aug_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n"
    )
    print(f"\nWarmup+augmented: {len(combined)} samples ({len(public_only)} public + {len(augmented)} domain)")
    print(f"Saved: {warmup_aug_path}")

    # Action distribution in warmup+augmented
    counts2 = Counter(r["expected_action"] for r in combined)
    for action, n in sorted(counts2.items(), key=lambda x: -x[1]):
        print(f"  {action:<22} {n:4d}  ({100*n/len(combined):.1f}%)")


if __name__ == "__main__":
    main()
