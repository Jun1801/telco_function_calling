from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class MockTelcoApi:
    def __init__(
        self,
        subscribers: list[dict[str, Any]],
        plans: list[dict[str, Any]] | None = None,
        network_status: list[dict[str, Any]] | None = None,
    ) -> None:
        self._subscribers = {subscriber["customer_id"]: copy.deepcopy(subscriber) for subscriber in subscribers}
        self._plans = {plan["plan_id"]: copy.deepcopy(plan) for plan in plans or []}
        self._network_status = {item["area_code"]: copy.deepcopy(item) for item in network_status or []}
        self._tickets: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "MockTelcoApi":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(data["subscribers"], data.get("plans", []), data.get("network_status", []))

    def get_subscriber(self, customer_id: str) -> dict[str, Any] | None:
        subscriber = self._subscribers.get(customer_id)
        return copy.deepcopy(subscriber) if subscriber else None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = arguments.get("customer_id")
        subscriber = self._subscribers.get(customer_id)
        if subscriber is None:
            raise ValueError(f"Subscriber not found: {customer_id}")

        if tool_name == "get_balance":
            return {"balance": subscriber["balance"], "outstanding_balance": subscriber["outstanding_balance"]}
        if tool_name == "get_current_plan":
            return {"plan_id": subscriber["plan_id"], "status": subscriber["status"]}
        if tool_name == "add_data_package":
            return {"package_code": arguments["package_code"], "status": "queued"}
        if tool_name == "change_plan":
            subscriber["plan_id"] = arguments["new_plan"]
            return {"plan_id": subscriber["plan_id"], "status": "changed"}
        if tool_name == "suspend_line":
            subscriber["status"] = "suspended"
            return {"status": "suspended", "reason": arguments["reason"]}
        if tool_name == "resume_line":
            subscriber["status"] = "active"
            return {"status": "active"}
        if tool_name == "report_lost_sim":
            subscriber["lost_sim_reported"] = True
            return {"lost_sim_reported": True}
        if tool_name == "update_billing_email":
            subscriber["billing_email"] = arguments["email"]
            return {"billing_email": subscriber["billing_email"]}
        if tool_name == "enable_roaming":
            subscriber["roaming_enabled"] = True
            return {"roaming_enabled": True, "country": arguments["country"]}
        if tool_name == "cancel_subscription":
            subscriber["status"] = "cancelled"
            return {"status": "cancelled"}
        if tool_name == "get_usage":
            return {"data_used_gb": subscriber.get("data_used_gb", 0), "voice_minutes": subscriber.get("voice_minutes", 0)}
        if tool_name == "pay_bill":
            amount = arguments["amount"]
            subscriber["outstanding_balance"] = max(0, subscriber["outstanding_balance"] - amount)
            return {"outstanding_balance": subscriber["outstanding_balance"], "paid_amount": amount}
        if tool_name == "activate_esim":
            subscriber["esim_active"] = True
            subscriber["eid"] = arguments["eid"]
            return {"esim_active": True, "eid": arguments["eid"]}
        if tool_name == "replace_sim":
            subscriber["sim_type"] = arguments["sim_type"]
            return {"sim_type": subscriber["sim_type"], "status": "replacement_queued"}
        if tool_name == "open_support_ticket":
            ticket = {
                "ticket_id": f"T{len(self._tickets) + 1:04d}",
                "customer_id": customer_id,
                "category": arguments["category"],
                "priority": arguments.get("priority", "normal"),
            }
            self._tickets.append(ticket)
            return ticket
        if tool_name == "close_support_ticket":
            return {"ticket_id": arguments["ticket_id"], "status": "closed"}
        if tool_name == "check_network_status":
            area_code = arguments.get("area_code")
            return copy.deepcopy(self._network_status.get(area_code, {"area_code": area_code, "status": "unknown"}))
        if tool_name == "list_available_plans":
            return {"plans": list(self._plans.values())}
        if tool_name == "send_otp":
            return {"otp_sent": True, "channel": arguments["channel"]}
        if tool_name == "verify_otp":
            return {"verified": arguments["otp"] == "123456"}
        if tool_name == "get_customer_profile":
            return {
                "customer_id": customer_id,
                "status": subscriber["status"],
                "plan_id": subscriber["plan_id"],
                "billing_email": subscriber["billing_email"],
            }
        if tool_name == "disable_roaming":
            subscriber["roaming_enabled"] = False
            return {"roaming_enabled": False}
        if tool_name == "apply_late_fee_waiver":
            return {"waiver_applied": True, "amount": arguments["amount"]}
        if tool_name == "increase_credit_limit":
            subscriber["credit_limit"] = arguments["new_limit"]
            return {"credit_limit": subscriber["credit_limit"]}
        if tool_name == "set_spending_limit":
            subscriber["spending_limit"] = arguments["limit"]
            return {"spending_limit": subscriber["spending_limit"]}
        if tool_name == "block_premium_sms":
            subscriber["premium_sms_blocked"] = True
            return {"premium_sms_blocked": True}
        if tool_name == "unblock_premium_sms":
            subscriber["premium_sms_blocked"] = False
            return {"premium_sms_blocked": False}
        if tool_name == "register_autopay":
            subscriber["autopay_enabled"] = True
            return {"autopay_enabled": True}
        if tool_name == "remove_autopay":
            subscriber["autopay_enabled"] = False
            return {"autopay_enabled": False}
        if tool_name == "transfer_ownership":
            subscriber["owner_id"] = arguments["new_owner_id"]
            return {"owner_id": subscriber["owner_id"]}

        raise ValueError(f"Unsupported tool: {tool_name}")
