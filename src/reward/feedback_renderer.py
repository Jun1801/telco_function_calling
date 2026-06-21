"""Render structured evaluator feedback into rich teacher-context text.

Plan §10 asks for feedback that is both machine-readable (structured errors) and
human-readable. The VPD teacher only ever saw a flat English string, so the
structured detail (which condition failed, expected vs actual, the corrective
action) was lost. This renderer rebuilds the text FROM the structured error
codes — independent of the source message — so it supports Vietnamese (`vi`,
default, for the real Viettel tools) and English (`en`, for the synthetic set).
"""

from __future__ import annotations

from typing import Any

_SUGGESTION = {
    "vi": {
        "ask_clarification": "hỏi lại người dùng giá trị hợp lệ (ask_clarification)",
        "abstain": "từ chối an toàn, không gọi hàm (abstain)",
        "fix_arguments": "sửa lại tham số rồi gọi hàm",
        "call_function": "gọi đúng hàm yêu cầu (call_function)",
        "call_functions": "gọi đủ các hàm cần thiết (call_functions)",
    },
    "en": {
        "ask_clarification": "ask the user for a valid value (ask_clarification)",
        "abstain": "abstain safely, do not call any tool (abstain)",
        "fix_arguments": "fix the arguments then call the tool",
        "call_function": "call the required function (call_function)",
        "call_functions": "issue all required calls (call_functions)",
    },
}

_HEADER = {
    "vi": {
        "schema_invalid": "Phản hồi có lỗi schema:",
        "contract_invalid": "Phản hồi vi phạm điều kiện nghiệp vụ:",
        "execution_failed": "Phản hồi gây lỗi khi thực thi:",
        "wrong_call": "Phản hồi gọi sai (hàm hoặc tham số):",
        "_default": "Phản hồi chưa đúng:",
    },
    "en": {
        "schema_invalid": "The response has schema errors:",
        "contract_invalid": "The response violates business preconditions:",
        "execution_failed": "The response failed at execution:",
        "wrong_call": "The response calls the wrong function or arguments:",
        "_default": "The response is incorrect:",
    },
}

_OK = {
    "vi": "Phản hồi hợp lệ: gọi hàm đã qua kiểm tra schema, hợp đồng và thực thi.",
    "en": "Valid response: the call passed schema, contract, and execution checks.",
}


def _line_for_code(code: str, path: str, expected: Any, actual: Any, lang: str) -> str | None:
    has_ea = expected is not None or actual is not None
    if lang == "vi":
        if code == "invalid_enum" and has_ea:
            return f"Tham số `{path}` không hợp lệ: Kỳ vọng một trong {expected}, thực tế {actual!r}."
        if code == "invalid_type" and has_ea:
            return f"Sai kiểu tham số `{path}`: Kỳ vọng kiểu {expected}, thực tế {actual!r}."
        if code == "missing_arg":
            return f"Thiếu tham số bắt buộc `{path}`."
        if code == "precondition_failed" and has_ea:
            return f"Vi phạm điều kiện nghiệp vụ tại `{path}`: Kỳ vọng {expected!r}, thực tế {actual!r}."
        if code == "permission_denied":
            return f"Thiếu quyền: thao tác yêu cầu khách hàng đã được xác minh (`{path}`)."
        if code == "deprecated_tool":
            return f"Hàm `{path}` đã ngừng hỗ trợ (deprecated)."
        if code == "unknown_tool":
            return f"Hàm `{path}` không tồn tại trong danh mục."
        if code == "wrong_function":
            base = f"Gọi sai hàm `{path}`."
            return base + (f" Nên dùng `{expected}`." if expected is not None else "")
        if code == "wrong_argument_value":
            base = f"Sai giá trị tham số `{path}`" + (f" (bạn dùng {actual!r})" if actual is not None else "") + "."
            return base + (f" Đúng phải là {expected!r}." if expected is not None else "")
        if code == "missing_argument":
            return f"Thiếu tham số `{path}` mà câu hỏi yêu cầu."
        if code == "extra_argument":
            return f"Thừa tham số `{path}`" + (f" (giá trị {actual!r})" if actual is not None else "") + " không cần thiết."
        if code == "invalid_code":
            return f"Mã `{path}` = {actual!r} không có trong danh mục mã hợp lệ."
        return None
    # en
    if code == "invalid_enum" and has_ea:
        return f"Argument `{path}` invalid: expected one of {expected}, got {actual!r}."
    if code == "invalid_type" and has_ea:
        return f"Argument `{path}` has wrong type: expected {expected}, got {actual!r}."
    if code == "missing_arg":
        return f"Missing required argument `{path}`."
    if code == "precondition_failed" and has_ea:
        return f"Precondition failed at `{path}`: expected {expected!r}, got {actual!r}."
    if code == "permission_denied":
        return f"Permission denied: the action requires a verified customer (`{path}`)."
    if code == "deprecated_tool":
        return f"Tool `{path}` is deprecated."
    if code == "unknown_tool":
        return f"Tool `{path}` does not exist in the registry."
    if code == "wrong_function":
        base = f"Wrong function `{path}` selected."
        return base + (f" Use `{expected}` instead." if expected is not None else "")
    if code == "wrong_argument_value":
        base = f"Wrong value for `{path}`" + (f" (you used {actual!r})" if actual is not None else "") + "."
        return base + (f" Should be {expected!r}." if expected is not None else "")
    if code == "missing_argument":
        return f"Missing argument `{path}` required by the request."
    if code == "extra_argument":
        return f"Unnecessary argument `{path}`" + (f" (value {actual!r})" if actual is not None else "") + "."
    if code == "invalid_code":
        return f"Code `{path}` = {actual!r} is not in the valid catalogue."
    return None


def _render_error(error: dict[str, Any], lang: str) -> str:
    code = error.get("code", "")
    path = error.get("path", "")
    detail = _line_for_code(code, path, error.get("expected"), error.get("actual"), lang)
    if detail is None:
        detail = error.get("message") or error.get("type", "")
    suggested = error.get("suggested_action")
    hint = _SUGGESTION.get(lang, {}).get(suggested) if suggested else None
    if hint:
        detail += (" → Gợi ý: " if lang == "vi" else " → Suggested: ") + hint + "."
    return detail


def render_teacher_feedback(feedback: dict[str, Any], lang: str = "vi") -> str:
    """Build a rich teacher-context feedback block from a structured feedback dict."""
    lang = "en" if lang == "en" else "vi"
    status = feedback.get("machine_status", "")
    errors = feedback.get("errors", [])

    if status == "ok" or not errors:
        return _OK[lang]

    header = _HEADER[lang].get(status, _HEADER[lang]["_default"])
    lines = [header]
    for error in errors:
        lines.append("• " + _render_error(error, lang))
    return "\n".join(lines)
