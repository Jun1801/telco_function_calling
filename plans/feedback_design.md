# Feedback Design

## 1. Nơi sinh feedback
- `src/evaluation/routing.py` route theo `source`: real (`real_tool_xlsx`) → `src/evaluation/real_evaluator.py`; synthetic → `evaluator.py` + `reward_feedback.py`.
- Mọi evaluator trả cùng shape: `{reward_soft, reward_strict, reward_total, feedback, metrics}`.
- `feedback` là phần để teacher (SDPO/VPD) và Feedback-SDFT học từ lỗi.

## 2. Cấu trúc `feedback`
```json
{
  "machine_status": "wrong_call",          // ok | schema_invalid | wrong_call | wrong_action | format_error
  "errors": [ { "...error item..." } ],     // máy đọc, structured
  "feedback_text": [ "Wrong value for location_code" ]  // người đọc, từng dòng
}
```
**Error item:** `{type, code, path, actual, [expected], message, suggested_action}`
(`expected` chỉ có khi `reveal_gold=True`).

## 3. Ba nguồn lỗi (real, schema-only) — theo thứ tự ưu tiên

**(0) Schema layer** (`SchemaValidator`) — ưu tiên cao nhất. Nếu schema sai → chỉ báo schema, bỏ qua so gold/code.
Codes: `invalid_enum`, `invalid_type`, `missing_arg`, `unknown_arg`, `pattern_mismatch`, `deprecated_tool`, `unknown_tool`.

**(A) Gold-diff** (`_gold_diff_errors`) — khi call đã schema-valid, so với gold call:
- `wrong_function` — sai hàm (argument diff bỏ qua).
- `wrong_argument_value` — sai giá trị tham số.
- `missing_argument` / `extra_argument` — thiếu/thừa tham số.
- `unnecessary_call` — call (parallel) không khớp gold nào → **1 tín hiệu duy nhất**, không spam per-arg.
- **`reveal_gold=False` (mặc định):** chỉ báo *sai ở đâu*, KHÔNG lộ giá trị gold → teacher phải tự suy giá trị đúng. `reveal_gold=True` chỉ dùng cho ablation.

**(B) Reference-code** (`_reference_code_errors`) — `location_code`/`kpi_code`/`unit_code` không có enum trong schema:
- `invalid_code` — mã không thuộc catalogue. Danh mục hợp lệ lấy từ **catalogue** (không phải gold) → không phải gold-leak.

## 4. Feedback theo action
| expected_action | Lỗi đặc thù | Chấm |
|---|---|---|
| `call_function` | (0)/(A)/(B) như trên | soft = 0.15 action + 0.30 func + 0.20 key_f1 + 0.20 val_acc + 0.15 schema; strict = all==1 |
| `call_functions` (parallel/multi ReAct) | set-by-name; call thừa → `unnecessary_call` | strict cần đủ số call + khớp hết |
| `ask_clarification` | `missing_slot_not_handled` (kèm `missing_slots`) | recall slot: `0.6*action + 0.4*recall` |
| `abstain` | `unsafe_or_forbidden_action` nếu không abstain | strict = abstain đúng |
| (mọi action) parse lỗi | `format_error/parse_error` | **reward 0** — KHÔNG tính là abstain hợp lệ |

## 5. `suggested_action` 
Map `_SUGGESTED_ACTION` → mỗi error gắn một trong: `ask_clarification`, `abstain`, `fix_arguments`,
`call_function`, `call_functions`, `fix_format`. (vd: `invalid_enum`→ask_clarification, `deprecated_tool`→abstain.)

## 6. Renderer cho teacher (`src/reward/feedback_renderer.py`)
`render_teacher_feedback(feedback, lang)` dựng lại **text giàu từ structured codes** (độc lập với `message`):
- Header theo `machine_status` + mỗi error 1 dòng `• …` + đuôi “→ Gợi ý: <suggested_action>”.

## 7. Feedback được tiêu dùng ở đâu
- **Feedback-SDFT** (`src/training/build_corrections.py`): `feedback_text` → `CORRECTION_REQUEST` → model tự sửa → chỉ giữ correction đạt reward==1.
- **SDPO** (`src/training/train_sdpo.py`): teacher context = prompt + `[Environment Feedback]` + sibling demo; student chỉ thấy prompt.
- **VPD-lite** (E/M): teacher thấy prompt+feedback, student chỉ prompt.

## 8. Ví dụ (real)
Sai giá trị, **không lộ gold** (mặc định):
```json
{"machine_status":"wrong_call",
 "errors":[{"type":"wrong_call","code":"wrong_argument_value","path":"arguments.location_code",
            "actual":"HCM","message":"Wrong value for location_code","suggested_action":"fix_arguments"}],
 "feedback_text":["Wrong value for location_code"]}
```
Render (vi): `Phản hồi gọi sai (hàm hoặc tham số):` / `• Sai giá trị tham số \`arguments.location_code\` (bạn dùng 'HCM'). → Gợi ý: sửa lại tham số rồi gọi hàm.`
(Bật `reveal_gold=True` mới thêm `"expected":"HNI"` và “Đúng phải là 'HNI'”.)


