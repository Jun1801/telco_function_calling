# Multi-Step Analysis & Improvement Ideas

## Kết quả M1b trên multi_step

- **Accuracy: 54.3%** (156 samples)
- func=1.00 — model luôn chọn đúng tool
- val_acc=0.62 — sai argument values 38% trường hợp
- Nguyên nhân: model không đọc đúng `station_code` từ observation của R1 để dùng làm `object_code` ở R2

## Root Cause

R2 instruction hiện tại:
```
[Kết quả bước 1 — danh sách trạm]:
[{"station_code": "EHB00022", ...}, {"station_code": "EHB00023", ...}]
Hãy dùng kết quả trên để tra cứu tiếp.
```

Vấn đề:
1. Model phải tự suy luận cần dùng `EHB00022` làm `object_code` — không explicit
2. `object_code` không có enum trong schema (vì có thể là location hoặc station) → model dễ hallucinate
3. Có nhiều station trong observation → ambiguous, model không biết chọn cái nào
4. Training data R2 chỉ có 520/4815 (10.8%) — ít hơn các split khác

## Tại sao không chỉ tăng data?

Tăng data = phải train lại, không phù hợp với thực tế ít dữ liệu. Cần giải pháp inference-time.

## Hướng cải thiện (không cần train thêm)

### Hướng 2 — Prompt Engineering cho R2

Thêm 1 dòng explicit vào R2 instruction khi detect `scenario == "react_step2"`:

```python
# Trong prompt_builder.py
if sample.get("scenario") == "react_step2":
    import re, json as _json
    obs_match = re.search(r'\[Kết quả bước 1[^\]]*\]:\s*(\[.*?\])', instruction, re.DOTALL)
    if obs_match:
        stations = _json.loads(obs_match.group(1))
        codes = [s["station_code"] for s in stations if "station_code" in s]
        instruction += f"\n\n⚠️ Sử dụng station_code từ kết quả trên làm object_code. Các mã hợp lệ: {codes}"
```

Kết quả R2 instruction:
```
... [Kết quả bước 1]: [{"station_code": "EHB00022",...}]
⚠️ Sử dụng station_code từ kết quả trên làm object_code. Các mã hợp lệ: ["EHB00022", "EHB00023", "EHB00024"]
```

**Ưu điểm:** Zero-shot, không train lại, test ngay.

---

### Hướng 3 — Agentic Pipeline

Thay vì R1 + R2 là 2 record độc lập với observation giả, orchestrator chạy thật:

```
User request
    ↓
Model → R1: call regional_station_info(location_code="THA", ...)
    ↓
[Orchestrator execute tool thật → kết quả thật]
    ↓
Kết quả: [{"station_code": "EHB00132", ...}]
    ↓
Model → R2: call radio_traffic(object_code="EHB00132", ...)
```

Implementation (`scripts/run_agentic_eval.py`):

```python
def run_agentic(user_request, model, tokenizer, kpi_mock):
    # Turn 1
    messages = build_system_prompt() + [{"role": "user", "content": user_request}]
    r1_output = generate(model, messages)
    r1_call = parse_model_output(r1_output)

    # Execute R1
    r1_result = kpi_mock.execute(r1_call)
    observation = format_observation(r1_result)

    # Turn 2 — append real observation vào conversation
    messages += [
        {"role": "assistant", "content": r1_output},
        {"role": "user", "content": observation + "\nHãy dùng kết quả trên để tra cứu tiếp."}
    ]
    r2_output = generate(model, messages)
    r2_call = parse_model_output(r2_output)

    return evaluate(r2_call, gold_r2_call)
```

**Ưu điểm:**
- Model nhận real execution result thay vì fake observation
- Full conversation context — giống production thật
- Model chỉ cần single-step reasoning tại mỗi turn
- Không cần train lại

**Đây là cách các production LLM function-calling thực sự hoạt động** (Claude Tool Use, OpenAI function calling).

---

## Đề xuất thực hiện

1. **Ngắn hạn (test ngay):** Hướng 2 — sửa `prompt_builder.py`, chạy lại eval M1b trên `multi_step`, đo improvement
2. **Dài hạn (đúng hướng):** Hướng 3 — viết `scripts/run_agentic_eval.py` với real tool execution loop

Kỳ vọng: Hướng 2 có thể đưa multi_step từ **54.3% → 70%+** chỉ bằng prompt fix.
