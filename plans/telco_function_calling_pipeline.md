# Schema-Generalized Telco Function Calling Agent

## 1. Tên đề tài đề xuất

> Schema-Generalized Telco Function Calling with Verifiable Feedback Self-Distillation and Bandit Routing

---

## 2. Bối cảnh và bài toán

**Telco** viết tắt của **Telecommunications**, tức lĩnh vực viễn thông. Trong đề tài này, Telco bao gồm các nghiệp vụ như:

- Quản lý thuê bao.
- Kiểm tra trạng thái thuê bao.
- Tra cứu gói cước hiện tại.
- Đăng ký / hủy gói cước.
- Kiểm tra hóa đơn.
- Kiểm tra chất lượng mạng 4G/5G.
- Tạo ticket báo lỗi.
- Tra cứu trạng thái xử lý sự cố.
- Kích hoạt eSIM.
- Đăng ký roaming.
- Tư vấn gói data phù hợp.

Ví dụ yêu cầu của người dùng:

```text
Tôi ở Đống Đa, Hà Nội, mạng 5G rất chậm, số thuê bao 0987654321. Tạo ticket giúp tôi.
```

Agent cần chuyển yêu cầu đó thành function call có cấu trúc:

```json
{
  "action": "call_function",
  "calls": [
    {
      "name": "create_trouble_ticket",
      "arguments": {
        "msisdn": "0987654321",
        "issue_type": "slow_data",
        "location": "Đống Đa, Hà Nội",
        "description": "Khách hàng báo mạng 5G rất chậm tại Đống Đa, Hà Nội."
      }
    }
  ]
}
```

Sau đó hệ thống thực thi hàm và trả lời lại người dùng.

---

## 3. Bài toán chính

Bài toán chính không chỉ là fine-tune model để gọi đúng một vài hàm, mà là:

> Làm sao để một SLM/LLM Agent có thể đọc schema của các hàm Telco, hiểu yêu cầu tự nhiên của người dùng, chọn đúng function, truyền đúng tham số, biết hỏi lại khi thiếu thông tin, biết từ chối khi yêu cầu không hợp lệ, và vẫn hoạt động tốt khi thêm function mới chưa từng xuất hiện trong quá trình training.

Bài toán này là:

> **Schema-Generalized Telco Function Calling**

Model không học thuộc tập hàm cố định, mà có khả năng tổng quát:

```text
Đọc schema hàm
→ hiểu intent người dùng
→ chọn function phù hợp
→ trích xuất argument
→ kiểm tra thiếu thông tin
→ hỏi lại / từ chối / gọi hàm
→ tự sửa lỗi dựa trên feedback kiểm chứng được
```

---

## 4. Vấn đề của SFT/DPO truyền thống

Nếu chỉ làm pipeline đơn giản:

```text
SFT → DPO → Benchmark
```

thì chưa đủ, vì:

1. **SFT** dễ học thuộc function đã thấy trong training.
2. **DPO** chủ yếu học từ cặp chosen/rejected cố định, không tận dụng tốt feedback chi tiết.
3. Khi thêm function mới, model có thể không hiểu schema mới nếu chưa từng train.
4. Function calling có feedback kiểm chứng rất giàu thông tin, nhưng DPO không khai thác hết.

Ví dụ model sinh sai:

```json
{
  "name": "create_trouble_ticket",
  "arguments": {
    "msisdn": "0987654321",
    "issue_type": "bad_network"
  }
}
```

Validator có thể sinh feedback rất cụ thể:

```json
{
  "schema_valid": false,
  "errors": [
    "Invalid enum value: issue_type must be one of no_signal, slow_data, cannot_call, cannot_receive_sms, billing_error.",
    "Missing required argument: location.",
    "Missing required argument: description."
  ],
  "suggested_action": "ask_clarification"
}
```

Thay vì chỉ cho reward thấp, ta dùng feedback này để model học cách tự sửa lỗi.

---

## 5. Ý tưởng chính

Pipeline đề xuất kết hợp 4 ý tưởng.

### 5.1. Schema-generalized function calling

Model phải gọi được cả **function mới chưa từng thấy trong training**, miễn là inference-time có schema của function đó.

Ví dụ train không có hàm:

```python
activate_esim(msisdn: str, eid: str, device_model: str)
```

Nhưng khi test, hệ thống đưa schema này vào prompt. User hỏi:

```text
Kích hoạt eSIM cho số 0987654321, EID là 8904903200000000001.
```

Model cần sinh đúng:

```json
{
  "action": "call_function",
  "calls": [
    {
      "name": "activate_esim",
      "arguments": {
        "msisdn": "0987654321",
        "eid": "8904903200000000001"
      }
    }
  ]
}
```

### 5.2. Verifiable rich feedback

Function calling có thể kiểm chứng tự động bằng:

- JSON parser.
- JSON schema validation.
- Type checking.
- Enum checking.
- Regex/pattern checking.
- Required argument checking.
- Business rule checking.
- Mock API execution.

Do đó ta không chỉ có scalar reward, mà còn có **rich feedback** chỉ rõ model sai ở đâu.

### 5.3. Feedback-guided self-distillation / SDPO-lite

Model tự sinh output, validator phát hiện lỗi, sau đó feedback được dùng để tạo corrected output. Corrected output này được distill lại vào model.

Luồng:

```text
Instruction + schema
→ model sinh function call
→ validator/executor trả feedback
→ teacher/self-correction sinh output đúng
→ train model để sinh output đúng trực tiếp từ instruction + schema
```

### 5.4. Contextual Bandit Strategy Routing

Không phải query nào cũng nên xử lý bằng một prompt cố định. Bandit chọn strategy phù hợp:

- Direct function call.
- Schema-aware reasoning.
- Plan-then-call.
- Self-correct once.
- Ask-clarification-biased.
- Abstain/safety-biased.

Bandit nhận context của query và học từ reward sau khi function call được validate/execute.

---

## 6. Pipeline tổng quan

```text
[1] Telco Tool Registry
        ↓
[2] Synthetic Data & Benchmark Generator
        ↓
[3] Seen / Unseen Tool Split
        ↓
[4] Tool Retriever + Schema-aware Reranker
        ↓
[5] Context / Slot / Intent Analyzer
        ↓
[6] Contextual Bandit Strategy Router
        ↓
[7] SLM Function Calling Model
        ↓
[8] Schema Validator + Business Rule Checker
        ↓
[9] Mock Telco Executor
        ↓
[10] Verifiable Reward + Rich Feedback Generator
        ↓
[11] Feedback-Guided Self-Distillation / SDPO-lite
        ↓
[12] Evaluation Dashboard + Demo Agent
```

Có thể chia thành 2 pha:

```text
Offline training/evaluation:
Schema → data → SFT → rollout → feedback → self-distillation → benchmark

Online inference/demo:
User query → retrieve tools → bandit chọn strategy → model gọi hàm → validate/execute → trả lời
```

---

# 7. Phân tích từng thành phần

## 7.1. Telco Tool Registry

### Mục đích

Tool Registry là kho chứa toàn bộ function/API Telco đã được chuẩn hóa.

### Input

Danh sách hàm thô:

```text
Tên hàm
Mô tả nghiệp vụ
Tham số
Kiểu dữ liệu
Ràng buộc
Ví dụ sử dụng
Risk level
Business dependencies
```

### Output

File `tools.json`:

```json
{
  "name": "create_trouble_ticket",
  "domain": "network_support",
  "description": "Tạo ticket báo lỗi dịch vụ viễn thông cho thuê bao.",
  "parameters": {
    "type": "object",
    "properties": {
      "msisdn": {
        "type": "string",
        "pattern": "^[0-9]{10}$",
        "description": "Số thuê bao 10 chữ số."
      },
      "issue_type": {
        "type": "string",
        "enum": ["no_signal", "slow_data", "cannot_call", "cannot_receive_sms", "billing_error"]
      },
      "location": {
        "type": "string"
      },
      "description": {
        "type": "string"
      }
    },
    "required": ["msisdn", "issue_type", "location", "description"]
  },
  "examples": [
    "Tạo ticket báo lỗi mạng chậm cho số 0987654321 ở Cầu Giấy."
  ],
  "risk_level": "medium",
  "dependencies": ["check_network_status"]
}
```

### Tác dụng

- Chuẩn hóa function schema.
- Cho phép thêm function mới mà không train lại toàn bộ.
- Cung cấp thông tin cho retriever.
- Cung cấp schema cho validator.
- Cung cấp metadata cho benchmark và reward.

### Tại sao chọn?

Vì nếu số lượng function lớn, không thể nhét toàn bộ schema vào prompt. Tool Registry giúp tách bài toán thành:

```text
retrieve đúng tool trước
→ model gọi hàm sau
```

---

## 7.2. Synthetic Data & Benchmark Generator

### Mục đích

Tạo dữ liệu function calling Telco mà không cần annotate thủ công quá nhiều.

### Input

```text
Tool Registry
Danh sách intent Telco
Template câu hỏi tiếng Việt
Scenario type
Business constraints
```

### Output

Train/eval samples dạng JSONL:

```json
{
  "instruction": "Tôi ở Cầu Giấy, mạng 5G rất chậm, số 0987654321. Tạo giúp tôi phiếu báo lỗi.",
  "available_functions": ["check_network_status", "create_trouble_ticket", "get_ticket_status"],
  "ground_truth": {
    "action": "call_function",
    "calls": [
      {
        "name": "create_trouble_ticket",
        "arguments": {
          "msisdn": "0987654321",
          "issue_type": "slow_data",
          "location": "Cầu Giấy",
          "description": "Khách hàng báo mạng 5G rất chậm tại Cầu Giấy."
        }
      }
    ]
  },
  "scenario_type": "single_step"
}
```

Negative samples:

```json
{
  "negative_call": {
    "name": "check_network_status",
    "arguments": {
      "location": "Cầu Giấy",
      "network_type": "5G"
    }
  },
  "error_type": "wrong_function_for_user_goal"
}
```

Preference/feedback data:

```json
{
  "chosen": {
    "name": "create_trouble_ticket",
    "arguments": {
      "msisdn": "0987654321",
      "issue_type": "slow_data",
      "location": "Cầu Giấy",
      "description": "Khách hàng báo mạng 5G rất chậm tại Cầu Giấy."
    }
  },
  "rejected": {
    "name": "create_trouble_ticket",
    "arguments": {
      "msisdn": "0987654321",
      "issue_type": "bad_network"
    }
  },
  "feedback": [
    "Invalid enum value for issue_type.",
    "Missing required argument: location.",
    "Missing required argument: description."
  ]
}
```

### Scenario types

| Scenario | Ý nghĩa |
|---|---|
| Single-step | Một query gọi một function |
| Multi-step | Cần nhiều bước liên tiếp |
| Parallel | Cần gọi nhiều hàm độc lập |
| Nested/dependent | Output hàm trước là input hàm sau |
| Stateful | Dựa vào context hội thoại trước |
| Ask-clarification | Thiếu tham số, cần hỏi lại |
| Abstention/refusal | Yêu cầu ngoài phạm vi hoặc nhạy cảm |
| Unseen-tool | Function chưa từng train |
| Expanded-library | Nhiều function distractor |

### Tác dụng

Dữ liệu phục vụ:

- SFT.
- Self-distillation.
- Reward evaluation.
- Benchmark.
- Unseen-tool testing.
- Hard negative analysis.

### Tại sao chọn?

Vì đề tài cần có cả dữ liệu train và benchmark có kiểm chứng. Nếu không có benchmark rõ, khó chứng minh model thực sự tốt hơn.

---

## 7.3. Seen / Unseen Tool Split

### Mục đích

Đánh giá khả năng generalize sang function mới.

### Input

Tool Registry gồm khoảng 40 function.

### Output

```text
25 seen functions: dùng để train
10 unseen functions: chỉ dùng để test
5 adversarial/hard-negative functions
50 distractor functions cho expanded-library test
```

### Tác dụng

Trả lời câu hỏi nghiên cứu:

> Khi thêm function mới, model có gọi được không nếu không train lại?

### Tại sao chọn?

Nếu train/test random theo sample, model có thể chỉ học thuộc function cũ. Seen/unseen split buộc model học kỹ năng đọc schema và suy luận trên function mới.

---

## 7.4. Tool Retriever + Schema-aware Reranker

### Mục đích

Khi số function lớn, không đưa toàn bộ schema vào prompt mà retrieve top-k function liên quan.

### Input

```text
User query
Tool Registry
Conversation state
Detected slots / intent
```

Ví dụ:

```text
Tôi ở Cầu Giấy, mạng 5G rất chậm, tạo ticket cho số 0987654321.
```

### Output

Top-k tools:

```json
[
  {
    "name": "create_trouble_ticket",
    "score": 0.92
  },
  {
    "name": "check_network_status",
    "score": 0.86
  },
  {
    "name": "get_ticket_status",
    "score": 0.52
  }
]
```

### Cách tính score

```text
score = semantic_similarity
      + keyword_score
      + intent_domain_score
      + slot_coverage_score
      + required_argument_compatibility
      - risk_penalty_if_sensitive
```

### Tác dụng

- Giảm token cost.
- Giảm nhiễu do quá nhiều schema.
- Scale lên 50–100+ functions.
- Tách lỗi retrieval và lỗi function calling.

### Tại sao chọn?

Đây là điều bắt buộc khi function library lớn. Model chỉ nên nhìn top-k schema thay vì toàn bộ tool registry.

---

## 7.5. Context / Slot / Intent Analyzer

### Mục đích

Phân tích query để biết:

```text
intent là gì
có slot nào
thiếu tham số nào
có cần multi-step không
request có nhạy cảm không
```

### Input

```text
Tạo ticket báo lỗi mạng giúp tôi.
```

### Output

```json
{
  "intent": "create_network_ticket",
  "slots": {
    "msisdn": null,
    "location": null,
    "issue_type": "network_issue",
    "description": null
  },
  "missing_required_slots": ["msisdn", "location", "description"],
  "is_multi_step": false,
  "is_sensitive": false,
  "recommended_action": "ask_clarification"
}
```

### Tác dụng

- Giúp detect thiếu thông tin.
- Cải thiện ask-back accuracy.
- Giảm unnecessary call.
- Hỗ trợ retriever bằng slot/domain features.
- Hỗ trợ bandit chọn strategy.

### Tại sao chọn?

Một lỗi lớn của function calling là model gọi API khi thiếu thông tin. Analyzer giúp hệ thống biết lúc nào nên hỏi lại thay vì gọi hàm bừa.

---

## 7.6. Contextual Bandit Strategy Router

### Mục đích

Chọn strategy phù hợp cho từng query thay vì dùng một prompt cố định.

### Input

Context features:

```json
{
  "query_length": 18,
  "has_msisdn": true,
  "has_location": true,
  "missing_slots_count": 0,
  "is_multi_step": false,
  "is_sensitive": false,
  "retrieval_confidence": 0.92,
  "schema_complexity": 0.4
}
```

### Output

Strategy được chọn:

```json
{
  "selected_arm": "direct_schema_call",
  "reason": "Query đủ thông tin, single-step, retrieval confidence cao."
}
```

### Các arms

| Arm | Strategy | Khi nào dùng |
|---|---|---|
| A1 | Direct schema call | Query đơn giản, đủ thông tin |
| A2 | Schema-aware reasoning | Function mới hoặc schema phức tạp |
| A3 | Plan-then-call | Multi-step hoặc nested |
| A4 | Self-correct once | Retrieval thấp hoặc schema validation dễ lỗi |
| A5 | Ask-clarification-biased | Thiếu slot |
| A6 | Abstain/safety-biased | Yêu cầu nhạy cảm/ngoài phạm vi |

### Reward update

```text
reward = schema_validity
       + execution_success
       + task_success
       + correct_ask_back
       - hallucinated_call
       - unnecessary_call
       - latency_penalty
```

### Tác dụng

- Query thiếu thông tin → chọn ask-back.
- Query multi-step → chọn plan.
- Function mới/schema phức tạp → chọn self-correct.
- Query nhạy cảm → chọn abstain.
- Tăng task success, giảm hallucinated/unnecessary calls.

### Tại sao chọn bandit?

Bandit là RL nhẹ, phù hợp timeline 1 tháng hơn PPO/GRPO. Nó dễ implement, dễ giải thích, dễ vẽ learning curve/regret/arm distribution.

---

## 7.7. SLM Function Calling Model

### Mục đích

Model chính sinh structured output.

### Input

Prompt sau khi đã retrieve top-k schema:

```text
You are a Telco function calling agent.

User query:
"Tôi ở Cầu Giấy, mạng 5G chậm, số 0987654321. Tạo ticket giúp tôi."

Available functions:
1. create_trouble_ticket(...)
2. check_network_status(...)
3. get_ticket_status(...)

Output JSON only:
{
  "action": "call_function" | "ask_clarification" | "abstain",
  "calls": [...]
}
```

### Output

```json
{
  "action": "call_function",
  "calls": [
    {
      "name": "create_trouble_ticket",
      "arguments": {
        "msisdn": "0987654321",
        "issue_type": "slow_data",
        "location": "Cầu Giấy",
        "description": "Khách hàng báo mạng 5G chậm tại Cầu Giấy."
      }
    }
  ]
}
```

### Model đề xuất

```text
Qwen2.5-3B-Instruct hoặc Qwen2.5-7B-Instruct + LoRA/QLoRA
```

Hoặc SLM 4B/8B tương đương nếu yêu cầu đề tài bắt buộc.

### Tác dụng

- Map natural language → structured function call.
- Đọc schema động.
- Extract argument.
- Quyết định call / ask / abstain.
- Hỗ trợ multi-step planning nhẹ.

### Tại sao chọn SLM?

Đề tài yêu cầu demo agent dùng SLM 4B/8B. Ngoài ra, dùng SLM giúp chứng minh distillation/routing có giá trị, thay vì chỉ prompt model API lớn.

---

## 7.8. Schema Validator + Business Rule Checker

### Mục đích

Kiểm tra output của model có hợp lệ không.

### Input

Model output:

```json
{
  "action": "call_function",
  "calls": [
    {
      "name": "create_trouble_ticket",
      "arguments": {
        "msisdn": "0987654321",
        "issue_type": "bad_network"
      }
    }
  ]
}
```

Tool schema:

```json
{
  "required": ["msisdn", "issue_type", "location", "description"],
  "issue_type_enum": ["no_signal", "slow_data", "cannot_call", "cannot_receive_sms", "billing_error"]
}
```

### Output

```json
{
  "valid": false,
  "errors": [
    {
      "type": "invalid_enum",
      "field": "issue_type",
      "message": "issue_type must be one of no_signal, slow_data, cannot_call, cannot_receive_sms, billing_error."
    },
    {
      "type": "missing_required_argument",
      "field": "location"
    },
    {
      "type": "missing_required_argument",
      "field": "description"
    }
  ]
}
```

### Business rules

Ví dụ:

```text
register_plan chỉ gọi khi subscriber_status = active
cancel_plan chỉ gọi nếu thuê bao đang dùng plan đó
get_sensitive_profile cần xác thực chủ thuê bao
create_ticket không được thiếu location
```

### Tác dụng

- Bắt lỗi JSON/schema.
- Tạo feedback chi tiết.
- Ngăn function call sai nghiệp vụ.
- Làm nguồn reward kiểm chứng được.

### Tại sao chọn?

Function calling có thể được kiểm chứng tự động rất rõ. Đây là lợi thế so với các bài toán alignment chỉ dựa vào LLM judge.

---

## 7.9. Mock Telco Executor

### Mục đích

Giả lập backend Telco để thực thi function call.

### Input

Validated function call:

```json
{
  "name": "get_current_plan",
  "arguments": {
    "msisdn": "0987654321"
  }
}
```

### Output

```json
{
  "success": true,
  "data": {
    "plan_id": "5G_MAX100",
    "plan_name": "5G Max 100GB",
    "price": 100000,
    "remaining_data_gb": 24.5
  }
}
```

Hoặc lỗi:

```json
{
  "success": false,
  "error": "subscriber_not_found"
}
```

### Tác dụng

- Đánh giá execution success.
- Tạo reward thực thi.
- Cho demo giống agent thật.
- Tạo multi-step dependency.

### Tại sao chọn mock executor?

Không cần API Telco thật. Mock executor đủ để chứng minh function calling, validation, reward và end-to-end demo.

---

## 7.10. Verifiable Reward + Rich Feedback Generator

### Mục đích

Biến kết quả validate/execute thành reward và feedback.

### Input

```text
Ground truth
Model output
Validation result
Execution result
Business rule result
```

### Output

Scalar reward:

```json
{
  "reward": 0.42,
  "components": {
    "action_correct": 1.0,
    "function_selection": 1.0,
    "argument_accuracy": 0.4,
    "schema_validity": 0.0,
    "execution_success": 0.0,
    "task_success": 0.0,
    "hallucinated_call_penalty": 0.0,
    "unnecessary_call_penalty": 0.0
  }
}
```

Rich feedback:

```json
{
  "feedback": [
    "The selected function is correct.",
    "Argument issue_type has invalid enum value.",
    "Missing required argument: location.",
    "Missing required argument: description.",
    "Because required arguments are missing, the correct action should be ask_clarification."
  ]
}
```

### Reward đề xuất

```text
R = 0.20 * action_correct
  + 0.20 * function_selection
  + 0.20 * argument_accuracy
  + 0.15 * schema_validity
  + 0.10 * execution_success
  + 0.10 * task_success
  + 0.05 * correct_ask_or_abstain
  - 0.10 * hallucinated_call
  - 0.10 * unnecessary_call
  - 0.05 * latency_penalty
```

### Tác dụng

Reward dùng cho:

- Metric evaluation.
- Bandit update.
- Self-distillation sample selection.
- Optional DPO/SDFT baseline.

Rich feedback dùng cho:

- SDPO-lite.
- Self-correction.
- Error analysis dashboard.

### Tại sao chọn rich feedback?

Scalar reward chỉ nói output tốt/xấu. Rich feedback nói model sai ở đâu và nên sửa thế nào. Điều này đặc biệt hợp với function calling vì schema/executor tạo được feedback tự động.

---

## 7.11. Feedback-Guided Self-Distillation / SDPO-lite

### Mục đích

Model học từ lỗi của chính nó, thay vì chỉ học từ ground truth cố định.

### Input

Rollout sai:

```json
{
  "instruction": "Tạo ticket báo lỗi mạng cho số 0987654321.",
  "schema": "create_trouble_ticket requires msisdn, issue_type, location, description",
  "wrong_output": {
    "action": "call_function",
    "calls": [
      {
        "name": "create_trouble_ticket",
        "arguments": {
          "msisdn": "0987654321",
          "issue_type": "slow_data"
        }
      }
    ]
  },
  "feedback": [
    "Missing required argument: location.",
    "Missing required argument: description.",
    "Correct action should be ask_clarification."
  ]
}
```

### Intermediate output

Teacher/self-correction output:

```json
{
  "action": "ask_clarification",
  "question": "Bạn cho mình xin khu vực gặp lỗi và mô tả cụ thể tình trạng mạng để tạo ticket nhé."
}
```

### Final training sample

```json
{
  "input": "instruction + schema",
  "target": {
    "action": "ask_clarification",
    "question": "Bạn cho mình xin khu vực gặp lỗi và mô tả cụ thể tình trạng mạng để tạo ticket nhé."
  }
}
```

### Tác dụng

- Model học sửa lỗi schema.
- Model học khi nào nên ask-back.
- Giảm hallucinated call.
- Tăng khả năng generalize sang function mới.
- Không cần human preference labels.

### Tại sao chọn SDPO-lite thay vì DPO làm main?

DPO chỉ biết chosen tốt hơn rejected. SDPO-lite tận dụng feedback chi tiết:

```text
thiếu field nào
sai enum nào
nên call hay ask
sai business rule nào
```

Trong function calling, feedback này có thể tạo tự động, nên đây là hướng tự nhiên và mới hơn.

---

# 8. Luồng offline training chi tiết

```text
Bước 1: Tạo Telco Tool Registry
Bước 2: Sinh train/eval data
Bước 3: Chia seen/unseen tools
Bước 4: Train SFT baseline trên seen tools
Bước 5: Cho SFT model rollout trên seen + unseen queries
Bước 6: Validator/executor sinh reward + feedback
Bước 7: Tạo corrected samples từ feedback
Bước 8: Train Feedback-SDFT / SDPO-lite
Bước 9: Train/evaluate contextual bandit router bằng reward logs
Bước 10: Đánh giá tất cả method
```

## Input offline

```text
tools.json
train.jsonl
eval_seen.jsonl
eval_unseen.jsonl
eval_expanded.jsonl
mock_telco_db.json
```

## Output offline

```text
SFT adapter
SDPO-lite adapter
reward logs
benchmark results
bandit policy
evaluation dashboard
```

---

# 9. Luồng online inference chi tiết

```text
User query
   ↓
Intent/slot analyzer
   ↓
Tool retriever lấy top-k schema
   ↓
Bandit chọn strategy
   ↓
Prompt builder
   ↓
SLM sinh function call
   ↓
Schema/business validator
   ↓
Nếu lỗi và strategy cho phép: self-correct once
   ↓
Mock executor
   ↓
Final response
   ↓
Log reward để update bandit
```

## Input online

```text
User query
Conversation state
Tool registry
Bandit policy
Model adapter
```

## Output online

```text
Final answer
Function call trace
Execution result
Reward score
Error/feedback nếu có
```

---

# 10. Phương pháp thực nghiệm đề xuất

## 10.1. Dataset setup

```text
40 Telco functions:
- 25 seen functions dùng để train
- 10 unseen functions chỉ dùng để test
- 5 adversarial/hard-negative functions
+ 50 distractor functions cho expanded-library test
```

Training samples:

```text
1,000–1,500 samples
```

Evaluation samples:

```text
300–500 samples
```

## 10.2. Methods so sánh

| Method | Vai trò |
|---|---|
| Prompt-only | Zero-shot schema reasoning |
| SFT | Học format JSON và seen tools |
| SFT + Retrieval | Kiểm tra tác dụng của top-k schema |
| SDFT | Self-distill từ high-reward outputs |
| Feedback-SDFT / SDPO-lite | Distill từ rich feedback |
| SDPO-lite + Bandit Router | Final system |
| DPO | Optional baseline cũ |

## 10.3. Evaluation sets

| Eval set | Mục tiêu |
|---|---|
| Seen tools | Học function cũ tốt không |
| Unseen tools | Function mới có gọi được không |
| Expanded library | Nhiều hàm nhiễu có còn chọn đúng không |
| Missing slot | Có hỏi lại đúng không |
| Multi-step | Có lập plan/gọi nhiều hàm đúng không |
| Safety/abstain | Có từ chối đúng không |

---

# 11. Metrics

## Function calling metrics

| Metric | Ý nghĩa |
|---|---|
| Function selection accuracy | Chọn đúng function |
| Argument accuracy | Truyền đúng tham số |
| Schema validity | JSON/schema hợp lệ |
| Execution success rate | Mock API chạy thành công |
| Task success rate | Hoàn thành yêu cầu người dùng |
| Hallucinated call rate | Gọi function không tồn tại |
| Unnecessary call rate | Gọi hàm khi đáng ra hỏi lại/từ chối |
| Ask-back accuracy | Hỏi lại đúng khi thiếu thông tin |
| Abstention accuracy | Từ chối đúng khi ngoài phạm vi/nhạy cảm |
| Latency | Thời gian phản hồi |
| Cost/query | Token hoặc chi phí/query |

## Retrieval metrics

| Metric | Ý nghĩa |
|---|---|
| Tool Recall@k | Gold function có nằm trong top-k không |
| MRR | Gold function xếp hạng cao không |
| Avg candidate tools | Trung bình số tools đưa vào prompt |
| Avg token cost | Token cost/query |

## Bandit metrics

| Metric | Ý nghĩa |
|---|---|
| Average reward | Reward trung bình |
| Cumulative reward | Tổng reward theo episode |
| Average regret | Regret trung bình |
| Arm distribution | Tần suất chọn từng strategy |
| Best arm by scenario | Strategy tốt nhất theo từng loại query |

---

# 12. Tính mới của đề tài

Đề tài không chỉ tối ưu function calling trên tập hàm cố định. Tính mới nằm ở:

## 12.1. Schema-generalized function calling

Model được đánh giá trên function mới chưa từng xuất hiện trong training.

## 12.2. Verifiable rich feedback

Feedback được sinh tự động từ schema validation, business rule và mock execution.

## 12.3. Feedback-guided self-distillation / SDPO-lite

Model học từ lỗi của chính nó bằng feedback chi tiết, thay vì chỉ học từ chosen/rejected như DPO.

## 12.4. Contextual bandit strategy routing

Bandit chọn strategy xử lý phù hợp theo query, thay vì dùng một prompt cố định.

## 12.5. Expanded tool library evaluation

Pipeline được kiểm tra khi số lượng function tăng lên, dùng retrieval để giảm token cost và tăng robustness.

---

# 13. Tại sao pipeline này phù hợp hơn DPO-only?

| Tiêu chí | DPO-only | Pipeline đề xuất |
|---|---|---|
| Tận dụng schema feedback | Yếu | Mạnh |
| Gọi function mới | Không đảm bảo | Có đánh giá và tối ưu trực tiếp |
| Scale nhiều hàm | Không giải quyết chính | Có retrieval + reranking |
| Xử lý thiếu tham số | Có thể học nhưng không rõ | Có slot analyzer + ask-back strategy |
| Tối ưu online strategy | Không | Có contextual bandit |
| Tính mới | Trung bình | Cao hơn |
| Khả thi 1 tháng | Khá | Khả thi nếu làm bản lite |

---

# 14. Kế hoạch

## Tuần 1 — Tool Registry + Data + Evaluator

Output:

```text
tools.json
mock_telco_db.json
train.jsonl
eval_seen.jsonl
eval_unseen.jsonl
eval_expanded.jsonl
validator.py
mock_executor.py
reward_scorer.py
```

## Tuần 2 — Retriever + SFT baseline

Output:

```text
hybrid_tool_retriever.py
SFT adapter
Prompt-only results
SFT results
Tool Recall@k results
```

## Tuần 3 — SDPO-lite + Bandit Router

Output:

```text
rollout logs
rich feedback data
corrected samples
SDPO-lite / Feedback-SDFT adapter
LinUCB / epsilon-greedy bandit router
bandit learning curves
```

## Tuần 4 — Demo + Evaluation + Report

Output:

```text
Streamlit/Gradio demo
Evaluation dashboard
Method comparison tables
Ablation study
Final report
```

---

# 15. Kết luận

Pipeline đề xuất:

```text
Telco Tool Registry
→ ToolACE-mini data generation
→ Seen/unseen function benchmark
→ Schema-aware tool retrieval
→ SFT schema-reading baseline
→ Verifiable validator/executor feedback
→ Feedback-guided self-distillation / SDPO-lite
→ Contextual bandit strategy router
→ Demo agent + benchmark dashboard
```

Lý do chọn pipeline này:

```text
1. Phù hợp bài toán function library lớn.
2. Giải quyết yêu cầu thêm function mới mà không train lại toàn bộ.
3. Có reward/feedback kiểm chứng được, không phụ thuộc LLM judge.
4. Có self-distillation/on-policy learning mới hơn DPO.
5. Có bandit RL đúng định hướng và dễ triển khai hơn PPO/GRPO.
6. Có benchmark rõ để chứng minh hiệu quả và tính mới.
```

Câu định vị đề tài:

> Đề tài không chỉ tối ưu function calling trên một tập hàm cố định, mà hướng tới schema-generalized Telco function calling, trong đó agent có thể gọi function mới nhờ đọc schema động, tự sửa lỗi từ phản hồi kiểm chứng được, và chọn chiến lược gọi hàm bằng contextual bandit routing.
