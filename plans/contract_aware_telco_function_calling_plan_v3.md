# Tech Plan v3 — Contract-Aware Self-Adaptive Telco Function Calling Agent

## 0. Phiên bản và thay đổi chính

Bản này nâng cấp từ hướng cũ:

> Schema-Generalized Telco Function Calling with Verifiable Feedback Self-Distillation and Bandit Routing

thành hướng mạnh hơn:

> **Contract-Aware Self-Adaptive Telco Function Calling with Verifiable Feedback Self-Distillation and Bandit Routing**

Điểm nâng cấp chính:

1. Không chỉ kiểm tra **JSON schema validity**, mà kiểm tra thêm **Tool Contract**:
   - Preconditions.
   - Postconditions.
   - Side effects.
   - Risk level.
   - Permission requirement.
   - Business constraints.

2. Không chỉ đánh giá function đã có, mà mô phỏng **Dynamic Tool Evolution**:
   - Thêm function mới.
   - Thay đổi schema.
   - Deprecated function cũ.
   - Thêm nhiều distractor tools.

3. Core self-distillation vẫn là **on-policy self-distillation**:
   - Chính SFT checkpoint hiện tại tự rollout.
   - Chính checkpoint đó tự sửa lỗi từ feedback.
   - Qwen3-Coder/Qwen3.5-9B chỉ là external-teacher ablation/bootstrap, không phải method chính.

---

## 1. Tên đề tài chốt

### Tiếng Việt

> **Tối ưu Function Calling cho Telco Agent trên tập hàm động bằng Tool Contract, phản hồi kiểm chứng, Self-Distillation và Bandit Routing**

### Tiếng Anh

> **Contract-Aware Self-Adaptive Telco Function Calling with Verifiable Feedback Self-Distillation and Bandit Routing**

---

## 2. Bài toán chính

Trong các hệ thống Telco thực tế, một function call không chỉ cần đúng JSON schema. Nó còn cần đúng nghiệp vụ.

Ví dụ, một lời gọi hàm có thể hợp lệ về mặt JSON:

```json
{
  "name": "register_plan",
  "arguments": {
    "msisdn": "0987654321",
    "plan_id": "5G_MAX100"
  }
}
```

Nhưng vẫn sai nghiệp vụ nếu:

```text
subscriber_status = suspended
customer_verified = false
plan_id không nằm trong danh sách gói hợp lệ cho vùng đó
```

Vì vậy, đề tài không chỉ giải quyết:

```text
Natural language → valid JSON function call
```

mà giải quyết bài toán rộng hơn:

```text
Natural language
→ retrieve function/schema/contract phù hợp
→ generate function call
→ validate schema
→ validate business contract
→ execute mock API
→ receive verifiable feedback
→ self-correct / self-distill
→ adapt to new or changed tools
```

---

## 3. Ý tưởng cốt lõi

### 3.1. Schema-generalized function calling

Model không học thuộc tập hàm cố định, mà học khả năng đọc schema động.

Khi thêm function mới như:

```python
activate_esim(msisdn: str, eid: str, device_model: str)
```

model vẫn có thể gọi đúng nếu schema được đưa vào inference.

---

### 3.2. Contract-aware function calling

Mỗi tool không chỉ có schema, mà có **Tool Contract**:

```json
{
  "name": "register_plan",
  "preconditions": [
    "subscriber_status == active",
    "customer_verified == true",
    "plan_id in available_plans"
  ],
  "postconditions": [
    "subscription.plan_id == requested_plan_id"
  ],
  "side_effects": [
    "change_subscription",
    "charge_fee",
    "send_sms_notification"
  ],
  "risk_level": "high",
  "permission_required": "customer_verified"
}
```

Function call đúng schema nhưng vi phạm contract vẫn bị xem là sai.

---

### 3.3. Verifiable rich feedback

Feedback được sinh tự động từ:

```text
JSON parse
schema validation
required argument check
type/enum/regex check
business precondition check
permission check
side-effect risk check
mock API execution
postcondition check
```

Ví dụ feedback:

```json
{
  "valid": false,
  "feedback": [
    "Function register_plan is schema-valid.",
    "Precondition failed: subscriber_status must be active, but current status is suspended.",
    "Permission failed: customer_verified is false.",
    "Correct action should be ask_clarification or abstain, not call_function."
  ]
}
```

---

### 3.4. Feedback-guided on-policy self-distillation

Core method không dùng model khác làm teacher chính.

Luồng đúng:

```text
Qwen3.5-SFT checkpoint
→ rollout
→ validator/contract checker/executor sinh feedback
→ chính Qwen3.5-SFT dùng feedback prompt để self-correct
→ lọc corrected output hợp lệ
→ fine-tune lại chính model đó
```

Qwen3-Coder/Qwen3.5-9B chỉ dùng tùy chọn cho external-teacher ablation.

---

### 3.5. Contextual bandit routing

Bandit chọn strategy xử lý phù hợp:

```text
direct_schema_call
schema_contract_reasoning
plan_then_call
self_correct_once
ask_clarification_biased
abstain_safety_biased
```

Dựa trên:

```text
missing slots
schema complexity
contract risk
tool novelty
retrieval confidence
sensitivity
latency/token cost
```

---

### 3.6. Dynamic tool evolution

Đề tài mô phỏng môi trường Telco API thay đổi:

```text
Phase 1: seen tools
Phase 2: add unseen tools
Phase 3: modify schema/contracts
Phase 4: deprecate old tools
Phase 5: add many distractor tools
```

Mục tiêu:

> Agent thích nghi với tool mới/schema mới mà không cần train lại toàn bộ từ đầu.

---

## 4. Pipeline tổng thể

```text
[1] Telco Tool Registry + Tool Contracts
        ↓
[2] Dynamic Synthetic Data & Benchmark Generator
        ↓
[3] Seen / Unseen / Evolving Tool Split
        ↓
[4] Schema- and Contract-aware Tool Retrieval
        ↓
[5] Context / Slot / Intent / Risk Analyzer
        ↓
[6] SFT Schema-Contract Reading Baseline
        ↓
[7] Schema Validator + Contract Checker
        ↓
[8] Mock Telco Executor
        ↓
[9] Verifiable Reward + Rich Feedback Generator
        ↓
[10] Feedback-Guided On-Policy Self-Distillation / SDPO-lite
        ↓
[11] PI-SA-CS-LinUCB Bandit Strategy Router
        ↓
[12] Demo Agent + Evaluation Dashboard
```

---

## 5. Tech stack tổng quan

| Thành phần | Stack đề xuất | Vai trò |
|---|---|---|
| Ngôn ngữ | Python | Toàn bộ pipeline |
| Tool registry | JSON Schema, Pydantic | Chuẩn hóa function schema |
| Tool contract | JSON/YAML contract spec + rule checker | Precondition/postcondition/side-effect/risk |
| Synthetic data | Python templates, optional LLM paraphraser | Sinh dữ liệu Telco |
| Mock database | SQLite hoặc DuckDB | Mock thuê bao/gói cước/ticket/billing |
| Mock executor | Python functions, optional FastAPI | Giả lập Telco APIs |
| Retrieval | BM25 + BGE-M3 + FAISS | Retrieve top-k tools |
| Reranking | BGE-reranker-v2-m3 | Rerank tool candidates |
| Main SLM | Qwen3.5-4B hoặc Qwen3.5-9B | Backbone chính |
| Coder/agent candidate | Qwen3-Coder / Qwen3-Coder-30B-A3B | Optional upper-bound/external teacher ablation |
| Stable baseline | Qwen2.5-Coder-7B-Instruct | Baseline ổn định |
| Community baseline | Qwopus3.5-4B/9B-Coder | Baseline agent/tool-use community |
| Fine-tuning | Unsloth + QLoRA hoặc TRL + PEFT | SFT, Feedback-SDFT |
| Inference | Transformers, optional vLLM | Generate function call |
| Structured output | vLLM guided JSON / Outlines / post-parse repair | Ép JSON format |
| Validation | jsonschema + Pydantic | Validate schema |
| Contract checking | Custom Python rule checker | Validate business contract |
| Reward | Python rule-based scorer | Verifiable reward |
| Bandit | Custom NumPy | PI-SA-CS-LinUCB |
| Orchestration | Python state machine, optional LangGraph | Điều phối agent |
| Demo | Streamlit + Pandas/Plotly | Chat, trace, dashboard |
| Tracking | JSONL logs, optional W&B/MLflow | Log thực nghiệm |

---

## 6. Model LLM/SLM

### 6.1. Main backbone: Qwen3.5-4B hoặc Qwen3.5-9B

**Vai trò**

- Backbone chính cho SFT.
- Dùng cho on-policy rollout.
- Dùng cho feedback-guided self-correction.
- Dùng trong final demo nếu chạy được.

**Lý do chọn**

- Mới hơn Qwen2.5.
- Có size 4B/9B, hợp yêu cầu SLM.
- Qwen3.5 model card có hướng dẫn tool call qua vLLM với `--enable-auto-tool-choice` và `--tool-call-parser qwen3_coder`.
- Phù hợp function calling/tool-use hiện đại.

**Khuyến nghị**

```text
GPU hạn chế:
  Main = Qwen3.5-4B

GPU khá:
  Main = Qwen3.5-9B

Baseline ổn định:
  Qwen2.5-Coder-7B-Instruct
```

---

### 6.2. Agentic/coder candidate: Qwen3-Coder

**Vai trò**

- Optional upper-bound baseline.
- Optional external-teacher ablation.
- Optional data bootstrap/paraphrase/correction generation.

**Không dùng làm teacher chính của self-distillation**

Nếu dùng Qwen3-Coder sinh correction, đó là:

```text
External-teacher distillation
```

không phải self-distillation.

---

### 6.3. Stable baseline: Qwen2.5-Coder-7B-Instruct

**Vai trò**

- Baseline ổn định.
- Fallback nếu Qwen3.5 khó setup.
- So sánh model thế hệ trước với model mới.

---

### 6.4. Community baseline: Qwopus3.5-4B/9B-Coder

**Vai trò**

- Baseline community thiên về agent/tool-use.
- Có thể dùng prompt-only hoặc external-teacher ablation.

**Lưu ý**

- Cần tự benchmark lại.
- Không phụ thuộc hoàn toàn vào model card.

---

## 7. Telco Tool Registry + Tool Contracts

### 7.1. Mục tiêu

Xây kho function Telco có cả:

```text
schema
description
examples
business contract
risk metadata
dependencies
```

### 7.2. Input

```text
function_name
domain
description
parameters
required fields
type constraints
enum values
regex/pattern
examples
risk level
permission requirement
preconditions
postconditions
side effects
dependencies
```

### 7.3. Output

`tools.json` hoặc `tools.yaml`.

Ví dụ:

```json
{
  "name": "register_plan",
  "domain": "plan_management",
  "description": "Đăng ký gói cước mới cho thuê bao.",
  "parameters": {
    "type": "object",
    "properties": {
      "msisdn": {
        "type": "string",
        "pattern": "^[0-9]{10}$"
      },
      "plan_id": {
        "type": "string"
      }
    },
    "required": ["msisdn", "plan_id"]
  },
  "contract": {
    "preconditions": [
      "subscriber_status == active",
      "customer_verified == true",
      "plan_id in available_plans"
    ],
    "postconditions": [
      "subscription.plan_id == requested_plan_id"
    ],
    "side_effects": [
      "change_subscription",
      "charge_fee",
      "send_sms_notification"
    ],
    "risk_level": "high",
    "permission_required": "customer_verified"
  },
  "examples": [
    "Đăng ký gói 5G_MAX100 cho số 0987654321."
  ]
}
```

### 7.4. Số lượng function

MVP:

```text
30–40 functions
```

Trong đó:

```text
10–15 functions có contract đầy đủ
25 seen functions dùng train
10 unseen functions dùng test
5 hard-negative/adversarial functions
+ 50 distractor tools cho expanded-library test
```

### 7.5. Nhóm function

```text
Subscriber:
- get_subscriber_profile
- check_subscriber_status
- verify_customer_identity

Plan/Data:
- list_available_plans
- recommend_data_plan
- register_plan
- cancel_plan

Network/Support:
- check_network_status
- check_cell_quality
- create_trouble_ticket
- get_ticket_status
- escalate_ticket

Billing:
- get_billing_summary
- get_payment_history

eSIM/Roaming:
- activate_esim
- get_roaming_packages
- register_roaming_package

Sensitive/Permission:
- get_sensitive_profile
- lock_lost_sim
- unlock_sim
```

### 7.6. Tác dụng

- Thêm function mới bằng schema + contract.
- Validate đúng JSON và đúng nghiệp vụ.
- Tạo feedback giàu hơn cho self-distillation.
- Tạo điểm mới khác biệt so với function calling thường.

---

## 8. Dynamic Synthetic Data & Benchmark Generator

### 8.1. Mục tiêu

Sinh dữ liệu cho:

```text
schema-valid function calling
contract-valid function calling
unseen tools
changed schema/contract
deprecated tools
distractor-heavy setting
```

### 8.2. Stack

```text
Python template generator
Faker hoặc script random
Optional LLM paraphraser
Pydantic/jsonschema validator
Contract checker
JSONL
```

### 8.3. Output

```text
train.jsonl
eval_seen.jsonl
eval_unseen.jsonl
eval_contract.jsonl
eval_evolution_phase2_new_tools.jsonl
eval_evolution_phase3_schema_changed.jsonl
eval_evolution_phase4_deprecated.jsonl
eval_expanded.jsonl
negative_samples.jsonl
feedback_samples.jsonl
```

### 8.4. Sample

```json
{
  "instruction": "Đăng ký gói 5G_MAX100 cho số 0987654321.",
  "available_functions": ["register_plan", "get_current_plan", "list_available_plans"],
  "mock_state": {
    "subscriber_status": "suspended",
    "customer_verified": true,
    "available_plans": ["5G_MAX100", "DATA70"]
  },
  "ground_truth": {
    "action": "ask_clarification",
    "question": "Thuê bao hiện đang bị tạm khóa nên chưa thể đăng ký gói mới. Bạn cần kích hoạt lại thuê bao trước."
  },
  "scenario_type": "contract_violation"
}
```

### 8.5. Scenario cần có

| Scenario | Ý nghĩa |
|---|---|
| Single-step | Một query gọi một function |
| Multi-step | Cần nhiều bước |
| Parallel | Gọi nhiều hàm độc lập |
| Nested/dependent | Output hàm trước làm input hàm sau |
| Stateful | Dựa vào context hội thoại |
| Ask-clarification | Thiếu slot bắt buộc |
| Abstention/refusal | Yêu cầu nhạy cảm/ngoài phạm vi |
| Contract violation | Function call đúng schema nhưng sai business contract |
| Side-effect risk | Function có side effect/risk cao |
| Unseen-tool | Function chưa từng train |
| New-tool phase | Tool mới được thêm |
| Schema-change phase | Tool cũ đổi schema |
| Deprecated-tool phase | Tool cũ bị deprecated |
| Expanded-library | Nhiều distractor tools |

### 8.6. Quy mô đề xuất

```text
Train: 1,000–1,500 samples
Eval seen: 200–300 samples
Eval unseen: 150–250 samples
Eval contract: 150–250 samples
Eval dynamic phases: 100–200 samples/phase
Eval expanded: 200–300 samples
Feedback/correction samples: 500–1,000 samples
```

---

## 9. Seen / Unseen / Evolving Tool Split

### 9.1. Mục tiêu

Đánh giá khả năng thích nghi với tool mới và tool thay đổi.

### 9.2. Setup

```text
Phase 1 — Seen tools:
  25 tools dùng train.

Phase 2 — New tools:
  thêm 10 unseen tools như activate_esim, register_roaming_package.

Phase 3 — Schema/contract changes:
  đổi required field, enum, precondition của một số tools.

Phase 4 — Deprecated tools:
  deprecated một số hàm cũ, model không được gọi nữa.

Phase 5 — Expanded library:
  thêm 50 distractor tools.
```

### 9.3. Metrics mới

```text
new_tool_success_rate
schema_change_robustness
contract_change_robustness
deprecated_tool_avoidance
adaptation_gain_after_self_distillation
performance_drop_after_tool_evolution
recovery_rate_after_feedback_distillation
```

---

## 10. Schema- and Contract-aware Tool Retrieval

### 10.1. Mục tiêu

Retrieve top-k tools dựa trên query, schema và contract.

### 10.2. Stack

```text
BM25: rank-bm25
Dense embedding: BGE-M3
Vector index: FAISS
Reranker: BGE-reranker-v2-m3
Hybrid fusion: Reciprocal Rank Fusion
```

### 10.3. Tool text để embedding

```text
tool_text =
  name
  + domain
  + description
  + parameter names/descriptions
  + examples
  + preconditions
  + side_effects
  + risk_level
  + keywords
```

### 10.4. Hybrid score

```text
score = 0.35 * dense_similarity
      + 0.20 * bm25_score
      + 0.15 * intent_domain_score
      + 0.15 * slot_coverage_score
      + 0.10 * contract_compatibility_score
      + 0.05 * risk_prior_score
```

### 10.5. Contract compatibility examples

```text
Nếu query yêu cầu thay đổi gói cước:
  ưu tiên tools có side_effect = change_subscription.

Nếu customer_verified = false:
  giảm điểm tools yêu cầu permission_required = customer_verified.

Nếu subscriber_status = suspended:
  giảm điểm register_plan.
```

### 10.6. Metrics

```text
Tool Recall@3
Tool Recall@5
MRR
Contract-aware Recall@k
Avg candidate tools
Avg prompt tokens
```

---

## 11. Context / Slot / Intent / Risk Analyzer

### 11.1. Mục tiêu

Phân tích query để hỗ trợ retrieval, contract checking và bandit routing.

### 11.2. Stack

```text
Rule-based regex + dictionaries
Optional small LLM prompt classifier
Python
```

### 11.3. Output

```json
{
  "intent": "register_plan",
  "slots": {
    "msisdn": "0987654321",
    "plan_id": "5G_MAX100"
  },
  "missing_required_slots": [],
  "is_multi_step": false,
  "is_sensitive": false,
  "side_effect_intent": true,
  "requires_permission": true,
  "risk_level_estimate": "high"
}
```

### 11.4. Tác dụng

- Tăng ask-back accuracy.
- Giảm unsafe function calls.
- Hỗ trợ contract-aware retrieval.
- Tạo context vector cho bandit.
- Phát hiện side-effect/risk trước khi gọi hàm.

---

## 12. SFT Schema-Contract Reading Baseline

### 12.1. Mục tiêu

Train model đọc schema và contract, sinh function call/action đúng.

### 12.2. Stack

```text
Transformers
Unsloth / PEFT
QLoRA
Optional vLLM for serving
```

### 12.3. Input prompt

```text
You are a Telco function calling agent.

User query:
"Đăng ký gói 5G_MAX100 cho số 0987654321."

Current state:
subscriber_status = suspended
customer_verified = true

Available functions:
1. register_plan(...)
   Contract:
   - precondition: subscriber_status == active
   - precondition: customer_verified == true
   - side_effect: change_subscription, charge_fee

2. get_current_plan(...)

Return JSON only:
{
  "action": "call_function" | "ask_clarification" | "abstain",
  "calls": [...],
  "question": "...",
  "reason": "..."
}
```

### 12.4. Output

```json
{
  "action": "ask_clarification",
  "question": "Thuê bao hiện đang bị tạm khóa nên chưa thể đăng ký gói mới. Bạn cần kích hoạt lại thuê bao trước.",
  "reason": "register_plan requires subscriber_status == active."
}
```

### 12.5. Training stages

```text
V0: Prompt-only baseline
V1: SFT on seen tools + contracts
V2: Feedback-Guided Self-Distillation / SDPO-lite
V3: Final model + bandit router
```

### 12.6. Fine-tuning đề xuất

```text
Framework:
  Unsloth + QLoRA
  hoặc TRL + PEFT nếu cần custom nhiều hơn

LoRA rank:
  16–64

Target modules:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Learning rate:
  1e-4 đến 2e-4

Epochs:
  1–3

Max sequence length:
  4k–8k tùy GPU
```

---

## 13. Structured Output Control

### Mục tiêu

Giảm lỗi JSON/function-call format.

### Stack

MVP:

```text
Prompt JSON-only
Post-parse repair
jsonschema validation
```

Nâng cao:

```text
vLLM guided_json
Outlines
XGrammar
```

### Tác dụng

- Tăng schema validity.
- Giảm parse error.
- Tách lỗi format với lỗi reasoning/contract.
- Hỗ trợ evaluation tự động.

---

## 14. Schema Validator + Contract Checker

### 14.1. Mục tiêu

Kiểm tra output hợp lệ về:

```text
JSON format
schema
business contract
permission
side-effect risk
postcondition
```

### 14.2. Stack

```text
jsonschema
Pydantic
Custom Python contract checker
```

### 14.3. Input

```json
{
  "action": "call_function",
  "calls": [
    {
      "name": "register_plan",
      "arguments": {
        "msisdn": "0987654321",
        "plan_id": "5G_MAX100"
      }
    }
  ]
}
```

Mock state:

```json
{
  "subscriber_status": "suspended",
  "customer_verified": true,
  "available_plans": ["5G_MAX100", "DATA70"]
}
```

### 14.4. Output

```json
{
  "schema_valid": true,
  "contract_valid": false,
  "errors": [
    {
      "type": "precondition_failed",
      "condition": "subscriber_status == active",
      "actual": "suspended"
    }
  ],
  "suggested_action": "ask_clarification"
}
```

### 14.5. Tác dụng

- Bắt lỗi mà schema validation không bắt được.
- Ngăn unsafe/side-effect calls.
- Tạo contract feedback cho self-distillation.
- Tạo metric mới: contract validity, precondition violation rate.

---

## 15. Mock Telco Executor

### Mục tiêu

Giả lập backend Telco để execute function call.

### Stack

```text
Python functions
SQLite hoặc DuckDB
Optional FastAPI
```

### Mock DB tables

```text
subscribers(msisdn, status, customer_type, region, verified)
plans(plan_id, price, data_gb, validity_days, eligible_regions)
subscriptions(msisdn, current_plan, remaining_data)
tickets(ticket_id, msisdn, issue_type, location, status)
network_status(location, network_type, status, outage_level)
billing(msisdn, month, amount, paid_status)
tool_versions(tool_name, version, status)
```

### Output example

```json
{
  "success": true,
  "data": {
    "ticket_id": "TT20260602001",
    "status": "created",
    "estimated_resolution_time": "24h"
  }
}
```

### Tác dụng

- Chấm execution success.
- Chấm postcondition.
- Tạo final response.
- Tạo multi-step dependencies.
- Hỗ trợ dynamic tool evolution.

---

## 16. Verifiable Reward + Rich Feedback Generator

### 16.1. Mục tiêu

Tạo scalar reward và feedback chi tiết từ schema, contract và execution.

### 16.2. Input

```text
Ground truth
Model output
Schema validation result
Contract checking result
Execution result
Business rule result
Latency/token info
```

### 16.3. Output

```json
{
  "reward": 0.37,
  "components": {
    "action_correct": 0.0,
    "function_selection": 1.0,
    "argument_accuracy": 1.0,
    "schema_validity": 1.0,
    "contract_validity": 0.0,
    "execution_success": 0.0,
    "task_success": 0.0,
    "unsafe_call_penalty": 1.0
  },
  "feedback": [
    "The selected function and arguments are schema-valid.",
    "Precondition failed: subscriber_status must be active.",
    "The function has high-risk side effect: change_subscription.",
    "Correct action should be ask_clarification or abstain."
  ]
}
```

### 16.4. Reward formula

```text
R = 0.15 * action_correct
  + 0.15 * function_selection
  + 0.15 * argument_accuracy
  + 0.10 * schema_validity
  + 0.15 * contract_validity
  + 0.10 * execution_success
  + 0.10 * task_success
  + 0.05 * correct_ask_or_abstain
  - 0.10 * hallucinated_call
  - 0.10 * unnecessary_call
  - 0.15 * unsafe_side_effect_call
  - 0.05 * deprecated_tool_call
  - 0.05 * latency_penalty
  - 0.05 * token_cost_penalty
```

### 16.5. Tác dụng

- Evaluation.
- Bandit update.
- Self-distillation sample selection.
- Tạo feedback giàu hơn schema-only.
- Không phụ thuộc LLM judge.

---

## 17. Feedback-Guided On-Policy Self-Distillation / SDPO-lite

### 17.1. Mục tiêu

Cải thiện model bằng lỗi của chính model, với feedback từ schema + contract + execution.

### 17.2. Điểm cần giữ chuẩn

```text
Self-distillation = chính SFT checkpoint hiện tại rollout và self-correct.
External teacher = optional ablation, không phải core method.
```

### 17.3. Core pipeline

```text
1. Train SFT model trên seen tools + contracts.
2. Dùng chính SFT model rollout trên train/unseen/evolution prompts.
3. Validator + contract checker + executor sinh reward và rich feedback.
4. Với output sai, tạo correction prompt.
5. Dùng chính SFT model + feedback prompt để sinh corrected output.
6. Validate corrected output bằng schema + contract checker.
7. Lọc corrected samples hợp lệ.
8. Fine-tune lại model trên corrected samples.
```

### 17.4. Correction prompt format

```text
Instruction:
{user_query}

Current state:
{mock_state}

Available schema and contracts:
{top_k_tool_schemas_and_contracts}

Wrong output:
{model_wrong_output}

Validator and contract feedback:
{rich_feedback}

Task:
Generate the corrected JSON output only.
```

### 17.5. Output corrected sample

```json
{
  "input": "instruction + state + schema + contract",
  "target": {
    "action": "ask_clarification",
    "question": "Thuê bao hiện đang bị tạm khóa nên chưa thể đăng ký gói mới. Bạn cần kích hoạt lại thuê bao trước.",
    "reason": "register_plan requires subscriber_status == active."
  }
}
```

### 17.6. Optional external-teacher ablation

```text
External-Teacher SDFT:
  Qwen3-Coder hoặc Qwen3.5-9B sinh corrected output.

Mục đích:
  So sánh self-distillation với teacher-student distillation.
```

### 17.7. Metrics riêng

```text
correction_success_rate
valid_corrected_sample_rate
contract_error_reduction
unsafe_call_reduction
unseen_tool_improvement
evolution_phase_recovery
```

---

## 18. PI-SA-CS-LinUCB Bandit Router

### 18.1. Thuật toán chính

**PI-SA-CS-LinUCB**

> Prior-Informed Schema-Aware Cost-Sensitive LinUCB

### 18.2. Arms

| Arm | Strategy | Khi nào tốt |
|---|---|---|
| A1 | Direct schema call | Query đơn giản, đủ thông tin, risk thấp |
| A2 | Schema-contract reasoning | Function mới, contract phức tạp |
| A3 | Plan-then-call | Multi-step/nested |
| A4 | Self-correct once | Retrieval thấp hoặc schema/contract dễ lỗi |
| A5 | Ask-clarification-biased | Thiếu slot/precondition chưa rõ |
| A6 | Abstain/safety-biased | Sensitive/high-risk/permission thiếu |

### 18.3. Context features

```text
query_length
has_msisdn
has_location
has_money
has_month
has_plan_id
has_ticket_id
missing_required_slots_count
slot_coverage_ratio
retrieval_confidence
top1_top2_tool_score_gap
num_candidate_tools
num_required_args
num_enum_fields
schema_complexity_score
contract_complexity_score
risk_level_score
side_effect_score
permission_missing
is_unseen_tool
is_new_tool_phase
is_schema_changed_tool
is_deprecated_candidate
is_multi_step
is_parallel
is_stateful
is_sensitive_request
estimated_prompt_tokens
estimated_latency
```

### 18.4. Score function

```text
score_a = θ_a^T x
        + α * sqrt(x^T A_a^{-1} x)
        + γ * prior_a(x)
        - β * cost_a(x)
```

### 18.5. Prior examples

```text
missing_slots_count cao → tăng prior ask-clarification-biased
contract_complexity cao → tăng prior schema-contract reasoning
side_effect_score cao → tăng prior abstain/safety-biased
permission_missing = true → tăng prior ask/abstain
is_multi_step = true → tăng prior plan-then-call
is_schema_changed_tool = true → tăng prior self-correct once
retrieval_confidence cao + risk thấp → tăng prior direct schema call
```

### 18.6. Baselines

```text
Fixed strategy
Epsilon-greedy
UCB1
LinUCB
PI-SA-CS-LinUCB
Optional Thompson Sampling
```

### 18.7. Metrics

```text
average_reward
cumulative_reward
average_regret
arm_selection_distribution
best_arm_by_scenario
contract_violation_rate_by_arm
unsafe_call_rate_by_arm
cost_per_success
```

---

## 19. Agent Orchestration

### Stack

MVP:

```text
Python state machine
```

Nâng cao:

```text
LangGraph
```

### Nodes

```text
analyze_query
retrieve_tools
bandit_route
build_prompt
generate_call
validate_schema
check_contract
self_correct_if_needed
execute_tool
check_postconditions
generate_final_answer
log_reward
```

### Online flow

```text
User query
   ↓
Intent/slot/risk analyzer
   ↓
Schema-contract-aware retriever top-k
   ↓
Bandit chọn strategy
   ↓
Prompt builder
   ↓
SLM generate function call/action
   ↓
Schema validator
   ↓
Contract checker
   ↓
Nếu lỗi và strategy = self_correct_once → sửa 1 lần
   ↓
Mock executor
   ↓
Postcondition checker
   ↓
Reward + feedback
   ↓
Final answer
   ↓
Log trace
```

---

## 20. Demo Stack

### Stack

```text
Streamlit
Pandas
Plotly
JSON viewer
```

### Tabs

```text
Tab 1: Chat Agent
Tab 2: Retrieved Tools + Contracts
Tab 3: Function Call Trace
Tab 4: Schema Validation
Tab 5: Contract Checking
Tab 6: Execution Result
Tab 7: Reward + Rich Feedback
Tab 8: Bandit Dashboard
Tab 9: Benchmark Results
```

### Demo scenario 1 — valid low-risk call

```text
Tôi ở Đống Đa, Hà Nội, mạng 5G rất chậm, số 0987654321. Tạo ticket giúp tôi.
```

Expected:

```text
create_trouble_ticket
schema valid
contract valid
execution success
```

### Demo scenario 2 — schema-valid but contract-invalid call

```text
Đăng ký gói 5G_MAX100 cho số 0987654321.
```

Mock state:

```text
subscriber_status = suspended
customer_verified = true
```

Expected:

```text
Không gọi register_plan.
Ask clarification / explain precondition failed.
```

### Demo scenario 3 — sensitive/permission

```text
Cho tôi xem thông tin thuê bao của số 0912345678 không phải của tôi.
```

Expected:

```text
abstain hoặc yêu cầu xác thực.
```

### Demo scenario 4 — new function

```text
Kích hoạt eSIM cho số 0987654321, EID là 8904903200000000001.
```

Expected:

```text
activate_esim được retrieve và gọi đúng dù unseen trong training.
```

### Demo scenario 5 — deprecated tool

Tool `old_register_plan` bị deprecated.

Expected:

```text
Không gọi old_register_plan.
Dùng register_plan_v2 hoặc ask clarification.
```

---

## 21. Evaluation Framework

### Stack

```text
Python evaluator
Pydantic/jsonschema parse
Contract checker
Pandas
scikit-learn metrics
Plotly
```

### Methods so sánh

| Method | Vai trò |
|---|---|
| Prompt-only | Zero-shot baseline |
| SFT | Học format, schema và seen tools |
| SFT + Retrieval | Kiểm tra tác dụng retrieval |
| SFT + Contract Checker only | Rule-based safety layer |
| Self-SDFT | Self-distill từ high-reward/self-corrected outputs |
| Feedback-SDFT / SDPO-lite | Distill từ rich schema+contract feedback |
| External-Teacher SDFT | Optional ablation |
| SDPO-lite + PI-SA-CS-LinUCB | Final system |
| Optional DPO | Baseline preference optimization cũ |

### Eval sets

| Eval set | Mục tiêu |
|---|---|
| Seen tools | Function đã train |
| Unseen tools | Function mới chưa train |
| Contract violation | Đúng schema nhưng sai contract |
| Side-effect risk | Gọi hàm có rủi ro cao |
| Dynamic phase new tools | Thêm tool mới |
| Dynamic phase schema changed | Schema/contract đổi |
| Dynamic phase deprecated | Tool bị deprecated |
| Expanded library | Nhiều distractor functions |
| Missing slot | Kiểm tra ask-back |
| Multi-step | Kiểm tra planning |
| Safety/abstain | Kiểm tra từ chối đúng |

### Metrics

Function calling:

```text
function_selection_accuracy
argument_accuracy
argument_f1
schema_validity
execution_success_rate
task_success_rate
hallucinated_call_rate
unnecessary_call_rate
ask_back_accuracy
abstention_accuracy
latency
token_cost
```

Contract-aware metrics:

```text
contract_validity
precondition_violation_rate
postcondition_success_rate
unsafe_side_effect_call_rate
permission_violation_rate
deprecated_tool_call_rate
contract_feedback_correction_rate
```

Tool evolution metrics:

```text
new_tool_success_rate
schema_change_robustness
contract_change_robustness
deprecated_tool_avoidance
adaptation_gain_after_self_distillation
performance_drop_after_tool_evolution
recovery_rate_after_feedback_distillation
```

Retrieval:

```text
tool_recall@3
tool_recall@5
MRR
contract_aware_recall@k
avg_candidate_tools
avg_prompt_tokens
```

Bandit:

```text
average_reward
cumulative_reward
average_regret
arm_selection_distribution
best_arm_by_scenario
contract_violation_rate_by_arm
cost_per_success
```

Distillation:

```text
correction_success_rate
feedback_utilization_rate
valid_corrected_sample_rate
seen_vs_unseen_improvement
contract_error_reduction
unsafe_call_reduction
```

---

## 22. Repo Structure

```text
telco-function-agent/
├── data/
│   ├── tools.json
│   ├── tool_contracts.json
│   ├── mock_telco_db.json
│   ├── train.jsonl
│   ├── eval_seen.jsonl
│   ├── eval_unseen.jsonl
│   ├── eval_contract.jsonl
│   ├── eval_evolution_new_tools.jsonl
│   ├── eval_evolution_schema_changed.jsonl
│   ├── eval_evolution_deprecated.jsonl
│   └── eval_expanded.jsonl
├── src/
│   ├── registry/
│   │   ├── tool_registry.py
│   │   └── contract_registry.py
│   ├── generation/
│   │   └── synth_data_generator.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── bge_retriever.py
│   │   └── hybrid_reranker.py
│   ├── analyzer/
│   │   └── slot_intent_risk_analyzer.py
│   ├── bandit/
│   │   ├── epsilon_greedy.py
│   │   ├── ucb.py
│   │   ├── linucb.py
│   │   └── pi_sa_cs_linucb.py
│   ├── model/
│   │   ├── prompt_builder.py
│   │   ├── inference.py
│   │   └── train_sft.py
│   ├── validation/
│   │   ├── schema_validator.py
│   │   ├── contract_checker.py
│   │   └── business_rules.py
│   ├── executor/
│   │   └── mock_telco_api.py
│   ├── reward/
│   │   └── reward_feedback.py
│   ├── distillation/
│   │   ├── rollout.py
│   │   ├── self_correction.py
│   │   ├── external_teacher_ablation.py
│   │   └── build_distill_dataset.py
│   └── evaluation/
│       └── evaluator.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── generate_data.py
│   ├── train_sft.sh
│   ├── run_rollout.py
│   ├── train_sdpo_lite.sh
│   └── run_eval.py
└── reports/
    └── results.md
```

---

## 23. Kế hoạch 1 tháng

### Tuần 1 — Registry + Contracts + Data + Validator

Output:

```text
tools.json
tool_contracts.json
mock_telco_db.json
train.jsonl
eval_seen.jsonl
eval_unseen.jsonl
eval_contract.jsonl
eval_evolution_*.jsonl
schema_validator.py
contract_checker.py
mock_executor.py
reward_scorer.py
```

Việc chính:

```text
- Thiết kế 30–40 Telco functions.
- Thêm contract đầy đủ cho 10–15 functions quan trọng.
- Chia seen/unseen/evolution tools.
- Sinh synthetic dataset.
- Viết schema validator, contract checker, mock executor.
- Viết reward function đầu tiên.
```

---

### Tuần 2 — Retrieval + SFT Baseline

Output:

```text
hybrid_tool_retriever.py
prompt-only results
SFT adapter
SFT results
Tool Recall@k results
Contract-aware Recall@k results
```

Việc chính:

```text
- Build BM25 + BGE-M3 + FAISS retrieval.
- Thêm contract-aware scoring.
- Optional reranker.
- Prompt-only baseline.
- Train QLoRA SFT trên seen tools + contracts.
- Đánh giá seen/unseen/contract ban đầu.
```

---

### Tuần 3 — Feedback-SDFT / SDPO-lite + Bandit

Output:

```text
rollout logs
rich feedback data
self-corrected samples
SDPO-lite adapter
PI-SA-CS-LinUCB router
bandit learning curves
```

Việc chính:

```text
- Rollout SFT model.
- Sinh feedback từ schema validator + contract checker + executor.
- Dùng chính SFT model để self-correct bằng feedback prompt.
- Validate và lọc self-corrected outputs.
- Train Feedback-SDFT / SDPO-lite.
- Implement epsilon-greedy, UCB, LinUCB, PI-SA-CS-LinUCB.
- Đánh giá bandit routing.
```

---

### Tuần 4 — Demo + Evaluation + Report

Output:

```text
Streamlit demo
Evaluation dashboard
Method comparison tables
Ablation study
Final report/slides
```

Việc chính:

```text
- Hoàn thiện demo agent.
- Tạo benchmark dashboard.
- So sánh methods.
- Optional external-teacher ablation nếu còn thời gian.
- Viết báo cáo.
- Chuẩn bị câu trả lời phản biện.
```

---

## 24. Scope tối thiểu để chắc hoàn thành

Must-have:

```text
1. Tool Registry 30–40 functions.
2. Tool Contract cho 10–15 functions quan trọng.
3. Seen/unseen/evolution split.
4. Synthetic dataset.
5. Hybrid retrieval.
6. SFT baseline.
7. Schema validator + contract checker + reward + feedback.
8. Feedback-Guided Self-Distillation / SDPO-lite bằng chính SFT checkpoint.
9. PI-SA-CS-LinUCB router.
10. Streamlit demo.
```

Nice-to-have:

```text
1. External-teacher ablation với Qwen3-Coder/Qwen3.5-9B.
2. DPO baseline.
3. BGE reranker.
4. vLLM guided JSON.
5. LangGraph orchestration.
6. Qwopus vs Qwen comparison.
7. Preference-tunable bandit mode.
8. Tool-use error memory.
```

Không nên làm:

```text
1. Full PPO/GRPO.
2. Full SDPO như paper.
3. Train neural reward model.
4. Real Telco backend.
5. Hàng nghìn functions.
6. Multi-agent phức tạp.
7. Gọi external teacher là self-distillation.
```

---

## 25. Chốt stack cuối cùng

```text
Model:
- Main: Qwen3.5-4B hoặc Qwen3.5-9B
- Agentic/coder candidate: Qwen3-Coder / Qwen3-Coder-30B-A3B nếu chạy được
- Stable baseline: Qwen2.5-Coder-7B-Instruct
- Community baseline: Qwopus3.5-4B/9B-Coder

Training:
- Unsloth + QLoRA
- Optional HF TRL + PEFT

Retrieval:
- BGE-M3
- BM25
- FAISS
- Optional BGE-reranker-v2-m3

Validation:
- Pydantic
- jsonschema
- custom business rules
- contract checker

Executor:
- Python mock functions
- SQLite/DuckDB
- Optional FastAPI

Optimization:
- Feedback-Guided On-Policy Self-Distillation / SDPO-lite
- PI-SA-CS-LinUCB
- Optional external-teacher ablation
- Optional DPO baseline

Inference:
- Transformers
- Optional vLLM guided JSON

Demo:
- Streamlit
- Pandas/Plotly

Tracking:
- JSONL logs
- Optional W&B/MLflow
```

---

## 26. Câu định vị cuối

> Đề tài tập trung vào contract-aware schema-generalized Telco function calling: thay vì chỉ fine-tune model để ghi nhớ tập hàm cố định hoặc chỉ tạo JSON hợp lệ, hệ thống học cách đọc schema và business contract động, retrieve function liên quan, tránh các function call vi phạm precondition/permission/side-effect, tự sửa lỗi từ phản hồi kiểm chứng bằng chính checkpoint hiện tại, và dùng contextual bandit để chọn chiến lược gọi hàm phù hợp cho từng query.

---

## 27. Nguồn tham khảo nhanh

- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3-Coder-30B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Qwen2.5-Coder-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen3-Coder collection: https://huggingface.co/collections/Qwen/qwen3-coder
- Qwopus-Coder collection: https://huggingface.co/collections/Jackrong/qwopus-coder
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- BGE-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- vLLM Qwen3.5 usage: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- Unsloth documentation: https://docs.unsloth.ai/
- Hugging Face TRL: https://huggingface.co/docs/trl/
- LangGraph: https://github.com/langchain-ai/langgraph
