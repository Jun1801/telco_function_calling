# Chiến lược triển khai tổng thể
# Contract-Guided Variational Policy Distillation cho Telco Function Calling

## 0. Mục tiêu của tài liệu

Tài liệu này mô tả kế hoạch kỹ thuật end-to-end cho đề tài:

> **Contract-Guided Variational Policy Distillation for Generalizable Telco Function Calling**

Tên tiếng Việt:

> **Tối ưu Function Calling cho Telco Agent trên tập hàm động bằng Tool Contract, phản hồi kiểm chứng, Variational Policy Distillation và Bandit Routing**

Kế hoạch kết hợp các ý tưởng chính từ:

- **ToolRL: Reward is All Tool Learning Needs**
  - Reward fine-grained cho tool selection và parameter correctness.
  - Correctness reward quan trọng hơn format reward.
  - Dynamic/progressive reward.
- **Tool-Zero**
  - Tool-name và parameter-name masking.
  - Multi-turn augmentation.
  - Soft-to-strict reward schedule.
  - Per-argument-value penalty.
  - Multi-tool collaboration reward.
- **Reinforcement Learning via Self-Distillation — SDPO**
  - Dùng model conditioned on rich feedback làm self-teacher.
  - Distill teacher token distribution vào student trên on-policy rollouts.
- **Learning from Language Feedback via Variational Policy Distillation — VPD**
  - Teacher không còn thụ động.
  - E-step cập nhật feedback-conditioned teacher.
  - M-step distill teacher vào student.
  - Dynamic trust region giữa teacher và current policy.
- **ToolACE**
  - Synthetic function-calling data generation.
  - Rule-based và execution-based verification.

---

# 1. Câu hỏi nghiên cứu

Đề tài cần trả lời các câu hỏi:

1. Model có gọi đúng function Telco đã thấy trong training không?
2. Model có gọi được function mới chỉ nhờ đọc schema ở inference không?
3. Tool-name masking có giúp model bớt học thuộc tên hàm không?
4. Contract-aware reward có giảm function call đúng schema nhưng sai nghiệp vụ không?
5. Rich language feedback có cải thiện policy tốt hơn scalar reward đơn thuần không?
6. VPD có tốt hơn SFT, Feedback-SDFT và passive SDPO không?
7. Progressive soft-to-strict reward có giúp training ổn định hơn strict reward từ đầu không?
8. Model có thích nghi được khi API mới được thêm, schema thay đổi hoặc tool bị deprecated không?
9. Contextual bandit có chọn được strategy phù hợp với từng query để tối ưu accuracy, safety và cost không?

---

# 2. Đóng góp kỹ thuật dự kiến

## Contribution 1 — Contract-aware Telco Function Calling Environment

Xây môi trường function calling kiểm chứng được gồm:

```text
JSON schema
business preconditions
postconditions
permissions
side effects
risk levels
tool dependencies
mock execution
```

Một function call chỉ được xem là đúng khi:

```text
schema valid
AND contract valid
AND execution success
AND task success
```

## Contribution 2 — Telco-ToolACE-mini Data Pipeline

Sinh dữ liệu:

```text
seen tools
unseen tools
renamed/masked tools
multi-turn clarification
contract violations
schema changes
deprecated tools
parallel/multi-step calls
hard negatives
```

Tất cả ground truth và negative samples được verify bằng:

```text
schema validator
contract checker
mock executor
```

## Contribution 3 — Tool-Generalization Curriculum

Kết hợp các ý tưởng Tool-Zero:

```text
real names
→ function-name masking
→ function + parameter masking
→ unseen tools
→ paraphrased schemas
→ schema changes
```

Mục tiêu là buộc model học semantics của description/schema thay vì ghi nhớ tên API.

## Contribution 4 — Progressive Tool Reward

Reward chuyển dần:

```text
soft component-level reward
→ strict JSON/AST + contract + execution reward
```

Reward được decomposed theo:

```text
action
function name
argument keys
argument values
schema
contract
execution
task
safety
```

## Contribution 5 — VPD-style Adaptive Self-Teacher

Teacher branch:

```text
nhìn prompt gốc
+ rollout
+ structured language feedback
```

Teacher được cập nhật trong E-step để diễn giải feedback tốt hơn.

Student branch:

```text
chỉ nhìn prompt gốc
```

Student học teacher distribution trong M-step.

---

# 3. Kiến trúc tổng thể

```text
Telco Tool Registry + Tool Contracts
        ↓
Telco-ToolACE-mini Data Generator
        ↓
Seen / Unseen / Evolution Benchmark
        ↓
Hybrid Tool Retrieval: BM25 + BGE-M3 + Reranker
        ↓
Context / Slot / Risk Analyzer
        ↓
Qwen3.5 SLM Function Calling Policy
        ↓
Schema Validator + Contract Checker + Mock Executor
        ↓
Scalar Reward + Structured Feedback
        ↓
SFT → Feedback-SDFT → SDPO → VPD-lite
        ↓
PI-SA-CS-LinUCB Strategy Router
        ↓
Demo + Evaluation Dashboard
```

---

# 4. Tech stack chốt

| Thành phần | Công nghệ |
|---|---|
| Language | Python 3.11 |
| Main model | Qwen3.5-4B hoặc Qwen3.5-9B |
| Stable baseline | Qwen2.5-Coder-7B-Instruct |
| Optional external teacher | Qwen3-Coder hoặc Qwen3.5-9B |
| Fine-tuning | Unsloth + QLoRA |
| Custom training | Hugging Face Transformers + TRL/PEFT |
| RL/VPD backend | ưu tiên `verl`, fallback custom PyTorch |
| Inference | Transformers; optional vLLM |
| Tool schemas | JSON Schema |
| Data models | Pydantic |
| Sparse retrieval | rank-bm25 |
| Dense retrieval | BGE-M3 |
| Vector index | FAISS |
| Reranker | BGE-reranker-v2-m3 |
| Mock DB | SQLite hoặc DuckDB |
| API simulator | Pure Python, optional FastAPI |
| Bandit | NumPy/PyTorch custom |
| Demo | Streamlit |
| Charts | Plotly/Pandas |
| Tracking | JSONL + W&B optional |

---

# 5. Model strategy

## 5.1. Main model

### Phương án ưu tiên

```text
Qwen3.5-4B
```

Lý do:

```text
đủ nhỏ để QLoRA
phù hợp timeline 1 tháng
output function call ngắn
dễ chạy nhiều ablation
```

### Phương án nếu GPU mạnh

```text
Qwen3.5-9B
```

### Baseline

```text
Qwen2.5-Coder-7B-Instruct
```

### External-teacher ablation

```text
Qwen3-Coder
hoặc Qwen3.5-9B nếu student là 4B
```

Lưu ý:

```text
External teacher không phải core self-distillation.
```

---

# 6. Thiết kế Tool Registry

## 6.1. Số lượng

MVP:

```text
30–40 tools
```

Trong đó:

```text
25 seen tools
10 unseen tools
5 hard-negative tools
10–15 tools có contract đầy đủ
50 synthetic distractor tools cho scale test
```

## 6.2. Tool schema

```json
{
  "name": "register_plan",
  "domain": "plan_management",
  "description": "Đăng ký một gói cước cho thuê bao.",
  "parameters": {
    "type": "object",
    "properties": {
      "msisdn": {
        "type": "string",
        "pattern": "^[0-9]{10}$",
        "description": "Số thuê bao 10 chữ số."
      },
      "plan_id": {
        "type": "string",
        "description": "Mã gói cước."
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
  "status": "active",
  "replacement_tool": null
}
```

---

# 7. Data strategy — Telco-ToolACE-mini

## 7.1. Public data warm-up

Có thể dùng subset:

```text
xLAM Function Calling 60K
ToolACE
APIGen-MT-5K
xLAM irrelevance
Hermes function calling optional
```

Không dùng toàn bộ ngay.

Khuyến nghị:

```text
10K–20K general tool-use samples
```

Mục tiêu:

```text
format
tool schema reading
general function selection
parallel calls
multi-turn
no-tool behavior
```

## 7.2. Telco custom data

MVP:

```text
1,000–1,500 Telco SFT samples
500–1,000 feedback/correction samples
```

Scenario distribution:

```text
25% single-step valid calls
15% missing-slot / ask-back
10% abstention / permission
20% contract-aware decisions
10% multi-step/dependency
5% parallel calls
10% name/parameter masking
5% schema change/deprecated tools
```

## 7.3. Masking curriculum

### Stage 1 — Real names

```json
{
  "name": "activate_esim",
  "parameters": {
    "msisdn": {},
    "eid": {}
  }
}
```

### Stage 2 — Mask function name

```json
{
  "name": "func_7",
  "description": "Kích hoạt eSIM cho thuê bao bằng EID."
}
```

### Stage 3 — Mask function và parameters

```json
{
  "name": "func_7",
  "parameters": {
    "param_1": {
      "description": "Số thuê bao"
    },
    "param_2": {
      "description": "Mã nhận dạng eSIM"
    }
  }
}
```

Tỉ lệ gợi ý:

```text
50% real names
25% function-name masked
15% function + parameter masked
10% renamed/paraphrased schema
```

---

# 8. Multi-turn augmentation

Áp dụng bốn kiểu.

## 8.1. Missing parameter

```text
User: Đăng ký gói data giúp tôi.
Agent: Bạn cho mình xin số thuê bao và mã gói.
User: 0987654321, gói 5G_MAX100.
Agent: call register_plan(...)
```

## 8.2. Delayed tool availability

```text
Turn 1:
register_plan chưa có.

Turn 2:
register_plan_v2 được thêm vào registry.
```

## 8.3. User correction

```text
Agent gọi sai plan_id.
User nói: Tôi muốn gói DATA70, không phải DATA120.
Agent sửa call.
```

## 8.4. Backend challenge

```text
Agent gọi register_plan.
Backend feedback: subscriber suspended.
Agent chuyển sang ask/abstain.
```

---

# 9. Reward design

## 9.1. Reward components

```text
R_format
R_action
R_function
R_argument_keys
R_argument_values
R_schema
R_contract
R_execution
R_task
```

Penalties:

```text
P_hallucinated
P_unnecessary
P_unsafe
P_deprecated
P_extra_calls
P_latency
P_tokens
```

## 9.2. Call matching

Với parallel/multiple calls, không so sánh cứng theo thứ tự.

Dùng Hungarian matching:

```text
predicted calls ↔ gold calls
```

Pair score:

```text
S_pair =
  w_name * name_match
+ w_key * argument_key_score
+ w_value * argument_value_score
```

## 9.3. Argument score

```text
R_arg_keys = F1(predicted argument keys, gold argument keys)
```

```text
R_arg_values = average field-level matching score
```

Field weighting:

```text
msisdn: high weight
plan_id: high weight
enum fields: high weight
free-text description: lower weight
```

## 9.4. Action-specific masking

### Gold = call_function

Chấm:

```text
function
arguments
schema
contract
execution
task
```

### Gold = ask_clarification

Chấm:

```text
action đúng
missing slots được hỏi đủ
không gọi tool
```

### Gold = abstain

Chấm:

```text
abstain đúng
reason liên quan permission/risk
không gọi unsafe function
```

## 9.5. Soft reward

Giai đoạn đầu:

```text
partial function match
partial argument match
domain match
schema partial validity
contract component correctness
```

## 9.6. Strict reward

Giai đoạn sau:

```text
valid JSON
correct action
exact/canonical function
all required args
schema valid
contract valid
execution success
task success
```

## 9.7. Progressive reward schedule

```text
R_t = (1 - lambda_t) * R_soft + lambda_t * R_strict
```

Sigmoid schedule:

```python
lambda_t = 1 / (1 + exp(-k * (progress - midpoint)))
```

Gợi ý:

```text
0–20% steps:
  70% soft, 30% strict

20–50%:
  chuyển dần

50–100%:
  10% soft, 90% strict
```

## 9.8. Reward proposal

```text
R_soft =
  0.10 action
+ 0.15 function/domain
+ 0.15 argument keys
+ 0.15 argument values
+ 0.10 schema partial
+ 0.15 contract partial
+ 0.10 execution progress
+ 0.10 task progress
```

```text
R_strict =
  0.05 format
+ 0.10 action
+ 0.15 function
+ 0.15 argument keys
+ 0.15 argument values
+ 0.10 schema
+ 0.15 contract
+ 0.10 execution
+ 0.05 task
```

Penalties:

```text
-0.25 unsafe call
-0.20 hallucinated tool
-0.15 deprecated tool
-0.10 unnecessary call
-0.05 extra tool call
-0.05 normalized token/latency cost
```

---

# 10. Structured feedback design

Output gồm hai phần:

```text
machine-readable feedback
human-readable language feedback
```

Example:

```json
{
  "reward": 0.31,
  "status": "failed",
  "correct_components": {
    "function_name": true,
    "argument_keys": true,
    "schema": true
  },
  "incorrect_components": {
    "contract": false,
    "execution": false
  },
  "errors": [
    {
      "type": "precondition_failed",
      "condition": "subscriber_status == active",
      "actual": "suspended"
    }
  ],
  "suggested_action": "ask_clarification",
  "feedback_text": [
    "The selected function and arguments are schema-valid.",
    "The subscriber is suspended, but register_plan requires an active subscriber.",
    "The correct action is ask_clarification, not call_function."
  ]
}
```

---

# 11. Training stages

## Stage 0 — Evaluator-first

Trước khi train:

```text
tool registry chạy được
schema validator chạy được
contract checker chạy được
mock executor chạy được
reward scorer chạy được
```

Output bắt buộc:

```text
evaluate(prediction, sample)
→ metrics
→ reward
→ feedback
```

## Stage 1 — Prompt-only baseline

Model:

```text
Qwen3.5-4B hoặc 9B
```

Đánh giá:

```text
seen
unseen
contract
masked names
ask-back
abstention
```

## Stage 2 — Minimal SFT

Data:

```text
10K–20K public general samples optional
+
800–1,500 Telco verified samples
```

Strategy:

```text
1 epoch trước
```

QLoRA config gợi ý:

```yaml
load_in_4bit: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 1.0e-4
num_train_epochs: 1
max_seq_length: 4096
warmup_ratio: 0.03
weight_decay: 0.01
gradient_accumulation_steps: 8
```

Output:

```text
M1 = SFT checkpoint
```

## Stage 3 — Feedback-SDFT baseline

Đây là baseline dễ triển khai, không gọi là SDPO.

Pipeline:

```text
M1 rollout
→ feedback
→ M1 generate corrected output
→ validate correction
→ SFT trên corrected samples
```

Output:

```text
M2 = Feedback-SDFT checkpoint
```

## Stage 4 — Standard SDPO

Rollout:

```text
K = 4 candidates/query
temperature = 0.8–1.0
```

Student context:

```text
original prompt
```

Teacher context:

```text
original prompt
+ environment feedback
+ successful sibling demonstration nếu có
```

Distillation:

```text
teacher logits trên original rollout tokens
student logits trên original rollout tokens
```

Loss:

```text
JSD hoặc reverse-KL
teacher stop-gradient
```

Stability:

```text
EMA teacher
top-k logits
importance ratio clipping
```

Output:

```text
M3 = passive SDPO checkpoint
```

## Stage 5 — VPD-lite

Đây là main research method.

### E-step

Mục tiêu:

```text
cập nhật feedback-conditioned teacher
```

Input:

```text
on-policy rollout batch
scalar outcomes
structured feedback
```

Positive trajectories:

```text
schema valid
contract valid
execution success
```

Negative trajectories:

```text
schema fail
contract fail
unsafe
deprecated
wrong function
wrong args
```

Teacher objective:

```text
tăng likelihood/score cho positive trajectories
giảm likelihood/score cho negative trajectories
giữ teacher gần current student bằng trust region
```

Có thể dùng:

```text
unpaired preference objective
hoặc binary outcome-weighted objective
```

### Dynamic reference

Teacher reference:

```text
current student checkpoint
```

không phải base model cố định.

Trust-region loss:

```text
KL(teacher || stop_gradient(student_current))
```

### M-step

Student distills teacher:

```text
teacher:
  prompt + feedback

student:
  prompt only
```

Loss:

```text
top-k JSD hoặc KL
```

### Update schedule

Gợi ý MVP:

```text
1 E-step mỗi 3–5 M-step
```

### VPD-lite configuration

```yaml
rollouts_per_prompt: 4
temperature: 1.0
teacher_update_frequency: 4
teacher_trust_region_beta: 0.02
distillation_top_k: 20
distillation_divergence: jsd
importance_clip: 2.0
max_generation_tokens: 256
```

---

# 12. Feedback curriculum

## Phase A — Schema feedback

```text
invalid JSON
missing field
wrong type
invalid enum
unknown function
```

## Phase B — Contract feedback

```text
precondition failed
permission missing
unsafe side effect
deprecated tool
```

## Phase C — Multi-tool feedback

```text
wrong order
missing dependency
wrong result propagation
unnecessary call
```

Training order:

```text
A → A+B → A+B+C
```

---

# 13. Bandit router

Chỉ làm sau khi model pipeline ổn.

Arms:

```text
direct_call
schema_contract_reasoning
plan_then_call
self_correct_once
ask_clarification_biased
abstain_safety_biased
```

Context:

```text
retrieval confidence
missing slots
schema complexity
contract complexity
risk level
tool novelty
multi-step flag
estimated cost
```

Main algorithm:

```text
PI-SA-CS-LinUCB
```

Reward:

```text
task reward
- latency
- token cost
- correction cost
```

---

# 14. Experimental matrix

| ID | Method |
|---|---|
| M0 | Prompt-only |
| M1 | Minimal SFT |
| M2 | SFT + masking |
| M3 | Feedback-SDFT |
| M4 | Standard SDPO |
| M5 | VPD-lite |
| M6 | VPD-lite + progressive reward |
| M7 | M6 + contextual bandit |

Nếu thiếu thời gian:

```text
M0
M1
M2
M3
M5
```

là bộ tối thiểu có câu chuyện tốt.

---

# 15. Ablation studies

## Reward ablation

```text
strict only
soft only
progressive soft-to-strict
```

## Masking ablation

```text
no masking
function-name masking
function + parameter masking
```

## Feedback ablation

```text
scalar reward only
language feedback only
structured + language feedback
```

## Distillation ablation

```text
Feedback-SDFT
SDPO
VPD-lite
```

## Contract ablation

```text
schema-only
schema + contract checker
schema + contract + self-distillation
```

## Teacher ablation

```text
passive teacher
adaptive teacher
external teacher
```

---

# 16. Evaluation datasets

```text
eval_seen.jsonl
eval_unseen.jsonl
eval_masked_tools.jsonl
eval_contract.jsonl
eval_missing_slot.jsonl
eval_abstention.jsonl
eval_multi_step.jsonl
eval_parallel.jsonl
eval_schema_changed.jsonl
eval_deprecated.jsonl
eval_expanded_library.jsonl
```

---

# 17. Metrics

## Core function-calling metrics

```text
function_selection_accuracy
argument_key_f1
argument_value_accuracy
schema_validity
execution_success_rate
task_success_rate
```

## Safety/contract metrics

```text
contract_validity
precondition_violation_rate
permission_violation_rate
unsafe_call_rate
deprecated_tool_call_rate
abstention_accuracy
ask_back_accuracy
```

## Generalization metrics

```text
unseen_tool_accuracy
masked_tool_accuracy
renamed_tool_robustness
schema_change_robustness
new_tool_adaptation_gain
```

## Distillation metrics

```text
teacher_student_jsd
correction_success_rate
feedback_utilization_rate
valid_correction_rate
learning_gain_per_rollout
```

## Efficiency metrics

```text
latency
tokens/query
GPU hours
cost per successful task
```

---

# 18. Repo structure

```text
telco-vpd-agent/
├── configs/
│   ├── sft.yaml
│   ├── sdpo.yaml
│   ├── vpd.yaml
│   └── bandit.yaml
├── data/
│   ├── tools.json
│   ├── contracts.json
│   ├── mock_db.sqlite
│   ├── train.jsonl
│   └── eval/
├── src/
│   ├── registry/
│   │   ├── tool_registry.py
│   │   └── contract_registry.py
│   ├── generation/
│   │   ├── template_generator.py
│   │   ├── masking.py
│   │   ├── multi_turn_augmentation.py
│   │   └── negative_sampler.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── reranker.py
│   ├── environment/
│   │   ├── schema_validator.py
│   │   ├── contract_checker.py
│   │   ├── executor.py
│   │   └── feedback_generator.py
│   ├── reward/
│   │   ├── call_matching.py
│   │   ├── soft_reward.py
│   │   ├── strict_reward.py
│   │   └── scheduler.py
│   ├── training/
│   │   ├── train_sft.py
│   │   ├── train_feedback_sdft.py
│   │   ├── train_sdpo.py
│   │   ├── train_vpd.py
│   │   └── teacher_update.py
│   ├── bandit/
│   │   └── pi_sa_cs_linucb.py
│   └── evaluation/
│       ├── evaluator.py
│       └── metrics.py
├── scripts/
│   ├── build_registry.py
│   ├── generate_data.py
│   ├── run_baseline.py
│   ├── run_rollouts.py
│   ├── train_all.sh
│   └── run_eval.py
├── app/
│   └── streamlit_app.py
└── reports/
    ├── results.csv
    └── error_analysis.md
```

---

# 19. Hướng dẫn thực hiện theo thứ tự

## Bước 1 — Tạo registry và contracts

Checklist:

```text
[ ] 30–40 tools
[ ] 10–15 contracts đầy đủ
[ ] active/deprecated/version fields
[ ] examples
[ ] replacement tools
```

## Bước 2 — Tạo mock environment

Checklist:

```text
[ ] subscribers
[ ] plans
[ ] subscriptions
[ ] tickets
[ ] network status
[ ] billing
[ ] tool versions
```

## Bước 3 — Viết evaluator

Phải có API:

```python
result = evaluate_prediction(
    sample=sample,
    prediction=prediction,
    tool_registry=registry,
    state=mock_state,
)
```

Output:

```python
{
    "reward_soft": float,
    "reward_strict": float,
    "reward_total": float,
    "feedback": dict,
    "metrics": dict,
}
```

## Bước 4 — Viết data generator

Checklist:

```text
[ ] templates
[ ] paraphrases
[ ] masking
[ ] multi-turn
[ ] hard negatives
[ ] unseen split
[ ] evolution split
[ ] verification
```

## Bước 5 — Chạy prompt-only

Không train trước khi có baseline.

Lưu:

```text
prediction
reward
feedback
latency
token count
```

## Bước 6 — Train minimal SFT

Train 1 epoch.

Sau đó kiểm tra:

```text
format improvement
seen accuracy
unseen degradation/improvement
contract behavior
```

## Bước 7 — Feedback-SDFT

Dùng để đảm bảo feedback pipeline hoạt động.

## Bước 8 — Standard SDPO

MVP:

```text
500 prompts
K=4
max 256 output tokens
top-k=20
```

## Bước 9 — VPD-lite

Chỉ bắt đầu sau khi SDPO chạy ổn.

Triển khai:

```text
teacher update function
dynamic reference KL
E/M schedule
```

## Bước 10 — Progressive reward

So sánh:

```text
strict from start
vs progressive schedule
```

## Bước 11 — Bandit

Chỉ thêm sau khi các strategies thật sự khác nhau.

## Bước 12 — Demo và error analysis

Demo cần hiển thị:

```text
retrieved tools
schema/contracts
model call
reward breakdown
feedback
teacher/student difference
final execution
```

---

# 20. Kế hoạch 4 tuần

## Tuần 1 — Environment và Data

```text
Day 1–2:
Tool registry + contracts

Day 3:
Mock DB + executor

Day 4:
Schema validator + contract checker

Day 5:
Reward + feedback generator

Day 6–7:
Synthetic dataset + eval sets
```

Deliverable:

```text
verified data
evaluator
prompt-only baseline
```

## Tuần 2 — SFT và Generalization

```text
Day 8:
Public dataset normalization

Day 9:
Telco data merge

Day 10:
Minimal SFT

Day 11:
Seen/unseen evaluation

Day 12:
Tool masking

Day 13:
Multi-turn augmentation

Day 14:
Error analysis
```

Deliverable:

```text
SFT checkpoint
masking ablation
```

## Tuần 3 — Feedback Learning

```text
Day 15:
Feedback-SDFT

Day 16:
Rollout infrastructure

Day 17–18:
Standard SDPO

Day 19:
SDPO evaluation

Day 20–21:
VPD-lite E/M implementation
```

Deliverable:

```text
Feedback-SDFT
SDPO
VPD-lite initial run
```

## Tuần 4 — Final Experiments

```text
Day 22:
Progressive reward

Day 23:
VPD full run

Day 24:
Ablations

Day 25:
Bandit router optional

Day 26:
Dynamic tool evolution evaluation

Day 27:
Streamlit demo

Day 28:
Report + slides
```

---

# 21. Scope control

## Must-have

```text
Tool registry
contracts
verified data
minimal SFT
fine-grained reward
structured feedback
Feedback-SDFT
masking evaluation
seen/unseen evaluation
```

## Strong target

```text
standard SDPO
progressive reward
```

## Main research target

```text
VPD-lite
```

## Optional

```text
bandit
external teacher
full dynamic evolution
```

Nếu VPD không chạy kịp:

```text
không gọi Feedback-SDFT là VPD/SDPO
báo cáo chính xác method đã làm
```

---

# 22. Rủi ro và fallback

## Rủi ro 1 — VPD quá nặng

Fallback:

```text
SDPO subset nhỏ
hoặc Feedback-SDFT + progressive reward
```

## Rủi ro 2 — Teacher không tốt hơn student

Xử lý:

```text
feedback curriculum
successful sibling rollout
EMA/dynamic reference
giảm E-step frequency
```

## Rủi ro 3 — Reward hacking

Xử lý:

```text
strict reward tăng sớm hơn
execution/contract reward nặng hơn format
test adversarial cases
```

## Rủi ro 4 — Model học thuộc tool names

Xử lý:

```text
masking
renaming
unseen split
schema paraphrase
```

## Rủi ro 5 — SFT làm giảm unseen generalization

Xử lý:

```text
1 epoch
mixed public + Telco data
masking
early stopping theo unseen eval
```

---

# 23. Tiêu chí thành công

MVP thành công nếu:

```text
SFT > prompt-only trên seen tools
masking cải thiện unseen/renamed tools
Feedback-SDFT giảm schema/contract errors
```

Strong result nếu:

```text
SDPO > Feedback-SDFT
progressive reward > strict-only
```

Research-level result nếu:

```text
VPD-lite > SDPO
đặc biệt trên unseen tools, contract violations và schema changes
```

---

# 24. Thứ tự ưu tiên cuối cùng

```text
1. Evaluator
2. Verified data
3. Minimal SFT
4. Tool masking
5. Fine-grained progressive reward
6. Feedback-SDFT
7. Standard SDPO
8. VPD-lite
9. Bandit
10. Demo
```

---

# 25. Tên method đề xuất

Tên chính:

> **Contract-Guided Variational Tool Distillation — CVTD**

Tên mô tả:

> A VPD-style adaptive self-distillation method for contract-aware and schema-generalized Telco function calling.

Chỉ nên claim là method mới sau khi thực sự triển khai được E-step teacher adaptation và M-step distribution distillation.

---

# 26. Tài liệu tham khảo chính

1. ToolRL: Reward is All Tool Learning Needs  
   https://arxiv.org/abs/2504.13958

2. Reinforcement Learning via Self-Distillation  
   https://arxiv.org/abs/2601.20802

3. Learning from Language Feedback via Variational Policy Distillation  
   https://arxiv.org/abs/2605.15113

4. Tool-Zero  
   https://arxiv.org/abs/2511.01934

5. ToolACE  
   https://arxiv.org/abs/2409.00920

6. ToolRL code  
   https://github.com/qiancheng0/ToolRL

7. SDPO code  
   https://github.com/lasgroup/SDPO
