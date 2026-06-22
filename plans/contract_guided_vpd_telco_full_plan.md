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
4. ~~Contract-aware reward có giảm function call đúng schema nhưng sai nghiệp vụ không?~~ **(BỎ — real KPI read-only, không có precondition/permission; "nghiệp vụ" thu về reference-code: mã hợp lệ trong catalogue)**
5. Rich language feedback có cải thiện policy tốt hơn scalar reward đơn thuần không?
6. VPD có tốt hơn SFT, Feedback-SDFT và passive SDPO không?
7. Progressive soft-to-strict reward có giúp training ổn định hơn strict reward từ đầu không?
8. Model có thích nghi được khi API mới được thêm, schema thay đổi hoặc tool bị deprecated không?
9. Contextual bandit có chọn được strategy phù hợp với từng query để tối ưu accuracy, safety và cost không?

---

# 2. Đóng góp kỹ thuật dự kiến

## Contribution 1 — Schema + Reference-aware Telco Function Calling Environment
Môi trường kiểm chứng (real):

```text
JSON schema (enum/type/required/pattern)
reference-code ∈ catalogue (location/kpi/unit/station)
gold-diff (function + arguments so gold dựng từ catalogue)
```

Một function call (real) đúng khi:

```text
schema valid  AND  reference-code valid  AND  khớp gold (function + args)
```

(Synthetic legacy — môi trường giao dịch: schema ∧ contract ∧ execution ∧ task; preconditions/postconditions/permissions/side_effects/risk/mock execution.)

## Contribution 2 — Real-KPI Data Pipeline (construct-then-paraphrase)

Sinh dữ liệu (real, §7):

```text
seen tools · unseen tools
renamed/masked tools
missing-slot / ask-clarification
multi-step (ReAct) · parallel calls
abstain (out-of-scope)
```

Gold dựng từ catalogue (không sai); verify bằng **Dual-Layer Verification**:

```text
schema validator
reference-code ∈ catalogue        (thay contract checker)
city↔code + date + dedupe + optional semantic LLM-judge
```

> (Synthetic-legacy thêm: contract violations · schema-change · deprecated · contract checker · mock executor.)

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
→ strict JSON/AST reward (real: + reference-code; synthetic-legacy: + contract + execution)
```

Reward được decomposed theo (real):

```text
action
function name
argument keys
argument values
schema
reference-code (mã ∈ catalogue)
```

(Synthetic-legacy có thêm: `contract · execution · task · safety` — KHÔNG dùng cho real read-only.)

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
Real KPI Tool Registry (26 hàm, read-only) + reference catalogues
        ↓
Real data generator (construct-then-paraphrase) + Dual-Layer Verification
        ↓
Seen / Unseen / Masked / Missing / Multi(ReAct) / Parallel / Abstain benchmark
        ↓
Qwen3-4B Function Calling Policy
        ↓
Schema Validator + Reference-code check   (KHÔNG contract/executor)
        ↓
Scalar Reward + Structured Feedback (gold-diff, reveal_gold off)
        ↓
SFT → Feedback-SDFT → SDPO → VPD-lite
        ↓
(synthetic-legacy / chưa làm: Contract Checker · Mock Executor · BM25+BGE retrieval · Bandit Router · Demo)
```

---

# 4. Tech stack chốt

| Thành phần | Công nghệ |
|---|---|
| Language | Python 3.11 |
| Main model | Qwen3-4B hoặc Qwen3-8B |
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
Qwen3 - 4B
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

```text
26 hàm KPI thật:
  18 seen   (huấn luyện + eval seen)
  8 unseen  (chỉ eval, zero-shot đọc schema):
    radio_kpi · thong_ke_cntt · top_tram_max · top_tram_min ·
    top_cell_max · top_cell_min · top_sub_attached_max · top_sub_attached_min
```

Catalogue đóng (gold dựng từ đây — xem §7.2):

```text
location_code : 48   (VNM/khu vực/tỉnh/huyện)
kpi_code      : 12
unit_code     : 9
station_code  : 144  (synthetic EHB000NN, phủ đủ 48 location)
```

## 6.2. Tool schema (real, read-only)

Enum được **inject** cho `location_code` / `kpi_code` / `unit_code` / `station_code` (đóng theo catalogue).
`object_code` **KHÔNG** enum (chấp nhận mã location HOẶC station — tập mở). Contract = stub rỗng (read-only).

```json
{
  "name": "vung_phu_province",
  "domain": "kpi_reporting",
  "description": "Tra cứu hiện trạng vùng phủ sóng của Viettel theo loại công nghệ 2G/3G/4G/5G.",
  "side_effect": "read",
  "split": "seen",
  "parameters": {
    "type": "object",
    "properties": {
      "location_code": {
        "type": "string",
        "enum": ["VNM", "KV1", "KV2", "KV3", "HNI", "HCM", "..."],
        "description": "Mã vị trí (huyện/tỉnh/khu vực/toàn quốc) — đóng theo catalogue 48 mã."
      },
      "tech_type": {
        "type": "string",
        "enum": ["2G", "3G", "4G", "5G"],
        "description": "Loại công nghệ cần tra cứu."
      }
    },
    "required": ["location_code", "tech_type"]
  },
  "status": "active",
  "replacement_tool": null
}
```

Contract (read-only → không precondition/side-effect):

```json
{"tool_name": "vung_phu_province", "preconditions": [], "side_effects": [], "permissions": []}
```

---

# 7. Data strategy — Real KPI (construct-then-paraphrase, ToolACE-adapted)

## 7.1. Public data warm-up

Public function-calling data (chuẩn hóa qua `scripts/prepare_public_warmup.py`, chỉ normalize — không auto-tải):

```text
ToolACE · Hermes-Function-Calling · APIGen-MT · xLAM
→ data/public_warmup_*.jsonl  (đã ở dạng messages)
```

Mục tiêu = **format priming** (đọc schema, chọn hàm, parallel, multi-turn, no-tool). Trộn vào SFT M1
một-stage gộp (warmup + real domain). Hiện có ~3K mẫu normalized; dùng subset.

## 7.2. Real custom data — construct-then-paraphrase

**Gold dựng từ catalogue đóng (§6.1) → gold KHÔNG THỂ SAI; LLM chỉ viết câu hỏi tiếng Việt.**
Dual-Layer Verification (rule: schema + code∈catalogue + date + city↔code word-boundary + length; optional
semantic LLM-judge) + dedupe Jaccard 0.85 + arg-cap 3. Canonical: `data/real_data/outputs-2/` (Kaggle vLLM).

Train **2613** mẫu (`data/sft_train_real.jsonl`):

```text
single_step_valid : 828
missing_slot       : 430   (ask_clarification)
multi_step (ReAct) : 520   (R1+R2, xem §8.2)
masking            : 350   (func_X, xem §7.3)
abstain            : 289
parallel           : 196
```

Eval **~1195** mẫu (7 split `data/eval_real_*.jsonl`):

```text
seen 250 · unseen 150 · masked 170 · missing_slot 234 · multi_step 156 · parallel 80 · abstain 155
```

(Audit độc lập `scripts/audit_real_data.py`: 0 defect mọi family.)

## 7.3. Masking curriculum (real)

Mẫu masking mang **`masked_tool`** = schema `func_X` nhúng trong prompt (gold gọi `func_X`); tool thật bị
shadow + tắt distractor fallback (xem `routing.build_sample_prompt`). Đã có sẵn trong train (350) + eval masked (170).

### Stage 1 — Real names
```json
{"name": "vung_phu_province", "parameters": {"location_code": {}, "tech_type": {}}}
```

### Stage 2 — Mask function name
```json
{"name": "func_3", "description": "Tra cứu hiện trạng vùng phủ sóng theo công nghệ.",
 "parameters": {"location_code": {}, "tech_type": {}}}
```

### Stage 3 — Mask function + parameters
```json
{"name": "func_3", "parameters": {"param_1": {"description": "Mã vị trí"},
 "param_2": {"description": "Loại công nghệ 2G/3G/4G/5G"}}}
```

Tỉ lệ gợi ý:

```text
50% real names
25% function-name masked
15% function + parameter masked
10% renamed/paraphrased schema
```

---

# 8. Multi-step & augmentation (real, read-only)

## 8.1. Missing-slot → ask_clarification (single-turn)

Bỏ một slot bắt buộc khỏi câu hỏi; gold = `ask_clarification(asked_slots=[...])`. Cặp `from_date`/`to_date`
luôn drop CÙNG nhau (một cụm thời gian). Ví dụ:

```text
User: Cho tôi vùng phủ sóng 5G ở Hà Nội.     # thiếu… (đủ slot ⇒ gọi luôn); nếu thiếu location/thời gian:
Gold: {"action": "ask_clarification", "asked_slots": ["from_date", "to_date"]}
```

## 8.2. ReAct multi-step (Cách B)

Phụ thuộc dữ liệu được phân rã thành **các record single-turn** (tái dùng loss single-turn, không cần multi-turn masking):

```text
R1: {"action":"call_function","call":{"tool_name":"regional_station_info","arguments":{...}}}
    → observation tổng hợp (src/executor/kpi_mock.py) trả MÃ TRẠM thật, ví dụ EHB00118
R2: {"action":"call_function","call":{"tool_name":"radio_traffic",
     "arguments":{"object_code":"EHB00118", ...}}}   # dùng mã trạm THẬT từ R1
```

## 8.3. Parallel calls

Nhiều hàm độc lập trong một lượt; gold = `call_functions(calls=[...])`; chấm set-by-name (xem §9, §10).

## 8.4. (Synthetic legacy) User-correction & backend-challenge

```text
Không áp dụng cho real read-only (không có giao dịch/đổi trạng thái/suspend).
Giữ làm reference cho domain synthetic.
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

Feedback giữ **hai phần** (machine-readable + human-readable), shape ổn định để teacher/VPD/Feedback-SDFT dùng chung:

```json
{
  "machine_status": "ok | schema_invalid | wrong_call | wrong_action | format_error",
  "errors": [
    {"type": "...", "code": "...", "path": "arguments.<key>",
     "actual": "...", "expected": "(chỉ khi reveal_gold=True)",
     "message": "...", "suggested_action": "..."}
  ],
  "feedback_text": ["dòng người đọc 1", "..."]
}
```

## 10.1. Ba nguồn lỗi (ưu tiên giảm dần)

```text
(0) Schema layer (SchemaValidator) — ưu tiên cao nhất, sai schema thì chỉ báo schema:
    invalid_enum · invalid_type · missing_arg · unknown_arg · pattern_mismatch · deprecated_tool · unknown_tool

(A) Gold-diff (so call schema-valid với gold):
    wrong_function · wrong_argument_value · missing_argument · extra_argument
    unnecessary_call  (call parallel không khớp gold → 1 tín hiệu, KHÔNG spam per-arg)
    → reveal_gold=False MẶC ĐỊNH: chỉ báo sai ở đâu, KHÔNG lộ giá trị gold
      (teacher phải tự suy). reveal_gold=True chỉ cho ablation.

(B) Reference-code (location_code/kpi_code/unit_code không có enum):
    invalid_code — mã ngoài catalogue (danh mục hợp lệ lấy từ catalogue, không phải gold → không leak)
```

## 10.2. Feedback theo action

```text
call_function      → (0)/(A)/(B)
call_functions     → set-by-name; call thừa → unnecessary_call
ask_clarification  → missing_slot_not_handled (kèm missing_slots) + chấm recall slot
abstain            → unsafe_or_forbidden_action nếu không abstain
parse lỗi          → format_error, reward 0 (KHÔNG tính là abstain hợp lệ)
```

`suggested_action` mỗi lỗi ∈ `{ask_clarification, abstain, fix_arguments, call_function, call_functions, fix_format}`.

## 10.3. Renderer cho teacher

`src/reward/feedback_renderer.render_teacher_feedback(feedback, lang)` dựng text giàu **từ structured codes**
(header theo machine_status + mỗi lỗi 1 dòng + gợi ý), **song ngữ `vi` (real, mặc định) / `en` (synthetic)**.
Đã **wire** vào teacher context của SDPO (`train_sdpo`) và correction (`build_corrections`) — chọn lang theo `record.source`.

## 10.4. Ví dụ (real, reveal_gold=False)

```json
{
  "machine_status": "wrong_call",
  "errors": [
    {"type": "wrong_call", "code": "wrong_argument_value", "path": "arguments.location_code",
     "actual": "HCM", "message": "Wrong value for location_code", "suggested_action": "fix_arguments"}
  ],
  "feedback_text": ["Wrong value for location_code"]
}
```

Render (vi): `Phản hồi gọi sai (hàm hoặc tham số):` → `• Sai giá trị tham số \`arguments.location_code\` (bạn dùng 'HCM'). → Gợi ý: sửa lại tham số rồi gọi hàm.`
(reveal_gold=True mới thêm `"expected": "HNI"` và “Đúng phải là 'HNI'”.)

---

# 11. Training stages

## Stage 0 — Evaluator-first

Trước khi train (real):

```text
tool registry chạy được
schema validator chạy được
reference-code check chạy được      (thay contract checker — read-only, không mock executor)
reward scorer + structured feedback chạy được
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
Qwen3 -4B hoặc 8B
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

Positive trajectories (real):

```text
schema valid
reference-code valid (mã ∈ catalogue)
khớp gold (function + arguments)
```

Negative trajectories (real):

```text
schema fail (invalid_enum/type/missing_arg/…)
invalid_code (mã ngoài catalogue)
wrong function
wrong args
sai action (nên ask_clarification / abstain)
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
invalid_enum · invalid_type · missing_arg · unknown_arg · pattern_mismatch · unknown_tool · deprecated_tool
(+ parse_error → format_error)
```

## Phase B — Business / reference feedback (real)
```text
invalid_code            (mã ngoài catalogue location/kpi/unit)
wrong_argument_value    (giá trị sai so gold; reveal_gold=False)
wrong_function          (chọn sai hàm)
missing_slot_not_handled / unsafe_or_forbidden_action  (sai action: nên ask/abstain)
— synthetic legacy: precondition_failed · permission_denied · deprecated/unsafe side effect
```

## Phase C — Multi-call feedback
```text
unnecessary_call        (call parallel thừa, không khớp gold nào)
missing_argument / extra_argument
ReAct dependency: R2 dùng sai mã trạm từ observation của R1
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
schema_reference_reasoning   (real: thay schema_contract_reasoning)
plan_then_call
self_correct_once
ask_clarification_biased
abstain_safety_biased
```

Context:

```text
missing slots
schema complexity
reference-code complexity   (real: thay contract complexity)
tool novelty
multi-step flag
estimated cost
(synthetic-legacy: retrieval confidence · risk level)
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

## ~~Contract ablation~~ → Reference-code ablation (real)

> Contract checker BỎ (read-only). Thay bằng ablation reference-code:

```text
schema-only
schema + reference-code check
schema + reference-code + self-distillation
```

## Teacher ablation

```text
passive teacher
adaptive teacher
external teacher
```

---

# 16. Evaluation datasets

**Real (7 split đang dùng, ~1195 mẫu):**

```text
eval_real_seen.jsonl      (250)
eval_real_unseen.jsonl    (150)
eval_real_masked.jsonl    (170)
eval_real_missing_slot.jsonl (234)
eval_real_multi_step.jsonl   (156)
eval_real_parallel.jsonl     (80)
eval_real_abstain.jsonl      (155)
```

(Synthetic-legacy 14 split: eval_seen/unseen/masked_tools/**contract**/missing_slot/abstention/multi_step/parallel/schema_changed/deprecated/evolution_*/expanded_library.)

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

## Safety metrics (real)

> BỎ `contract_validity` / `precondition_violation_rate` / `permission_violation_rate` / `unsafe_call_rate` /
> `deprecated_tool_call_rate` (read-only, không contract). Real chỉ giữ:

```text
abstention_accuracy   (abstain đúng khi ngoài scope)
ask_back_accuracy     (hỏi đủ slot khi thiếu)
reference_code_validity   (mã ∈ catalogue — thay cho contract_validity)
```

(Synthetic-legacy: contract_validity · precondition/permission_violation_rate · unsafe/deprecated_call_rate.)

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
Feedback-SDFT giảm schema + reference-code (mã sai) errors    (real: KHÔNG còn contract errors)
```

Strong result nếu:

```text
SDPO > Feedback-SDFT
progressive reward > strict-only
```

Research-level result nếu:

```text
VPD-lite > SDPO
đặc biệt trên unseen tools, masked tools và sai mã/sai action (real)
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

> A VPD-style adaptive self-distillation method for **schema- and reference-grounded**, schema-generalized Telco function calling.

> **Lưu ý (real):** đã BỎ contract-aware (read-only). "Contract-Guided" giờ hiểu theo nghĩa **được dẫn dắt bởi
> schema + reference-code (catalogue) + gold-diff**, KHÔNG phải business contract giao dịch. Nếu muốn tên chính xác
> hơn có thể đổi thành **Schema-/Reference-Guided VTD** — cần bạn xác nhận trước khi đổi title toàn dự án.

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
