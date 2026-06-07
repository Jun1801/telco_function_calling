# Detailed Implementation Plan

This plan maps `plans/contract_guided_vpd_telco_full_plan.md` into concrete repository tasks. The order is strict: do not train a model until the evaluator, verified data, and baseline logging are working.

## 0. Research Targets

The implementation must support these claims:

- Seen-tool accuracy improves after SFT.
- Masking improves unseen/renamed tool generalization.
- Contract-aware reward reduces schema-valid but business-invalid calls.
- Structured language feedback improves over scalar-only reward.
- VPD-lite improves over Feedback-SDFT or standard SDPO on at least one hard split.
- Progressive reward is more stable than strict-only reward.
- Bandit routing improves reward/cost/safety tradeoff when multiple strategies exist.

Minimum method matrix:

```text
M0 prompt-only
M1 minimal SFT
M2 SFT + masking
M3 Feedback-SDFT
M5 VPD-lite
```

Strong target matrix:

```text
M0 prompt-only
M1 minimal SFT
M2 SFT + masking
M3 Feedback-SDFT
M4 standard SDPO
M5 VPD-lite
M6 VPD-lite + progressive reward
M7 M6 + contextual bandit
```

## 1. Evaluator-First Environment

Goal: build a deterministic environment where every function call can be scored before any model training.

Deliverables:

- `data/tools.json`
- `data/tool_contracts.json`
- `data/mock_telco_db.json`
- `src/registry/tool_registry.py`
- `src/registry/contract_registry.py`
- `src/validation/schema_validator.py`
- `src/validation/contract_checker.py`
- `src/executor/mock_telco_api.py`
- `src/evaluation/evaluator.py`
- `src/reward/reward_feedback.py`

Tool registry requirements:

- 30-40 real telco tools.
- 25 seen tools.
- 10 unseen/evolution tools.
- 5 hard-negative or deprecated tools.
- 10-15+ tools with full contracts.
- Each tool must include `domain`, `description`, `parameters`, `required`, `status`, `deprecated`, `risk_level`, and `side_effect`.

Contract requirements:

- preconditions;
- permissions;
- side effects;
- risk level;
- replacement tools for deprecated APIs where relevant.

Evaluator API:

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

Success criteria:

- `python -m pytest` passes.
- `python scripts/run_eval.py` validates every generated JSONL sample.
- Evaluator handles `call_function`, `ask_clarification`, and `abstain`.

## 2. Telco-ToolACE-Mini Data Pipeline

Data generation must follow ToolACE-style generation, not a hand-written sample list.

### 2.1 Tool Self-Evolution Synthesis

Implement in:

- `src/generation/toolace_mini.py`
- future split: `src/generation/masking.py`

Generate tool variants:

- real names;
- function-name masked, e.g. `func_7`;
- function + parameter masked, e.g. `param_1`, `param_2`;
- renamed tools;
- paraphrased schemas;
- schema-changed tools;
- deprecated tools with replacement.

Curriculum ratio:

```text
50% real names
25% function-name masked
15% function + parameter masked
10% renamed/paraphrased schema
```

### 2.2 Interactive Scenario/Dialog Generation

Implement scenario families:

```text
25% single-step valid calls
15% missing-slot / ask-back
10% abstention / permission
20% contract-aware decisions
10% multi-step / dependency
5% parallel calls
10% name/parameter masking
5% schema-change / deprecated tools
```

Required scenarios:

- valid read/write;
- missing required argument;
- invalid enum/type;
- unknown/hallucinated tool;
- deprecated tool;
- permission denied;
- subscriber status contract violation;
- prepaid/postpaid contract violation;
- ask clarification;
- abstain;
- multi-step dependency;
- parallel tool calls;
- unseen tool;
- expanded library with distractors.

### 2.3 Dual-Layer Validation

Every generated candidate must pass verification before writing JSONL.

Validation layers:

1. schema validator;
2. contract checker;
3. mock executor;
4. evaluator reward/feedback check.

Each final sample must include:

```json
{
  "id": "...",
  "source": "telco_toolace_mini",
  "split": "...",
  "scenario": "...",
  "instruction": "...",
  "expected_action": "...",
  "gold_call": {},
  "prediction": {},
  "expected_status": "...",
  "toolace_validation": {}
}
```

Output files:

- `data/train.jsonl`
- `data/eval_seen.jsonl`
- `data/eval_unseen.jsonl`
- `data/eval_masked_tools.jsonl`
- `data/eval_contract.jsonl`
- `data/eval_missing_slot.jsonl`
- `data/eval_abstention.jsonl`
- `data/eval_multi_step.jsonl`
- `data/eval_parallel.jsonl`
- `data/eval_schema_changed.jsonl`
- `data/eval_deprecated.jsonl`
- `data/eval_expanded_library.jsonl`

Commands:

```powershell
python scripts/generate_data.py
python scripts/run_eval.py
```

## 3. Public Warm-Up Dataset Loader

Goal: add optional general function-calling warm-up data without losing Telco-specific verification. The source list is maintained in `plans/dataset_plans.md`; this implementation plan should follow that file instead of redefining dataset policy.

Dataset roles from `plans/dataset_plans.md`:

```text
General SFT:
- Salesforce/xlam-function-calling-60k
- Team-ACE/ToolACE
- Salesforce/APIGen-MT-5k
- xLAM irrelevance 7.5K
- Hermes Function Calling v1 optional

Domain SFT:
- Telco-ToolACE-mini custom data

Self-distillation:
- on-policy Telco rollout + contract feedback

Evaluation only:
- BFCL
- Telco seen/unseen/contract/evolution benchmark
```

Deliverables:

- `src/generation/public_warmup_loader.py`
- `scripts/prepare_public_warmup.py`
- `data/public_warmup_subset.jsonl`

Scope:

- start with 1K-5K samples;
- do not download the full dataset by default;
- normalize to the same training format;
- keep `source` values such as `xlam`, `toolace`, `apigen_mt`, `xlam_irrelevance`, or `hermes_fc`;
- keep separate from Telco eval splits.

Success criteria:

- loader can normalize at least one General SFT source first, preferably Team-ACE/ToolACE or xLAM;
- malformed samples are skipped with a count;
- no public sample enters Telco contract eval.

## 4. Reward Design

Implement fine-grained reward components:

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

Soft reward:

```text
0.10 action
0.15 function/domain
0.15 argument keys
0.15 argument values
0.10 schema partial
0.15 contract partial
0.10 execution progress
0.10 task progress
```

Strict reward:

```text
0.05 format
0.10 action
0.15 function
0.15 argument keys
0.15 argument values
0.10 schema
0.15 contract
0.10 execution
0.05 task
```

Penalty weights:

```text
-0.25 unsafe call
-0.20 hallucinated tool
-0.15 deprecated tool
-0.10 unnecessary call
-0.05 extra tool call
-0.05 normalized token/latency cost
```

Multi-call scoring:

- implement Hungarian matching for predicted calls vs gold calls;
- pair score = name match + argument key score + argument value score;
- do not require exact order for parallel calls.

Deliverables:

- `src/reward/soft_reward.py`
- `src/reward/strict_reward.py`
- `src/reward/call_matching.py`
- `src/reward/scheduler.py`

## 5. Progressive Reward Schedule

Implement:

```text
R_t = (1 - lambda_t) * R_soft + lambda_t * R_strict
```

Schedule:

```text
0-20% steps: 70% soft, 30% strict
20-50% steps: transition
50-100% steps: 10% soft, 90% strict
```

Success criteria:

- compare `strict_only`, `soft_only`, and `progressive`;
- report reward hacking cases;
- progressive should reduce early instability or improve hard-split metrics.

## 6. Prompt-Only Baseline

Do this before SFT.

Deliverables:

- `src/model/prompt_builder.py`
- `src/model/output_parser.py`
- `scripts/run_baseline.py`
- `reports/prompt_only_results.jsonl`
- `reports/error_analysis.md`

Evaluate on:

- seen;
- unseen;
- contract;
- masked tools;
- missing slot;
- abstention;
- schema changed;
- deprecated.

Log:

- prompt;
- tools shown;
- raw model output;
- parsed prediction;
- reward;
- feedback;
- latency;
- token count if available.

## 7. Minimal SFT

Only start after prompt-only results exist.

Config:

```yaml
model: Qwen3.5-4B
fallback_model: Qwen2.5-Coder-7B-Instruct
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

Deliverables:

- `configs/sft.yaml`
- `src/training/train_sft.py`
- `scripts/train_sft.py`
- `reports/sft_results.jsonl`

Success criteria:

- SFT > prompt-only on seen tools;
- no severe drop on unseen/contract splits;
- format validity improves.

## 8. Feedback-SDFT Baseline

This is not SDPO/VPD. Name it accurately.

Pipeline:

```text
M1 rollout
-> evaluator feedback
-> M1 self-corrects with feedback prompt
-> verifier filters corrected outputs
-> SFT on corrected samples
```

Deliverables:

- `scripts/run_rollouts.py`
- `src/training/build_corrections.py`
- `src/training/train_feedback_sdft.py`
- `data/corrections.jsonl`
- `reports/feedback_sdft_results.jsonl`

Success criteria:

- valid correction rate is measured;
- schema/contract error rate decreases vs SFT;
- failed corrections are logged.

## 9. Standard SDPO

MVP setup:

```yaml
rollouts_per_prompt: 4
temperature: 0.8
max_generation_tokens: 256
distillation_top_k: 20
divergence: jsd
teacher_stop_gradient: true
importance_clip: 2.0
```

Teacher context:

```text
original prompt
+ environment feedback
+ successful sibling demonstration if available
```

Student context:

```text
original prompt only
```

Deliverables:

- `configs/sdpo.yaml`
- `src/training/train_sdpo.py`
- `reports/sdpo_results.jsonl`

Success criteria:

- SDPO > Feedback-SDFT, or the report explains the failure;
- teacher/student divergence is logged.

## 10. VPD-Lite Main Method

VPD-lite is the main research target.

E-step:

- update feedback-conditioned teacher;
- increase positive trajectories;
- decrease negative trajectories;
- keep teacher close to current student with dynamic reference KL.

M-step:

- student sees prompt only;
- teacher sees prompt + feedback;
- distill teacher distribution to student.

Config:

```yaml
rollouts_per_prompt: 4
temperature: 1.0
teacher_update_frequency: 4
teacher_trust_region_beta: 0.02
distillation_top_k: 20
distillation_divergence: jsd
importance_clip: 2.0
max_generation_tokens: 256
e_steps_per_cycle: 1
m_steps_per_cycle: 3
```

Deliverables:

- `configs/vpd.yaml`
- `src/training/train_vpd.py`
- `src/training/teacher_update.py`
- `reports/vpd_results.jsonl`

Success criteria:

- VPD-lite > SDPO on at least one hard split: unseen, contract, schema-change, masked;
- if not, report exact failure mode.

## 11. Masking Curriculum

Implement and evaluate:

- no masking;
- function-name masking;
- function + parameter masking;
- renamed/paraphrased schema.

Deliverables:

- `src/generation/masking.py`
- `data/eval_masked_tools.jsonl`
- `reports/masking_ablation.jsonl`

Success criteria:

- masking improves unseen/renamed robustness or reduces memorization of API names.

## 12. Bandit Router

Only implement after strategies differ meaningfully.

Arms:

```text
direct_call
schema_contract_reasoning
plan_then_call
self_correct_once
ask_clarification_biased
abstain_safety_biased
```

Context features:

```text
retrieval_confidence
missing_slots
schema_complexity
contract_complexity
risk_level
tool_novelty
multi_step_flag
estimated_cost
```

Reward:

```text
task_reward - latency - token_cost - correction_cost
```

Deliverables:

- `configs/bandit.yaml`
- `src/bandit/pi_sa_cs_linucb.py`
- `reports/bandit_results.jsonl`

Success criteria:

- better reward/cost tradeoff;
- unsafe call rate does not increase.

## 13. Ablation Plan

Reward ablation:

- strict only;
- soft only;
- progressive.

Masking ablation:

- no masking;
- function-name masking;
- function + parameter masking.

Feedback ablation:

- scalar reward only;
- language feedback only;
- structured + language feedback.

Distillation ablation:

- Feedback-SDFT;
- SDPO;
- VPD-lite.

Contract ablation:

- schema-only;
- schema + contract checker;
- schema + contract + self-distillation.

Teacher ablation:

- passive teacher;
- adaptive teacher;
- external teacher.

## 14. Metrics

Core:

- `function_selection_accuracy`
- `argument_key_f1`
- `argument_value_accuracy`
- `schema_validity`
- `execution_success_rate`
- `task_success_rate`

Safety/contract:

- `contract_validity`
- `precondition_violation_rate`
- `permission_violation_rate`
- `unsafe_call_rate`
- `deprecated_tool_call_rate`
- `abstention_accuracy`
- `ask_back_accuracy`

Generalization:

- `unseen_tool_accuracy`
- `masked_tool_accuracy`
- `renamed_tool_robustness`
- `schema_change_robustness`
- `new_tool_adaptation_gain`

Distillation:

- `teacher_student_jsd`
- `correction_success_rate`
- `feedback_utilization_rate`
- `valid_correction_rate`
- `learning_gain_per_rollout`

Efficiency:

- latency;
- tokens/query;
- GPU hours;
- cost per successful task.

## 15. Four-Week Execution Schedule

Week 1: Environment and Data

```text
Day 1-2: registry + contracts
Day 3: mock DB + executor
Day 4: schema validator + contract checker
Day 5: reward + feedback generator
Day 6-7: Telco-ToolACE-mini + eval sets
```

Week 2: SFT and Generalization

```text
Day 8: public ToolACE loader
Day 9: Telco/public data merge
Day 10: prompt-only baseline
Day 11: minimal SFT
Day 12: seen/unseen evaluation
Day 13: masking curriculum
Day 14: error analysis
```

Week 3: Feedback Learning

```text
Day 15: Feedback-SDFT
Day 16: rollout infrastructure
Day 17-18: standard SDPO
Day 19: SDPO evaluation
Day 20-21: VPD-lite E/M implementation
```

Week 4: Final Experiments

```text
Day 22: progressive reward
Day 23: VPD full run
Day 24: ablations
Day 25: bandit router optional
Day 26: dynamic tool evolution evaluation
Day 27: Streamlit demo
Day 28: report + slides
```

## 16. Fallback Rules

If VPD-lite is too heavy:

- run SDPO on a small subset;
- or report Feedback-SDFT + progressive reward honestly.

If teacher is not better than student:

- use feedback curriculum;
- add successful sibling rollout;
- lower teacher update frequency;
- strengthen trust-region KL.

If reward hacking appears:

- increase strict reward earlier;
- weight execution/contract higher than format;
- add adversarial eval cases.

If SFT hurts unseen tools:

- reduce epochs;
- increase masking;
- mix public ToolACE data;
- early stop on unseen eval.

## 17. Current Priority

The next implementation steps are:

1. Finish Telco-ToolACE-mini as a true framework module, not a static sample list.
2. Add public warm-up loader following `plans/dataset_plans.md`.
3. Normalize train format for SFT.
4. Implement prompt builder and prompt-only baseline.
5. Only then start minimal SFT.
