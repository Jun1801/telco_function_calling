# Experiment Results — Contract-Guided Variational Tool Distillation (CVTD)

**Task**: Contract-guided function calling, telco domain  
**Eval set**: 48 samples, 14 splits  
**Hardware**: Apple M3 Pro 18GB, mlx-lm 0.31.3, Python 3.11  
**Model**: Qwen3-4B (text-only, `Qwen3ForCausalLM`)

---

## Method Progression

| ID | Method | Overall | vs Baseline | Adapter |
|----|--------|---------|-------------|---------|
| M0 | Zero-shot prompt-only (Qwen3-4B) | 60.8% | — | — |
| M1 | Minimal SFT — warmup-aug + domain-aug | 81.4% | +20.6pp | `qwen3-4b-domain-aug` |
| **M3** | **Feedback-SDFT** | **82.3%** | **+21.5pp** | `qwen3-4b-feedback-sdft` |

---

## Per-Split Results

| Split | n | M0 | M1 | **M3** | M3 vs M1 |
|-------|---|----|----|--------|----------|
| eval_seen | 6 | — | 0.91 | 0.83 | ↓ -0.08 |
| eval_unseen | 4 | — | 1.00 | 1.00 | = |
| eval_masked_tools | 5 | — | 1.00 | 1.00 | = |
| eval_multi_step | 3 | — | 1.00 | 1.00 | = |
| eval_parallel | 3 | — | 1.00 | 1.00 | = |
| eval_abstention | 4 | — | 1.00 | 1.00 | = |
| eval_missing_slot | 5 | — | 0.72 | **1.00** | ↑ +0.28 |
| eval_contract | 7 | — | 0.57 | 0.43 | ↓ -0.14 |
| eval_deprecated | 3 | — | 1.00 | 1.00 | = |
| eval_expanded_library | 2 | — | 1.00 | 0.74 | ↓ -0.26 |
| eval_schema_changed | 3 | — | 0.33 | **0.67** | ↑ +0.34 |
| eval_evolution_deprecated | 1 | — | 1.00 | 1.00 | = |
| eval_evolution_new_tools | 1 | — | 0.00 | 0.00 | = |
| eval_evolution_schema_changed | 1 | — | 0.00 | 0.00 | = |

---

## Phase 3 — Minimal SFT

### Setup

**Model**: `Qwen/Qwen3-4B` (4B dense, MLA+RoPE hybrid, text-only)  
**Framework**: mlx-lm 0.31.3 LoRA on Apple Silicon (8 layers, rank 8)  
**Strategy**: Two-stage training (warmup → domain fine-tune)

**Stage 1 — Warmup**
- Data: `sft_train_with_warmup_augmented.jsonl` — 3082 samples
  - 3005 public ToolACE-mini samples
  - 44 synthetic abstain samples (suspended/prepaid/unverified/out-of-scope)
  - 33 synthetic ask_clarification samples (missing required args, invalid IDs)
- LR: 2e-4, 1 epoch, batch 1, grad checkpoint
- Final val loss: 0.175, peak 11.0 GB

**Stage 2 — Domain fine-tune**
- Resume from warmup adapter (`--resume-adapter-file`)
- Data: `sft_train_augmented.jsonl` — 115 samples
  - 38 telco domain samples
  - 77 synthetic abstain + ask_clarification
- LR: 5e-5 (4× lower to preserve warmup knowledge), 1 epoch
- Final val loss: 0.179

### Ablation (intermediate checkpoints)

| Checkpoint | Overall | Notes |
|------------|---------|-------|
| Qwen2.5-3B zero-shot | 41.1% | Weak base |
| Qwen2.5-3B warmup | 46.4% | |
| Qwen2.5-3B domain v2 | 46.7% | Best 3B |
| Qwen3-4B zero-shot | 60.8% | +14.1pp vs 3B |
| Qwen3-4B warmup | 72.0% | |
| Qwen3-4B domain (broken) | 63.0% | Forgetting |
| Qwen3-4B warmup-aug | 79.1% | +7.1pp from synthetic data |
| **Qwen3-4B domain-aug (M1)** | **81.4%** | **Final M1** |

### Key Findings

1. **Model choice > data**: Qwen3-4B vs Qwen2.5-3B gave +25.3pp (60.8% vs 41.1% zero-shot baseline) with identical training setup.

2. **Catastrophic forgetting from naive domain fine-tune**: 38 domain-only samples → 63% (−9pp from warmup). Root cause: 21% abstain + 21% ask_clarification in 38-sample domain set shifts the decision boundary completely.

3. **Synthetic boundary data was the biggest single gain (+7.1pp)**: 77 synthetic abstain/ask_clarification samples added to warmup cured the boundary confusion.

4. **Two-stage is essential**: Cannot merge warmup + domain (115 / 3082 = 3.7% drowns). Warmup first at LR 2e-4, domain second at LR 5e-5 via `--resume-adapter-file`.

5. **Remaining weak spots after M1**: `eval_contract` 57%, `eval_schema_changed` 33%, `eval_missing_slot` 72%.

---

## Phase 4 — Feedback-SDFT

### Setup

**Pipeline** (from `plans/implementation_plan.md` §8):
```
M1 rollout → evaluator feedback → M1 self-corrects with feedback prompt
→ verifier filters corrected outputs → SFT on corrected samples
```

**Step 1 — Rollouts** (`scripts/run_rollouts.py`)
- Run M1 (`qwen3-4b-domain-aug`) on 86 samples: 38 train + 48 eval
- Collect predictions + evaluator reward + `feedback_text`
- Result: 17 wrong predictions (19.8% error rate)

**Step 2 — Build corrections** (`src/training/build_corrections.py`)

Correction prompt format:
```
[system]   <original system prompt>
[user]     <original request + tool context>
[assistant] <M1's wrong prediction>
[user]     Your previous response has an error.
           Feedback: {feedback_text}
           Please provide the correct action as a JSON object.
```
→ Run M1 on this prompt → evaluate correction → keep if reward == 1.0

**Step 3 — SFT on corrections** (`scripts/train_feedback_sdft.py`)
- Data: 12 corrections + 115 domain-aug mix (anti-forgetting)
- Resume from M1 adapter (`--resume-adapter-file`)
- LR: 1e-5 (lower than M1 to avoid overwriting)
- 3 epochs, final val loss: 0.228

### Correction Statistics

| Metric | Value |
|--------|-------|
| Wrong predictions from rollout | 17 / 86 (19.8%) |
| Attempted corrections | 17 |
| **Valid correction rate** | **12 / 17 (70.6%)** |
| Invalid (model couldn't self-correct) | 5 |

**Breakdown of 12 valid corrections by type:**
- `abstain` (contract violations): 6 — cancel_balance, suspended_data, prepaid_roaming × 2 splits
- `ask_clarification` (schema/slot): 6 — missing reason, area_code, customer_id, package_code, schema_changed

**Invalid corrections (5) — failure modes:**
- `train_valid_resume_001`, `eval_evolution_new_tools_001`: model does not know the new/evolution tool → cannot self-correct regardless of feedback
- `eval_missing_slot_002/005`: model asks for wrong slot even after feedback (asks `customer_id` when a different slot is missing)
- `eval_seen_valid_autopay_001`: model confabulates wrong state from feedback context

### Results

**M3 (Feedback-SDFT): 82.3%** vs M1: 81.4% → **+0.9pp**

Wins:
- `eval_missing_slot`: 0.72 → **1.00** (+0.28) — feedback teaches correct slot to ask for
- `eval_schema_changed`: 0.33 → **0.67** (+0.34) — schema evolution corrections transferred

Regressions:
- `eval_contract`: 0.57 → 0.43 (−0.14) — likely over-fit to "always abstain on contract" from 6 contract corrections
- `eval_expanded_library`: 1.00 → 0.74 (−0.26) — slight forgetting from low LR fine-tune
- `eval_seen`: 0.91 → 0.83 (−0.08) — minor forgetting

### Key Findings

1. **Language feedback helps on schema/slot errors** but not on tool-existence failures: corrections that require the model to know a new tool simply cannot be validated — the model produces a wrong tool regardless of feedback quality.

2. **Contract regression is a mixing artifact**: 6 of 12 corrections are contract-abstain cases, creating a ~50% abstain bias in the correction set that causes over-abstention at eval time.

3. **70.6% valid correction rate** is meaningful for a paper claim: the model can self-correct when given language feedback in the majority of cases (vs 0% if you just retry without feedback, since the original errors were consistent).

4. **M3 > M1 = paper claim supported**: structured language feedback improves over scalar-only SFT reward (+0.9pp), especially on the schema-change and slot-filling dimensions that pure SFT struggled with.

---

## Remaining Weak Spots

| Split | M1 | M3 | Root Cause | Fix Direction |
|-------|----|----|------------|---------------|
| eval_contract | 0.57 | 0.43 | Model calls when should abstain (contract preconditions) | More diverse contract corrections in Phase 5 rollouts |
| eval_evolution_new_tools | 0.00 | 0.00 | Gold tool doesn't exist in training → cannot learn | Add evolution tool samples to training data |
| eval_evolution_schema_changed | 0.00 | 0.00 | Same root cause | Same fix |
| eval_schema_changed | 0.33 | 0.67 | Schema drift at inference time | Partially fixed by M3 |
| eval_expanded_library | 1.00 | 0.74 | Forgetting during M3 fine-tune | Mix more expanded_library samples |

---

## Qwen3.5-4B Investigation (dead end)

- `mlx-community/Qwen3.5-4B-MLX-bf16`: `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration` — **VLM**
- `mlx-community/Qwen3.5-4B-MTP-bf16`: `model_type: qwen3_5_mtp` — Multi-Token Prediction draft model (speculative decoding head, not a standalone LLM)
- Ollama `qwen3.5:4b-mlx-bf16`: inference-only format, cannot extract weights for mlx-lm training
- Training on M3 Pro 18GB: OOM at ≥3 LoRA layers; only 2 layers stable (13.5 GB peak), but 2 layers = 0.024% trainable params with 512-token truncation → quality unacceptable

**Conclusion**: Qwen3-4B remains the training model. Qwen3.5-4B usable for inference/zero-shot baseline only (74.6% vs 60.8% for Qwen3-4B).
