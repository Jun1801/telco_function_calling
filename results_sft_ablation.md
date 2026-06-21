# Phase 3 — Minimal SFT: Results Summary

**Task**: Contract-guided function calling for telco domain  
**Eval set**: 48 samples across 14 splits  
**Hardware**: Apple M3 Pro 18GB, mlx-lm 0.29.1 (LoRA, 8 layers)

---

## Overall Progression

| # | Model | Overall | Notes |
|---|-------|---------|-------|
| 0 | Qwen3-4B zero-shot (baseline) | 60.8% | No training |
| 1 | Qwen2.5-3B zero-shot | 41.1% | Weaker base |
| 2 | Qwen2.5-3B warmup (3005 samples) | 46.4% | +5.3pp |
| 3 | Qwen2.5-3B domain fine-tune | 43.7% | Forgetting, no gain |
| 4 | Qwen2.5-3B domain v2 (1 epoch, LR 5e-5) | 46.7% | Best for 3B |
| 5 | **Qwen3-4B warmup** (3005 samples) | **72.0%** | Big jump, switched model |
| 6 | Qwen3-4B domain fine-tune (38 samples) | 63.0% | **Regression** — boundary confused |
| 7 | **Qwen3-4B warmup-aug** (+77 synthetic) | **79.1%** | +7.1pp, fixed abstain/clarify |
| 8 | **Qwen3-4B domain-aug** (115 samples) | **81.4%** | **Best result** |

Total gain over baseline: **+20.6pp**

---

## Per-Split Breakdown (key checkpoints)

| Split | n | Baseline | Warmup | Domain (broken) | Warmup-aug | **Domain-aug** |
|-------|---|----------|--------|-----------------|------------|----------------|
| eval_seen | 6 | – | 0.74 | 0.74 | 0.91 | **0.91** |
| eval_unseen | 4 | – | 1.00 | 1.00 | 1.00 | **1.00** |
| eval_masked_tools | 5 | – | 1.00 | 1.00 | 1.00 | **1.00** |
| eval_multi_step | 3 | – | 1.00 | 1.00 | 1.00 | **1.00** |
| eval_parallel | 3 | – | 1.00 | 1.00 | 1.00 | **1.00** |
| eval_abstention | 4 | – | 1.00 | **0.00** | 1.00 | **1.00** |
| eval_missing_slot | 5 | – | 0.72 | 0.86 | 1.00 | 0.72 |
| eval_contract | 7 | – | 0.29 | 0.00 | 0.43 | **0.57** |
| eval_deprecated | 3 | – | 0.67 | 0.33 | 1.00 | **1.00** |
| eval_expanded_library | 2 | – | 0.74 | 0.74 | 0.74 | **1.00** |
| eval_schema_changed | 3 | – | 0.33 | 0.67 | 0.00 | 0.33 |
| eval_evolution_deprecated | 1 | – | 0.00 | 0.00 | 0.00 | **1.00** |
| eval_evolution_new_tools | 1 | – | 1.00 | 1.00 | 1.00 | 0.00 |
| eval_evolution_schema_changed | 1 | – | 0.00 | 1.00 | 0.00 | 0.00 |

---

## Training Setup (Final: qwen3-4b-domain-aug)

**Stage 1 — Warmup**
- Model: `Qwen/Qwen3-4B` (text-only, `Qwen3ForCausalLM`)
- Data: `sft_train_with_warmup_augmented.jsonl` — 3082 samples
  - 3005 public ToolACE-mini samples (call_function heavy)
  - 44 synthetic abstain samples
  - 33 synthetic ask_clarification samples (extra slot-missing scenarios)
- Epochs: 1, LR: 2e-4, batch: 1, LoRA layers: 8
- Final val loss: 0.175, Peak mem: 11.0 GB

**Stage 2 — Domain Fine-tune**
- Resume from: warmup adapter (`--resume-adapter-file`)
- Data: `sft_train_augmented.jsonl` — 115 samples
  - 38 telco domain samples
  - 44 synthetic abstain + 33 synthetic ask_clarification
- Epochs: 1, LR: 5e-5 (4x lower to avoid overwriting warmup)
- Final val loss: 0.179

---

## Key Findings

### 1. Model choice matters more than data
Switching Qwen2.5-3B → Qwen3-4B gave +25.6pp (46.7% → 72.0%) with identical training data.  
Qwen3-4B uses MLA + RoPE hybrid, significantly better instruction following out of the box.

### 2. Catastrophic forgetting from naive domain fine-tune
Training only on 38 domain samples from scratch destroyed generalization:
- `eval_abstention`: 1.00 → **0.00** (model always tries to call a function)
- `eval_contract`: 0.29 → **0.00**
- Overall: 72.0% → **63.0%** (−9pp regression)

Root cause: 38 domain samples = 21% abstain, 21% ask_clarification — completely different distribution from real world, shifts the decision boundary.

### 3. Synthetic data fixed the boundary problem (+7.1pp)
Generated 77 synthetic samples (44 abstain + 33 ask_clarification) covering:
- Suspended/prepaid subscribers, unverified customers, missing prerequisites
- 18 tools with missing required args, invalid customer IDs, invalid arg types

Adding these to warmup data: warmup-aug 79.1% vs warmup 72.0%.

### 4. Two-stage training is necessary
Cannot merge warmup + domain into one pass:
- Domain samples get drowned (115 / 3082 = 3.7%)
- Warmup first gives the "foundation", domain fine-tune at low LR adds specifics without overwriting

### 5. Remaining weak spots
| Split | Score | Reason |
|-------|-------|--------|
| eval_schema_changed | 0.33 | Schema evolution requires reinterpretation, not just pattern matching |
| eval_evolution_schema_changed | 0.00 | Only 1 sample, but same root cause |
| eval_evolution_new_tools | 0.00 | Model calls old tool instead of new equivalent |
| eval_missing_slot | 0.72 | Domain fine-tune shifted abstain/ask_clarify boundary slightly |
| eval_contract | 0.57 | Contract preconditions need more diverse training signal |

---

## Qwen3.5-4B Investigation (dead end)

Checked `mlx-community/Qwen3.5-4B-MLX-bf16`:
- `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`
- Has `video_preprocessor_config.json`, `preprocessor_config.json` → **VLM**, not text-only
- mlx-lm 0.29.1 does not support `qwen3_5` architecture for LoRA training
- Cannot upgrade to mlx-lm 0.30+ because it requires `transformers>=5.0.0` (max available on Python 3.9 is 4.57.6)

**Decision**: Stay with Qwen3-4B as primary model for Phase 4+.

---

## Next Steps (Phase 4 — Feedback-SDFT)

Priority targets based on current gaps:
1. `eval_schema_changed` / `eval_evolution_*` — teach model to handle schema drift via language feedback
2. `eval_contract` (0.57) — richer contract violation explanations as training signal
3. `eval_missing_slot` regression — add more diverse ask_clarification samples to fine-tune data

Phase 4 approach: instead of scalar reward (1.0/0.0), use the textual `feedback` field from the evaluator as a training signal — train model to predict correct action *given* the feedback on its wrong prediction.
