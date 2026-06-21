# CLAUDE.md

Guidance for Claude Code working in this repo. Keep this file lean — detailed specs live in `plans/contract_guided_vpd_telco_full_plan.md` and the `configs/` YAMLs.

## Project Purpose

**Contract-Guided Variational Policy Distillation for Generalizable Telco Function Calling (CVTD).**
Combines ideas from ToolRL, Tool-Zero, SDPO, VPD, ToolACE.

Research questions:
1. Call the correct telco API on **seen** tools after SFT?
2. Call **unseen** tools by reading schema alone at inference?
3. Does **tool-name masking** reduce API-name memorization?
4. Does **contract-aware reward** cut schema-valid but business-invalid calls?
5. Does **rich language feedback** beat scalar reward alone?
6. Does **VPD-lite** beat SFT, Feedback-SDFT, passive SDPO?
7. Is **progressive soft→strict reward** more stable than strict-only?
8. Can the model adapt when APIs change / schemas evolve / tools deprecate?
9. Does **contextual bandit routing** improve accuracy/safety/cost?

## Current Status — Real-Tool Pivot

Pivoted from the synthetic 82-tool set to **26 real Viettel KPI functions** (`data/Function.xlsx`), Vietnamese, read-only.
- **Canonical clean data: `data/real_data/outputs-2/`** (Kaggle vLLM). `data/*.jsonl` = that mirror + local ReAct decomposition of `multi_step`; both kept in sync and audited clean.
- Dataset: **2613 train / ~1195 eval**. Gold args are **constructed from closed catalogues** (gold cannot be wrong; the LLM only writes the query).
- **Independent audit (`scripts/audit_real_data.py`) reports 0 defects** across all families. Tests: **86 pass**; `run_eval` gold self-check **1243/1243**.
- Generation runs on **Kaggle** (vLLM/transformers) — the M3 Pro throttles/hangs on large local MLX runs.
- No M-model retrained on real data yet (M0–M7 scores below are synthetic).

## Data

**Real KPI (active):**
- `data/real_tools.json` — 26 tools (18 seen / 8 unseen); enums injected for `location_code`/`kpi_code`/`unit_code`/`station_code` (NOT `object_code` — it accepts location OR station)
- `data/real_tool_contracts.json` (read-only stubs), `data/real_reference_codes.json` (loc 48 / kpi 12 / unit 9), `data/real_station_catalogue.json` (144 stations `EHB000NN` → all 48 locations)
- `data/sft_train_real.jsonl` — 2613 (single 828, missing_slot 430, multi_step-react 520, masking 350, abstain 289, parallel 196)
- `data/eval_real_{seen 250, unseen 150, masked 170, missing_slot 234, multi_step 156, parallel 80, abstain 155}.jsonl`

**Synthetic (M0–M7 history):** `data/tools.json` (82), `data/tool_contracts.json`, `data/mock_telco_db.json`, 14 `eval_*.jsonl` splits, `data/sft_*.jsonl`.

## Output Format

```json
{"action": "call_function", "call": {"tool_name": "...", "arguments": {...}}}
{"action": "call_functions", "calls": [{"tool_name": "...", "arguments": {...}}, ...]}
{"action": "ask_clarification", "asked_slots": ["..."]}
{"action": "abstain", "reason": "..."}
```

## Commands

```bash
python -m pytest                                   # tests (use python3 if 3.11 lacks pytest)
python scripts/run_eval.py                         # gold self-check (synthetic + real)

# Real-tool pipeline
python scripts/parse_function_xlsx.py              # Function.xlsx → registries + catalogues
python3.11 scripts/generate_real_data.py --scale 1.0   # generate+verify+split (local MLX; prod data via Kaggle)
python3.11 scripts/build_multistep_react.py        # multi_step → ReAct R1/R2 (in-place)
python3 scripts/audit_real_data.py data            # independent cleanliness audit (also: data/real_data/outputs-2)
python3 scripts/repair_missing_slot_dates.py data data/real_data/outputs-2  # deterministic bug-9 repair

# Eval an adapter (real splits route to the schema-only evaluator)
python3.11 scripts/run_baseline.py --backend mlx --model Qwen/Qwen3-4B --adapter <dir> --splits real

# Two-stage SFT (Python 3.11, mlx-lm 0.31+; --mask-prompt on by default)
python3.11 scripts/train_sft_mlx.py --model Qwen/Qwen3-4B --train-file <train> --eval-file <eval> \
    --epochs 1 --learning-rate 2e-4 --output-dir outputs/sft_mlx/<warmup>           # warmup
python3.11 scripts/train_sft_mlx.py ... --learning-rate 5e-5 --resume-from <warmup> # domain fine-tune
```

## Architecture

- **Evaluation** — `src/evaluation/routing.py` dispatches by `source`: real KPI samples (`source=="real_tool_xlsx"`) → `real_evaluator.py` (schema-only, no executor/contracts; gold-diff + reference-code feedback, masked embedded schema, multi-call set matching); else the 4-layer `evaluator.py` (schema → contract → mock execution → task success). Both return `reward_soft`, `reward_strict`, `feedback`, metrics. `parse_error` always scores 0 (never a valid abstain). Shared entry points: `build_sample_prompt` / `evaluate_sample`.
- **Validation** — `src/validation/{schema_validator,contract_checker}.py`.
- **Real-data generation** — `real_arg_sampler.py` (valid gold args), `real_tool_llm_generator.py` (`gen_*` per family, MLX), `real_tool_verifier.py` (rule + optional semantic + dedupe), orchestrated by `scripts/generate_real_data.py`. Multi-step = **ReAct (Cách B)**: single-turn records, R2 uses the REAL station code from `src/executor/kpi_mock.py`. Kaggle mirror: `kaggle/generate_real_data_vllm.py` (keep in parity).
- **Model** — `prompt_builder.py` (injects tool schemas + contracts; `extra_tools` injects a masked `func_X` schema-only and shadows the real tool; synthetic fallbacks are synthetic-registry-only). `output_parser.py` strips `<think>`, returns `parse_error` on failure.
- **Registries** — `tool_registry.py`, `contract_registry.py` (load from the JSON files above).
- **Training** — `src/training/{build_corrections,train_feedback_sdft,train_sdpo,train_vpd,teacher_update}.py`.

## Training Stack

Apple Silicon M3 Pro 18GB · Python 3.11 · `Qwen/Qwen3-4B` (text-only `Qwen3ForCausalLM`) · `mlx-lm` 0.31.3 + LoRA (8 layers, rank 8). `Qwen3.5-4B` is a VLM — not trainable on 18GB. Two-stage SFT: warmup (LR 2e-4, format + abstain/clarify boundaries) → domain fine-tune (LR 5e-5, `--resume-from`).

## Experimental Matrix (synthetic dataset)

| ID | Method | Status | Score |
|----|--------|--------|-------|
| M0 | Prompt-only baseline (Qwen3-4B zero-shot) | ✅ | 60.8% |
| M1 | Minimal SFT (warmup-aug + domain-aug) | ✅ | **83.5%** |
| M2 | SFT + masking curriculum | ⏳ | — |
| M3 | Feedback-SDFT | ⏳ | — |
| M4 | Standard SDPO | ✅ | 47.9% |
| M5 | VPD-lite | ✅ | **80.9%** |
| M6 | VPD-lite + progressive reward | ⏳ | — |
| M7 | M6 + contextual bandit | ⏳ | — |

Result so far: **M1 > M5 > M4**. M4 collapsed from JSD distillation on a too-narrow (seen-only) rollout set. M5 needs diverse rollout coverage to beat SFT.

## Methods (terse)

- **Feedback-SDFT (M3)** — supervised correction baseline (NOT SDPO/VPD): M1 rollout → evaluator feedback → ask model to correct → keep only reward==1.0 corrections → SFT. Files: `run_rollouts.py`, `build_corrections.py`, `train_feedback_sdft.py`.
- **SDPO (M4)** — K=4 rollouts, top-k JSD distillation; teacher sees prompt+feedback+sibling demo, student sees prompt only; `success_threshold` skips all-failed groups. Config: `configs/sdpo.yaml`.
- **VPD-lite (M5/M6, main method)** — E-step updates a feedback-conditioned teacher (trust-region KL vs current student); M-step distills teacher→student (teacher sees feedback, student doesn't); top-k JSD. Config: `configs/vpd.yaml`.
- **Reward** — fine-grained components (format/action/function/args/schema/contract/execution/task). *Design (Tier-2, not yet wired): penalties (unsafe/hallucinated/deprecated/unnecessary/extra/cost), progressive soft→strict schedule, Hungarian §9.2 matching. Current parallel scoring uses order-independent set-by-name matching.*

## Tier-1 fixes (done, regression-tested in `tests/test_tier1_bug_fixes.py`)

parse_error → reward 0; deprecated keeps `schema_validity=1`; masking injects `func_X` + shadows real tool; unmatched parallel → one `unnecessary_call`; `--mask-prompt` on; `train_sdpo` skips all-failed groups; `run_rollouts` routes real; **bug-9 missing_slot date-pair** (repaired in committed data); kaggle↔local parity.

## Scope Constraints (from AGENTS.md)

- Don't call Feedback-SDFT "SDPO" or "VPD" — name methods accurately.
- Don't start a later phase before its prerequisites exist; follow `plans/implementation_plan.md`.
- Every generated sample must pass dual-layer validation (schema + contract + executor + evaluator, or the real audit).
- No real telco backend / PPO/GRPO / neural reward model / heavy multi-agent framework for the MVP.
- Never commit API keys, credentials, or real customer data. Use `uv` (not bare pip). Use `python3.11` for MLX. Use the Shadow File technique for large file rewrites.
