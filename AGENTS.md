# Repository Guidelines

## Source of Truth

All implementation work must follow `plans/contract_guided_vpd_telco_full_plan.md`. Use `plans/implementation_plan.md` as the task-level execution map and `plans/dataset_plans.md` for public warm-up dataset policy. Do not introduce model training, routing, demo work, or new abstractions that skip the plan order.

Required order:

```text
Evaluator -> Verified ToolACE-style data -> Prompt-only baseline -> Minimal SFT
-> Feedback-SDFT -> SDPO -> VPD-lite -> Progressive reward -> Bandit -> Demo
```

## Project Structure & Module Organization

Use `src/` for Python modules, `data/` for tool registries, contracts, mock state, and JSONL datasets, `scripts/` for repeatable commands, `tests/` for automated tests, `configs/` for training/router configs, `reports/` for metrics and analysis, and `app/` for the Streamlit demo.

Current core areas:

- `src/registry/`: tool and contract registries.
- `src/validation/`: schema validator and contract checker.
- `src/executor/`: mock telco API.
- `src/evaluation/`: evaluator API and metrics.
- `src/reward/`: reward and feedback logic.
- `src/generation/`: Telco-ToolACE-mini and data generation.

## Development Commands

Run from the repository root:

```powershell
python scripts/generate_data.py
python scripts/run_eval.py
python -m pytest
```

`generate_data.py` must generate verified JSONL datasets. `run_eval.py` must validate expected actions/statuses through the evaluator. `pytest` must pass before moving to the next phase.

## Data Generation Rules

Data generation must be ToolACE-style, not a static hand-written sample list. Follow these stages:

1. Tool Self-Evolution Synthesis: real names, masked names, masked parameters, renamed schemas, deprecated tools.
2. Interactive Scenario/Dialog Generation: valid calls, missing-slot, abstention, contract violations, hard negatives, multi-step, parallel, unseen/evolution tools.
3. Dual-Layer Validation: schema validator, contract checker, mock executor, reward/feedback evaluator.

Every generated sample must include verification metadata and must pass `scripts/run_eval.py`.

Public warm-up data must follow `plans/dataset_plans.md`: xLAM, ToolACE, APIGen-MT, xLAM irrelevance, and optional Hermes are General SFT sources. Telco-ToolACE-mini is Domain SFT. BFCL and Telco benchmarks are evaluation-only.

## Coding Style & Naming Conventions

Use Python 3.11+ style with 4-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Prefer typed function signatures and small modules aligned to plan components. Keep JSON keys stable and explicit: `tool_name`, `arguments`, `expected_action`, `expected_status`, `toolace_validation`, `metrics`, and `feedback`.

## Testing Guidelines

Add focused `pytest` tests for every behavior change. Prioritize:

- schema validation;
- contract checking;
- mock execution;
- reward and feedback;
- evaluator metrics;
- ToolACE-mini generation;
- split leakage and verification metadata.

Bug fixes and logic changes should include a regression test. Generated cache files must not be committed.

## Scope Control

Do not start SFT before prompt-only baseline exists. Do not call Feedback-SDFT “SDPO” or “VPD”. Do not implement VPD-lite before rollout and Feedback-SDFT are working. Do not add a real telco backend, PPO/GRPO, neural reward model, or complex multi-agent framework for the MVP.

## Security & Configuration

Never commit API keys, model credentials, real customer data, or private telco records. Mock data in `data/` must remain synthetic. Keep optional tracking credentials such as W&B or MLflow in environment variables or ignored local files.
