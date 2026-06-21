"""
Phase 5 — SDPO rollout generation.

Generates K rollouts per training prompt, evaluates each, and identifies
best sibling per prompt group. Output feeds into src/training/train_sdpo.py.

Output: data/sdpo_rollouts.jsonl
Each record:
  id, split, instruction, prompt (student messages), expected_action, gold_call,
  rollouts: [{response, prediction, reward, feedback, log_probs_per_token}]
  best_rollout_idx: index of highest-reward rollout (or -1 if all failed)
  avg_reward: average reward across K rollouts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction
from src.evaluation.real_evaluator import evaluate_real_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.model.output_parser import parse_model_output
from src.model.prompt_builder import build_prompt_messages
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry

REAL_SOURCE = "real_tool_xlsx"  # marks samples from data/real_tools.json (read-only KPI tools)


def main() -> None:
    args = _parse_args()
    data_dir = ROOT / "data"

    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    mock_api = MockTelcoApi.from_file(data_dir / "mock_telco_db.json")

    # Real read-only KPI tools (separate registry + schema-only evaluator + code tables).
    real_registry = real_contracts = real_refs = None
    if (data_dir / "real_tools.json").exists():
        real_registry = ToolRegistry.from_file(data_dir / "real_tools.json")
        real_contracts = ContractRegistry.from_file(data_dir / "real_tool_contracts.json")
        ref_path = data_dir / "real_reference_codes.json"
        real_refs = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None

    model, tokenizer = _load_model(args)

    samples = _load_samples(data_dir, args.splits)
    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} prompts from {args.splits} — generating K={args.k} rollouts each")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(samples):
        is_real = sample.get("source") == REAL_SOURCE
        if is_real and real_registry is None:
            raise SystemExit("Real sample found but data/real_tools.json is missing.")

        if is_real:
            extra = [sample["masked_tool"]] if sample.get("masked_tool") else None
            prompt = build_prompt_messages(sample, real_registry, real_contracts, extra_tools=extra)
        else:
            prompt = build_prompt_messages(sample, tool_registry, contract_registry)
        rollouts = []

        for k in range(args.k):
            raw_output = _generate(model, tokenizer, prompt, args.max_tokens, args.temperature)
            prediction = parse_model_output(raw_output)
            if is_real:
                result = evaluate_real_prediction(sample, prediction, real_registry, references=real_refs)
            else:
                result = evaluate_prediction(sample, prediction, tool_registry, mock_api, contract_registry)

            rollouts.append({
                "k": k,
                "raw_output": raw_output,
                "prediction": prediction,
                "reward": result["reward_total"],
                "feedback": result["feedback"],
            })

        rewards = [r["reward"] for r in rollouts]
        avg_reward = sum(rewards) / len(rewards)

        # best sibling = highest reward rollout (for teacher context)
        best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
        best_reward = rewards[best_idx]

        record = {
            "id": sample["id"],
            "split": sample.get("split", "train"),
            "instruction": sample.get("instruction", ""),
            "expected_action": sample.get("expected_action", ""),
            "gold_call": sample.get("gold_call"),
            "gold_calls": sample.get("gold_calls") or sample.get("gold_steps"),
            "missing_slots": sample.get("missing_slots", []),
            "prompt": prompt,
            "rollouts": rollouts,
            "best_rollout_idx": best_idx if best_reward >= args.success_threshold else -1,
            "avg_reward": avg_reward,
        }
        records.append(record)

        status_parts = [f"r{j}={rewards[j]:.2f}" for j in range(args.k)]
        print(f"[{i+1}/{len(samples)}] {sample['id']}  avg={avg_reward:.2f}  [{', '.join(status_parts)}]")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_any_success = sum(1 for r in records if r["best_rollout_idx"] >= 0)
    n_mixed = sum(1 for r in records if 0 < r["avg_reward"] < 1.0)
    print(f"\nSaved {len(records)} prompt groups → {out_path}")
    print(f"Prompts with ≥1 success (has teacher demo): {n_any_success}/{len(records)}")
    print(f"Mixed prompts (0 < avg_reward < 1.0, useful for distillation): {n_mixed}/{len(records)}")


def _generate(model: Any, tokenizer: Any, prompt: list[dict], max_tokens: int, temperature: float) -> str:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)
    return mlx_generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, verbose=False)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from mlx_lm import load
    except ImportError:
        raise SystemExit("mlx-lm not found. Run with python3.11.")
    print(f"Loading model {args.model} with adapter {args.adapter} ...")
    return load(args.model, adapter_path=args.adapter)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def _load_samples(data_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if "train" in splits:
        samples += _read_jsonl(data_dir / "train.jsonl")
    if "eval" in splits:  # synthetic eval splits only (exclude eval_real_*)
        for path in sorted(data_dir.glob("eval_*.jsonl")):
            if path.name.startswith("eval_real_"):
                continue
            samples += _read_jsonl(path)
    if "real_train" in splits:
        samples += _read_jsonl(data_dir / "sft_train_real.jsonl")
    if "real_eval" in splits:
        samples += _read_jsonl(data_dir / "eval_real_seen.jsonl")
        samples += _read_jsonl(data_dir / "eval_real_unseen.jsonl")
    return samples


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5: generate K rollouts per training prompt.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-feedback-sdft"),
                   help="Teacher adapter (M3) for rollout generation.")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "eval", "real_train", "real_eval"])
    p.add_argument("--k", type=int, default=4, help="Rollouts per prompt (K).")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="Minimum reward for a rollout to be used as teacher demo.")
    p.add_argument("--limit", type=int, default=0, help="Cap number of prompts (smoke testing).")
    p.add_argument("--output", default=str(ROOT / "data" / "sdpo_rollouts.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    main()
