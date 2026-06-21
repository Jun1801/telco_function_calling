"""
Run M1 model on train + eval data and save all predictions with reward/feedback.
Output: data/rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import build_sample_prompt, evaluate_sample, load_real_assets
from src.model.output_parser import parse_model_output
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def main() -> None:
    args = _parse_args()
    data_dir = ROOT / "data"

    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    real_assets = load_real_assets(data_dir)
    generator = _build_mlx_generator(args)

    samples = _load_samples(data_dir, args.splits)
    print(f"Loaded {len(samples)} samples from {args.splits}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(samples):
        prompt = build_sample_prompt(sample, tool_registry, contract_registry, real_assets)
        raw_output = generator.generate(sample, prompt)
        prediction = parse_model_output(raw_output)
        result = evaluate_sample(
            sample, prediction, tool_registry, contract_registry, data_dir, real_assets
        )

        record = {
            "id": sample["id"],
            "split": sample.get("split", "train"),
            "scenario": sample.get("scenario", ""),
            "expected_action": sample.get("expected_action", ""),
            "gold_call": sample.get("gold_call"),
            "gold_calls": sample.get("gold_calls") or sample.get("gold_steps"),
            "missing_slots": sample.get("missing_slots", []),
            "prompt": prompt,
            "raw_output": raw_output,
            "prediction": prediction,
            "reward": result["reward_total"],
            "feedback": result["feedback"],
            "metrics": result["metrics"],
        }
        records.append(record)

        status = "OK" if record["reward"] == 1.0 else "FAIL"
        print(f"[{i+1}/{len(samples)}] {status} reward={record['reward']:.2f}  {sample['id']}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_wrong = sum(1 for r in records if r["reward"] < 1.0)
    print(f"\nSaved {len(records)} rollouts → {out_path}")
    print(f"Wrong predictions: {n_wrong}/{len(records)} ({n_wrong/len(records)*100:.1f}%)")
    print(f"Correction candidates: {n_wrong}")


def _load_samples(data_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    samples = []
    if "train" in splits:
        path = data_dir / "train.jsonl"
        if path.exists():
            samples.extend(json.loads(l) for l in path.open() if l.strip())
    if "eval" in splits:  # synthetic eval only; real tools use run_sdpo_rollouts
        for path in sorted(data_dir.glob("eval_*.jsonl")):
            if path.name.startswith("eval_real_"):
                continue
            samples.extend(json.loads(l) for l in path.open() if l.strip())
    return samples


def _build_mlx_generator(args: argparse.Namespace) -> Any:
    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        raise SystemExit("mlx-lm not found. Run with python3.11.")

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    class _Gen:
        def generate(self, sample: dict[str, Any], prompt: list[dict[str, str]]) -> str:
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            return mlx_generate(model, tokenizer, prompt=text,
                                max_tokens=args.max_tokens, verbose=False)

    return _Gen()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M1 model rollouts on train+eval data.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-domain-aug"),
                   help="Path to M1 adapter directory.")
    p.add_argument("--splits", nargs="+", default=["train", "eval"],
                   choices=["train", "eval"],
                   help="Which data splits to run rollouts on.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--output", default=str(ROOT / "data" / "rollouts.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    main()
