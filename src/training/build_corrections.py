"""
Build feedback-correction training pairs from rollouts.

For each wrong prediction:
  1. Construct correction prompt:
       [system] <original system>
       [user]   <original request>
       [asst]   <wrong prediction>        ← what model said
       [user]   Error: {feedback}. Correct it.
  2. Run model inference → get correction candidate
  3. Evaluate correction via evaluator
  4. Keep only corrections where reward == 1.0

Output format (SFT messages): same multi-turn structure, last assistant turn = correction target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction
from src.evaluation.routing import REAL_SOURCE
from src.executor.mock_telco_api import MockTelcoApi
from src.model.output_parser import parse_model_output
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry
from src.reward.feedback_renderer import render_teacher_feedback


CORRECTION_REQUEST = (
    "Your previous response has an error.\n"
    "Feedback: {feedback}\n\n"
    "Please provide the correct action as a JSON object."
)


def build_corrections(
    rollouts_path: Path,
    output_path: Path,
    generator: Any,
    tool_registry: ToolRegistry,
    contract_registry: ContractRegistry,
    mock_api: MockTelcoApi,
) -> dict[str, int]:
    rollouts = [json.loads(l) for l in rollouts_path.open() if l.strip()]
    wrong = [r for r in rollouts if r["reward"] < 1.0]
    print(f"Wrong predictions to correct: {len(wrong)}/{len(rollouts)}")

    corrections = []
    stats = {"attempted": 0, "valid": 0, "invalid": 0}

    for r in wrong:
        lang = "vi" if r.get("source") == REAL_SOURCE else "en"
        feedback_text = _format_feedback(r["feedback"], lang)
        correction_prompt = r["prompt"] + [
            {"role": "assistant", "content": json.dumps(r["prediction"], ensure_ascii=False)},
            {"role": "user", "content": CORRECTION_REQUEST.format(feedback=feedback_text)},
        ]

        raw_correction = generator.generate(correction_prompt)
        correction_pred = parse_model_output(raw_correction)

        # Build a minimal sample dict for the evaluator
        eval_sample = {
            "id": r["id"],
            "split": r["split"],
            "expected_action": r["expected_action"],
            "gold_call": r.get("gold_call"),
            "gold_calls": r.get("gold_calls"),
            "missing_slots": r.get("missing_slots", []),
        }
        result = evaluate_prediction(
            eval_sample, correction_pred, tool_registry, mock_api, contract_registry
        )
        stats["attempted"] += 1

        status = "VALID" if result["reward_total"] == 1.0 else "INVALID"
        print(f"  [{status}] {r['id']}  correction: {json.dumps(correction_pred)[:80]}")

        if result["reward_total"] == 1.0:
            stats["valid"] += 1
            # SFT training sample: full multi-turn conversation ending in correct action
            sft_messages = r["prompt"] + [
                {"role": "assistant", "content": json.dumps(r["prediction"], ensure_ascii=False)},
                {"role": "user", "content": CORRECTION_REQUEST.format(feedback=feedback_text)},
                {"role": "assistant", "content": json.dumps(correction_pred, ensure_ascii=False)},
            ]
            corrections.append({
                "id": f"{r['id']}_correction",
                "source": "feedback_sdft",
                "split": r["split"],
                "original_reward": r["reward"],
                "correction_reward": result["reward_total"],
                "feedback_text": feedback_text,
                "messages": sft_messages,
            })
        else:
            stats["invalid"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for c in corrections:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nCorrections: {stats['valid']} valid / {stats['attempted']} attempted "
          f"({stats['valid']/max(stats['attempted'],1)*100:.1f}% valid correction rate)")
    print(f"Saved → {output_path}")
    return stats


def _format_feedback(feedback: dict[str, Any], lang: str = "vi") -> str:
    # Rich structured rendering so the model sees which arg/value is wrong + the
    # suggested action, not just a flat string. Falls back for empty feedback.
    if feedback.get("errors"):
        return render_teacher_feedback(feedback, lang)
    texts = feedback.get("feedback_text", [])
    if texts:
        return " ".join(texts)
    return "The response was incorrect."


def main() -> None:
    args = _parse_args()

    data_dir = ROOT / "data"
    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    mock_api = MockTelcoApi.from_file(data_dir / "mock_telco_db.json")
    generator = _build_mlx_generator(args)

    build_corrections(
        rollouts_path=Path(args.rollouts),
        output_path=Path(args.output),
        generator=generator,
        tool_registry=tool_registry,
        contract_registry=contract_registry,
        mock_api=mock_api,
    )


def _build_mlx_generator(args: argparse.Namespace) -> Any:
    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        raise SystemExit("mlx-lm not found. Run with python3.11.")

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    class _Gen:
        def generate(self, prompt: list[dict[str, str]]) -> str:
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            return mlx_generate(model, tokenizer, prompt=text,
                                max_tokens=args.max_tokens, verbose=False)

    return _Gen()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build feedback-correction pairs from rollouts.")
    p.add_argument("--rollouts", default=str(ROOT / "data" / "rollouts.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data" / "corrections.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-domain-aug"))
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main()
