"""Build feedback-correction training pairs from rollouts (HF Transformers port).

For each failed rollout:
  1. Build correction prompt: original messages + wrong answer + "Error: {feedback}. Correct it."
  2. Generate correction using M1b model
  3. Evaluate correction
  4. Keep only reward == 1.0

Output: data/corrections_m3.jsonl — multi-turn messages compatible with train_sft.py

python src/training/build_corrections_hf.py \
  --rollouts data/rollouts_m3.jsonl \
  --output data/corrections_m3.jsonl \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import evaluate_sample, load_real_assets
from src.model.output_parser import parse_model_output
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry
from src.reward.feedback_renderer import render_teacher_feedback

CORRECTION_REQUEST = (
    "Your previous response has an error.\n"
    "Feedback: {feedback}\n\n"
    "Please provide the correct action as a JSON object."
)


def _format_feedback(feedback: dict, lang: str = "vi") -> str:
    if feedback.get("errors"):
        return render_teacher_feedback(feedback, lang)
    texts = feedback.get("feedback_text", [])
    return " ".join(texts) if texts else "The response was incorrect."


def main() -> None:
    args = _parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_dir = ROOT / "data"
    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    real_assets = load_real_assets(data_dir)

    print(f"Loading {args.model} + {args.adapter} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rollouts = [json.loads(l) for l in Path(args.rollouts).open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rollouts)} failed rollouts to correct.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"attempted": 0, "valid": 0}

    with out_path.open("w", encoding="utf-8") as fout:
        for r in rollouts:
            lang = "vi" if r.get("source") == "real_tool_xlsx" else "en"
            feedback_text = _format_feedback(r["feedback"], lang)

            correction_messages = r["prompt"] + [
                {"role": "assistant", "content": json.dumps(r["prediction"], ensure_ascii=False)},
                {"role": "user", "content": CORRECTION_REQUEST.format(feedback=feedback_text)},
            ]

            prompt_text = tokenizer.apply_chat_template(
                correction_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            enc = tokenizer(prompt_text, return_tensors="pt").to(base.device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            raw_correction = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            correction_pred = parse_model_output(raw_correction)

            eval_sample = {
                "id": r["id"],
                "source": r.get("source"),
                "split": r.get("split", "train"),
                "expected_action": r.get("expected_action"),
                "gold_call": r.get("gold_call"),
                "gold_calls": r.get("gold_calls"),
                "missing_slots": r.get("missing_slots", []),
            }
            result = evaluate_sample(
                eval_sample, correction_pred, tool_registry, contract_registry, data_dir, real_assets
            )

            stats["attempted"] += 1
            status = "VALID" if result["reward_total"] == 1.0 else "INVALID"
            print(f"  [{status}] {r['id']}  {json.dumps(correction_pred)[:80]}")

            if result["reward_total"] == 1.0:
                stats["valid"] += 1
                sft_messages = correction_messages + [
                    {"role": "assistant", "content": json.dumps(correction_pred, ensure_ascii=False)},
                ]
                record = {
                    "id": f"{r['id']}_correction",
                    "source": "feedback_sdft",
                    "split": r.get("split", "train"),
                    "original_reward": r["reward"],
                    "correction_reward": 1.0,
                    "feedback_text": feedback_text,
                    "messages": sft_messages,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    rate = stats["valid"] / max(stats["attempted"], 1) * 100
    print(f"\nCorrections: {stats['valid']}/{stats['attempted']} valid ({rate:.1f}%)")
    print(f"Saved → {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default=str(ROOT / "data" / "rollouts_m3.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data" / "corrections_m3.jsonl"))
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    main()
