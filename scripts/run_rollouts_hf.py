"""Single-pass greedy rollouts from M1b (temperature=0) for M3 Feedback-SDFT.

Saves only failed predictions (reward < 1.0). These become input to
build_corrections_hf.py which generates correction pairs.

python scripts/run_rollouts_hf.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --input data/sft_train_real.jsonl \
  --output data/rollouts_m3.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import build_sample_prompt, evaluate_sample, load_real_assets
from src.model.output_parser import parse_model_output
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


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

    samples = [json.loads(l) for l in Path(args.input).open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(samples)} samples. Running greedy rollouts ...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            messages = build_sample_prompt(sample, tool_registry, contract_registry, real_assets)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
            raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            prediction = parse_model_output(raw)
            result = evaluate_sample(
                sample, prediction, tool_registry, contract_registry, data_dir, real_assets
            )

            reward = result["reward_total"]
            if reward >= 1.0:
                ok += 1
            else:
                failed += 1
                record = {
                    "id": sample["id"],
                    "source": sample.get("source"),
                    "split": sample.get("split", "train"),
                    "expected_action": sample.get("expected_action"),
                    "gold_call": sample.get("gold_call"),
                    "gold_calls": sample.get("gold_calls") or sample.get("gold_steps"),
                    "missing_slots": sample.get("missing_slots", []),
                    "prompt": messages,
                    "raw_output": raw,
                    "prediction": prediction,
                    "reward": reward,
                    "feedback": result["feedback"],
                    "metrics": result["metrics"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(samples)}] ok={ok} failed={failed}", flush=True)

    total = ok + failed
    print(f"\nDone. ok={ok} failed={failed} ({failed/max(total,1)*100:.1f}% failure rate)")
    print(f"Failures (correction candidates) → {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--input", default=str(ROOT / "data" / "sft_train_real.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data" / "rollouts_m3.jsonl"))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
