"""Generate K rollouts per training prompt using HF Transformers (GPU).

Used by both M4 (SDPO) and M5 (VPD-lite). Outputs rollout groups with
per-rollout reward/feedback so teacher and student can be trained offline.

python scripts/run_sdpo_rollouts_hf.py \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --input data/sft_train_real.jsonl \
  --output data/sdpo_rollouts_m4.jsonl \
  --k 4 --temperature 0.8
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
    tools_path = data_dir / "tools.json"
    contracts_path = data_dir / "tool_contracts.json"
    tool_registry = ToolRegistry.from_file(tools_path) if tools_path.exists() else ToolRegistry([])
    contract_registry = ContractRegistry.from_file(contracts_path) if contracts_path.exists() else ContractRegistry([])
    real_assets = load_real_assets(data_dir)

    print(f"Loading model {args.model} + adapter {args.adapter} ...")
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
    print(f"Loaded {len(samples)} training samples. Generating K={args.k} rollouts each ...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    skipped = kept = 0

    if args.resume and out_path.exists():
        print(f"Resuming from existing output: {out_path}")
        with out_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    processed_ids.add(record["id"])
                    kept += 1
                except Exception:
                    pass
        print(f"Found {len(processed_ids)} already processed samples.")

    open_mode = "a" if (args.resume and out_path.exists()) else "w"
    with out_path.open(open_mode, encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            if sample["id"] in processed_ids:
                continue
            messages = build_sample_prompt(sample, tool_registry, contract_registry, real_assets)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            enc = tokenizer(prompt_text, return_tensors="pt").to(base.device)
            prompt_len = enc["input_ids"].shape[1]

            rollouts = []
            for _ in range(args.k):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                prediction = parse_model_output(raw)
                result = evaluate_sample(
                    sample, prediction, tool_registry, contract_registry, data_dir, real_assets
                )
                rollouts.append({
                    "raw_output": raw,
                    "prediction": prediction,
                    "reward": result["reward_total"],
                    "feedback": result["feedback"],
                    "metrics": result["metrics"],
                })

            rewards = [r["reward"] for r in rollouts]
            avg_reward = sum(rewards) / len(rewards)

            # Skip groups where all rollouts fail — no learning signal for JSD
            if avg_reward == 0.0:
                skipped += 1
                continue

            best_idx = max(range(args.k), key=lambda i: rewards[i])
            record = {
                "id": sample["id"],
                "source": sample.get("source"),
                "expected_action": sample.get("expected_action"),
                "prompt_messages": messages,
                "rollouts": rollouts,
                "best_idx": best_idx,
                "avg_reward": avg_reward,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(samples)}] kept={kept} skipped={skipped}", flush=True)

    print(f"Done. Groups kept: {kept}, skipped (all-fail): {skipped} → {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--input", default=str(ROOT / "data" / "sft_train_real.jsonl"))
    p.add_argument("--output", default=str(ROOT / "data" / "sdpo_rollouts.jsonl"))
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--success-threshold", type=float, default=1.0)
    p.add_argument("--resume", action="store_true", help="Resume from existing output if present")
    return p.parse_args()


if __name__ == "__main__":
    main()
