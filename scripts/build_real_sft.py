"""Build the M1 SFT training file for the REAL KPI domain (Colab A100 / transformers).

`data/sft_train_real.jsonl` holds RAW samples; `scripts/train_sft.py` (trl SFTTrainer) needs
chat `messages`. We format each real sample with the SAME prompt path used at eval
(`routing.build_sample_prompt`) so train/inference prompts match exactly, append the gold
action as the assistant turn, then mix in a public-warmup subset (already in messages form,
for general function-calling format priming) — a single-stage merged SFT per master-plan §Phase1.

Outputs (gitignored, derived):
  data/sft_train_real_with_warmup.jsonl   real-domain (messages) + warmup subset, shuffled
  data/sft_eval_real_messages.jsonl       small held-out from real train (SFT val-loss only)

  python3 scripts/build_real_sft.py --warmup-n 10000 --eval-n 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import build_sample_prompt, load_real_assets
from src.generation.sft_formatter import _assistant_payload
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry

DATA = ROOT / "data"


def _format_real(sample: dict, tool_reg, contract_reg, real_assets) -> dict:
    messages = build_sample_prompt(sample, tool_reg, contract_reg, real_assets)  # [system, user]
    payload = _assistant_payload(sample)  # asserts no <from_step_1> leak
    messages = messages + [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]
    return {"id": sample["id"], "source": sample.get("source"),
            "expected_action": sample["expected_action"], "messages": messages}


def _project_warmup(row: dict) -> dict:
    return {"id": row["id"], "source": row.get("source", "warmup"),
            "expected_action": row.get("expected_action", "call_function"), "messages": row["messages"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-n", type=int, default=10000, help="public-warmup samples to mix in (0 = none)")
    ap.add_argument("--eval-n", type=int, default=256, help="held-out real-domain samples for SFT val loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-file", default=str(DATA / "public_warmup_all.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tools_path = DATA / "tools.json"
    contracts_path = DATA / "tool_contracts.json"
    tool_reg = ToolRegistry.from_file(tools_path) if tools_path.exists() else ToolRegistry([])
    contract_reg = ContractRegistry.from_file(contracts_path) if contracts_path.exists() else ContractRegistry([])
    real_assets = load_real_assets(DATA)
    if real_assets is None:
        sys.exit("real_tools.json missing — run scripts/parse_function_xlsx.py first")

    raw = [json.loads(l) for l in (DATA / "sft_train_real.jsonl").open(encoding="utf-8") if l.strip()]
    domain = [_format_real(s, tool_reg, contract_reg, real_assets) for s in raw]
    rng.shuffle(domain)
    eval_rows, train_domain = domain[: args.eval_n], domain[args.eval_n:]

    warmup = []
    wfile = Path(args.warmup_file)
    if args.warmup_n and wfile.exists():
        pool = [json.loads(l) for l in wfile.open(encoding="utf-8") if l.strip()]
        rng.shuffle(pool)
        warmup = [_project_warmup(r) for r in pool[: args.warmup_n]]
    elif args.warmup_n:
        print(f"WARNING: {wfile} missing — building domain-only train (no warmup mix)")

    train = train_domain + warmup
    rng.shuffle(train)

    out_train = DATA / "sft_train_real_with_warmup.jsonl"
    out_eval = DATA / "sft_eval_real_messages.jsonl"
    with out_train.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_eval.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train: {len(train)} (real-domain {len(train_domain)} + warmup {len(warmup)}) -> {out_train.name}")
    print(f"eval:  {len(eval_rows)} held-out real -> {out_eval.name}")


if __name__ == "__main__":
    main()
