"""Build data/eval_real_masked.jsonl from data/eval_real_seen.jsonl (RQ3 — masking).

Deterministic, no LLM. Masks the function name to func_X and embeds the masked
tool schema in the sample so the model must read the description/schema, not the
name. Derived from eval_real_seen (already held out of training) so there is NO
train/eval leak — the base instruction is itself a held-out eval query.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    tools = {t["name"]: t for t in json.loads((DATA / "real_tools.json").read_text(encoding="utf-8"))}
    seen = [json.loads(l) for l in (DATA / "eval_real_seen.jsonl").open(encoding="utf-8") if l.strip()]

    masked = []
    for i, base in enumerate(seen):
        if base.get("expected_action") != "call_function":
            continue
        tool = tools.get(base["gold_call"]["tool_name"])
        if tool is None:
            continue
        name = f"func_{i + 1}"
        masked.append({
            "id": f"real_eval_mask_{i:03d}",
            "source": "real_tool_xlsx",
            "split": "eval_real_masked",
            "scenario": "function_name_masking",
            "scenario_family": "masking",
            "instruction": f"Dùng {name} để: {base['instruction']}",
            "customer_verified": True,
            "expected_action": "call_function",
            "generator": "make_real_eval_masked",
            "gold_call": {"tool_name": name, "arguments": copy.deepcopy(base["gold_call"]["arguments"])},
            "masked_tool": {
                "name": name,
                "description": tool["description"],  # the signal the model must read
                "parameters": copy.deepcopy(tool["parameters"]),
                "status": "active",
                "deprecated": False,
            },
        })

    out = DATA / "eval_real_masked.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in masked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(masked)} masked eval samples → {out}")


if __name__ == "__main__":
    main()
